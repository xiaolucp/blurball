"""
Evaluate trajectory event MLP using PREDICTED ball positions from BlurBall model.

Pipeline:
  1. Load BlurBall (ball detection) model
  2. For each event frame, run ball detection on 9 frames (5 before + center + 3 after)
  3. Extract (x, y) from heatmap via soft-argmax
  4. Feed trajectory + table features into trained MLP
  5. Compare with GT labels
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from trajectory_event import (
    TrajectoryEventMLP, interpolate_missing, get_table_bbox_from_seg,
    EVENT_MAP, OFFSETS, IMG_W, IMG_H
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def soft_argmax_2d(heatmap):
    """Extract (x, y) from heatmap via soft-argmax.
    
    Args:
        heatmap: [H, W] tensor
    Returns:
        (x, y) in pixel coordinates, confidence
    """
    h, w = heatmap.shape
    # Apply sigmoid to convert logits to probabilities
    hm = torch.sigmoid(heatmap)
    
    # Confidence = max value
    conf = hm.max().item()
    
    if conf < 0.1:
        return 0.0, 0.0, 0.0  # No detection
    
    # Soft-argmax
    hm_flat = hm.reshape(-1)
    hm_soft = F.softmax(hm_flat * 10, dim=0)  # temperature=10 for sharper peak
    
    coords_y = torch.arange(h, dtype=torch.float32, device=heatmap.device)
    coords_x = torch.arange(w, dtype=torch.float32, device=heatmap.device)
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
    
    x = (hm_soft * grid_x.reshape(-1)).sum().item()
    y = (hm_soft * grid_y.reshape(-1)).sum().item()
    
    return x, y, conf


def load_ball_model(ckpt_path, device='cuda'):
    """Load pre-trained BlurBall model."""
    from omegaconf import OmegaConf
    from models.blurball import BlurBall
    
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'model', 'blurball.yaml')
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg['frames_in'] = 3
    cfg['frames_out'] = 3
    cfg = OmegaConf.create(cfg)
    
    model = BlurBall(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model


def predict_ball_position(ball_model, frame_paths, center_idx, device='cuda',
                          img_size=(288, 512)):
    """Run ball detection on 3 consecutive frames centered at center_idx.
    
    Returns (x, y, conf) in original image coordinates.
    """
    h, w = img_size
    
    # Load 3 frames: center_idx-1, center_idx, center_idx+1
    frames = []
    for offset in [-1, 0, 1]:
        idx = center_idx + offset
        if 0 <= idx < len(frame_paths):
            img = cv2.imread(frame_paths[idx])
            if img is None:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        frames.append(img)
    
    # Stack: [3, H, W, 3] -> [1, 9, H, W]
    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = ball_model(input_tensor)
        # out is dict: {0: [1, 3, H, W]}
        if isinstance(out, dict):
            heatmap = out[0][0, 1]  # center frame (index 1 of 3)
        else:
            heatmap = out[0, 1]
    
    x, y, conf = soft_argmax_2d(heatmap)
    
    # Scale back to original image coordinates
    x_orig = x / w * orig_w
    y_orig = y / h * orig_h
    
    return x_orig, y_orig, conf


def build_predicted_samples(ball_model, data_root, split='training',
                            min_coverage=5, device='cuda'):
    """Build trajectory samples using PREDICTED ball positions."""
    ann_root = os.path.join(data_root, split, 'annotations')
    images_root = os.path.join(data_root, split, 'images')
    games = sorted([d for d in os.listdir(ann_root) if d.startswith('game_')])
    
    samples = []
    skipped = 0
    
    for game in tqdm(games, desc=f'Predicting [{split}]'):
        ann_dir = os.path.join(ann_root, game)
        img_dir = os.path.join(images_root, game)
        ep = os.path.join(ann_dir, 'events_markup.json')
        if not os.path.exists(ep):
            continue
        
        with open(ep) as f:
            events = json.load(f)
        
        # List all frame paths
        all_frames = sorted([os.path.join(img_dir, fn)
                            for fn in os.listdir(img_dir) if fn.endswith('.jpg')])
        n_frames = len(all_frames)
        
        seg_dir = os.path.join(ann_dir, 'segmentation_masks')
        
        # Pre-compute table bbox
        game_table_bbox = None
        if os.path.isdir(seg_dir):
            for fn in sorted(os.listdir(seg_dir))[:20]:
                fid = int(fn.replace('.png', ''))
                bbox = get_table_bbox_from_seg(seg_dir, fid)
                if bbox is not None:
                    game_table_bbox = bbox
                    break
        
        for fid_str, event_str in events.items():
            fid = int(fid_str)
            event_label = EVENT_MAP.get(event_str, 2)
            
            # Check bounds
            if fid - 5 < 1 or fid + 3 >= n_frames - 1:
                skipped += 1
                continue
            
            # Predict ball position for each of the 9 offsets
            positions = []
            coverage = 0
            for offset in OFFSETS:
                target_fid = fid + offset
                x, y, conf = predict_ball_position(
                    ball_model, all_frames, target_fid, device=device)
                
                if conf > 0.3:
                    positions.append((x / IMG_W, y / IMG_H, 1.0))
                    coverage += 1
                else:
                    positions.append((0.0, 0.0, 0.0))
            
            if coverage < min_coverage:
                skipped += 1
                continue
            
            # Interpolate missing
            positions = interpolate_missing(positions)
            
            # Table features
            table_bbox = None
            if os.path.isdir(seg_dir):
                table_bbox = get_table_bbox_from_seg(seg_dir, fid)
            if table_bbox is None:
                table_bbox = game_table_bbox
            
            if table_bbox is not None:
                tx1, ty1, tx2, ty2 = table_bbox
                table_feat = [tx1/IMG_W, ty1/IMG_H, tx2/IMG_W, ty2/IMG_H,
                              (tx1+tx2)/2/IMG_W, (ty1+ty2)/2/IMG_H]
            else:
                table_feat = [0.0] * 6
            
            samples.append({
                'positions': positions,
                'table_feat': table_feat,
                'event': event_label,
                'game': game,
                'frame_id': fid,
            })
    
    log.info(f"[{split}] Built {len(samples)} predicted trajectory samples "
             f"(skipped {skipped})")
    return samples


def make_features(positions, table_feat):
    """Convert positions + table_feat to feature vector (same as TrajectoryEventDataset)."""
    pos_flat = []
    for x, y, v in positions:
        pos_flat.extend([x, y, v])
    
    velocities = []
    for i in range(1, len(positions)):
        x1, y1, v1 = positions[i]
        x0, y0, v0 = positions[i-1]
        if v0 > 0.3 and v1 > 0.3:
            velocities.extend([x1 - x0, y1 - y0])
        else:
            velocities.extend([0.0, 0.0])
    
    accels = []
    for i in range(1, len(velocities) // 2):
        dvx = velocities[i*2] - velocities[(i-1)*2]
        dvy = velocities[i*2+1] - velocities[(i-1)*2+1]
        accels.extend([dvx, dvy])
    
    cx, cy, cv = positions[5]
    if len(table_feat) == 6 and table_feat[4] > 0:
        table_rel = [cx - table_feat[4], cy - table_feat[5]]
        table_half = [1.0 if cx < table_feat[4] else 0.0]
        tw = max(table_feat[2] - table_feat[0], 0.01)
        th = max(table_feat[3] - table_feat[1], 0.01)
        table_norm = [(cx - table_feat[0]) / tw, (cy - table_feat[1]) / th]
    else:
        table_rel = [0.0, 0.0]
        table_half = [0.5]
        table_norm = [0.5, 0.5]
    
    features = pos_flat + velocities + accels + table_feat + table_rel + table_half + table_norm
    return features


def evaluate(mlp_model, samples, device='cuda'):
    """Evaluate MLP on predicted trajectory samples."""
    mlp_model.eval()
    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()
    
    for s in samples:
        features = make_features(s['positions'], s['table_feat'])
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = mlp_model(feat_tensor)
            pred = logits.argmax(1).item()
        
        label = s['event']
        total += 1
        class_total[label] += 1
        if pred == label:
            correct += 1
            class_correct[label] += 1
    
    acc = correct / total if total > 0 else 0
    class_acc = {}
    for c in range(3):
        class_acc[c] = class_correct[c] / class_total[c] if class_total[c] > 0 else 0
    
    return acc, class_acc, class_total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/liuzhenlu/cyclex/TOTNet/dataset')
    parser.add_argument('--ball_ckpt', default='../weights/blurball_models/blurball_best')
    parser.add_argument('--mlp_ckpt', default='../checkpoints/traj_event/best.pth')
    parser.add_argument('--min_coverage', type=int, default=5)
    parser.add_argument('--split', default='training', help='training or test')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    
    # Load ball detection model
    log.info("Loading ball detection model...")
    ball_model = load_ball_model(args.ball_ckpt, device=device)
    
    # Load MLP
    log.info("Loading trajectory MLP...")
    mlp_ckpt = torch.load(args.mlp_ckpt, map_location=device, weights_only=False)
    input_dim = 68
    mlp_model = TrajectoryEventMLP(input_dim=input_dim, num_classes=3).to(device)
    mlp_model.load_state_dict(mlp_ckpt['state_dict'])
    log.info(f"MLP trained at epoch {mlp_ckpt['epoch']}, "
             f"val_acc={mlp_ckpt.get('val_acc', 0)*100:.1f}%")
    
    # Build predicted samples
    log.info(f"Running ball detection on {args.split} set...")
    samples = build_predicted_samples(
        ball_model, args.data_root, args.split,
        min_coverage=args.min_coverage, device=device)
    
    # Also build GT samples for comparison
    from trajectory_event import build_trajectory_samples
    gt_samples = build_trajectory_samples(
        args.data_root, args.split, min_coverage=args.min_coverage)
    
    # Evaluate with predicted positions
    log.info("\n=== PREDICTED positions ===")
    acc, class_acc, class_total = evaluate(mlp_model, samples, device)
    log.info(f"Acc: {acc*100:.1f}%")
    log.info(f"  bounce: {class_acc[0]*100:.1f}% ({class_total[0]} samples)")
    log.info(f"  net:    {class_acc[1]*100:.1f}% ({class_total[1]} samples)")
    log.info(f"  empty:  {class_acc[2]*100:.1f}% ({class_total[2]} samples)")
    
    # Evaluate with GT positions
    log.info("\n=== GT positions (reference) ===")
    acc_gt, class_acc_gt, class_total_gt = evaluate(mlp_model, gt_samples, device)
    log.info(f"Acc: {acc_gt*100:.1f}%")
    log.info(f"  bounce: {class_acc_gt[0]*100:.1f}% ({class_total_gt[0]} samples)")
    log.info(f"  net:    {class_acc_gt[1]*100:.1f}% ({class_total_gt[1]} samples)")
    log.info(f"  empty:  {class_acc_gt[2]*100:.1f}% ({class_total_gt[2]} samples)")


if __name__ == '__main__':
    main()
