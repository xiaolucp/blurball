"""
Trajectory-based Event Detection v2: Train with PREDICTED ball positions.

Step 1: Run ball detection on all event frames, cache predicted positions
Step 2: Train MLP on predicted positions (closing the domain gap)
"""

import os
import sys
import json
import logging
import random
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from trajectory_event import (
    TrajectoryEventMLP, interpolate_missing, get_table_bbox_from_seg,
    TrajectoryEventDataset, EVENT_MAP, OFFSETS, IMG_W, IMG_H
)
from traj_event_predicted import soft_argmax_2d, load_ball_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def cache_predicted_positions(ball_model, data_root, split, cache_path, device='cuda'):
    """Run ball detection on all event frames, save to cache."""
    ann_root = os.path.join(data_root, split, 'annotations')
    images_root = os.path.join(data_root, split, 'images')
    games = sorted([d for d in os.listdir(ann_root) if d.startswith('game_')])
    
    img_size = (288, 512)
    h, w = img_size
    all_predictions = {}  # {game: {frame_id: (x_norm, y_norm, conf)}}
    
    for game in tqdm(games, desc=f'Ball detection [{split}]'):
        ann_dir = os.path.join(ann_root, game)
        img_dir = os.path.join(images_root, game)
        ep = os.path.join(ann_dir, 'events_markup.json')
        if not os.path.exists(ep):
            continue
        
        with open(ep) as f:
            events = json.load(f)
        
        all_frames = sorted([fn for fn in os.listdir(img_dir) if fn.endswith('.jpg')])
        n_frames = len(all_frames)
        
        game_preds = {}
        
        # Collect all frame IDs we need to predict
        needed_fids = set()
        for fid_str in events:
            fid = int(fid_str)
            for offset in OFFSETS:
                needed_fids.add(fid + offset)
        
        # Batch predict: for each needed frame, run 3-frame ball detection
        for target_fid in sorted(needed_fids):
            if target_fid < 1 or target_fid >= n_frames - 1:
                game_preds[target_fid] = (0.0, 0.0, 0.0)
                continue
            
            frames = []
            for offset in [-1, 0, 1]:
                idx = target_fid + offset
                fp = os.path.join(img_dir, all_frames[idx])
                img = cv2.imread(fp)
                if img is None:
                    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img.shape[:2]
                img = cv2.resize(img, (w, h))
                img = img.astype(np.float32) / 255.0
                frames.append(img)
            
            stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
            input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = ball_model(input_tensor)
                if isinstance(out, dict):
                    heatmap = out[0][0, 1]
                else:
                    heatmap = out[0, 1]
            
            x, y, conf = soft_argmax_2d(heatmap)
            # Scale to original coords then normalize
            x_norm = (x / w * orig_w) / IMG_W
            y_norm = (y / h * orig_h) / IMG_H
            game_preds[target_fid] = (x_norm, y_norm, conf)
        
        all_predictions[game] = game_preds
    
    with open(cache_path, 'wb') as f:
        pickle.dump(all_predictions, f)
    log.info(f"Cached predictions to {cache_path}")
    return all_predictions


def build_predicted_trajectory_samples(data_root, split, predictions,
                                       min_coverage=3, conf_threshold=0.3,
                                       noise_std=0.0):
    """Build trajectory samples from cached predicted positions."""
    ann_root = os.path.join(data_root, split, 'annotations')
    games = sorted([d for d in os.listdir(ann_root) if d.startswith('game_')])
    
    samples = []
    skipped = 0
    
    for game in games:
        ann_dir = os.path.join(ann_root, game)
        ep = os.path.join(ann_dir, 'events_markup.json')
        if not os.path.exists(ep) or game not in predictions:
            continue
        
        with open(ep) as f:
            events = json.load(f)
        
        game_preds = predictions[game]
        seg_dir = os.path.join(ann_dir, 'segmentation_masks')
        
        # Table bbox
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
            
            positions = []
            coverage = 0
            for offset in OFFSETS:
                target_fid = fid + offset
                if target_fid in game_preds:
                    x, y, conf = game_preds[target_fid]
                    if conf > conf_threshold:
                        # Add noise for augmentation
                        if noise_std > 0:
                            x += random.gauss(0, noise_std)
                            y += random.gauss(0, noise_std)
                        positions.append((x, y, 1.0))
                        coverage += 1
                    else:
                        positions.append((0.0, 0.0, 0.0))
                else:
                    positions.append((0.0, 0.0, 0.0))
            
            if coverage < min_coverage:
                skipped += 1
                continue
            
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
    
    log.info(f"[{split}] Built {len(samples)} samples (skipped {skipped})")
    return samples


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for feat, label in loader:
        feat, label = feat.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(feat)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feat.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += feat.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    preds_all, labels_all = [], []
    for feat, label in loader:
        feat, label = feat.to(device), label.to(device)
        logits = model(feat)
        loss = criterion(logits, label)
        total_loss += loss.item() * feat.size(0)
        pred = logits.argmax(1)
        correct += (pred == label).sum().item()
        total += feat.size(0)
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(label.cpu().tolist())
    
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(preds_all, labels_all):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1
    
    class_acc = {}
    for c in range(3):
        class_acc[c] = class_correct[c] / class_total[c] if class_total[c] > 0 else 0
    
    return total_loss / total, correct / total, class_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_root', default='/home/liuzhenlu/cyclex/TOTNet/dataset')
    parser.add_argument('--ball_ckpt', default='../weights/blurball_models/blurball_best')
    parser.add_argument('--cache_dir', default='../checkpoints/traj_event_v2')
    parser.add_argument('--save_dir', default='../checkpoints/traj_event_v2')
    parser.add_argument('--min_coverage', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--skip_cache', action='store_true', help='Use existing cache')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    
    # Step 1: Cache predicted positions
    cache_path = os.path.join(args.cache_dir, 'predicted_positions.pkl')
    if os.path.exists(cache_path) and args.skip_cache:
        log.info(f"Loading cached predictions from {cache_path}")
        with open(cache_path, 'rb') as f:
            predictions = pickle.load(f)
    else:
        log.info("Running ball detection to cache positions...")
        ball_model = load_ball_model(args.ball_ckpt, device=device)
        predictions = cache_predicted_positions(
            ball_model, args.data_root, 'training', cache_path, device=device)
        del ball_model
        torch.cuda.empty_cache()
    
    # Step 2: Build training samples from predicted positions
    train_samples = build_predicted_trajectory_samples(
        args.data_root, 'training', predictions,
        min_coverage=args.min_coverage)
    
    dist = Counter(s['event'] for s in train_samples)
    log.info(f"Class dist: bounce={dist[0]}, net={dist[1]}, empty={dist[2]}")
    
    # Split
    train_s, val_s = train_test_split(
        train_samples, test_size=0.15, random_state=args.seed, shuffle=True)
    
    train_ds = TrajectoryEventDataset(train_s, augment=True)
    val_ds = TrajectoryEventDataset(val_s, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Class weights
    counts = [dist[0], dist[1], dist[2]]
    total = sum(counts)
    weights = torch.tensor([total / (3 * max(c, 1)) for c in counts], dtype=torch.float32).to(device)
    log.info(f"Class weights: {weights.tolist()}")
    
    # Model
    sample_feat, _ = train_ds[0]
    input_dim = sample_feat.shape[0]
    log.info(f"Feature dim: {input_dim}")
    
    model = TrajectoryEventMLP(
        input_dim=input_dim, 
        hidden_dims=(256, 128, 64),  # slightly bigger
        num_classes=3, 
        dropout=0.4
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, class_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch+1}/{args.epochs}  LR={lr:.2e}")
        log.info(f"  Train | loss={train_loss:.4f} acc={train_acc*100:.1f}%")
        log.info(f"  Val   | loss={val_loss:.4f} acc={val_acc*100:.1f}% "
                 f"bounce={class_acc[0]*100:.1f}% net={class_acc[1]*100:.1f}% empty={class_acc[2]*100:.1f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best.pth'))
            log.info(f"  ★ Best (val_loss={val_loss:.4f}, acc={val_acc*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final: evaluate best model on full predicted dataset (as "test")
    log.info("\n=== Final evaluation (best model) ===")
    ckpt = torch.load(os.path.join(args.save_dir, 'best.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    
    # Evaluate on val set
    val_loss, val_acc, class_acc = eval_epoch(model, val_loader, criterion, device)
    log.info(f"Val  | acc={val_acc*100:.1f}% "
             f"bounce={class_acc[0]*100:.1f}% net={class_acc[1]*100:.1f}% empty={class_acc[2]*100:.1f}%")
    
    # Also evaluate GT-trained MLP on predicted data for comparison
    gt_mlp_path = '../checkpoints/traj_event/best.pth'
    if os.path.exists(gt_mlp_path):
        log.info("\n--- Comparison: GT-trained MLP on predicted data ---")
        gt_ckpt = torch.load(gt_mlp_path, map_location=device, weights_only=False)
        gt_model = TrajectoryEventMLP(input_dim=input_dim, num_classes=3).to(device)
        gt_model.load_state_dict(gt_ckpt['state_dict'])
        gt_loss, gt_acc, gt_class = eval_epoch(gt_model, val_loader, criterion, device)
        log.info(f"GT-MLP on pred data | acc={gt_acc*100:.1f}% "
                 f"bounce={gt_class[0]*100:.1f}% net={gt_class[1]*100:.1f}% empty={gt_class[2]*100:.1f}%")


if __name__ == '__main__':
    main()
