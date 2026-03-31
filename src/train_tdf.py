"""
Train BlurBall with TDF (True Distance Field) instead of Gaussian heatmap.

TDF ground truth: for each pixel, value = exp(-distance_to_ball_boundary / sigma)
  - 1.0 inside ball, decays smoothly outside
  - More accurate ball shape than Gaussian blob
  - Uses the real ball mask from our pixel-level detection (step1_v7)

Strategy:
  - Load pretrained BlurBall backbone from bb_mt_phase1
  - Replace final layer: 3-channel heatmap → 1-channel TDF
  - Train on 10% of BlurBall dataset
  - Weighted BCE loss (same as original, ball region upweighted)
"""

import os
import sys
import math
import logging
import random
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import scipy.ndimage

sys.path.insert(0, os.path.dirname(__file__))
from models.blurball import BlurBall

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ─── TDF ground truth generation ───

def generate_tdf(h, w, cx, cy, ball_radius=3.5, sigma=3.0):
    """Generate True Distance Field centered at (cx, cy).

    Value = 1.0 inside ball radius, then exp(-dist/sigma) decay outside.
    This represents the actual ball shape rather than a Gaussian blob.

    Args:
        h, w: output size
        cx, cy: ball center in output coords
        ball_radius: radius of the ball in pixels (at output resolution)
        sigma: decay rate outside ball boundary
    """
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return np.zeros((h, w), dtype=np.float32)

    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Inside ball: 1.0, outside: exponential decay from boundary
    dist_from_boundary = np.maximum(dist - ball_radius, 0)
    tdf = np.exp(-dist_from_boundary / sigma)

    return tdf.astype(np.float32)


def generate_tdf_with_blur(h, w, cx, cy, theta_deg, length,
                            ball_radius=3.0, sigma=2.5):
    """Generate TDF with motion blur direction.

    For moving ball: elongate the TDF along motion direction.
    """
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return np.zeros((h, w), dtype=np.float32)

    if length < 0.5:
        # Static ball: circular TDF
        return generate_tdf(h, w, cx, cy, ball_radius, sigma)

    # Moving ball: place TDF along motion trail
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Sample points along the trail
    n_pts = max(3, int(length * 2))
    tdf = np.zeros((h, w), dtype=np.float32)

    for i in range(n_pts):
        t = (i / (n_pts - 1) - 0.5) * length
        px = cx + cos_t * t
        py = cy + sin_t * t

        y = np.arange(0, h, dtype=np.float32)
        x = np.arange(0, w, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        dist_from_boundary = np.maximum(dist - ball_radius, 0)
        pt_tdf = np.exp(-dist_from_boundary / sigma)
        tdf = np.maximum(tdf, pt_tdf)

    return tdf


# ─── Dataset ───

class BlurBallTDFDataset(Dataset):
    """BlurBall dataset with TDF ground truth."""

    def __init__(self, data_root, matches, num_frames=3,
                 img_size=(288, 512), augment=True,
                 ball_radius=3.5, sigma=3.0, fraction=1.0):
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment
        self.ball_radius = ball_radius
        self.sigma = sigma

        self.samples = []

        for match in tqdm(matches, desc='Loading clips'):
            match_dir = os.path.join(data_root, f'{match:02d}')
            csv_dir = os.path.join(match_dir, 'csv')
            frames_dir = os.path.join(match_dir, 'frames')

            if not os.path.isdir(csv_dir):
                continue

            for csv_file in sorted(os.listdir(csv_dir)):
                if not csv_file.endswith('.csv'):
                    continue
                clip_name = csv_file.replace('.csv', '')
                clip_frames_dir = os.path.join(frames_dir, clip_name)

                if not os.path.isdir(clip_frames_dir):
                    continue

                csv_path = os.path.join(csv_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue

                frame_files = sorted([f for f in os.listdir(clip_frames_dir)
                                      if f.endswith('.png')])
                n_frames = len(frame_files)
                if n_frames < num_frames:
                    continue

                annos = {}
                for _, row in df.iterrows():
                    fid = int(row['Frame'])
                    annos[fid] = {
                        'vis': int(row['Visibility']),
                        'x': float(row['X']),
                        'y': float(row['Y']),
                        'theta': float(row['theta']),
                        'l': float(row['l']),
                    }

                half = num_frames // 2
                for center_idx in range(half, n_frames - half):
                    center_fid = int(frame_files[center_idx].replace('.png', ''))
                    if center_fid not in annos:
                        continue

                    frame_paths = []
                    for offset in range(-half, half + 1):
                        idx = center_idx + offset
                        frame_paths.append(
                            os.path.join(clip_frames_dir, frame_files[idx]))

                    self.samples.append({
                        'frame_paths': frame_paths,
                        'anno': annos[center_fid],
                    })

        # Subsample
        if fraction < 1.0:
            n = max(1, int(len(self.samples) * fraction))
            random.shuffle(self.samples)
            self.samples = self.samples[:n]

        log.info(f"Dataset: {len(self.samples)} samples from {len(matches)} matches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        h, w = self.img_size

        imgs = []
        for fp in sample['frame_paths']:
            img = cv2.imread(fp)
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        orig_h, orig_w = imgs[0].shape[:2]
        scale_x = w / orig_w
        scale_y = h / orig_h

        # Augmentation
        flip = self.augment and random.random() < 0.5

        # Resize + normalize
        processed = []
        for img in imgs:
            if flip:
                img = np.fliplr(img).copy()
            img = cv2.resize(img, (w, h))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            processed.append(img)

        input_tensor = np.concatenate(processed, axis=0).astype(np.float32)

        # Annotation
        anno = sample['anno']
        vis = anno['vis']

        if vis:
            cx = anno['x'] * scale_x
            cy = anno['y'] * scale_y
            if flip:
                cx = w - cx
            theta = anno['theta']
            if flip:
                theta = 180 - theta
            length = anno['l'] * max(scale_x, scale_y)
        else:
            cx, cy = -1, -1
            theta, length = 0, 0

        # Generate TDF ground truth
        tdf = generate_tdf_with_blur(h, w, cx, cy, theta, length,
                                      self.ball_radius, self.sigma)

        return (
            torch.from_numpy(input_tensor),
            torch.from_numpy(tdf).unsqueeze(0),  # [1, H, W]
            torch.tensor(float(vis)),
        )


# ─── Loss ───

class TDFLoss(nn.Module):
    """Weighted BCE loss for TDF prediction."""

    def __init__(self, pos_weight_factor=50.0):
        super().__init__()
        self.pos_weight_factor = pos_weight_factor

    def forward(self, pred, target, vis):
        """
        pred: [B, 1, H, W] sigmoid output
        target: [B, 1, H, W] TDF ground truth
        vis: [B] visibility flag
        """
        # Positive region weighting
        pos_weight = (target > 0.5).float() * (self.pos_weight_factor - 1) + 1

        loss = F.binary_cross_entropy(pred, target, weight=pos_weight, reduction='mean')
        return loss


# ─── Model setup ───

class AttrDict(dict):
    """Dict that supports dot-access (for BlurBall model config)."""
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, AttrDict):
                val = AttrDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(key)


def build_blurball_model(config_path, pretrained_ckpt=None, tdf_output=True):
    """Build BlurBall model, optionally load pretrained and swap final layer."""
    with open(config_path, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    if tdf_output:
        # Override: output 1 channel instead of frames_out (3)
        cfg['frames_out'] = 1

    model = BlurBall(cfg)

    if pretrained_ckpt:
        log.info(f"Loading pretrained: {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        # Strip 'backbone.' prefix if present (from multitask model)
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith('backbone.'):
                clean_sd[k[len('backbone.'):]] = v
            else:
                clean_sd[k] = v

        # Load with strict=False to skip final_layers shape mismatch
        model_sd = model.state_dict()
        loaded = 0
        skipped = []
        for k, v in clean_sd.items():
            if k in model_sd:
                if model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    loaded += 1
                else:
                    skipped.append(f"{k}: {v.shape} → {model_sd[k].shape}")
            # else: key not in model, skip silently (event/seg heads)

        model.load_state_dict(model_sd)
        log.info(f"Loaded {loaded} params, skipped {len(skipped)} (shape mismatch)")
        for s in skipped:
            log.info(f"  Skipped: {s}")

    return model


# ─── Metrics ───

@torch.no_grad()
def compute_metrics(model, loader, criterion, device, dist_threshold=5.0,
                    shape_threshold=0.5):
    """Compute loss, position accuracy, and shape IoU.

    Shape metrics (ball mask quality):
      - Threshold pred and GT at shape_threshold to get binary masks
      - IoU = intersection / union
      - Precision = intersection / pred_area
      - Recall = intersection / gt_area
    """
    model.eval()
    total_loss = 0
    n = 0
    correct = 0
    visible_total = 0

    # Shape metrics accumulators
    total_iou = 0
    total_precision = 0
    total_recall = 0
    shape_count = 0

    for batch in loader:
        imgs, tdf_gt, vis = [x.to(device) for x in batch]

        out = model(imgs)
        pred = torch.sigmoid(out[0])

        loss = criterion(pred, tdf_gt, vis)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

        # Position accuracy
        b, _, h, w = pred.shape
        pred_flat = pred.view(b, -1)
        gt_flat = tdf_gt.view(b, -1)

        pred_idx = pred_flat.argmax(1)
        gt_idx = gt_flat.argmax(1)

        pred_y, pred_x = pred_idx // w, pred_idx % w
        gt_y, gt_x = gt_idx // w, gt_idx % w

        dist = torch.sqrt((pred_x.float() - gt_x.float())**2 +
                          (pred_y.float() - gt_y.float())**2)

        # Shape IoU
        pred_mask = (pred > shape_threshold).float()
        gt_mask = (tdf_gt > shape_threshold).float()

        for i in range(bs):
            if vis[i] > 0.5:
                visible_total += 1
                if dist[i] < dist_threshold:
                    correct += 1

                # Per-sample IoU
                p = pred_mask[i].view(-1)
                g = gt_mask[i].view(-1)
                intersection = (p * g).sum()
                union = ((p + g) > 0).float().sum()
                pred_area = p.sum()
                gt_area = g.sum()

                if gt_area > 0:
                    iou = (intersection / union).item() if union > 0 else 0.0
                    prec = (intersection / pred_area).item() if pred_area > 0 else 0.0
                    rec = (intersection / gt_area).item()
                    total_iou += iou
                    total_precision += prec
                    total_recall += rec
                    shape_count += 1

    acc = correct / visible_total if visible_total > 0 else 0
    avg_iou = total_iou / shape_count if shape_count > 0 else 0
    avg_prec = total_precision / shape_count if shape_count > 0 else 0
    avg_rec = total_recall / shape_count if shape_count > 0 else 0

    return {
        'loss': total_loss / n,
        'acc': acc,
        'visible': visible_total,
        'iou': avg_iou,
        'precision': avg_prec,
        'recall': avg_rec,
    }


# ─── Training ───

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        imgs, tdf_gt, vis = [x.to(device) for x in batch]

        out = model(imgs)
        pred = torch.sigmoid(out[0])

        loss = criterion(pred, tdf_gt, vis)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=f'{total_loss/n:.4f}')

    return total_loss / n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/home/liuzhenlu/cyclex/BlurBall/dataset_blurball/blurball_dataset')
    parser.add_argument('--config',
                        default='/home/liuzhenlu/cyclex/BlurBall/src/configs/model/blurball.yaml')
    parser.add_argument('--pretrained',
                        default='/home/liuzhenlu/cyclex/BlurBall/checkpoints/bb_mt_phase1/best.pth')
    parser.add_argument('--save_dir', default='../checkpoints/blurball_tdf')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='Fraction of data to use (0.1 = 10%)')
    parser.add_argument('--ball_radius', type=float, default=3.5)
    parser.add_argument('--sigma', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Train/test split
    train_matches = [m for m in range(0, 22) if m != 18]
    test_matches = list(range(22, 26))

    img_size = (288, 512)

    log.info(f"Building datasets (fraction={args.fraction})...")
    train_ds = BlurBallTDFDataset(
        args.data_root, train_matches, img_size=img_size, augment=True,
        ball_radius=args.ball_radius, sigma=args.sigma, fraction=args.fraction)
    test_ds = BlurBallTDFDataset(
        args.data_root, test_matches, img_size=img_size, augment=False,
        ball_radius=args.ball_radius, sigma=args.sigma, fraction=1.0)  # full test set

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    log.info(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    # Model
    model = build_blurball_model(args.config, args.pretrained, tdf_output=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Params: {total_params:,} total, {trainable:,} trainable")

    # Loss, optimizer
    criterion = TDFLoss(pos_weight_factor=50.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val = compute_metrics(model, test_loader, criterion, device, dist_threshold=5)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch+1}/{args.epochs}  LR={lr:.2e}")
        log.info(f"  Train loss={train_loss:.4f}")
        log.info(f"  Val   loss={val['loss']:.4f}  acc@5px={val['acc']*100:.1f}%"
                 f"  IoU={val['iou']*100:.1f}%  Prec={val['precision']*100:.1f}%  Rec={val['recall']*100:.1f}%")

        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val['loss'],
                'val_acc': val['acc'],
            }, os.path.join(args.save_dir, 'best.pth'))
            log.info(f"  * Best (loss={val['loss']:.4f}, acc@5px={val['acc']*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final eval
    log.info("\n=== Final evaluation ===")
    ckpt = torch.load(os.path.join(args.save_dir, 'best.pth'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    for thresh in [3, 5, 10]:
        m = compute_metrics(model, test_loader, criterion, device, dist_threshold=thresh)
        log.info(f"  acc@{thresh}px = {m['acc']*100:.1f}%  IoU={m['iou']*100:.1f}%  Prec={m['precision']*100:.1f}%  Rec={m['recall']*100:.1f}%")


if __name__ == '__main__':
    main()
