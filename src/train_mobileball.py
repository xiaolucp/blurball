"""
Train MobileBall v2 on BlurBall dataset.

Dataset format:
  {root}/{match}/frames/{clip}/{00000.png, ...}
  {root}/{match}/csv/{clip}.csv  (Frame, Visibility, X, Y, theta, l)

Joint supervision:
  - Heatmap: Gaussian blob at ball center (BCE loss)
  - l: motion blur length (Smooth L1)
  - theta: blur direction as (cos, sin) unit vector (Smooth L1)
"""

import os
import sys
import json
import math
import logging
import random
import glob
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.mobileball_v2 import MobileBallV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ─── Heatmap generation ───

def gaussian_heatmap(h, w, cx, cy, sigma=2.5):
    """Generate Gaussian heatmap centered at (cx, cy)."""
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=np.float32)
    
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    
    hm = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return hm.astype(np.float32)


# ─── Dataset ───

class BlurBallDataset(Dataset):
    """BlurBall dataset for MobileBall v2 training.
    
    Returns 3 consecutive frames as input, with center frame's annotation
    as supervision target.
    """
    
    def __init__(self, data_root, matches, num_frames=3, 
                 img_size=(288, 512), augment=True, sigma=2.5):
        self.num_frames = num_frames
        self.img_size = img_size  # (H, W)
        self.augment = augment
        self.sigma = sigma
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
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
                
                # Load CSV
                csv_path = os.path.join(csv_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                except:
                    continue
                
                # List frames
                frame_files = sorted(os.listdir(clip_frames_dir))
                frame_files = [f for f in frame_files if f.endswith('.png')]
                n_frames = len(frame_files)
                
                if n_frames < num_frames:
                    continue
                
                # Build index: frame_id → annotation
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
                
                # Create samples: center frame = index 1 (of 0,1,2 for 3-frame)
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
        
        log.info(f"Dataset: {len(self.samples)} samples from {len(matches)} matches")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        h, w = self.img_size
        
        # Load frames
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
        
        # Augmentation: random horizontal flip
        flip = False
        if self.augment and random.random() < 0.5:
            flip = True
            imgs = [np.fliplr(img) for img in imgs]
        
        # Augmentation: color jitter
        if self.augment and random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            for i in range(len(imgs)):
                img = imgs[i].astype(np.float32)
                img = img * brightness
                img = (img - img.mean()) * contrast + img.mean()
                imgs[i] = np.clip(img, 0, 255).astype(np.uint8)
        
        # Resize and normalize
        processed = []
        for img in imgs:
            img = cv2.resize(img, (w, h))
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)  # [3, H, W]
            processed.append(img)
        
        # Stack frames: [num_frames*3, H, W]
        input_tensor = np.concatenate(processed, axis=0).astype(np.float32)
        
        # Annotation for center frame
        anno = sample['anno']
        vis = anno['vis']
        
        if vis:
            # Scale coordinates to model input space
            cx = anno['x'] * scale_x
            cy = anno['y'] * scale_y
            
            if flip:
                cx = w - cx
            
            # Length (in pixels, scaled)
            length = anno['l'] * max(scale_x, scale_y)
            
            # Theta (degrees → cos, sin)
            theta_deg = anno['theta']
            if flip:
                theta_deg = 180 - theta_deg  # flip direction
            theta_rad = math.radians(theta_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
        else:
            cx, cy = -1, -1
            length = 0
            cos_t, sin_t = 0, 0
        
        # Generate heatmap target (full resolution)
        hm = gaussian_heatmap(h, w, cx, cy, sigma=self.sigma)
        
        # Length map: value at ball location
        length_map = np.zeros((h, w), dtype=np.float32)
        if vis and 0 <= int(cy) < h and 0 <= int(cx) < w:
            # Paint length in gaussian-weighted region around ball
            length_map = hm * length
        
        # Direction map: (cos, sin) at ball location
        dir_map = np.zeros((2, h, w), dtype=np.float32)
        if vis and 0 <= int(cy) < h and 0 <= int(cx) < w:
            dir_map[0] = hm * cos_t
            dir_map[1] = hm * sin_t
        
        # Visibility flag
        vis_tensor = np.float32(vis)
        
        return (
            torch.from_numpy(input_tensor),
            torch.from_numpy(hm).unsqueeze(0),        # [1, H, W]
            torch.from_numpy(length_map).unsqueeze(0), # [1, H, W]
            torch.from_numpy(dir_map),                  # [2, H, W]
            torch.tensor(vis_tensor),
        )


# ─── Loss ───

class MobileBallLoss(nn.Module):
    """Joint loss for heatmap + length + direction."""
    
    def __init__(self, hm_weight=1.0, length_weight=0.5, dir_weight=0.5):
        super().__init__()
        self.hm_weight = hm_weight
        self.length_weight = length_weight
        self.dir_weight = dir_weight
    
    def forward(self, hm_pred, l_pred, dir_pred, hm_gt, l_gt, dir_gt, vis):
        """
        All predictions and targets: [B, C, H, W]
        vis: [B] binary mask
        """
        # Heatmap: weighted BCE (all samples)
        pos_weight = (hm_gt > 0.5).float() * 49 + 1  # 50:1 weight
        hm_loss = F.binary_cross_entropy(hm_pred, hm_gt, weight=pos_weight, 
                                          reduction='mean')
        
        # Length and direction: only at ball location (weighted by GT heatmap)
        # This focuses the loss on the ball region, not the entire image
        vis_mask = vis.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        # Use GT heatmap as spatial weight (ball region only)
        hm_weight = hm_gt  # [B, 1, H, W], peaks at ball center
        
        if vis.sum() > 0:
            # Weighted L1: only penalize at ball location
            weight_sum = (hm_weight * vis_mask).sum().clamp(min=1e-6)
            l_loss = (F.smooth_l1_loss(l_pred, l_gt, reduction='none') * hm_weight * vis_mask).sum() / weight_sum
            dir_loss = (F.smooth_l1_loss(dir_pred, dir_gt, reduction='none') * hm_weight * vis_mask).sum() / weight_sum
        else:
            l_loss = torch.tensor(0.0, device=hm_pred.device)
            dir_loss = torch.tensor(0.0, device=hm_pred.device)
        
        total = self.hm_weight * hm_loss + self.length_weight * l_loss + self.dir_weight * dir_loss
        return total, hm_loss, l_loss, dir_loss


# ─── Metrics ───

@torch.no_grad()
def compute_metrics(model, loader, criterion, device, dist_threshold=10.0):
    """Compute loss and ball detection accuracy."""
    model.eval()
    total_loss = 0
    total_hm = 0
    total_l = 0
    total_dir = 0
    n = 0
    
    correct = 0  # within dist_threshold pixels
    visible_total = 0
    
    for batch in loader:
        imgs, hm_gt, l_gt, dir_gt, vis = [x.to(device) for x in batch]
        hm_pred, l_pred, dir_pred = model(imgs)
        
        loss, hm_loss, l_loss, dir_loss = criterion(
            hm_pred, l_pred, dir_pred, hm_gt, l_gt, dir_gt, vis)
        
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_hm += hm_loss.item() * bs
        total_l += l_loss.item() * bs
        total_dir += dir_loss.item() * bs
        n += bs
        
        # Position accuracy
        b, _, h, w = hm_pred.shape
        hm_flat = hm_pred.view(b, -1)
        
        # Predicted position (argmax)
        max_idx = hm_flat.argmax(1)
        pred_y = (max_idx // w).float()
        pred_x = (max_idx % w).float()
        
        # GT position (argmax of GT heatmap)
        gt_flat = hm_gt.view(b, -1)
        gt_max_idx = gt_flat.argmax(1)
        gt_y = (gt_max_idx // w).float()
        gt_x = (gt_max_idx % w).float()
        
        dist = torch.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        
        for i in range(bs):
            if vis[i] > 0.5:
                visible_total += 1
                if dist[i] < dist_threshold:
                    correct += 1
    
    acc = correct / visible_total if visible_total > 0 else 0
    return {
        'loss': total_loss / n,
        'hm_loss': total_hm / n,
        'l_loss': total_l / n,
        'dir_loss': total_dir / n,
        'acc': acc,
        'visible_total': visible_total,
    }


# ─── Training ───

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    
    for batch in loader:
        imgs, hm_gt, l_gt, dir_gt, vis = [x.to(device) for x in batch]
        
        hm_pred, l_pred, dir_pred = model(imgs)
        loss, _, _, _ = criterion(hm_pred, l_pred, dir_pred, hm_gt, l_gt, dir_gt, vis)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    
    return total_loss / n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', 
                        default='/home/liuzhenlu/cyclex/BlurBall/dataset_blurball/blurball_dataset')
    parser.add_argument('--save_dir', default='../checkpoints/mobileball_v2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img_h', type=int, default=288)
    parser.add_argument('--img_w', type=int, default=512)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    
    # Train/test split (same as BlurBall paper config, skip match 18 which has no videos)
    train_matches = [m for m in range(0, 22) if m != 18]  # 00-21 (excl 18)
    test_matches = list(range(22, 26))  # 22-25
    
    img_size = (args.img_h, args.img_w)
    
    log.info("Building training dataset...")
    train_ds = BlurBallDataset(args.data_root, train_matches, img_size=img_size, augment=True)
    log.info("Building test dataset...")
    test_ds = BlurBallDataset(args.data_root, test_matches, img_size=img_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    log.info(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    
    # Model
    model = MobileBallV2(num_frames=3, pretrained_backbone=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = MobileBallLoss(hm_weight=1.0, length_weight=0.5, dir_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = compute_metrics(model, test_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch+1}/{args.epochs}  LR={lr:.2e}")
        log.info(f"  Train | loss={train_loss:.4f}")
        log.info(f"  Val   | loss={val_metrics['loss']:.4f} "
                 f"hm={val_metrics['hm_loss']:.4f} "
                 f"l={val_metrics['l_loss']:.4f} "
                 f"dir={val_metrics['dir_loss']:.4f} "
                 f"acc@10px={val_metrics['acc']*100:.1f}%")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['acc'],
            }, os.path.join(args.save_dir, 'best.pth'))
            log.info(f"  ★ Best (loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final eval
    log.info("\n=== Final evaluation (best model) ===")
    ckpt = torch.load(os.path.join(args.save_dir, 'best.pth'), 
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    final = compute_metrics(model, test_loader, criterion, device, dist_threshold=10)
    log.info(f"Best epoch {ckpt['epoch']+1}")
    log.info(f"  acc@10px = {final['acc']*100:.1f}%")
    
    # Also check acc@5px
    final5 = compute_metrics(model, test_loader, criterion, device, dist_threshold=5)
    log.info(f"  acc@5px  = {final5['acc']*100:.1f}%")


if __name__ == '__main__':
    main()
