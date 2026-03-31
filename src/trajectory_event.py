"""
Trajectory-based Event Detection (quick prototype).

Input: 9 ball positions (5 before + center + 3 after) + table region features
Output: event class (bounce=0, net=1, empty=2)

No heatmaps, no CNN — just coordinates + MLP.
"""

import os
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

EVENT_MAP = {'bounce': 0, 'net': 1, 'empty_event': 2}
NUM_OFFSETS = 9  # -5, -4, -3, -2, -1, 0, +1, +2, +3
OFFSETS = list(range(-5, 4))  # [-5, -4, -3, -2, -1, 0, 1, 2, 3]

# Image dimensions (TTNet default)
IMG_W, IMG_H = 1920, 1080


def get_table_bbox_from_seg(seg_dir, frame_id):
    """Extract table bounding box from segmentation mask."""
    seg_path = os.path.join(seg_dir, f'{frame_id}.png')
    if not os.path.exists(seg_path):
        return None
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None
    # Table is green channel in TTNet seg masks
    if len(seg.shape) == 3:
        table_mask = seg[:, :, 1] > 127  # G channel = table
    else:
        return None
    ys, xs = np.where(table_mask)
    if len(xs) == 0:
        return None
    return (xs.min(), ys.min(), xs.max(), ys.max())  # (x1, y1, x2, y2)


def build_trajectory_samples(data_root, split='training', min_coverage=6):
    """Build trajectory samples from TTNet annotations.
    
    Each sample: 9 ball positions + table info + event label.
    """
    images_root = os.path.join(data_root, split, 'images')
    ann_root = os.path.join(data_root, split, 'annotations')
    games = sorted([d for d in os.listdir(ann_root) if d.startswith('game_')])

    samples = []
    skipped = 0

    for game in games:
        ann_dir = os.path.join(ann_root, game)
        bp = os.path.join(ann_dir, 'ball_markup.json')
        ep = os.path.join(ann_dir, 'events_markup.json')
        if not os.path.exists(bp) or not os.path.exists(ep):
            continue

        with open(bp) as f:
            ball = json.load(f)
        with open(ep) as f:
            events = json.load(f)

        seg_dir = os.path.join(ann_dir, 'segmentation_masks')
        
        # Pre-compute a representative table bbox for this game
        # (use first available seg mask)
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

            # Collect ball positions for 9 frames
            positions = []  # [(x, y, vis), ...]
            coverage = 0
            for offset in OFFSETS:
                s = str(fid + offset)
                if s in ball and ball[s]['x'] >= 0 and ball[s]['y'] >= 0:
                    positions.append((ball[s]['x'] / IMG_W, ball[s]['y'] / IMG_H, 1.0))
                    coverage += 1
                else:
                    positions.append((0.0, 0.0, 0.0))

            if coverage < min_coverage:
                skipped += 1
                continue

            # Interpolate missing positions
            positions = interpolate_missing(positions)

            # Table bbox for this event frame (try specific frame, fallback to game-level)
            table_bbox = None
            if os.path.isdir(seg_dir):
                table_bbox = get_table_bbox_from_seg(seg_dir, fid)
            if table_bbox is None:
                table_bbox = game_table_bbox
            
            # Normalize table bbox
            if table_bbox is not None:
                tx1, ty1, tx2, ty2 = table_bbox
                table_feat = [tx1/IMG_W, ty1/IMG_H, tx2/IMG_W, ty2/IMG_H,
                              (tx1+tx2)/2/IMG_W, (ty1+ty2)/2/IMG_H]  # + center
            else:
                table_feat = [0.0] * 6

            samples.append({
                'positions': positions,     # 9 x (x, y, vis)
                'table_feat': table_feat,   # 6 floats
                'event': event_label,
                'game': game,
                'frame_id': fid,
            })

    log.info(f"[{split}] Built {len(samples)} trajectory samples "
             f"(skipped {skipped} low-coverage), {len(games)} games")
    return samples


def interpolate_missing(positions):
    """Linear interpolation for missing ball positions."""
    n = len(positions)
    result = list(positions)
    
    # Extract x, y, vis arrays
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    vis = [p[2] for p in positions]
    
    # Find valid indices
    valid_idx = [i for i in range(n) if vis[i] > 0.5]
    
    if len(valid_idx) < 2:
        return result  # Can't interpolate with < 2 points
    
    # Interpolate x and y
    for i in range(n):
        if vis[i] > 0.5:
            continue
        # Find nearest valid before and after
        before = [j for j in valid_idx if j < i]
        after = [j for j in valid_idx if j > i]
        
        if before and after:
            b, a = before[-1], after[0]
            t = (i - b) / (a - b)
            new_x = xs[b] + t * (xs[a] - xs[b])
            new_y = ys[b] + t * (ys[a] - ys[b])
            result[i] = (new_x, new_y, 0.5)  # 0.5 = interpolated
        elif before:
            # Extrapolate forward
            b = before[-1]
            if len(before) >= 2:
                b2 = before[-2]
                vx = xs[b] - xs[b2]
                vy = ys[b] - ys[b2]
                result[i] = (xs[b] + vx * (i - b), ys[b] + vy * (i - b), 0.3)
            else:
                result[i] = (xs[b], ys[b], 0.3)
        elif after:
            # Extrapolate backward
            a = after[0]
            if len(after) >= 2:
                a2 = after[1]
                vx = xs[a2] - xs[a]
                vy = ys[a2] - ys[a]
                result[i] = (xs[a] - vx * (a - i), ys[a] - vy * (a - i), 0.3)
            else:
                result[i] = (xs[a], ys[a], 0.3)
    
    return result


class TrajectoryEventDataset(Dataset):
    """Dataset for trajectory-based event classification."""
    
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        positions = s['positions']  # 9 x (x, y, vis)
        table_feat = s['table_feat']  # 6 floats
        
        # Flatten positions: [x0,y0,v0, x1,y1,v1, ..., x8,y8,v8] = 27
        pos_flat = []
        for x, y, v in positions:
            pos_flat.extend([x, y, v])
        
        # Compute velocity features (dx, dy between consecutive frames)
        velocities = []
        for i in range(1, len(positions)):
            x1, y1, v1 = positions[i]
            x0, y0, v0 = positions[i-1]
            if v0 > 0.3 and v1 > 0.3:
                velocities.extend([x1 - x0, y1 - y0])
            else:
                velocities.extend([0.0, 0.0])
        # 8 velocity pairs = 16 floats
        
        # Compute acceleration features
        accels = []
        for i in range(1, len(velocities) // 2):
            dvx = velocities[i*2] - velocities[(i-1)*2]
            dvy = velocities[i*2+1] - velocities[(i-1)*2+1]
            accels.extend([dvx, dvy])
        # 7 accel pairs = 14 floats
        
        # Table-relative ball position for center frame (offset=0, index=5)
        cx, cy, cv = positions[5]
        if len(table_feat) == 6 and table_feat[4] > 0:
            # Relative to table center
            table_rel = [cx - table_feat[4], cy - table_feat[5]]
            # Which half of table (left/right of center)
            table_half = [1.0 if cx < table_feat[4] else 0.0]
            # Relative to table bounds
            tw = max(table_feat[2] - table_feat[0], 0.01)
            th = max(table_feat[3] - table_feat[1], 0.01)
            table_norm = [(cx - table_feat[0]) / tw, (cy - table_feat[1]) / th]
        else:
            table_rel = [0.0, 0.0]
            table_half = [0.5]
            table_norm = [0.5, 0.5]
        
        # Combine all features
        # 27 (pos) + 16 (vel) + 14 (accel) + 6 (table) + 2 (table_rel) + 1 (half) + 2 (table_norm)
        features = pos_flat + velocities + accels + table_feat + table_rel + table_half + table_norm
        # Total: 27 + 16 + 14 + 6 + 2 + 1 + 2 = 68
        
        # Augmentation: horizontal flip
        if self.augment and random.random() < 0.3:
            features = self._hflip(features, positions)
        
        feat_tensor = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(s['event'], dtype=torch.long)
        
        return feat_tensor, label
    
    def _hflip(self, features, positions):
        """Flip x coordinates."""
        # Simple: flip all x values (1 - x)
        f = list(features)
        # Position x values at indices 0, 3, 6, ..., 24
        for i in range(0, 27, 3):
            f[i] = 1.0 - f[i]
        # Velocity dx at indices 27, 29, 31, ..., 41
        for i in range(27, 43, 2):
            f[i] = -f[i]
        # Accel dx
        for i in range(43, 57, 2):
            f[i] = -f[i]
        # Table features: flip x coords
        f[57] = 1.0 - f[57]  # tx1
        f[59] = 1.0 - f[59]  # tx2
        f[61] = 1.0 - f[61]  # table center x
        # Table rel
        f[63] = -f[63]
        # Table half
        f[65] = 1.0 - f[65]
        # Table norm x
        f[66] = 1.0 - f[66]
        return f


class TrajectoryEventMLP(nn.Module):
    """Simple MLP for event classification from trajectory features."""
    
    def __init__(self, input_dim=68, hidden_dims=(128, 64, 32), num_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


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
    
    # Per-class accuracy
    from collections import Counter
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(preds_all, labels_all):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1
    
    class_acc = {}
    for c in range(3):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c]
        else:
            class_acc[c] = 0.0
    
    return total_loss / total, correct / total, class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_root', default='/home/liuzhenlu/cyclex/TOTNet/dataset')
    parser.add_argument('--min_coverage', type=int, default=5,
                        help='Min frames with ball position (out of 9)')
    parser.add_argument('--save_dir', default='checkpoints/traj_event')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Build samples
    train_samples = build_trajectory_samples(
        args.data_root, 'training', min_coverage=args.min_coverage)
    test_samples = build_trajectory_samples(
        args.data_root, 'test', min_coverage=args.min_coverage)

    # Class distribution
    from collections import Counter
    dist = Counter(s['event'] for s in train_samples)
    log.info(f"Train class dist: bounce={dist[0]}, net={dist[1]}, empty={dist[2]}")

    # Split
    train_s, val_s = train_test_split(
        train_samples, test_size=0.15, random_state=args.seed, shuffle=True)

    train_ds = TrajectoryEventDataset(train_s, augment=True)
    val_ds = TrajectoryEventDataset(val_s, augment=False)
    test_ds = TrajectoryEventDataset(test_samples, augment=False) if test_samples else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Compute class weights for imbalanced data
    counts = [dist[0], dist[1], dist[2]]
    total = sum(counts)
    weights = torch.tensor([total / (3 * max(c, 1)) for c in counts], dtype=torch.float32).to(device)
    log.info(f"Class weights: {weights.tolist()}")

    # Model
    # Check feature dim
    sample_feat, _ = train_ds[0]
    input_dim = sample_feat.shape[0]
    log.info(f"Feature dim: {input_dim}")

    model = TrajectoryEventMLP(input_dim=input_dim, num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

    # Test evaluation
    if test_ds:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        # Load best
        ckpt = torch.load(os.path.join(args.save_dir, 'best.pth'), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        test_loss, test_acc, test_class_acc = eval_epoch(model, test_loader, criterion, device)
        log.info(f"\n=== TEST ===")
        log.info(f"  loss={test_loss:.4f} acc={test_acc*100:.1f}%")
        log.info(f"  bounce={test_class_acc[0]*100:.1f}% net={test_class_acc[1]*100:.1f}% "
                 f"empty={test_class_acc[2]*100:.1f}%")


if __name__ == '__main__':
    main()
