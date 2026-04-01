"""
Train TrajectoryEventModel standalone — no vision backbone, no video reading.

All inputs are pre-computed:
  - Ball positions: predicted_positions.json (644K frames)
  - Table keypoints: table_keypoints_13.json (13 points per game)
  - Event labels: TTNet annotations (bounce/net/empty)

Training is pure MLP on numbers → should be seconds per epoch.
"""

import os
import sys
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models.trajectory_event_model import TrajectoryEventModel, EVENT_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

EVENT_MAP = {'bounce': 0, 'net': 1, 'empty_event': 2}


class EventDataset(Dataset):
    """Pure numeric dataset: trajectory + table keypoints → event label."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['traj'], dtype=torch.float32),      # [T, 3]
            torch.tensor(s['table_kp'], dtype=torch.float32),  # [13, 2]
            torch.tensor(s['event'], dtype=torch.long),         # scalar
        )


def build_samples(ann_root, split, pred_positions, table_keypoints,
                  traj_before=5, traj_after=3, img_w=1920, img_h=1080,
                  model_w=512, model_h=288):
    """Build training samples from pre-computed data."""
    games_dir = os.path.join(ann_root, split, 'annotations')
    if not os.path.isdir(games_dir):
        return []

    games = sorted([d for d in os.listdir(games_dir)
                    if os.path.isdir(os.path.join(games_dir, d))])

    samples = []
    for game in games:
        ann_dir = os.path.join(games_dir, game)

        # Load event annotations
        events_path = os.path.join(ann_dir, 'events_markup.json')
        if not os.path.exists(events_path):
            continue
        with open(events_path) as f:
            events = json.load(f)

        # Get predicted ball positions for this game
        pred_key = f'{split}/{game}'
        ball_pos = pred_positions.get(pred_key, {})

        # Get table keypoints (scale from original to model input coords)
        tkp_data = table_keypoints.get(pred_key)
        if tkp_data is None:
            continue
        kps_raw = tkp_data['keypoints']  # [[x, y, vis], ...] × 13
        res = tkp_data['resolution']  # [w, h]
        # Scale to model input coords
        sx = model_w / res[0]
        sy = model_h / res[1]
        table_kp = [[kp[0] * sx, kp[1] * sy] for kp in kps_raw]

        for fid_str, event_str in events.items():
            fid = int(fid_str)
            event_label = EVENT_MAP.get(event_str, 2)

            # Build trajectory from predicted positions
            traj = []
            for offset in range(-traj_before, traj_after + 1):
                t_fid = str(fid + offset)
                if t_fid in ball_pos:
                    p = ball_pos[t_fid]
                    conf = p.get('conf', 0)
                    if conf > 0.3:
                        # Scale from original to model input coords
                        tx = p['x'] * sx
                        ty = p['y'] * sy
                        traj.append([tx, ty, 1.0])
                    else:
                        traj.append([0.0, 0.0, 0.0])
                else:
                    traj.append([0.0, 0.0, 0.0])

            samples.append({
                'traj': traj,
                'table_kp': table_kp,
                'event': event_label,
                'game': game,
                'frame': fid,
            })

    log.info(f"[{split}] {len(samples)} samples from {len(games)} games")
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/liuzhenlu/cyclex/TOTNet/dataset')
    parser.add_argument('--pred_positions', default='checkpoints/blurball_tdf/predicted_positions.json')
    parser.add_argument('--table_keypoints', default='checkpoints/table_keypoints_13.json')
    parser.add_argument('--save_dir', default='checkpoints/event_standalone')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--traj_before', type=int, default=5)
    parser.add_argument('--traj_after', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-computed data
    with open(args.pred_positions) as f:
        pred_pos = json.load(f)
    # Convert frame ids to strings
    for key in pred_pos:
        pred_pos[key] = {str(k): v for k, v in pred_pos[key].items()}
    log.info(f"Ball positions: {sum(len(v) for v in pred_pos.values())} frames")

    with open(args.table_keypoints) as f:
        table_kp = json.load(f)
    log.info(f"Table keypoints: {len(table_kp)} games")

    # Build samples
    traj_len = args.traj_before + 1 + args.traj_after
    train_samples = build_samples(args.data_root, 'training', pred_pos, table_kp,
                                   args.traj_before, args.traj_after)
    test_samples = build_samples(args.data_root, 'test', pred_pos, table_kp,
                                  args.traj_before, args.traj_after)

    # Split training into train/val
    random.shuffle(train_samples)
    n_val = max(1, len(train_samples) // 7)
    val_samples = train_samples[:n_val]
    train_samples = train_samples[n_val:]

    train_ds = EventDataset(train_samples)
    val_ds = EventDataset(val_samples)
    test_ds = EventDataset(test_samples) if test_samples else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size) if test_ds else None

    log.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Model
    model = TrajectoryEventModel(traj_len=traj_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {total_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss, correct, n = 0, 0, 0
        for traj, tkp, label in train_loader:
            traj, tkp, label = traj.to(device), tkp.to(device), label.to(device)
            logits = model(traj, table_kp=tkp)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * traj.size(0)
            correct += (logits.argmax(1) == label).sum().item()
            n += traj.size(0)
        train_acc = correct / n
        train_loss = total_loss / n

        # Val
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for traj, tkp, label in val_loader:
                traj, tkp, label = traj.to(device), tkp.to(device), label.to(device)
                logits = model(traj, table_kp=tkp)
                val_correct += (logits.argmax(1) == label).sum().item()
                val_n += traj.size(0)
        val_acc = val_correct / val_n

        # Test
        test_acc = 0
        if test_loader:
            test_correct, test_n = 0, 0
            with torch.no_grad():
                for traj, tkp, label in test_loader:
                    traj, tkp, label = traj.to(device), tkp.to(device), label.to(device)
                    logits = model(traj, table_kp=tkp)
                    test_correct += (logits.argmax(1) == label).sum().item()
                    test_n += traj.size(0)
            test_acc = test_correct / test_n

        scheduler.step()

        best_mark = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'traj_len': traj_len, 'num_classes': 3},
                'epoch': epoch,
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, os.path.join(args.save_dir, 'trajectory_event.pth'))
            best_mark = ' *'

        log.info(f"E{epoch:2d} | loss={train_loss:.4f} train={train_acc*100:.1f}% "
                 f"val={val_acc*100:.1f}% test={test_acc*100:.1f}%{best_mark}")

    log.info(f"\nBest val acc: {best_val_acc*100:.1f}%")


if __name__ == '__main__':
    main()
