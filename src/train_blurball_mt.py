"""
BlurBall Multi-Task Training on TTNet dataset.

Phase 1: Freeze backbone, train event + seg heads (from BlurBall checkpoint)
Phase 2: Unfreeze backbone, joint fine-tune all tasks

Usage:
  # Phase 1
  python train_blurball_mt.py --phase 1 --epochs 15 --lr 1e-3 --batch_size 8 \
    --blurball_ckpt weights/blurball_models/blurball_best \
    --save_dir checkpoints/bb_mt_phase1

  # Phase 2
  python train_blurball_mt.py --phase 2 --epochs 20 --lr 1e-4 --batch_size 4 \
    --resume checkpoints/bb_mt_phase1/best.pth \
    --save_dir checkpoints/bb_mt_phase2
"""

import os
import sys
import argparse
import logging
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.blurball_multitask import build_blurball_multitask
from dataloaders.ttnet_dataset import build_samples, TTNetMultiTaskDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Combined loss: ball heatmap + event CE + seg CE."""
    def __init__(self, w_ball=1.0, w_event=3.0, w_seg=0.5):
        super().__init__()
        self.w_ball = w_ball
        self.w_event = w_event
        self.w_seg = w_seg
        self.event_ce = nn.CrossEntropyLoss()
        self.seg_ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, ball_hm, event_logits, seg_logits,
                ball_target, vis_target, event_target, seg_target):
        losses = {}
        total = 0.0

        # Ball heatmap loss (weighted BCE)
        if ball_hm is not None:
            # ball_hm: [B, frames_out, H, W], use center frame
            center = ball_hm.shape[1] // 2
            pred = ball_hm[:, center]  # [B, H, W]
            # Weighted BCE: positive pixels are rare
            pos_weight = (vis_target > 0.5).float() * 50.0 + 1.0  # weight visible samples
            bce = F.binary_cross_entropy_with_logits(
                pred, ball_target,
                reduction='none'
            )
            # Weight: emphasize visible frames
            weight = vis_target.unsqueeze(-1).unsqueeze(-1) * 10.0 + 1.0
            ball_loss = (bce * weight).mean()
            losses['ball'] = ball_loss
            total += self.w_ball * ball_loss

        # Event loss
        if event_logits is not None:
            event_loss = self.event_ce(event_logits, event_target)
            losses['event'] = event_loss
            total += self.w_event * event_loss

        # Seg loss
        if seg_logits is not None:
            seg_loss = self.seg_ce(seg_logits, seg_target)
            losses['seg'] = seg_loss
            total += self.w_seg * seg_loss

        losses['total'] = total
        return total, losses


def train_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch):
    model.train()
    metrics = {'total': 0, 'ball': 0, 'event': 0, 'seg': 0, 'event_acc': 0, 'n': 0}

    for batch in tqdm(loader, desc=f'Train E{epoch}'):
        inp, ball_hm, vis, event, seg, traj = [x.to(device) for x in batch]

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            ball_pred, event_pred, seg_pred = model(inp, traj=traj)

        total, loss_d = loss_fn(ball_pred, event_pred, seg_pred,
                                ball_hm, vis, event, seg)

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = inp.size(0)
        metrics['total'] += loss_d['total'].item() * bs
        metrics['ball'] += loss_d.get('ball', torch.tensor(0)).item() * bs
        metrics['event'] += loss_d.get('event', torch.tensor(0)).item() * bs
        metrics['seg'] += loss_d.get('seg', torch.tensor(0)).item() * bs
        if event_pred is not None:
            acc = (event_pred.argmax(1) == event).float().sum().item()
            metrics['event_acc'] += acc
        metrics['n'] += bs

    n = metrics['n']
    return {k: metrics[k] / n for k in ['total', 'ball', 'event', 'seg', 'event_acc']}


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, epoch, prefix='Val'):
    model.eval()
    metrics = {'total': 0, 'ball': 0, 'event': 0, 'seg': 0,
               'event_acc': 0, 'seg_pixel_acc': 0, 'seg_pixels': 0, 'n': 0}

    for batch in tqdm(loader, desc=f'{prefix} E{epoch}'):
        inp, ball_hm, vis, event, seg, traj = [x.to(device) for x in batch]

        with torch.autocast(device_type='cuda'):
            ball_pred, event_pred, seg_pred = model(inp, traj=traj)

        total, loss_d = loss_fn(ball_pred, event_pred, seg_pred,
                                ball_hm, vis, event, seg)

        bs = inp.size(0)
        metrics['total'] += loss_d['total'].item() * bs
        metrics['ball'] += loss_d.get('ball', torch.tensor(0)).item() * bs
        metrics['event'] += loss_d.get('event', torch.tensor(0)).item() * bs
        metrics['seg'] += loss_d.get('seg', torch.tensor(0)).item() * bs
        if event_pred is not None:
            metrics['event_acc'] += (event_pred.argmax(1) == event).float().sum().item()
        if seg_pred is not None:
            seg_cls = seg_pred.argmax(1)
            metrics['seg_pixel_acc'] += (seg_cls == seg).float().sum().item()
            metrics['seg_pixels'] += seg.numel()
        metrics['n'] += bs

    n = metrics['n']
    result = {k: metrics[k] / n for k in ['total', 'ball', 'event', 'seg', 'event_acc']}
    result['seg_pixel_acc'] = metrics['seg_pixel_acc'] / max(metrics['seg_pixels'], 1) * 100
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=3)
    parser.add_argument('--data_root', default='/home/liuzhenlu/cyclex/TOTNet/dataset')
    parser.add_argument('--blurball_ckpt', default=None, help='BlurBall pretrained weights')
    parser.add_argument('--resume', default=None, help='Resume from multi-task checkpoint')
    parser.add_argument('--save_dir', default='checkpoints/bb_mt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--event_head', type=str, default='trajectory',
                        choices=['cascade', 'trajectory'],
                        help='Event head type: cascade (old) or trajectory (new)')
    parser.add_argument('--traj_before', type=int, default=5,
                        help='Number of frames before center for trajectory')
    parser.add_argument('--traj_after', type=int, default=3,
                        help='Number of frames after center for trajectory')
    parser.add_argument('--pred_positions', type=str, default=None,
                        help='Path to predicted ball positions JSON (use model predictions instead of GT)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}, Phase: {args.phase}")

    # Build model
    event_type = getattr(args, 'event_head', 'cascade')
    traj_len = getattr(args, 'traj_before', 5) + 1 + getattr(args, 'traj_after', 3)
    model = build_blurball_multitask(num_frames=args.num_frames, device='cpu',
                                      event_head_type=event_type, traj_len=traj_len)

    # Load weights
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    elif args.blurball_ckpt:
        log.info(f"Loading BlurBall backbone from {args.blurball_ckpt}")
        model.load_blurball_checkpoint(args.blurball_ckpt)

    if args.phase == 1:
        model.freeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Phase 1: backbone frozen, {trainable:,} trainable params")
    else:
        model.unfreeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Phase 2: all {trainable:,} params trainable")

    model = model.to(device)

    # Dataset
    # Load predicted positions if provided
    pred_pos = None
    if args.pred_positions and os.path.exists(args.pred_positions):
        import json as json_mod
        with open(args.pred_positions) as f:
            pred_pos = json_mod.load(f)
        # Keys in pred_pos are frame_id as int, convert to str for lookup
        for key in pred_pos:
            pred_pos[key] = {str(k): v for k, v in pred_pos[key].items()}
        log.info(f"Loaded predicted positions from {args.pred_positions} "
                 f"({sum(len(v) for v in pred_pos.values())} frames)")
    elif args.pred_positions:
        log.warning(f"Predicted positions file not found: {args.pred_positions}, using GT")

    train_samples = build_samples(args.data_root, 'training', num_frames=args.num_frames,
                                  require_event=True,
                                  traj_before=args.traj_before, traj_after=args.traj_after,
                                  predicted_positions=pred_pos)
    test_samples = build_samples(args.data_root, 'test', num_frames=args.num_frames,
                                 require_event=True,
                                 traj_before=args.traj_before, traj_after=args.traj_after,
                                 predicted_positions=pred_pos)

    # Split training into train/val
    train_s, val_s = train_test_split(train_samples, test_size=0.15,
                                      random_state=args.seed, shuffle=True)

    train_ds = TTNetMultiTaskDataset(train_s, augment=True)
    val_ds = TTNetMultiTaskDataset(val_s, augment=False)
    test_ds = TTNetMultiTaskDataset(test_samples, augment=False) if test_samples else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True) if test_ds else None

    log.info(f"Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_samples) if test_samples else 0}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler()
    loss_fn = MultiTaskLoss(w_ball=1.0, w_event=3.0, w_seg=0.5)

    best_val_loss = float('inf')
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        log.info(f"{'='*60}")
        log.info(f"Epoch {epoch}/{args.epochs}  LR={lr:.2e}")

        train_m = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, epoch)
        val_m = eval_epoch(model, val_loader, loss_fn, device, epoch, 'Val')

        test_m = None
        if test_loader:
            test_m = eval_epoch(model, test_loader, loss_fn, device, epoch, 'Test')

        scheduler.step()

        log.info(f"Train | loss={train_m['total']:.4f} ball={train_m['ball']:.4f} "
                 f"event={train_m['event']:.4f} seg={train_m['seg']:.4f} "
                 f"event_acc={train_m['event_acc']*100:.1f}%")
        log.info(f"Val   | loss={val_m['total']:.4f} ball={val_m['ball']:.4f} "
                 f"event={val_m['event']:.4f} seg={val_m['seg']:.4f} "
                 f"event_acc={val_m['event_acc']*100:.1f}% seg_pixel={val_m['seg_pixel_acc']:.1f}%")
        if test_m:
            log.info(f"Test  | loss={test_m['total']:.4f} "
                     f"event_acc={test_m['event_acc']*100:.1f}% seg_pixel={test_m['seg_pixel_acc']:.1f}%")

        entry = {'epoch': epoch, 'lr': lr, 'train': train_m, 'val': val_m}
        if test_m:
            entry['test'] = test_m
        history.append(entry)

        # Save
        is_best = val_m['total'] < best_val_loss
        if is_best:
            best_val_loss = val_m['total']
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_m,
            }, os.path.join(args.save_dir, 'best.pth'))
            log.info(f"★ Best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

        # Early stopping
        if no_improve >= args.patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    # Save history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    log.info(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == '__main__':
    main()
