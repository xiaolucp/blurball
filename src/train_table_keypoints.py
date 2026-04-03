"""
Train table keypoint regression head on TTNet data.

Automatically extracts GT keypoints from segmentation masks:
  - Table: 4 corners from convex hull of red region
  - Net: 2 endpoints from blue region bounding box

Uses frozen BlurBall backbone, only trains the keypoint head.
"""

import os
import sys
import json
import random
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.blurball_multitask import BlurBallMultiTask
from models.table_keypoint_head import TableKeypointHead, KEYPOINT_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── Extract GT keypoints from seg masks ───

def extract_keypoints_from_seg(seg_bgr):
    """Extract 6 table keypoints from a TTNet segmentation mask.

    Args:
        seg_bgr: [128, 320, 3] BGR segmentation mask
    Returns:
        keypoints: [12] normalized coords (x1,y1,...,x6,y6) in [0,1], or None if invalid
    """
    h, w = seg_bgr.shape[:2]

    # Table = red (BGR: 0, 0, 255)
    table_mask = (seg_bgr[:, :, 2] > 200).astype(np.uint8)
    # Net = blue (BGR: 255, 0, 0)
    net_mask = (seg_bgr[:, :, 0] > 200).astype(np.uint8)

    if table_mask.sum() < 50:
        return None

    # Table: convex hull → 4-point approximation
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    approx = cv2.approxPolyDP(hull, 0.015 * cv2.arcLength(hull, True), True)

    if len(approx) < 4:
        return None

    # Sort into 4 corners: need to identify near/far, left/right
    pts = [tuple(p[0]) for p in approx]

    # If more than 4, take the 4 most extreme points
    if len(pts) > 4:
        # Use min area rect as fallback
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        pts = [tuple(b.astype(int)) for b in box]

    pts = np.array(pts, dtype=np.float32)

    # Sort: top points (far) have smaller y, bottom (near) have larger y
    sorted_by_y = pts[pts[:, 1].argsort()]
    top_pts = sorted_by_y[:2]  # far (smaller y)
    bottom_pts = sorted_by_y[2:]  # near (larger y)

    # Within each pair, sort by x
    top_pts = top_pts[top_pts[:, 0].argsort()]  # far-left, far-right
    bottom_pts = bottom_pts[bottom_pts[:, 0].argsort()]  # near-left, near-right

    far_left = top_pts[0]
    far_right = top_pts[1]
    near_left = bottom_pts[0]
    near_right = bottom_pts[1]

    # Net: from blue mask
    if net_mask.sum() > 5:
        net_contours, _ = cv2.findContours(net_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if net_contours:
            net_cnt = max(net_contours, key=cv2.contourArea)
            nx, ny, nw, nh = cv2.boundingRect(net_cnt)
            net_left = np.array([nx, ny + nh // 2], dtype=np.float32)
            net_right = np.array([nx + nw, ny + nh // 2], dtype=np.float32)
        else:
            # Estimate net from table midpoints
            net_left = (far_left + near_left) / 2
            net_right = (far_right + near_right) / 2
    else:
        net_left = (far_left + near_left) / 2
        net_right = (far_right + near_right) / 2

    # Normalize to [0, 1]
    keypoints = np.array([
        near_left[0] / w, near_left[1] / h,
        near_right[0] / w, near_right[1] / h,
        far_left[0] / w, far_left[1] / h,
        far_right[0] / w, far_right[1] / h,
        net_left[0] / w, net_left[1] / h,
        net_right[0] / w, net_right[1] / h,
    ], dtype=np.float32)

    return keypoints


# ─── Dataset ───

class TableKeypointDataset(Dataset):
    """Dataset for table keypoint training."""

    def __init__(self, data_root, split='training', games=None,
                 num_frames=3, img_size=(288, 512), augment=False):
        self.img_size = img_size
        self.num_frames = num_frames
        self.augment = augment
        self.samples = []

        videos_root = os.path.join(data_root, split, 'videos')
        ann_root = os.path.join(data_root, split, 'annotations')

        if games is None:
            games = sorted([d for d in os.listdir(ann_root)
                            if os.path.isdir(os.path.join(ann_root, d))])

        for game in tqdm(games, desc=f'Loading {split}'):
            video_path = os.path.join(videos_root, f'{game}.mp4')
            ann_dir = os.path.join(ann_root, game)
            seg_dir = os.path.join(ann_dir, 'segmentation_masks')

            if not os.path.isfile(video_path) or not os.path.isdir(seg_dir):
                continue

            # Get keypoints from first available seg mask (table is static)
            seg_files = sorted(os.listdir(seg_dir))
            keypoints = None
            for sf in seg_files[:10]:  # try first 10
                seg = cv2.imread(os.path.join(seg_dir, sf))
                if seg is None:
                    continue
                kp = extract_keypoints_from_seg(seg)
                if kp is not None:
                    keypoints = kp
                    break

            if keypoints is None:
                log.warning(f"No valid table keypoints for {game}")
                continue

            # Get frame count
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Sample frames (every 50th frame since table is static)
            half = num_frames // 2
            for fid in range(half, n_frames - half, 50):
                self.samples.append({
                    'video_path': video_path,
                    'frame_id': fid,
                    'keypoints': keypoints,
                })

        log.info(f"TableKeypointDataset: {len(self.samples)} samples from {len(games)} games")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        h, w = self.img_size

        # Read 3 frames from video
        cap = cv2.VideoCapture(s['video_path'])
        fid = s['frame_id']
        frames = []
        for offset in [-1, 0, 1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid + offset))
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (w, h))
            frame = frame.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
        cap.release()

        inp = np.concatenate(frames, axis=0).astype(np.float32)
        keypoints = s['keypoints'].copy()

        # Augmentation: random horizontal flip
        if self.augment and random.random() < 0.5:
            # Flip input frames
            inp = inp[:, :, ::-1].copy()
            # Flip keypoint x coordinates
            for i in range(0, 12, 2):
                keypoints[i] = 1.0 - keypoints[i]
            # Swap left↔right pairs
            # near_left ↔ near_right, far_left ↔ far_right, net_left ↔ net_right
            for i in range(0, 12, 4):
                keypoints[i], keypoints[i+2] = keypoints[i+2], keypoints[i]
                keypoints[i+1], keypoints[i+3] = keypoints[i+3], keypoints[i+1]

        return (
            torch.from_numpy(inp),
            torch.from_numpy(keypoints),
        )


# ─── Training ───

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/liuzhenlu/cyclex/TOTNet/dataset')
    parser.add_argument('--backbone_ckpt', default='checkpoints/bb_mt_tdf_p1/best.pth')
    parser.add_argument('--save_dir', default='checkpoints/table_keypoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Build model: backbone + keypoint head
    from omegaconf import OmegaConf
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'model', 'blurball.yaml')
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    model = BlurBallMultiTask(cfg, tasks=('ball', 'seg'), num_frames=3)

    # Load backbone weights
    ckpt = torch.load(args.backbone_ckpt, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd_clean = {k: v for k, v in sd.items()
                if not k.startswith('event_head') and not k.startswith('traj_event_head')}
    model.load_state_dict(sd_clean, strict=False)
    log.info("Backbone loaded")

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    if model.seg_head:
        for p in model.seg_head.parameters():
            p.requires_grad = False

    # Add keypoint head
    kp_head = TableKeypointHead()
    model.kp_head = kp_head
    model = model.to(device)

    trainable = sum(p.numel() for p in kp_head.parameters())
    log.info(f"Keypoint head: {trainable:,} trainable params")

    # Dataset
    train_ds = TableKeypointDataset(args.data_root, 'training', augment=True)
    test_ds = TableKeypointDataset(args.data_root, 'test', augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    log.info(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Training
    optimizer = torch.optim.AdamW(kp_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        kp_head.train()
        train_loss = 0
        n = 0

        for batch in tqdm(train_loader, desc=f'Train E{epoch}'):
            inp, kp_gt = batch
            inp, kp_gt = inp.to(device), kp_gt.to(device)

            with torch.no_grad():
                y_list = model.backbone_forward(inp)

            kp_pred = kp_head(y_list)
            loss = F.mse_loss(kp_pred, kp_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inp.size(0)
            n += inp.size(0)

        train_loss /= n

        # Eval
        model.eval()
        kp_head.eval()
        val_loss = 0
        val_n = 0
        pixel_errors = []

        with torch.no_grad():
            for batch in test_loader:
                inp, kp_gt = batch
                inp, kp_gt = inp.to(device), kp_gt.to(device)

                y_list = model.backbone_forward(inp)
                kp_pred = kp_head(y_list)
                loss = F.mse_loss(kp_pred, kp_gt)

                val_loss += loss.item() * inp.size(0)
                val_n += inp.size(0)

                # Pixel error (in seg resolution 320x128)
                pred_px = kp_pred.cpu().numpy().reshape(-1, 6, 2)
                gt_px = kp_gt.cpu().numpy().reshape(-1, 6, 2)
                pred_px[:, :, 0] *= 320
                pred_px[:, :, 1] *= 128
                gt_px[:, :, 0] *= 320
                gt_px[:, :, 1] *= 128
                errs = np.sqrt(((pred_px - gt_px) ** 2).sum(axis=2)).mean(axis=1)
                pixel_errors.extend(errs.tolist())

        val_loss /= val_n
        mean_err = np.mean(pixel_errors)
        scheduler.step()

        log.info(f"Epoch {epoch}/{args.epochs}  Train loss={train_loss:.6f}  "
                 f"Val loss={val_loss:.6f}  Mean pixel error={mean_err:.1f}px")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': kp_head.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'mean_pixel_error': mean_err,
            }, os.path.join(args.save_dir, 'table_keypoints.pth'))
            log.info(f"  * Best (error={mean_err:.1f}px)")

    log.info(f"Done. Best val loss={best_val_loss:.6f}")


if __name__ == '__main__':
    main()
