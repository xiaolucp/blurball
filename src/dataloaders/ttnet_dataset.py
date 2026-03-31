"""
TTNet dataset adapter for BlurBall multi-task training.

TTNet data format:
  dataset/training/images/game_X/img_XXXXXX.jpg
  dataset/training/annotations/game_X/ball_markup.json   {frame_id: {x,y}}
  dataset/training/annotations/game_X/events_markup.json {frame_id: event_str}
  dataset/training/annotations/game_X/segmentation_masks/NNN.png

Produces samples of N consecutive frames with:
  - Ball position (x, y) + visibility for center frame
  - Event label (bounce=0, net=1, empty_event=2) for center frame
  - Segmentation mask (128x320, 4 classes) for center frame
"""

import os
import json
import random
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

EVENT_MAP = {'bounce': 0, 'net': 1, 'empty_event': 2}


def load_game_annotations(ann_dir):
    """Load ball_markup, events_markup, and seg mask paths for one game."""
    with open(os.path.join(ann_dir, 'ball_markup.json')) as f:
        ball = json.load(f)  # {str(frame_id): {x, y}}
    with open(os.path.join(ann_dir, 'events_markup.json')) as f:
        events = json.load(f)  # {str(frame_id): event_str}

    seg_dir = os.path.join(ann_dir, 'segmentation_masks')
    seg_frames = set()
    if os.path.isdir(seg_dir):
        for fn in os.listdir(seg_dir):
            if fn.endswith('.png'):
                seg_frames.add(int(fn.replace('.png', '')))

    return ball, events, seg_frames, seg_dir


def build_samples(data_root, split='training', games=None, num_frames=3,
                  img_size=(288, 512), require_event=True,
                  traj_before=5, traj_after=3,
                  predicted_positions=None):
    """Build sample list from TTNet dataset.

    Each sample is centered on a frame that has an event annotation.
    This ensures balanced training on event-bearing frames.

    Args:
        require_event: If True, only sample frames with event annotations.
                      If False, sample all frames with ball annotations (many more empty_event).
        traj_before: number of frames before center to include in trajectory
        traj_after: number of frames after center to include in trajectory
        predicted_positions: dict {split/game: {frame_id: {x,y,conf}}}
                            If provided, use predicted ball positions for trajectory.
    """
    videos_root = os.path.join(data_root, split, 'videos')
    ann_root = os.path.join(data_root, split, 'annotations')

    if games is None:
        # Discover games from annotations directory
        games = sorted([d for d in os.listdir(ann_root)
                        if os.path.isdir(os.path.join(ann_root, d))])

    half_left = (num_frames - 1) // 2
    half_right = num_frames // 2
    samples = []

    for game in games:
        video_path = os.path.join(videos_root, f'{game}.mp4')
        ann_dir = os.path.join(ann_root, game)

        if not os.path.isdir(ann_dir):
            log.warning(f"Skipping {game}: missing annotations")
            continue
        if not os.path.isfile(video_path):
            log.warning(f"Skipping {game}: missing video {video_path}")
            continue

        ball, events, seg_frames, seg_dir = load_game_annotations(ann_dir)

        # Get frame count from video
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if require_event:
            # Center on event frames
            target_frames = sorted([int(k) for k in events.keys()])
        else:
            # Center on all ball-annotated frames
            target_frames = sorted([int(k) for k in ball.keys()])

        for fid in target_frames:
            # Check we have enough context frames (for both image input and trajectory)
            ctx_left = max(half_left, traj_before)
            ctx_right = max(half_right, traj_after)
            if fid - ctx_left < 0 or fid + ctx_right >= n_frames:
                continue

            # Ball annotation for center frame
            fid_str = str(fid)
            if fid_str in ball:
                bx, by = ball[fid_str]['x'], ball[fid_str]['y']
                visible = 1
            else:
                bx, by = 0, 0
                visible = 0

            # Event label
            if fid_str in events:
                event_str = events[fid_str]
                event_label = EVENT_MAP.get(event_str, 2)
            else:
                event_label = 2  # empty_event

            # Seg mask path (may not exist for all frames)
            seg_path = os.path.join(seg_dir, f'{fid}.png') if fid in seg_frames else None

            # Frame indices (centered on event frame, for ball detection input)
            frame_ids = [fid + offset for offset in range(-half_left, half_right + 1)]

            # Trajectory: ball positions for frames [fid-traj_before, ..., fid+traj_after]
            # Use predicted positions if available, else fall back to GT
            pred_key = f'{split}/{game}'
            pred_game = predicted_positions.get(pred_key, {}) if predicted_positions else {}

            traj = []
            for offset in range(-traj_before, traj_after + 1):
                t_fid = fid + offset
                t_fid_str = str(t_fid)

                if pred_game and t_fid_str in pred_game:
                    # Use model's predicted position
                    p = pred_game[t_fid_str]
                    tx, ty = p['x'], p['y']
                    tvis = 1 if p.get('conf', 0) > 0.3 else 0
                    if not tvis:
                        tx, ty = 0, 0
                elif t_fid_str in ball:
                    # Fallback to GT
                    tx, ty = ball[t_fid_str]['x'], ball[t_fid_str]['y']
                    tvis = 1 if tx >= 0 and ty >= 0 else 0
                    if not tvis:
                        tx, ty = 0, 0
                else:
                    tx, ty, tvis = 0, 0, 0
                traj.append((tx, ty, tvis))

            samples.append({
                'video_path': video_path,
                'frame_ids': frame_ids,
                'ball_xy': (bx, by),
                'visible': visible,
                'event': event_label,
                'seg_path': seg_path,
                'game': game,
                'frame_id': fid,
                'traj': traj,
            })

    log.info(f"[{split}] Built {len(samples)} samples from {len(games)} games")
    return samples


class TTNetMultiTaskDataset(Dataset):
    """Dataset for BlurBall multi-task training on TTNet data."""

    def __init__(self, samples, img_size=(288, 512), seg_size=(128, 320),
                 num_classes_seg=4, augment=False):
        self.samples = samples
        self.img_size = img_size  # (H, W)
        self.seg_size = seg_size
        self.num_classes_seg = num_classes_seg
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _read_video_frames(self, video_path, frame_ids):
        """Read specific frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load frames from video
        frames = self._read_video_frames(s['video_path'], s['frame_ids'])

        orig_h, orig_w = frames[0].shape[:2]

        # Augmentation: random horizontal flip
        do_hflip = self.augment and random.random() < 0.3

        # Resize + normalize
        processed = []
        for img in frames:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            if do_hflip:
                img = img[:, ::-1, :].copy()
            img = img.astype(np.float32) / 255.0
            processed.append(img)

        # Stack: [N, H, W, 3] → [N*3, H, W]
        stacked = np.concatenate([p.transpose(2, 0, 1) for p in processed], axis=0)
        input_tensor = torch.from_numpy(stacked)

        # Ball position: scale to model output size
        bx, by = s['ball_xy']
        vis = s['visible']
        if vis:
            bx_scaled = bx / orig_w * self.img_size[1]
            by_scaled = by / orig_h * self.img_size[0]
            if do_hflip:
                bx_scaled = self.img_size[1] - bx_scaled
        else:
            bx_scaled, by_scaled = 0.0, 0.0

        # Generate ball heatmap (Gaussian, sigma=5)
        ball_hm = self._make_heatmap(bx_scaled, by_scaled, vis,
                                     self.img_size[0], self.img_size[1])

        # Event label
        event_label = s['event']

        # Segmentation mask
        if s['seg_path'] and os.path.exists(s['seg_path']):
            seg = cv2.imread(s['seg_path'], cv2.IMREAD_UNCHANGED)
            if seg is None:
                seg = np.zeros(self.seg_size, dtype=np.int64)
            else:
                # TTNet seg masks are 128x320 RGB with values 0 or 255
                # Convert to class indices: need to figure out channel mapping
                seg = self._parse_seg_mask(seg)
                if do_hflip:
                    seg = seg[:, ::-1].copy()
        else:
            seg = np.zeros(self.seg_size, dtype=np.int64)

        # Trajectory: scale positions to model input coords
        # traj is list of (x, y, vis) for surrounding frames
        traj_raw = s.get('traj', [(0, 0, 0)])
        traj_scaled = []
        for tx, ty, tv in traj_raw:
            if tv and tx >= 0 and ty >= 0:
                sx = tx / orig_w * self.img_size[1]
                sy = ty / orig_h * self.img_size[0]
                if do_hflip:
                    sx = self.img_size[1] - sx
                traj_scaled.append([sx, sy, 1.0])
            else:
                traj_scaled.append([0.0, 0.0, 0.0])
        traj_tensor = torch.tensor(traj_scaled, dtype=torch.float32)  # [T, 3]

        return (
            input_tensor,                                    # [N*3, H, W]
            torch.from_numpy(ball_hm).float(),               # [H, W]
            torch.tensor(vis, dtype=torch.float32),           # scalar
            torch.tensor(event_label, dtype=torch.long),      # scalar
            torch.from_numpy(seg).long(),                     # [Hs, Ws]
            traj_tensor,                                      # [T, 3] (x, y, vis)
        )

    def _make_heatmap(self, cx, cy, vis, h, w, sigma=5):
        """Generate Gaussian heatmap."""
        hm = np.zeros((h, w), dtype=np.float32)
        if not vis:
            return hm
        cx, cy = int(round(cx)), int(round(cy))
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return hm
        # Generate gaussian
        size = sigma * 6 + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = size // 2, size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Paste
        left = max(0, cx - size // 2)
        right = min(w, cx + size // 2 + 1)
        top = max(0, cy - size // 2)
        bottom = min(h, cy + size // 2 + 1)

        g_left = max(0, size // 2 - cx)
        g_right = g_left + (right - left)
        g_top = max(0, size // 2 - cy)
        g_bottom = g_top + (bottom - top)

        hm[top:bottom, left:right] = np.maximum(
            hm[top:bottom, left:right], g[g_top:g_bottom, g_left:g_right]
        )
        return hm

    def _parse_seg_mask(self, seg):
        """Parse TTNet segmentation mask to class indices.
        
        TTNet masks are 128x320 RGB:
        - Background (black): [0,0,0] → class 0
        - Person (non-zero patterns): → class 1  
        - Table: → class 2
        - Scoreboard: → class 3
        
        Since TTNet masks use binary (0/255) per channel, decode as:
        - R channel: person
        - G channel: table  
        - B channel: scoreboard
        """
        if len(seg.shape) == 3:
            h, w, c = seg.shape
            # Resize if needed
            if h != self.seg_size[0] or w != self.seg_size[1]:
                seg = cv2.resize(seg, (self.seg_size[1], self.seg_size[0]),
                                interpolation=cv2.INTER_NEAREST)
            
            # Binary threshold each channel
            result = np.zeros((self.seg_size[0], self.seg_size[1]), dtype=np.int64)
            # BGR order in OpenCV
            b, g, r = seg[:,:,0] > 127, seg[:,:,1] > 127, seg[:,:,2] > 127
            
            # Priority: scoreboard > table > person > background
            result[r] = 1  # person (red channel)
            result[g] = 2  # table (green channel) 
            result[b] = 3  # scoreboard (blue channel)
            return result
        else:
            # Grayscale — treat nonzero as class 1
            if seg.shape[0] != self.seg_size[0] or seg.shape[1] != self.seg_size[1]:
                seg = cv2.resize(seg, (self.seg_size[1], self.seg_size[0]),
                                interpolation=cv2.INTER_NEAREST)
            result = np.zeros(self.seg_size, dtype=np.int64)
            result[seg > 127] = 1
            return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data_root = '/home/liuzhenlu/cyclex/TOTNet/dataset'

    samples = build_samples(data_root, 'training', num_frames=3, require_event=True)
    print(f"Total event-centered samples: {len(samples)}")

    # Class distribution
    from collections import Counter
    events = Counter(s['event'] for s in samples)
    print(f"Events: bounce={events[0]}, net={events[1]}, empty={events[2]}")
    has_seg = sum(1 for s in samples if s['seg_path'] is not None)
    print(f"With seg mask: {has_seg}/{len(samples)}")

    # Test loading
    ds = TTNetMultiTaskDataset(samples[:10], augment=True)
    inp, hm, vis, ev, seg = ds[0]
    print(f"Input: {inp.shape}, HM: {hm.shape}, Vis: {vis}, Event: {ev}, Seg: {seg.shape}")
    print(f"Seg unique: {torch.unique(seg).tolist()}")
