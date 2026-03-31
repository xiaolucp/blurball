"""
End-to-end inference pipeline.

Processes a video and outputs:
  - Ball position per frame (x, y, confidence)
  - Table boundaries (left_edge, right_edge, net_x, top_y, bottom_y)
  - Event detection (bounce/net/empty with frame index)
  - Which side of the table the bounce happened on

Usage:
    from pipeline import Pipeline
    pipe = Pipeline('checkpoints/blurball_ball_seg.pth',
                    'checkpoints/trajectory_event.pth')
    results = pipe.process_video('video.mp4')
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from load_models import load_ball_seg_model, load_event_model
from models.trajectory_event_model import EVENT_NAMES


class TableExtractor:
    """Extract table boundaries from segmentation output."""

    def __init__(self, table_class=2):
        self.table_class = table_class

    def extract(self, seg_logits):
        """
        Args:
            seg_logits: [1, 4, H, W] raw seg output

        Returns:
            dict with table boundaries in seg resolution:
              left_x, right_x: table left/right edges
              net_x: estimated net position (center of table)
              top_y, bottom_y: table top/bottom edges
              valid: whether table was detected
        """
        seg_probs = F.softmax(seg_logits, dim=1)
        table_mask = (seg_probs[0, self.table_class] > 0.5).cpu().numpy()  # [H, W]

        if table_mask.sum() < 50:
            return {'valid': False}

        # Find bounding box of table region
        ys, xs = np.where(table_mask)
        left_x = int(xs.min())
        right_x = int(xs.max())
        top_y = int(ys.min())
        bottom_y = int(ys.max())
        net_x = (left_x + right_x) // 2
        seg_h, seg_w = table_mask.shape

        return {
            'left_x': left_x,
            'right_x': right_x,
            'net_x': net_x,
            'top_y': top_y,
            'bottom_y': bottom_y,
            'seg_h': seg_h,
            'seg_w': seg_w,
            'valid': True,
        }

    def ball_side(self, ball_x, table_info, img_w):
        """Determine which side of the table the ball is on.

        Args:
            ball_x: ball x coordinate in image pixels
            table_info: output from extract()
            img_w: image width

        Returns:
            'left', 'right', or 'unknown'
        """
        if not table_info['valid']:
            return 'unknown'

        # Scale net_x from seg resolution to image resolution
        scale_x = img_w / table_info['seg_w']
        net_x_img = table_info['net_x'] * scale_x

        if ball_x < net_x_img:
            return 'left'
        else:
            return 'right'


class Pipeline:
    """End-to-end video processing pipeline."""

    def __init__(self, ball_seg_ckpt, event_ckpt, device=None,
                 img_size=(288, 512), conf_threshold=0.3,
                 traj_before=5, traj_after=3):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.img_size = img_size  # (H, W)
        self.conf_threshold = conf_threshold
        self.traj_before = traj_before
        self.traj_after = traj_after
        self.traj_len = traj_before + 1 + traj_after

        # Load models
        self.model_bb = load_ball_seg_model(ball_seg_ckpt, device)
        self.model_ev = load_event_model(event_ckpt, device)
        self.table_extractor = TableExtractor()

    def _preprocess_frame(self, frame_bgr):
        """BGR frame → normalized tensor [3, H, W]."""
        h, w = self.img_size
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))
        frame = frame.astype(np.float32) / 255.0
        return frame.transpose(2, 0, 1)  # [3, H, W]

    def _extract_ball_pos(self, ball_hm, orig_w, orig_h):
        """Extract ball (x, y, conf) from heatmap in original image coords."""
        h, w = self.img_size
        # Use center frame of the heatmap output
        center = ball_hm.shape[1] // 2
        hm = torch.sigmoid(ball_hm[0, center]).cpu().numpy()

        conf = float(hm.max())
        if conf < self.conf_threshold:
            return 0.0, 0.0, conf, False

        py, px = np.unravel_index(hm.argmax(), hm.shape)
        # Scale to original image coords
        x = float(px) / w * orig_w
        y = float(py) / h * orig_h
        return x, y, conf, True

    def _extract_seg_context(self, seg_logits):
        """Extract segmentation class proportions [4]."""
        seg_probs = F.softmax(seg_logits, dim=1)
        return seg_probs.mean(dim=(2, 3))  # [1, 4]

    def process_video(self, video_path, max_frames=None):
        """Process a video end-to-end.

        Returns:
            dict with:
              - ball_positions: list of {frame, x, y, conf, visible}
              - table_info: table boundaries
              - events: list of {frame, event, side, probs}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"Processing {video_path}: {total_frames} frames, {orig_w}x{orig_h}, {fps:.1f}fps")

        # Read and process all frames
        ball_positions = []
        seg_contexts = []
        table_info = None
        frame_buffer = []  # sliding window of preprocessed frames

        for fid in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            processed = self._preprocess_frame(frame_bgr)
            frame_buffer.append(processed)

            # Need at least 3 frames for model input
            if len(frame_buffer) < 3:
                ball_positions.append({
                    'frame': fid, 'x': 0, 'y': 0, 'conf': 0, 'visible': False
                })
                seg_contexts.append(torch.zeros(1, 4))
                continue

            # Build 3-frame input from buffer (prev, curr, curr or prev, curr, next)
            idx = len(frame_buffer) - 1
            prev = frame_buffer[max(0, idx - 1)]
            curr = frame_buffer[idx]
            # Use current as "next" since true next isn't available yet
            inp = np.concatenate([prev, curr, curr], axis=0)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(self.device)

            # Forward
            with torch.no_grad():
                ball_hm, _, seg_logits = self.model_bb(inp_tensor)

            # Extract ball position
            x, y, conf, visible = self._extract_ball_pos(ball_hm, orig_w, orig_h)
            ball_positions.append({
                'frame': fid, 'x': x, 'y': y, 'conf': conf, 'visible': visible
            })

            # Extract seg context
            seg_ctx = self._extract_seg_context(seg_logits)
            seg_contexts.append(seg_ctx.cpu())

            # Extract table info (do once, use first valid detection)
            if table_info is None or not table_info.get('valid', False):
                table_info = self.table_extractor.extract(seg_logits)

            # Free old frames to save memory (keep last 2)
            if len(frame_buffer) > 3:
                frame_buffer = frame_buffer[-2:]

        cap.release()

        # Scale table info to image coords
        if table_info and table_info['valid']:
            sx = orig_w / table_info['seg_w']
            sy = orig_h / table_info['seg_h']
            table_info_img = {
                'left_x': table_info['left_x'] * sx,
                'right_x': table_info['right_x'] * sx,
                'net_x': table_info['net_x'] * sx,
                'top_y': table_info['top_y'] * sy,
                'bottom_y': table_info['bottom_y'] * sy,
                'valid': True,
            }
        else:
            table_info_img = {'valid': False}

        # Run event detection on sliding windows
        events = []
        for center in range(self.traj_before, len(ball_positions) - self.traj_after):
            # Build trajectory
            traj = []
            for offset in range(-self.traj_before, self.traj_after + 1):
                bp = ball_positions[center + offset]
                vis = 1.0 if bp['visible'] else 0.0
                traj.append([bp['x'], bp['y'], vis])

            traj_tensor = torch.tensor([traj], dtype=torch.float32).to(self.device)
            seg_ctx = seg_contexts[center].to(self.device)

            with torch.no_grad():
                logits = self.model_ev(traj_tensor, seg_ctx)
                probs = F.softmax(logits, dim=1)

            pred_class = probs.argmax(1).item()
            pred_prob = probs[0, pred_class].item()

            if pred_class != 2 and pred_prob > 0.5:  # not empty, confident
                bp = ball_positions[center]
                side = self.table_extractor.ball_side(
                    bp['x'], table_info or {}, orig_w)

                events.append({
                    'frame': center,
                    'event': EVENT_NAMES[pred_class],
                    'prob': round(pred_prob, 3),
                    'side': side,
                    'ball_x': bp['x'],
                    'ball_y': bp['y'],
                })

        print(f"Done: {len(ball_positions)} frames, {len(events)} events detected")
        print(f"Table: {table_info_img}")

        return {
            'ball_positions': ball_positions,
            'table': table_info_img,
            'events': events,
            'video_info': {
                'width': orig_w, 'height': orig_h,
                'fps': fps, 'total_frames': total_frames,
            },
        }


if __name__ == '__main__':
    pipe = Pipeline(
        ball_seg_ckpt='checkpoints/blurball_ball_seg.pth',
        event_ckpt='checkpoints/trajectory_event.pth',
    )

    # Process a test video
    video = '/home/liuzhenlu/cyclex/TOTNet/dataset/test/videos/test_1.mp4'
    if os.path.exists(video):
        results = pipe.process_video(video, max_frames=200)

        print(f"\n=== Results ===")
        print(f"Ball visible: {sum(1 for b in results['ball_positions'] if b['visible'])}/{len(results['ball_positions'])} frames")
        print(f"Table: {results['table']}")
        print(f"Events ({len(results['events'])}):")
        for e in results['events']:
            print(f"  Frame {e['frame']}: {e['event']} ({e['prob']:.0%}) on {e['side']} side")
    else:
        print(f"Video not found: {video}")
        print("Usage: python pipeline.py")
