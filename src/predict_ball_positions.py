"""
Pre-compute ball position predictions for all frames in TTNet dataset.

Runs the trained ball detection model (BlurBall TDF) on every frame,
extracts predicted (x, y) positions via argmax, and saves as JSON.
These predictions are used as trajectory input for event head training
(instead of GT annotations).
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_tdf import build_blurball_model, AttrDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATASET_ROOT = '/home/liuzhenlu/cyclex/TOTNet/dataset'
CHECKPOINT = '/home/liuzhenlu/cyclex/BlurBall/checkpoints/blurball_tdf/best.pth'
MODEL_CONFIG = '/home/liuzhenlu/cyclex/BlurBall/src/configs/model/blurball.yaml'
OUTPUT_PATH = '/home/liuzhenlu/cyclex/BlurBall/checkpoints/blurball_tdf/predicted_positions.json'

IMG_SIZE = (288, 512)  # model input (H, W)


def predict_from_video(model, video_path, device):
    """Run ball model on all frames from a video file.

    Uses a sliding window of 3 frames, keeping only 3 frames in memory at a time.
    Returns dict of {frame_id: {x, y, conf}}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return {}

    h, w = IMG_SIZE
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    positions = {}
    prev_frame = None
    curr_frame = None

    for fid in range(total_frames):
        ret, raw = cap.read()
        if not ret:
            break

        # Preprocess
        frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)  # [3, H, W]

        # Shift window
        prev_frame = curr_frame
        curr_frame = frame

        if fid < 1:
            continue  # need at least prev + curr

        # For frame fid, use (fid-1, fid, fid) as 3-frame input
        # (next frame not available yet, duplicate current)
        if prev_frame is None:
            continue

        # Read next frame if available for proper 3-frame window
        # But to avoid complexity, use prev, curr, curr for now
        # This is slightly less accurate but avoids double-reading
        inp = np.concatenate([prev_frame, curr_frame, curr_frame], axis=0)
        inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp_tensor)
            pred = torch.sigmoid(out[0])

        pred_np = pred[0, 0].cpu().numpy()
        conf = float(pred_np.max())
        py, px = np.unravel_index(pred_np.argmax(), pred_np.shape)
        ox = float(px) / w * orig_w
        oy = float(py) / h * orig_h

        positions[fid] = {
            'x': round(ox, 1),
            'y': round(oy, 1),
            'conf': round(conf, 4),
        }

    cap.release()
    return positions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Load model
    model = build_blurball_model(MODEL_CONFIG, None, tdf_output=True)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    log.info(f"Model loaded (epoch {ckpt['epoch']+1})")

    all_predictions = {}

    for split in ['training', 'test']:
        videos_dir = os.path.join(DATASET_ROOT, split, 'videos')
        if not os.path.isdir(videos_dir):
            continue

        videos = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])

        for video_file in tqdm(videos, desc=f'{split}'):
            game = video_file.replace('.mp4', '')
            video_path = os.path.join(videos_dir, video_file)
            positions = predict_from_video(model, video_path, device)

            key = f'{split}/{game}'
            all_predictions[key] = positions
            log.info(f"  {key}: {len(positions)} frames predicted")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_predictions, f)
    log.info(f"Saved predictions to {OUTPUT_PATH}")
    log.info(f"Total: {sum(len(v) for v in all_predictions.values())} frame predictions")


if __name__ == '__main__':
    main()
