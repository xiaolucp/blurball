"""
Visualize TDF predictions on a test video clip.

For each frame:
  1. Run model to get TDF prediction
  2. Threshold to get ball mask
  3. Draw: green contour + semi-transparent fill on original frame
  4. Also draw GT ball position (red dot) for comparison
  5. Save as video
"""

import sys
import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_tdf import build_blurball_model, AttrDict

# Config
DATASET_ROOT = '/home/liuzhenlu/cyclex/BlurBall/dataset_blurball/blurball_dataset'
CHECKPOINT = '/home/liuzhenlu/cyclex/BlurBall/checkpoints/blurball_tdf/best.pth'
MODEL_CONFIG = '/home/liuzhenlu/cyclex/BlurBall/src/configs/model/blurball.yaml'
OUTPUT_DIR = '/home/liuzhenlu/cyclex/BlurBall/output/vis_tdf'

MATCH = 22       # test set
CLIP = '001'
IMG_SIZE = (288, 512)  # model input (H, W)
THRESHOLD = 0.5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = build_blurball_model(MODEL_CONFIG, None, tdf_output=True)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f}")

    # Load frames
    frames_dir = os.path.join(DATASET_ROOT, f'{MATCH:02d}', 'frames', CLIP)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    print(f"Clip: match {MATCH}, clip {CLIP}, {len(frame_files)} frames")

    # Load GT annotations
    csv_path = os.path.join(DATASET_ROOT, f'{MATCH:02d}', 'csv', f'{CLIP}.csv')
    df = pd.read_csv(csv_path)
    gt_annos = {}
    for _, row in df.iterrows():
        fid = int(row['Frame'])
        gt_annos[fid] = {
            'vis': int(row['Visibility']),
            'x': float(row['X']),
            'y': float(row['Y']),
        }

    # Process frames
    h, w = IMG_SIZE
    vis_frames = []

    for i in tqdm(range(len(frame_files)), desc='Processing'):
        # Need 3 consecutive frames as input (center = current)
        center_idx = i
        if center_idx < 1:
            center_idx = 1
        if center_idx >= len(frame_files) - 1:
            center_idx = len(frame_files) - 2

        # Load 3 frames
        imgs = []
        for offset in [-1, 0, 1]:
            idx = center_idx + offset
            fp = os.path.join(frames_dir, frame_files[idx])
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        orig_h, orig_w = imgs[0].shape[:2]

        # Preprocess
        processed = []
        for img in imgs:
            img_r = cv2.resize(img, (w, h))
            img_f = img_r.astype(np.float32) / 255.0
            img_f = img_f.transpose(2, 0, 1)
            processed.append(img_f)

        input_tensor = np.concatenate(processed, axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            out = model(input_tensor)
            pred = torch.sigmoid(out[0])  # [1, 1, H, W]

        pred_np = pred[0, 0].cpu().numpy()  # [H, W]

        # Resize prediction to original resolution
        pred_orig = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold to get mask
        mask = (pred_orig > THRESHOLD).astype(np.uint8)

        # Draw on original frame
        frame_bgr = cv2.imread(os.path.join(frames_dir, frame_files[i]))
        vis_frame = frame_bgr.copy()

        # Green semi-transparent fill
        green_overlay = vis_frame.copy()
        green_overlay[mask > 0] = (0, 255, 0)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, green_overlay, 0.3, 0)

        # Green contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 1)

        # Predicted center (argmax)
        pred_resized = cv2.resize(pred_np, (orig_w, orig_h))
        py, px = np.unravel_index(pred_resized.argmax(), pred_resized.shape)
        cv2.circle(vis_frame, (px, py), 3, (255, 255, 0), -1)  # cyan dot = pred center

        # GT annotation
        fid = int(frame_files[i].replace('.png', ''))
        if fid in gt_annos and gt_annos[fid]['vis']:
            gx = int(round(gt_annos[fid]['x']))
            gy = int(round(gt_annos[fid]['y']))
            cv2.circle(vis_frame, (gx, gy), 3, (0, 0, 255), -1)  # red dot = GT

        # Heatmap inset (top-right corner)
        hm_vis = (pred_orig * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        inset_h, inset_w = orig_h // 4, orig_w // 4
        hm_small = cv2.resize(hm_color, (inset_w, inset_h))
        vis_frame[5:5+inset_h, orig_w-inset_w-5:orig_w-5] = hm_small

        # Label
        cv2.putText(vis_frame, f'Frame {fid}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, 'Green=pred mask  Red=GT  Cyan=pred center',
                    (10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        vis_frames.append(vis_frame)

    # Save video
    out_path = os.path.join(OUTPUT_DIR, f'match{MATCH:02d}_{CLIP}_tdf.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))
    for f in vis_frames:
        writer.write(f)
    writer.release()
    print(f"\nVideo saved: {out_path} ({len(vis_frames)} frames, {fps}fps)")

    # Also save a few sample frames as images
    sample_indices = [0, len(vis_frames)//4, len(vis_frames)//2,
                      3*len(vis_frames)//4, len(vis_frames)-1]
    for idx in sample_indices:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'sample_{idx:04d}.png'), vis_frames[idx])
    print(f"Sample frames saved to {OUTPUT_DIR}/sample_*.png")


if __name__ == '__main__':
    main()
