"""
Visualize TDF predictions at multiple thresholds on a single frame,
to see if ground balls can be detected at lower thresholds.
"""

import sys
import os
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_tdf import build_blurball_model

DATASET_ROOT = '/home/liuzhenlu/cyclex/BlurBall/dataset_blurball/blurball_dataset'
CHECKPOINT = '/home/liuzhenlu/cyclex/BlurBall/checkpoints/blurball_tdf/best.pth'
MODEL_CONFIG = '/home/liuzhenlu/cyclex/BlurBall/src/configs/model/blurball.yaml'
OUTPUT_DIR = '/home/liuzhenlu/cyclex/BlurBall/output/vis_tdf'

MATCH = 22
CLIP = '001'
IMG_SIZE = (288, 512)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_blurball_model(MODEL_CONFIG, None, tdf_output=True)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    frames_dir = os.path.join(DATASET_ROOT, f'{MATCH:02d}', 'frames', CLIP)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

    # Pick a frame where ground balls are visible
    test_frames = [100, 200, 300, 400, 500]
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]

    h, w = IMG_SIZE

    for fi in test_frames:
        # Load 3 frames
        center = max(1, min(fi, len(frame_files) - 2))
        imgs = []
        for offset in [-1, 0, 1]:
            fp = os.path.join(frames_dir, frame_files[center + offset])
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        orig_h, orig_w = imgs[0].shape[:2]

        processed = []
        for img in imgs:
            img_r = cv2.resize(img, (w, h))
            img_f = img_r.astype(np.float32) / 255.0
            img_f = img_f.transpose(2, 0, 1)
            processed.append(img_f)

        input_tensor = np.concatenate(processed, axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_tensor)
            pred = torch.sigmoid(out[0])

        pred_np = pred[0, 0].cpu().numpy()
        pred_orig = cv2.resize(pred_np, (orig_w, orig_h))

        # Print heatmap stats
        print(f"\nFrame {fi}: pred range [{pred_orig.min():.4f}, {pred_orig.max():.4f}]")
        print(f"  Pixels > 0.5: {(pred_orig > 0.5).sum()}")
        print(f"  Pixels > 0.3: {(pred_orig > 0.3).sum()}")
        print(f"  Pixels > 0.2: {(pred_orig > 0.2).sum()}")
        print(f"  Pixels > 0.1: {(pred_orig > 0.1).sum()}")
        print(f"  Pixels > 0.05: {(pred_orig > 0.05).sum()}")
        print(f"  Pixels > 0.02: {(pred_orig > 0.02).sum()}")

        # Create multi-threshold visualization
        frame_bgr = cv2.imread(os.path.join(frames_dir, frame_files[fi]))
        panels = []

        for thresh in thresholds:
            vis = frame_bgr.copy()
            mask = (pred_orig > thresh).astype(np.uint8)

            # Green fill
            green = vis.copy()
            green[mask > 0] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 0.6, green, 0.4, 0)

            # Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

            # Label
            n_pixels = mask.sum()
            n_blobs = len(contours)
            cv2.putText(vis, f'thresh={thresh}  {n_blobs} blobs  {n_pixels}px',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            panels.append(vis)

        # Also add raw heatmap
        hm_vis = (pred_orig * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        cv2.putText(hm_color, 'Raw heatmap', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(hm_color)

        # Arrange: 2 rows of panels
        # Row 1: thresh 0.5, 0.3, 0.2, heatmap
        # Row 2: thresh 0.1, 0.05, 0.02, original
        row1 = np.hstack([panels[0], panels[1], panels[2], panels[6]])
        row2 = np.hstack([panels[3], panels[4], panels[5], frame_bgr])
        grid = np.vstack([row1, row2])

        # Resize to reasonable size
        scale = 0.5
        grid_small = cv2.resize(grid, None, fx=scale, fy=scale)

        out_path = os.path.join(OUTPUT_DIR, f'multi_thresh_frame{fi:04d}.png')
        cv2.imwrite(out_path, grid_small)
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
