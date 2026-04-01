"""
Extract 13 table keypoints from each TTNet video using UpliftingTableTennis.

For each video: read a few frames, run table detection, average/filter the results.
Save as JSON: {split/game: {keypoints: [[x,y,vis]*13], resolution: [w,h]}}

The table is static per video, so we only need to run this once per game.
"""

import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

# Add UpliftingTableTennis to path
UTT_PATH = '/home/liuzhenlu/cyclex/UpliftingTableTennis'
sys.path.insert(0, UTT_PATH)

DATASET_ROOT = '/home/liuzhenlu/cyclex/TOTNet/dataset'
OUTPUT_PATH = '/home/liuzhenlu/cyclex/BlurBall/checkpoints/table_keypoints_13.json'

KEYPOINT_NAMES = [
    'close_left', 'close_right', 'center_left', 'center_right',
    'far_left', 'far_right', 'net_left_bot', 'net_right_bot',
    'net_center_bot', 'net_left_top', 'net_right_top',
    'close_center', 'far_center',
]


def extract_from_video(detector, video_path, n_samples=10):
    """Run table detection on a few frames, return averaged keypoints."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames evenly across video
    indices = np.linspace(100, total - 100, n_samples, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        return None, w, h

    # Predict on all sampled frames
    keypoints, _ = detector.predict(frames)  # (N, 13, 3)

    # Average visible keypoints across frames
    avg_kp = np.zeros((13, 3), dtype=np.float32)
    for i in range(13):
        visible = keypoints[:, i, 2] > 0.5
        if visible.sum() > 0:
            avg_kp[i, 0] = keypoints[visible, i, 0].mean()
            avg_kp[i, 1] = keypoints[visible, i, 1].mean()
            avg_kp[i, 2] = 1.0
        else:
            avg_kp[i, 2] = 0.0

    return avg_kp, w, h


def main():
    from interface import TableDetector

    detector = TableDetector(model_name='segformerpp_b2')
    print("Table detector loaded")

    results = {}

    for split in ['training', 'test']:
        videos_dir = os.path.join(DATASET_ROOT, split, 'videos')
        if not os.path.isdir(videos_dir):
            continue

        videos = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])

        for vf in tqdm(videos, desc=split):
            game = vf.replace('.mp4', '')
            video_path = os.path.join(videos_dir, vf)

            kp, w, h = extract_from_video(detector, video_path)

            key = f'{split}/{game}'
            if kp is not None:
                results[key] = {
                    'keypoints': kp.tolist(),
                    'resolution': [w, h],
                    'keypoint_names': KEYPOINT_NAMES,
                }
                n_vis = int((kp[:, 2] > 0.5).sum())
                print(f"  {key}: {n_vis}/13 keypoints detected, res={w}x{h}")
            else:
                print(f"  {key}: FAILED")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    os.chdir(UTT_PATH)
    main()
