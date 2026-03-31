"""
Load exported models for inference.

Usage:
    from load_models import load_ball_seg_model, load_event_model

    # 1. Ball detection + segmentation
    model_bb = load_ball_seg_model('checkpoints/blurball_ball_seg.pth')
    ball_hm, _, seg = model_bb(input_frames)  # input: [B, 9, 288, 512]

    # 2. Event classification
    model_ev = load_event_model('checkpoints/trajectory_event.pth')
    logits = model_ev(traj, seg_ctx)  # traj: [B, 9, 3], seg_ctx: [B, 4]
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models.blurball_multitask import build_blurball_multitask
from models.trajectory_event_model import TrajectoryEventModel, EVENT_NAMES


def load_ball_seg_model(checkpoint_path, device='cpu'):
    """Load BlurBall ball detection + segmentation model.

    Input:  [B, 9, 288, 512] — 3 RGB frames concatenated
    Output: ball_heatmap [B, 3, 288, 512], None, seg_logits [B, 4, 128, 320]
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    model = build_blurball_multitask(
        num_frames=config.get('num_frames', 3),
        device=device,
        tasks=tuple(config.get('tasks', ['ball', 'seg'])),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"BlurBall+Seg loaded: {total:,} params")
    return model


def load_event_model(checkpoint_path, device='cpu'):
    """Load trajectory event classification model.

    Input:  traj [B, 9, 3] (x, y, vis), seg_ctx [B, 4] (optional)
    Output: logits [B, 3] (bounce/net/empty)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    model = TrajectoryEventModel(
        traj_len=config.get('traj_len', 9),
        num_classes=config.get('num_classes', 3),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"TrajectoryEventModel loaded: {total:,} params")
    return model


if __name__ == '__main__':
    import torch.nn.functional as F

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load both models
    model_bb = load_ball_seg_model('checkpoints/blurball_ball_seg.pth', device)
    model_ev = load_event_model('checkpoints/trajectory_event.pth', device)

    # Simulate inference
    frames = torch.randn(1, 9, 288, 512).to(device)
    with torch.no_grad():
        ball_hm, _, seg = model_bb(frames)

    print(f"\n--- Ball+Seg inference ---")
    print(f"Ball heatmap: {list(ball_hm.shape)}, max={ball_hm.max():.3f}")
    print(f"Seg logits:   {list(seg.shape)}")

    # Extract ball position from heatmap
    hm = torch.sigmoid(ball_hm[0, 1])  # center frame
    py, px = (hm == hm.max()).nonzero(as_tuple=False)[0].tolist()
    print(f"Predicted ball: ({px}, {py})")

    # Extract seg class proportions
    seg_probs = F.softmax(seg, dim=1)
    seg_ctx = seg_probs.mean(dim=(2, 3))  # [B, 4]
    print(f"Seg proportions: {seg_ctx[0].tolist()}")

    # Simulate 9-frame trajectory
    traj = torch.zeros(1, 9, 3).to(device)
    for i in range(9):
        traj[0, i] = torch.tensor([px + i * 10, py - abs(i - 4) * 20, 1.0])

    with torch.no_grad():
        logits = model_ev(traj, seg_ctx)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(1).item()

    print(f"\n--- Event inference ---")
    print(f"Event: {EVENT_NAMES[pred]} (probs: bounce={probs[0,0]:.2f} net={probs[0,1]:.2f} empty={probs[0,2]:.2f})")
