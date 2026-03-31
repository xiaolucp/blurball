"""
Trajectory Event Model — standalone event classifier.

A lightweight model that classifies table tennis events (bounce/net/empty)
from ball trajectory sequences. Fully independent from the vision backbone.

Input:
  - traj: [B, T, 3] — ball positions (x, y, visibility) for T consecutive frames
  - seg_ctx: [B, 4] — segmentation class proportions (background/person/table/scoreboard)
    (optional, can be zeros if not available)

Output:
  - event_logits: [B, 3] — bounce / net / empty_event

Architecture:
  1. Normalize trajectory to [0,1] range
  2. 1D Conv encoder: extract temporal patterns from position sequence
  3. Handcrafted features: velocity (dx, dy) and acceleration (ddx, ddy)
  4. Fuse conv features + motion features + seg context
  5. MLP classifier → 3 classes

Model size: ~49K parameters
Inference: <1ms on CPU (just MLP, no vision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEventModel(nn.Module):
    """Standalone event classifier from ball trajectory."""

    def __init__(self, traj_len=9, num_seg_classes=4, num_classes=3,
                 hidden_dim=64, dropout_p=0.3,
                 input_width=512, input_height=288):
        super().__init__()
        self.traj_len = traj_len
        self.input_width = input_width
        self.input_height = input_height

        # 1D Conv trajectory encoder
        self.traj_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Handcrafted motion features dimension
        # velocity: 2 * (T-1), acceleration: 2 * (T-2)
        vel_feat_dim = 2 * (traj_len - 1) + 2 * (traj_len - 2)

        # Seg context: class proportions
        seg_feat_dim = num_seg_classes

        # Classifier
        total_dim = 64 + vel_feat_dim + seg_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, traj, seg_ctx=None):
        """
        Args:
            traj: [B, T, 3] — (x, y, visibility) per frame, in pixel coords
            seg_ctx: [B, 4] — segmentation class proportions (optional)
        Returns:
            event_logits: [B, num_classes]
        """
        B = traj.size(0)

        # Normalize positions to [0, 1]
        traj_norm = traj.clone()
        traj_norm[:, :, 0] = traj_norm[:, :, 0] / self.input_width
        traj_norm[:, :, 1] = traj_norm[:, :, 1] / self.input_height

        # 1D Conv features
        traj_conv_in = traj_norm.permute(0, 2, 1)  # [B, 3, T]
        traj_feat = self.traj_encoder(traj_conv_in).squeeze(-1)  # [B, 64]

        # Velocity and acceleration
        pos = traj_norm[:, :, :2]       # [B, T, 2]
        vis = traj_norm[:, :, 2:3]      # [B, T, 1]

        vel = pos[:, 1:] - pos[:, :-1]  # [B, T-1, 2]
        vel_mask = vis[:, 1:] * vis[:, :-1]
        vel = vel * vel_mask

        acc = vel[:, 1:] - vel[:, :-1]  # [B, T-2, 2]
        acc_mask = vel_mask[:, 1:] * vel_mask[:, :-1]
        acc = acc * acc_mask

        motion_feat = torch.cat([vel.reshape(B, -1), acc.reshape(B, -1)], dim=1)

        # Seg context (default zeros if not provided)
        if seg_ctx is None:
            seg_ctx = torch.zeros(B, 4, device=traj.device)

        # Fuse and classify
        fused = torch.cat([traj_feat, motion_feat, seg_ctx], dim=1)
        return self.classifier(fused)

    @torch.no_grad()
    def predict(self, traj, seg_ctx=None):
        """Convenience method: returns class index and probabilities."""
        self.eval()
        logits = self.forward(traj, seg_ctx)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1)
        return pred_class, probs


EVENT_NAMES = {0: 'bounce', 1: 'net', 2: 'empty'}


def build_trajectory_event_model(traj_len=9, device='cpu', checkpoint=None):
    """Build and optionally load a trained model."""
    model = TrajectoryEventModel(traj_len=traj_len).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        # Extract trajectory event head weights if from multi-task model
        traj_sd = {}
        for k, v in sd.items():
            if k.startswith('traj_event_head.'):
                traj_sd[k.replace('traj_event_head.', '')] = v

        if traj_sd:
            model.load_state_dict(traj_sd)
        else:
            # Try loading directly
            model.load_state_dict(sd, strict=False)

    total = sum(p.numel() for p in model.parameters())
    print(f"TrajectoryEventModel: {total:,} params, traj_len={traj_len}")
    return model


if __name__ == '__main__':
    model = build_trajectory_event_model(traj_len=9)

    # Test forward
    traj = torch.randn(4, 9, 3)  # 4 samples, 9 frames, (x, y, vis)
    traj[:, :, 0] = traj[:, :, 0] * 512   # x in pixel range
    traj[:, :, 1] = traj[:, :, 1] * 288   # y in pixel range
    traj[:, :, 2] = (traj[:, :, 2] > 0).float()  # visibility binary

    seg_ctx = torch.tensor([[0.6, 0.15, 0.2, 0.05]] * 4)  # seg class proportions

    logits = model(traj, seg_ctx)
    pred, probs = model.predict(traj, seg_ctx)

    print(f"Input:  traj {list(traj.shape)}, seg_ctx {list(seg_ctx.shape)}")
    print(f"Output: logits {list(logits.shape)}")
    print(f"Predictions: {[EVENT_NAMES[p.item()] for p in pred]}")
    print(f"Probs: {probs[0].tolist()}")

    # Size
    import os, tempfile
    tmp = os.path.join(tempfile.gettempdir(), 'traj_event.pth')
    torch.save(model.state_dict(), tmp)
    size_kb = os.path.getsize(tmp) / 1024
    print(f"\nModel file: {size_kb:.0f} KB")
