"""
BlurBall Multi-Task: BlurBall backbone (HRNet+SE) + Cascaded Heads

Architecture (cascaded):
  1. Ball heatmap head  → ball position (from backbone deconv)
  2. Segmentation head  → table/net/person regions (from backbone multi-scale)
  3. Event head (cascade) → bounce/net/empty, conditioned on:
     - backbone features (branch2+3)
     - ball heatmap (where is the ball?)
     - seg map (where is the table/net?)

The cascade allows event detection to leverage spatial context from
ball position and table geometry.
"""

import os
import logging
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blurball import BlurBall
from models.trajectory_event_model import TrajectoryEventModel

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class CascadeEventHead(nn.Module):
    """Cascade event classification: bounce / net / empty_event.

    Inputs:
      - backbone features (branch2: 64ch + branch3: 128ch) → 192ch
      - ball heatmap (num_frames ch) → spatial ball info
      - seg map (num_seg_classes ch) → table/net regions

    The heatmap and seg map are encoded with small convs, then
    concatenated with backbone features before global pooling + FC.
    """
    def __init__(self, backbone_channels=192, num_frames=3, num_seg_classes=4,
                 num_classes=3, dropout_p=0.3):
        super().__init__()
        # Encode ball heatmap → 16ch feature
        self.hm_encoder = nn.Sequential(
            nn.Conv2d(num_frames, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # Encode seg map → 16ch feature
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(num_seg_classes, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # Total: backbone(192) + hm(16) + seg(16) = 224
        total_ch = backbone_channels + 16 + 16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, backbone_feat, ball_hm, seg_logits):
        """
        Args:
            backbone_feat: [B, 192, H, W] (branch2+3 concatenated)
            ball_hm: [B, num_frames, H_hm, W_hm] (ball heatmap)
            seg_logits: [B, num_seg_classes, 128, 320] (seg output)
        """
        target_size = backbone_feat.shape[2:]

        # Resize heatmap and seg to match backbone feature size
        hm = F.interpolate(ball_hm, size=target_size, mode='bilinear', align_corners=False)
        hm_feat = self.hm_encoder(hm)

        seg = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        seg_feat = self.seg_encoder(seg.detach())  # detach to avoid unstable early gradients

        fused = torch.cat([backbone_feat, hm_feat, seg_feat], dim=1)
        fused = self.gap(fused)
        fused = fused.view(fused.size(0), -1)
        return self.fc(fused)


class TrajectoryEventHead(nn.Module):
    """Event classification from ball trajectory + segmentation context.

    Uses temporal ball positions (前5帧+当前+后3帧 = 9 positions) to
    classify bounce/net/empty. Ball trajectory reveals motion pattern:
    - Bounce: ball changes vertical direction near table surface
    - Net: ball stops/changes near net position
    - Empty: no special pattern

    Inputs:
      - traj: [B, T, 3] ball positions (x, y, visibility) for T frames
      - seg_logits: [B, num_seg_classes, Hs, Ws] segmentation prediction
        → extract table/net spatial features as context

    Architecture:
      1. Trajectory encoder: 1D Conv over time → temporal features
      2. Seg context: GAP over seg map → class-wise spatial summary
      3. Fuse trajectory + seg context → FC classifier
    """
    def __init__(self, traj_len=9, num_seg_classes=4, num_classes=3,
                 hidden_dim=64, dropout_p=0.3):
        super().__init__()
        self.traj_len = traj_len

        # Trajectory features: (x, y, vis) per frame + derived features
        # Derived: dx, dy (velocity), ddx, ddy (acceleration)
        # Raw: 3 * T, Velocity: 2 * (T-1), Accel: 2 * (T-2)
        # But simpler: just use 1D conv on the raw (x, y, vis) sequence
        self.traj_encoder = nn.Sequential(
            # [B, 3, T] → [B, 32, T]
            nn.Conv1d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # [B, 32, T] → [B, 64, T]
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # [B, 64, 1]
        )

        # Velocity/acceleration features (handcrafted)
        # velocity: T-1 values of (dx, dy), acceleration: T-2 values of (ddx, ddy)
        vel_feat_dim = 2 * (traj_len - 1) + 2 * (traj_len - 2)  # 16 + 14 = 30 for T=9

        # Seg context: per-class spatial statistics
        # For each seg class: mean activation → num_seg_classes features
        seg_feat_dim = num_seg_classes

        # Fuse: traj(64) + velocity(30) + seg(4)
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

    def forward(self, traj, seg_logits):
        """
        Args:
            traj: [B, T, 3] - (x, y, visibility) for each frame
            seg_logits: [B, C, Hs, Ws] - segmentation prediction
        Returns:
            event_logits: [B, num_classes]
        """
        B = traj.size(0)

        # 1. Trajectory conv features
        # Normalize positions to [0, 1] range for stability
        traj_norm = traj.clone()
        traj_norm[:, :, 0] = traj_norm[:, :, 0] / 512.0  # x / width
        traj_norm[:, :, 1] = traj_norm[:, :, 1] / 288.0  # y / height
        # [B, T, 3] → [B, 3, T] for Conv1d
        traj_conv_in = traj_norm.permute(0, 2, 1)
        traj_feat = self.traj_encoder(traj_conv_in).squeeze(-1)  # [B, 64]

        # 2. Velocity and acceleration (handcrafted)
        pos = traj_norm[:, :, :2]  # [B, T, 2]
        vis = traj_norm[:, :, 2:3]  # [B, T, 1]
        # Velocity: pos[t+1] - pos[t], masked by visibility
        vel = pos[:, 1:] - pos[:, :-1]  # [B, T-1, 2]
        vel_mask = vis[:, 1:] * vis[:, :-1]  # both frames must be visible
        vel = vel * vel_mask
        # Acceleration: vel[t+1] - vel[t]
        acc = vel[:, 1:] - vel[:, :-1]  # [B, T-2, 2]
        acc_mask = vel_mask[:, 1:] * vel_mask[:, :-1]
        acc = acc * acc_mask

        vel_flat = vel.reshape(B, -1)  # [B, 2*(T-1)]
        acc_flat = acc.reshape(B, -1)  # [B, 2*(T-2)]
        motion_feat = torch.cat([vel_flat, acc_flat], dim=1)  # [B, 30]

        # 3. Segmentation context: mean activation per class
        seg_soft = F.softmax(seg_logits.detach(), dim=1)  # [B, C, H, W]
        seg_ctx = seg_soft.mean(dim=(2, 3))  # [B, C] - class proportions

        # 4. Fuse and classify
        fused = torch.cat([traj_feat, motion_feat, seg_ctx], dim=1)
        return self.classifier(fused)


class SegmentationHead(nn.Module):
    """Lightweight segmentation: fuse all 4 HRNet branches → pixel-wise classes.
    Output: [B, num_classes, 128, 320] matching TTNet seg mask size.
    """
    def __init__(self, branch_channels=(16, 32, 64, 128), num_classes=4, seg_size=(128, 320)):
        super().__init__()
        self.seg_size = seg_size
        total_ch = sum(branch_channels)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(total_ch, 64, 1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.seg_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, y_list):
        h, w = y_list[0].shape[2:]
        up = []
        for feat in y_list:
            if feat.shape[2] != h or feat.shape[3] != w:
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            up.append(feat)
        fused = torch.cat(up, dim=1)
        fused = self.fuse_conv(fused)
        seg = self.seg_conv(fused)
        if seg.shape[2] != self.seg_size[0] or seg.shape[3] != self.seg_size[1]:
            seg = F.interpolate(seg, size=self.seg_size, mode='bilinear', align_corners=False)
        return seg


class BlurBallMultiTask(nn.Module):
    """BlurBall backbone + multi-task heads."""

    def __init__(self, cfg, num_event_classes=3, num_seg_classes=4,
                 seg_size=(128, 320), tasks=('ball', 'event', 'seg'),
                 num_frames=3, event_head_type='cascade', traj_len=9):
        super().__init__()
        self.tasks = tasks
        self.event_head_type = event_head_type
        self.backbone = BlurBall(cfg)

        stage4_ch = cfg['MODEL']['EXTRA']['STAGE4']['NUM_CHANNELS']  # [16,32,64,128]

        # Head 2: Segmentation (runs before event)
        self.seg_head = SegmentationHead(
            branch_channels=stage4_ch,
            num_classes=num_seg_classes,
            seg_size=seg_size,
        ) if 'seg' in tasks else None

        # Head 3: Event classification
        if 'event' not in tasks:
            self.event_head = None
            self.traj_event_head = None
        elif event_head_type == 'trajectory':
            self.event_head = None
            self.traj_event_head = TrajectoryEventModel(
                traj_len=traj_len,
                num_seg_classes=num_seg_classes,
                num_classes=num_event_classes,
            )
        else:
            # Legacy cascade head
            self.event_head = CascadeEventHead(
                backbone_channels=stage4_ch[2] + stage4_ch[3],
                num_frames=num_frames,
                num_seg_classes=num_seg_classes,
                num_classes=num_event_classes,
            )
            self.traj_event_head = None

    def backbone_forward(self, x):
        """Run backbone up to stage4, return y_list (4 branches)."""
        bb = self.backbone
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.conv2(x)
        x = bb.bn2(x)
        x = bb.relu(x)
        x = bb.layer1(x)

        x_list = []
        for i in range(bb.stage2_cfg['NUM_BRANCHES']):
            if bb.transition1[i] is not None:
                x_list.append(bb.transition1[i](x))
            else:
                x_list.append(x)
        y_list = bb.stage2(x_list)

        x_list = []
        for i in range(bb.stage3_cfg['NUM_BRANCHES']):
            if bb.transition2[i] is not None:
                x_list.append(bb.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = bb.stage3(x_list)

        x_list = []
        for i in range(bb.stage4_cfg['NUM_BRANCHES']):
            if bb.transition3[i] is not None:
                x_list.append(bb.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = bb.stage4(x_list)
        return y_list

    def forward(self, x, traj=None, table_kp=None):
        """
        Args:
            x: [B, N*3, H, W] input frames
            traj: [B, T, 3] ball trajectory (x, y, vis) - only for trajectory event head
            table_kp: [B, 13, 2] table keypoints in pixel coords (optional)
        """
        y_list = self.backbone_forward(x)

        # Head 1: Ball heatmap (original BlurBall deconv head)
        ball_heatmap = None
        if 'ball' in self.tasks:
            bb = self.backbone
            y_out = {}
            for scale in bb._out_scales:
                feat = y_list[scale]
                for i in range(bb.num_deconvs):
                    feat = bb.deconv_layers[i][scale](feat)
                y_out[scale] = bb.final_layers[scale](feat)
            ball_heatmap = y_out[0]  # [B, frames_out, H, W]

        # Head 2: Segmentation
        seg_logits = None
        if self.seg_head is not None:
            seg_logits = self.seg_head(y_list)

        # Head 3: Event classification
        event_logits = None
        if self.traj_event_head is not None and traj is not None:
            # Trajectory-based event head
            sg = seg_logits if seg_logits is not None else torch.zeros(
                x.size(0), 4, 128, 320, device=x.device)
            # Extract seg context for trajectory event head
            seg_ctx = F.softmax(sg.detach(), dim=1).mean(dim=(2, 3))  # [B, 4]
            event_logits = self.traj_event_head(traj, seg_ctx, table_kp)
        elif self.event_head is not None:
            # Legacy cascade event head
            b2 = F.interpolate(y_list[2], size=y_list[3].shape[2:],
                               mode='bilinear', align_corners=False)
            backbone_feat = torch.cat([b2, y_list[3]], dim=1)

            hm = ball_heatmap if ball_heatmap is not None else torch.zeros(
                x.size(0), 3, y_list[0].shape[2], y_list[0].shape[3], device=x.device)
            sg = seg_logits if seg_logits is not None else torch.zeros(
                x.size(0), 4, 128, 320, device=x.device)

            event_logits = self.event_head(backbone_feat, hm, sg)

        return ball_heatmap, event_logits, seg_logits

    def load_blurball_checkpoint(self, ckpt_path):
        """Load pre-trained BlurBall (ball-only) weights into backbone."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # BlurBall saves as {'model_state_dict': ..., 'epoch': ..., ...}
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        # Build candidate state dict
        candidate_sd = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if not isinstance(v, torch.Tensor):
                continue
            candidate_sd[f'backbone.{k}'] = v

        # Filter out shape-mismatched keys (e.g. conv1 when num_frames differs)
        model_sd = self.state_dict()
        new_sd = {}
        skipped = []
        for k, v in candidate_sd.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped.append(k)
            else:
                new_sd[k] = v

        if skipped:
            logger.info(f"Skipped {len(skipped)} shape-mismatched keys: {skipped}")

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded BlurBall ckpt: {len(new_sd)} keys loaded, "
                    f"{len(skipped)} skipped (shape), "
                    f"{len(missing)} missing, {len(unexpected)} unexpected")
        return missing, unexpected

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen")


def build_blurball_multitask(num_frames=3, device='cuda',
                             tasks=('ball', 'event', 'seg'),
                             event_head_type='cascade', traj_len=9):
    """Build multi-task BlurBall from blurball.yaml config."""
    from omegaconf import OmegaConf
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'model', 'blurball.yaml')
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg['frames_in'] = num_frames
    cfg['frames_out'] = num_frames
    # BlurBall's _make_deconv_layers uses dot notation (cfg.MODEL.EXTRA)
    # Convert to OmegaConf DictConfig for attribute access
    cfg = OmegaConf.create(cfg)

    model = BlurBallMultiTask(cfg, tasks=tasks, num_frames=num_frames,
                              event_head_type=event_head_type,
                              traj_len=traj_len).to(device)
    return model


if __name__ == '__main__':
    model = build_blurball_multitask(num_frames=3, device='cpu')
    total = sum(p.numel() for p in model.parameters())
    bb = sum(p.numel() for p in model.backbone.parameters())
    ev = sum(p.numel() for p in model.event_head.parameters()) if model.event_head else 0
    sg = sum(p.numel() for p in model.seg_head.parameters()) if model.seg_head else 0
    print(f"Total: {total:,}  Backbone: {bb:,}  Event: {ev:,}  Seg: {sg:,}")

    x = torch.randn(2, 9, 288, 512)
    ball, event, seg = model(x)
    print(f"Ball: {ball.shape}  Event: {event.shape}  Seg: {seg.shape}")
