"""
MobileBall: MobileNetV3-Large backbone + BlurBall-style ball detection.

Core idea from BlurBall:
  - Multi-frame input (3 consecutive frames → 9ch)
  - Deconv head → heatmap output for ball position
  - Gaussian heatmap supervision

Backbone: MobileNetV3-Large (modified first conv for 9ch input)
Head: Lightweight deconv upsampler → [B, num_frames, H, W] heatmap

Target: iOS deployment via CoreML
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class BallHeatmapHead(nn.Module):
    """Lightweight deconv head: upsample backbone features to heatmap.
    
    Takes feature map from backbone and upsamples to input resolution
    for ball position heatmap prediction.
    """
    def __init__(self, in_channels, out_channels, target_size=(288, 512)):
        super().__init__()
        self.target_size = target_size
        
        # Progressive upsample: reduce channels while upsampling
        # From 1/32 resolution back to full resolution
        self.up = nn.Sequential(
            # Stage 1: reduce channels
            nn.Conv2d(in_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 2: deconv 2x
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3: deconv 2x
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 4: deconv 2x
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Final 1x1 conv to output channels
        self.final = nn.Conv2d(16, out_channels, 1)
    
    def forward(self, x):
        x = self.up(x)
        x = self.final(x)
        # Resize to exact target if needed (deconv may not match exactly)
        if x.shape[2] != self.target_size[0] or x.shape[3] != self.target_size[1]:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x


class BallHeatmapHeadWithSkip(nn.Module):
    """Deconv head with skip connections from backbone intermediate features.
    
    Uses features from stride-4 and stride-8 levels for better localization.
    """
    def __init__(self, feat_channels, skip_channels_s4, skip_channels_s8,
                 out_channels, target_size=(288, 512)):
        super().__init__()
        self.target_size = target_size
        
        # Process deep features (stride 32 → stride 8)
        self.up1 = nn.Sequential(
            nn.Conv2d(feat_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Fuse with stride-8 skip (80ch from features[7-10])
        self.fuse_s8 = nn.Sequential(
            nn.Conv2d(64 + skip_channels_s8, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Upsample stride 8 → stride 4
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Fuse with stride-4 skip (40ch from features[4-6])
        self.fuse_s4 = nn.Sequential(
            nn.Conv2d(32 + skip_channels_s4, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Final upsample stride 4 → stride 2 → stride 1
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv2d(16, out_channels, 1)
    
    def forward(self, feat_deep, feat_s8, feat_s4):
        # Deep features → stride 8
        x = self.up1(feat_deep)
        
        # Fuse with stride-8 features
        if x.shape[2:] != feat_s8.shape[2:]:
            x = F.interpolate(x, size=feat_s8.shape[2:], mode='bilinear', align_corners=False)
        x = self.fuse_s8(torch.cat([x, feat_s8], dim=1))
        
        # Upsample to stride 4
        x = self.up2(x)
        
        # Fuse with stride-4 features
        if x.shape[2:] != feat_s4.shape[2:]:
            x = F.interpolate(x, size=feat_s4.shape[2:], mode='bilinear', align_corners=False)
        x = self.fuse_s4(torch.cat([x, feat_s4], dim=1))
        
        # Final upsample
        x = self.up3(x)
        x = self.final(x)
        
        if x.shape[2] != self.target_size[0] or x.shape[3] != self.target_size[1]:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x


class MobileBall(nn.Module):
    """MobileNetV3-Large backbone + ball detection head.
    
    Input: [B, num_frames*3, H, W] (concatenated RGB frames)
    Output: [B, num_frames, H, W] (ball position heatmap per frame)
    
    Features extracted at 3 scales for skip connections:
      - stride 4: features[4] output (40ch, H/4, W/4) — fine detail
      - stride 8: features[7] output (80ch, H/8, W/8) — mid level
      - stride 32: features[16] output (960ch, H/32, W/32) — deep semantics
    """
    
    def __init__(self, num_frames=3, img_size=(288, 512), use_skip=True,
                 pretrained_backbone=True):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.use_skip = use_skip
        
        # Load MobileNetV3-Large backbone
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        
        # Modify first conv: 3ch → 9ch (num_frames * 3)
        old_conv = mobilenet.features[0][0]  # Conv2d(3, 16, ...)
        new_conv = nn.Conv2d(
            num_frames * 3, 16,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Initialize: repeat pretrained 3ch weights for each frame group
        if pretrained_backbone:
            with torch.no_grad():
                old_weight = old_conv.weight  # [16, 3, k, k]
                # Tile to cover num_frames * 3 input channels
                repeats = (num_frames * 3 + 2) // 3  # ceil division
                new_weight = old_weight.repeat(1, repeats, 1, 1)[:, :num_frames*3, :, :]
                new_weight = new_weight / num_frames  # scale down
                new_conv.weight.copy_(new_weight)
        
        mobilenet.features[0][0] = new_conv
        
        # Extract backbone stages
        # features[0-3]: stride 2-4 (16→24ch)
        # features[4-6]: stride 8 (40ch) — skip_s4
        # features[7-12]: stride 16 (80→112ch) — skip_s8
        # features[13-16]: stride 32 (160→960ch) — deep
        self.stage1 = mobilenet.features[:4]    # → stride 4, 24ch
        self.stage2 = mobilenet.features[4:7]   # → stride 8, 40ch
        self.stage3 = mobilenet.features[7:13]  # → stride 16, 112ch
        self.stage4 = mobilenet.features[13:]   # → stride 32, 960ch
        
        # Ball detection head
        if use_skip:
            self.ball_head = BallHeatmapHeadWithSkip(
                feat_channels=960,
                skip_channels_s4=40,    # stage2 output
                skip_channels_s8=112,   # stage3 output
                out_channels=num_frames,
                target_size=img_size,
            )
        else:
            self.ball_head = BallHeatmapHead(
                in_channels=960,
                out_channels=num_frames,
                target_size=img_size,
            )
    
    def forward(self, x):
        # Backbone forward with intermediate features
        x1 = self.stage1(x)    # [B, 24, H/4, W/4]
        x2 = self.stage2(x1)   # [B, 40, H/8, W/8]
        x3 = self.stage3(x2)   # [B, 112, H/16, W/16]
        x4 = self.stage4(x3)   # [B, 960, H/32, W/32]
        
        # Ball heatmap
        if self.use_skip:
            heatmap = self.ball_head(x4, x3, x2)
        else:
            heatmap = self.ball_head(x4)
        
        return heatmap  # [B, num_frames, H, W]
    
    def get_backbone_features(self, x):
        """Extract backbone features (for future multi-task use)."""
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return {'s4': x1, 's8': x2, 's16': x3, 's32': x4}


def build_mobileball(num_frames=3, img_size=(288, 512), use_skip=True,
                     pretrained=True, device='cuda'):
    """Build MobileBall model."""
    model = MobileBall(
        num_frames=num_frames,
        img_size=img_size,
        use_skip=use_skip,
        pretrained_backbone=pretrained,
    ).to(device)
    return model


if __name__ == '__main__':
    import time
    
    model = build_mobileball(num_frames=3, device='cpu', use_skip=True)
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.stage1.parameters()) + \
               sum(p.numel() for p in model.stage2.parameters()) + \
               sum(p.numel() for p in model.stage3.parameters()) + \
               sum(p.numel() for p in model.stage4.parameters())
    head = sum(p.numel() for p in model.ball_head.parameters())
    print(f"Total: {total:,}  Backbone: {backbone:,}  Head: {head:,}")
    
    # Test forward
    x = torch.randn(2, 9, 288, 512)
    out = model(x)
    print(f"Input: {list(x.shape)} → Output: {list(out.shape)}")
    
    # Speed test
    model.eval()
    x = torch.randn(1, 9, 288, 512)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(x)
    
    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.time()
        with torch.no_grad():
            model(x)
        times.append(time.time() - t0)
    
    avg_ms = np.mean(times) * 1000 if 'numpy' in dir() else sum(times)/len(times)*1000
    print(f"CPU inference: {avg_ms:.1f} ms/frame")
    
    # No-skip version for comparison
    model_noskip = build_mobileball(num_frames=3, device='cpu', use_skip=False)
    total_ns = sum(p.numel() for p in model_noskip.parameters())
    print(f"\nNo-skip version: {total_ns:,} params")
