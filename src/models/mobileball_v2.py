"""
MobileBall v2: MobileNetV3-Large + SE cross-frame attention + joint prediction.

Architecture:
  Input: 3帧拼接 → 9ch × H × W
    ↓
  SE 跨帧注意力 (9→9, channel weighting across frames)
    ↓
  MobileNetV3-Large Encoder (first 4 layers)
    ├── features[0-1] → feat_1/4  (16ch, H/4×W/4)  ← skip
    └── features[2-3] → feat_1/8  (24ch, H/8×W/8)  ← main feature
    ↓
  Lightweight Decoder
    ├── Up×2 + concat(feat_1/4) → 32ch, H/4×W/4
    └── Up×2                    → 32ch, H/2×W/2  ← output resolution
    ↓
  Joint Prediction Head (1×1 Conv)
    ├── Heatmap  (1ch, sigmoid)   — ball location confidence
    ├── l        (1ch, relu)      — blur length (motion magnitude)
    └── θ        (2ch, unit vec)  — blur direction (cos, sin)

Target: iOS CoreML deployment
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class CrossFrameSE(nn.Module):
    """Squeeze-and-Excitation across input channels (frames)."""
    def __init__(self, channels=9, reduction=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LightDecoder(nn.Module):
    """Lightweight decoder with skip connection.
    
    feat_1/8 (24ch) → Up×2 + concat(feat_1/4, 16ch) → 32ch → Up×2 → 32ch (H/2×W/2)
    """
    def __init__(self, main_ch=24, skip_ch=16, out_ch=32):
        super().__init__()
        self.up1_conv = nn.Sequential(
            nn.Conv2d(main_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up2_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, feat_main, feat_skip):
        x = F.interpolate(feat_main, size=feat_skip.shape[2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, feat_skip], dim=1)
        x = self.up1_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2_conv(x)
        return x


class JointPredictionHead(nn.Module):
    """Joint prediction: heatmap + blur length + blur direction."""
    def __init__(self, in_ch=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.heatmap_conv = nn.Conv2d(in_ch, 1, 1)
        self.length_conv = nn.Conv2d(in_ch, 1, 1)
        self.direction_conv = nn.Conv2d(in_ch, 2, 1)
    
    def forward(self, x):
        feat = self.shared(x)
        heatmap = torch.sigmoid(self.heatmap_conv(feat))
        length = F.relu(self.length_conv(feat))
        direction = self.direction_conv(feat)
        direction = F.normalize(direction, p=2, dim=1)
        return heatmap, length, direction


class MobileBallV2(nn.Module):
    """MobileBall v2: MobileNetV3-Large (shallow) + SE + joint prediction.
    
    Input:  [B, 9, H, W]  (3 frames × 3 RGB)
    Output: heatmap [B, 1, H/2, W/2]
            length  [B, 1, H/2, W/2]
            theta   [B, 2, H/2, W/2]
    """
    
    def __init__(self, num_frames=3, pretrained_backbone=True):
        super().__init__()
        self.num_frames = num_frames
        in_channels = num_frames * 3

        # 1. SE cross-frame attention
        self.cross_frame_se = CrossFrameSE(channels=in_channels, reduction=3)
        
        # 2. MobileNetV3-Large backbone
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        
        # Modify first conv: 3ch → 9ch
        old_conv = mobilenet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, 16,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        if pretrained_backbone:
            with torch.no_grad():
                old_w = old_conv.weight
                repeats = (in_channels + 2) // 3
                new_w = old_w.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                new_w = new_w / num_frames
                new_conv.weight.copy_(new_w)
        mobilenet.features[0][0] = new_conv
        
        # MobileNetV3-Large first 4 layers:
        #   features[0]: Conv+BN+HSwish → [16, H/2, W/2]
        #   features[1]: InvertedResidual → [16, H/4, W/4]
        #   features[2]: InvertedResidual → [24, H/8, W/8]
        #   features[3]: InvertedResidual → [24, H/8, W/8]
        self.encoder_a = mobilenet.features[:2]   # → [B, 16, H/4, W/4]
        self.encoder_b = mobilenet.features[2:4]  # → [B, 24, H/8, W/8]
        
        # 3. Decoder
        self.decoder = LightDecoder(main_ch=24, skip_ch=16, out_ch=32)
        
        # 4. Joint prediction head
        self.head = JointPredictionHead(in_ch=32)
    
    def forward(self, x):
        x = self.cross_frame_se(x)
        feat_s4 = self.encoder_a(x)
        feat_s8 = self.encoder_b(feat_s4)
        feat = self.decoder(feat_s8, feat_s4)
        heatmap, length, direction = self.head(feat)
        return heatmap, length, direction
    
    def predict_position(self, x):
        """Run forward and extract ball (x, y) via soft-argmax."""
        heatmap, length, direction = self.forward(x)
        b, _, h, w = heatmap.shape
        hm = heatmap.view(b, -1)
        hm_soft = F.softmax(hm * 20, dim=1)
        
        coords_y = torch.arange(h, dtype=torch.float32, device=x.device)
        coords_x = torch.arange(w, dtype=torch.float32, device=x.device)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        
        pred_x = (hm_soft * grid_x.reshape(-1).unsqueeze(0)).sum(1) * 2
        pred_y = (hm_soft * grid_y.reshape(-1).unsqueeze(0)).sum(1) * 2
        conf = heatmap.view(b, -1).max(1)[0]
        
        return pred_x, pred_y, conf


def build_mobileball_v2(num_frames=3, pretrained=True, device='cuda'):
    return MobileBallV2(num_frames=num_frames, pretrained_backbone=pretrained).to(device)


if __name__ == '__main__':
    import time
    
    device = 'cpu'
    model = build_mobileball_v2(num_frames=3, pretrained=True, device=device)
    
    total = sum(p.numel() for p in model.parameters())
    se = sum(p.numel() for p in model.cross_frame_se.parameters())
    enc_a = sum(p.numel() for p in model.encoder_a.parameters())
    enc_b = sum(p.numel() for p in model.encoder_b.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    
    print(f"=== MobileBall v2 (Large backbone) ===")
    print(f"Total:      {total:>10,}")
    print(f"  SE:       {se:>10,}")
    print(f"  Enc A:    {enc_a:>10,}")
    print(f"  Enc B:    {enc_b:>10,}")
    print(f"  Decoder:  {dec:>10,}")
    print(f"  Head:     {head:>10,}")
    
    x = torch.randn(2, 9, 288, 512)
    hm, l, theta = model(x)
    print(f"\nInput:     {list(x.shape)}")
    print(f"Heatmap:   {list(hm.shape)}  range=[{hm.min():.3f}, {hm.max():.3f}]")
    print(f"Length:    {list(l.shape)}  range=[{l.min():.3f}, {l.max():.3f}]")
    print(f"Direction: {list(theta.shape)}  norm={theta.norm(dim=1).mean():.4f}")
    
    px, py, conf = model.predict_position(x)
    print(f"\nPred pos: x={px[0]:.1f}, y={py[0]:.1f}, conf={conf[0]:.3f}")
    
    model.eval()
    x = torch.randn(1, 9, 288, 512)
    for _ in range(3):
        with torch.no_grad(): model(x)
    
    times = []
    for _ in range(30):
        t0 = time.time()
        with torch.no_grad(): model(x)
        times.append(time.time() - t0)
    avg_ms = sum(times) / len(times) * 1000
    print(f"\nCPU inference: {avg_ms:.1f} ms/frame")
    
    torch.save(model.state_dict(), '/tmp/mobileball_v2_test.pth')
    size_mb = os.path.getsize('/tmp/mobileball_v2_test.pth') / 1024 / 1024
    print(f"Model file: {size_mb:.1f} MB")
