"""
Table Keypoint Regression Head.

Predicts 6 keypoints (12 coordinates) of the table tennis table:
  1. near-left:  bottom-left corner of table
  2. near-right: bottom-right corner of table
  3. far-left:   top-left corner of table
  4. far-right:  top-right corner of table
  5. net-left:   left end of net
  6. net-right:  right end of net

Input:  backbone multi-scale features (4 branches from HRNet)
Output: [B, 12] — (x, y) for each of 6 keypoints, normalized to [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

KEYPOINT_NAMES = [
    'near_left', 'near_right', 'far_left', 'far_right',
    'net_left', 'net_right',
]


class TableKeypointHead(nn.Module):
    """Regress 6 table keypoints from backbone features."""

    def __init__(self, branch_channels=(16, 32, 64, 128), num_keypoints=6):
        super().__init__()
        self.num_keypoints = num_keypoints
        total_ch = sum(branch_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(total_ch, 64, 1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_keypoints * 2),
            nn.Sigmoid(),  # output in [0, 1] range
        )

    def forward(self, y_list):
        """
        Args:
            y_list: list of 4 branch features from HRNet backbone
        Returns:
            keypoints: [B, num_keypoints * 2] — (x1,y1, x2,y2, ...) normalized to [0,1]
        """
        h, w = y_list[0].shape[2:]
        up = []
        for feat in y_list:
            if feat.shape[2] != h or feat.shape[3] != w:
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            up.append(feat)
        fused = torch.cat(up, dim=1)
        fused = self.fuse(fused)
        pooled = self.pool(fused).flatten(1)
        return self.regressor(pooled)

    def decode_keypoints(self, output, img_w, img_h):
        """Convert normalized [0,1] output to pixel coordinates.

        Args:
            output: [B, 12] or [12] normalized coordinates
            img_w, img_h: target image dimensions

        Returns:
            dict mapping keypoint_name → (x, y) in pixel coords
        """
        if output.dim() == 1:
            output = output.unsqueeze(0)
        coords = output[0].detach().cpu().numpy()
        result = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            x = coords[i * 2] * img_w
            y = coords[i * 2 + 1] * img_h
            result[name] = (float(x), float(y))
        return result
