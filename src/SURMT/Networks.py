import torch
import torch.nn as nn

from .utils import SegmentationHead, GLCAttention, DecoderStage, ParallelWienerFusion
from .RMT import PatchMerging


class SURMT(nn.Module):
    def __init__(self, classes=2, channels=3):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.input_channels = channels

        # Stem: 简单3×3卷积嵌入
        self.stem = nn.Sequential(
            nn.Conv2d(channels, filters[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
        )

        # Encoder: GLCAttention + PatchMerging (4× → 8× → 16× → 32×)
        self.attn1 = GLCAttention(filters[0])
        self.down1 = PatchMerging(filters[0], filters[1])

        self.attn2 = GLCAttention(filters[1])
        self.down2 = PatchMerging(filters[1], filters[2])

        self.attn3 = GLCAttention(filters[2])
        self.down3 = PatchMerging(filters[2], filters[3])

        # Center
        self.center = GLCAttention(filters[3])
        self.hsa = GLCAttention(filters[3])

        # Skip connection fusion
        self.patch_skip_enhance = ParallelWienerFusion(channels=filters[0])

        # Decoder
        self.decoder_stage3 = DecoderStage(filters[3], filters[2], skip_channels=filters[2])
        self.decoder_stage2 = DecoderStage(filters[2], filters[1], skip_channels=filters[1])
        self.decoder_stage1 = DecoderStage(filters[1], filters[0], skip_channels=filters[0])

        # Auxiliary losses
        self.bottom_loss = nn.Sequential(
            nn.Conv2d(filters[3], classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=32, mode='bilinear'),
        )

        self.before_final_decoder_loss = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.GELU(),
            nn.Conv2d(filters[0], filters[0], 1),
            nn.BatchNorm2d(filters[0]),
            nn.GELU(),
            nn.ConvTranspose2d(filters[0], filters[0]//2, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(filters[0]//2),
            nn.GELU(),
            nn.Conv2d(filters[0]//2, filters[0]//2, 3, 1, 1),
            nn.BatchNorm2d(filters[0]//2),
            nn.GELU(),
            nn.ConvTranspose2d(filters[0]//2, filters[0]//4, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(filters[0]//4),
            nn.GELU(),
            nn.Conv2d(filters[0]//4, classes, 3, 1, 1),
        )

        self.segmentation_head = SegmentationHead(in_channels=filters[0], out_channels=classes)

    def forward(self, x):
        # Stem: 4×下采样
        x = self.stem(x)                     # (B, 64, H/4, W/4)

        patch_features = x                   # 保存给 skip connection

        # Encoder
        e1 = self.attn1(x)                   # (B, 64, H/4, W/4)
        e1_down = self.down1(e1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, 128, H/8, W/8)

        e2 = self.attn2(e1_down)             # (B, 128, H/8, W/8)
        e2_down = self.down2(e2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, 256, H/16, W/16)

        e3 = self.attn3(e2_down)             # (B, 256, H/16, W/16)
        e3_down = self.down3(e3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, 512, H/32, W/32)

        # Center
        e4 = self.center(e3_down)            # (B, 512, H/32, W/32)
        hsa_feature = self.hsa(e4)           # (B, 512, H/32, W/32)

        # Decoder
        d3 = self.decoder_stage3(hsa_feature, e3)   # (B, 256, H/16, W/16)
        d2 = self.decoder_stage2(d3, e2)            # (B, 128, H/8, W/8)
        d1 = self.decoder_stage1(d2, e1)            # (B, 64, H/4, W/4)

        patch_features_enhanced = self.patch_skip_enhance(patch_features, d1)

        out = self.segmentation_head(d1, patch_features_enhanced)

        return out
