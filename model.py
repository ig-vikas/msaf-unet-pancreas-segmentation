"""Four matched U-Net variants for axial pancreas segmentation."""

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("unet", "attention", "fusion", "attention_fusion")


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
    )


def resize(features, size):
    return F.interpolate(features, size=size, mode="bilinear", align_corners=False)


class AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.skip_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.decoder_projection = nn.Conv2d(channels, hidden, 1)
        self.weight = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, decoder):
        combined = self.skip_projection(skip) + self.decoder_projection(decoder)
        weight = torch.sigmoid(self.weight(F.relu(combined)))
        return skip * weight


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, channels, attention):
        super().__init__()
        self.project = nn.Conv2d(in_channels, channels, 1)
        self.gate = AttentionGate(channels) if attention else None
        self.conv = conv_block(channels * 2, channels)

    def forward(self, features, skip):
        features = resize(self.project(features), skip.shape[-2:])
        if self.gate is not None:
            skip = self.gate(skip, features)
        return self.conv(torch.cat((features, skip), dim=1))


class PancreasUNet(nn.Module):
    def __init__(self, variant="attention_fusion"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Choose one of {VARIANTS}.")
        attention = variant in ("attention", "attention_fusion")
        fusion = variant in ("fusion", "attention_fusion")
        self.pool = nn.MaxPool2d(2)
        self.enc1 = conv_block(1, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.center = conv_block(64, 128)
        self.dec3 = DecoderBlock(128, 64, attention)
        self.dec2 = DecoderBlock(64, 32, attention)
        self.dec1 = DecoderBlock(32, 16, attention)
        self.fuse = None
        if fusion:
            self.coarse = nn.Conv2d(64, 8, 1)
            self.middle = nn.Conv2d(32, 8, 1)
            self.fuse = conv_block(32, 16)
        self.output = nn.Conv2d(16, 3, 1)

    def forward(self, images):
        enc1 = self.enc1(images)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        center = self.center(self.pool(enc3))
        dec3 = self.dec3(center, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        if self.fuse is not None:
            size = dec1.shape[-2:]
            coarse = resize(self.coarse(dec3), size)
            middle = resize(self.middle(dec2), size)
            dec1 = self.fuse(torch.cat((dec1, middle, coarse), dim=1))
        return self.output(dec1)
