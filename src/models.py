import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone


# -----------------------
# 1. Tiny U-Net backbone
# -----------------------
class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_bn(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_bn(1, 16)
        self.enc2 = conv_bn(16, 32)
        self.enc3 = conv_bn(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = conv_bn(64 + 32, 32)
        self.dec1 = conv_bn(32 + 16, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = torch.cat([nn.functional.interpolate(e3, scale_factor=2, mode="bilinear"), e2], 1)
        d2 = self.dec2(d2)
        d1 = torch.cat([nn.functional.interpolate(d2, scale_factor=2, mode="bilinear"), e1], 1)
        d1 = self.dec1(d1)
        return self.out(d1)

class SpheroidUNet(nn.Module):
    def __init__(self, in_channels=2, base=16):
        super().__init__()
        self.enc1 = self._block(in_channels, base)   # 2ch -> 16
        self.enc2 = self._block(base, 32)
        self.enc3 = self._block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = self._block(64 + 32, 32)
        self.dec1 = self._block(32 + base, base)     # was conv_bn(32+16) — wrong concat size
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = torch.cat([nn.functional.interpolate(e3, scale_factor=2, mode="bilinear"), e2], 1)
        d2 = self.dec2(d2)
        d1 = torch.cat([nn.functional.interpolate(d2, scale_factor=2, mode="bilinear"), e1], 1)
        d1 = self.dec1(d1)
        return self.out(d1)


class PatternUNet(nn.Module):
    """4-level U-Net for the pattern (denoising) model.

    Full-resolution skip connections and double convolutions give sharper
    borders than SmallUNet. Works at `input_size` (default 512).
    """
    input_size = 512

    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = SpheroidUNet._block(1, c[0])
        self.enc2 = SpheroidUNet._block(c[0], c[1])
        self.enc3 = SpheroidUNet._block(c[1], c[2])
        self.enc4 = SpheroidUNet._block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.dec3 = SpheroidUNet._block(c[3] + c[2], c[2])
        self.dec2 = SpheroidUNet._block(c[2] + c[1], c[1])
        self.dec1 = SpheroidUNet._block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    @staticmethod
    def _up(x):
        return nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self._up(e4), e3], 1))
        d2 = self.dec2(torch.cat([self._up(d3), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2), e1], 1))
        return self.out(d1)
# ----------------------


def extract_state_dict(loaded_checkpoint):
    """
    Backward-compatible extractor:
    - old format: plain state_dict
    - new format: checkpoint dict with model_state_dict
    """
    if isinstance(loaded_checkpoint, dict) and "model_state_dict" in loaded_checkpoint:
        return loaded_checkpoint["model_state_dict"]
    return loaded_checkpoint


def load_pattern_model(model_path: str | Path, map_location: str="cpu"):
    """Load pattern detection model in path

    Parameters
    ----------
    model_path : str | Path
        path
    map_location : str, optional
        where to use the compute, by default "cpu"

    Returns
    -------
    nn.model
        model
    """
    checkpoint = torch.load(model_path, map_location=map_location)
    arch = checkpoint.get("arch", "SmallUNet") if isinstance(checkpoint, dict) else "SmallUNet"
    model = PatternUNet() if arch == "PatternUNet" else SmallUNet()
    model.input_size = checkpoint.get("input_size", 256) if isinstance(checkpoint, dict) else 256
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, state_dict

def load_spheroid_model(model_path: str | Path, map_location: str="cpu"):
    model = SpheroidUNet(in_channels=2)
    checkpoint = torch.load(model_path, map_location=map_location)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, state_dict

