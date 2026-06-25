"""CNN models for the P4 CNRPark+EXT occupancy experiment.

Two models, differing by exactly ONE architectural change:

  BaselineCNN      -> 3 conv blocks (32->64->128) + MaxPool + Flatten-FC head.
  ExperimentalCNN  -> 4 conv blocks (32->64->128->256) + MaxPool + Flatten-FC head
                      with a wider input dim (256*9*9 vs 128*18*18).

The single change is depth: the experimental model adds one more conv
block (128->256). The head uses the same `Flatten`->FC mechanism as
baseline — only the input dim grows because the deeper stack produces a
wider final feature map (256 channels instead of 128). No BatchNorm, no
dropout, no other regularization — so the comparison isolates 'depth'
vs. the baseline's 'shallower conv stack'. Loss, optimizer, split,
augmentation, and everything else are identical (set in train.py / dataset.py).
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Plain 3-block conv classifier. Input 3x150x150 -> 1 logit."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 150 -> 75
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 75 -> 37
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 37 -> 18
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),                     # single logit for BCEWithLogits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ExperimentalCNN(nn.Module):
    """BaselineCNN + a 4th conv block + global average pooling. The ONE change.

    Differs from BaselineCNN by exactly one architectural change: the conv
    stack is deeper (4 blocks, 32->64->128->256). No BatchNorm, no dropout — so the comparison
    isolates 'deeper' vs. the baseline's
    'shallower + flatten-FC head'. Loss, optimizer, split, augmentation,
    and everything else are identical (set in train.py / dataset.py).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 150 -> 75
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 75 -> 37
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 37 -> 18
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # NEW: 4th conv block
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 18 -> 9

        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 9 * 9, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),                     # single logit for BCEWithLogits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def get_model(name: str) -> nn.Module:
    """Factory used by train.py. Raises on unknown name instead of guessing."""
    if name == "baseline":
        return BaselineCNN()
    if name == "experimental":
        return ExperimentalCNN()
    raise ValueError(f"unknown model '{name}'; expected 'baseline' or 'experimental'")


if __name__ == "__main__":
    # one runnable check — shapes must flow end-to-end.
    for n in ("baseline", "experimental"):
        m = get_model(n)
        out = m(torch.zeros(2, 3, 150, 150))
        assert out.shape == (2, 1), f"{n} bad shape {out.shape}"
        print(f"{n}: OK -> {tuple(out.shape)}")