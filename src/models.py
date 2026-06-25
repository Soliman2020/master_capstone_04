"""CNN models for the P4 CNRPark+EXT occupancy experiment.

Two models, differing by exactly ONE change (rubric requirement):

  BaselineCNN      -> plain conv stack, no normalization, no dropout.
  ExperimentalCNN  -> SAME architecture + BatchNorm2d after each conv.

The single change is normalization: tests whether BatchNorm's internal
covariate stabilization improves val accuracy / convergence on this dataset.
Everything else (depth, channels, pooling, head, loss) is identical so the
comparison is fair. train.py runs both and plots loss/accuracy curves.
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Plain 3-block conv classifier. Input 3x150x150 -> 1 logit."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 150 -> 75
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 75 -> 37
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
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
    """BaselineCNN + BatchNorm2d after every conv. The ONE change."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Head identical to baseline (same dims) -> only the conv stack changed.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
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