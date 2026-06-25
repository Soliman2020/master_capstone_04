"""CNRPark dataset loading for the P4 CNN baseline vs. one-change experiment.

Dataset home: http://cnrpark.it/
On-disk layout used here (original CNRPark, not EXT):
    <patches_dir>/
        A/busy/*.jpg     -> label 1 (occupied)
        A/free/*.jpg     -> label 0 (empty)
        B/busy/*.jpg     -> label 1
        B/free/*.jpg     -> label 0

Each subfolder named busy/free is scanned; the parent folder name provides
the label. ~12.5k 150x150 patches, balanced-ish across (camera, label).

This module only reads + preprocesses; it owns no model logic (see models.py).
SEED=42 so the train/val/test split is reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

SEED = 42
IMG_SIZE = 150  # native patch resolution; keep it -> no resize distortion
NUM_CLASSES = 1  # binary: one logit, BCEWithLogitsLoss (occupied vs empty)

# busy -> 1 (occupied), free -> 0 (empty)
LABEL_MAP = {"busy": 1, "free": 0}


def scan_patches(patches_dir: str | Path) -> List[Tuple[Path, int]]:
    """Walk <patches_dir>/<camera>/<busy|free>/*.jpg and return [(path, label), ...].

    Skips files whose grandparent folder name is not in LABEL_MAP so an
    accidental .DS_Store or stray README folder is ignored.
    """
    base = Path(patches_dir)
    entries: List[Tuple[Path, int]] = []
    for img_path in base.glob("*/*/*.jpg"):
        status = img_path.parent.name.lower()  # busy | free
        if status not in LABEL_MAP:
            continue
        entries.append((img_path, LABEL_MAP[status]))
    if not entries:
        raise FileNotFoundError(f"no labeled patches under {base} (expected */<busy|free>/*.jpg)")
    return entries


def get_transforms(train: bool) -> transforms.Compose:
    """Image preprocessing. Train adds light augmentation (the rubric's
    'one change' for the experimental run is architectural, not augmentation,
    so augmentation is shared by both runs to keep the comparison clean)."""
    common = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # ImageNet-style normalize -> matches transfer-learning conventions later
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        # Random flips only: parking orientation is arbitrary, color jitter would
        # fight the weather/lighting signal we actually want the model to see.
        common = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] + common
    return transforms.Compose(common)


class CNRParkDataset(Dataset):
    """Torch Dataset over CNRPark patches labeled by folder name.

    Args:
        entries: list of (image_path, label_int) from scan_patches().
        train: selects augmentation vs. eval transforms.
    """

    def __init__(self, entries: List[Tuple[Path, int]], train: bool = True):
        self.entries = entries
        self.transform = get_transforms(train=train)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.entries[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        # float scalar target for BCEWithLogitsLoss
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y.unsqueeze(0)


def build_dataloaders(
    patches_dir: str | Path,
    batch_size: int = 32,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders from a single patches folder.

    CNRPark (original) doesn't ship a held-out test LIST, so we carve
    val + test out of the full set with a SEED=42 random_split. Test is
    only touched at the end, val is used for model selection.

    Splits sum to (1 - val_frac - test_frac) for train. Shuffled once
    with the same generator, so both BaselineCNN and ExperimentalCNN see
    identical data ordering.
    """
    entries = scan_patches(patches_dir)
    n = len(entries)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(entries, [n_train, n_val, n_test], generator=g)

    # Wrap each subset with the right transform set without mutating shared state.
    def wrap(subset, train_flag: bool) -> Dataset:
        sub_entries = [entries[i] for i in subset.indices]
        return CNRParkDataset(sub_entries, train=train_flag)

    train_loader = DataLoader(wrap(train_ds, train_flag=True), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(wrap(val_ds, train_flag=False), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(wrap(test_ds, train_flag=False), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader