"""Training loop for the CNRPark+EXT CNN experiment.

Trains BaselineCNN and ExperimentalCNN under identical settings and prints
side-by-side val/test metrics. Saves loss/accuracy curves for the notebook.

Usage (from project_04_DeepLearning/):
    python src/train.py --patches-dir data/CNRPark-Patches-150x150 \
                        --epochs 10 --batch-size 32

Loss/accuracy CSVs are written to reports/metrics_<run>.csv; curves are
plotted from those files in the notebook.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import build_dataloaders
from models import get_model

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
) -> tuple[float, float]:
    """One pass over `loader`. Returns (mean_loss, accuracy)."""
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad()
        logit = model(x)
        loss = criterion(logit, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        # threshold at 0 -> occupied if logit > 0, else empty
        pred = (logit.detach() > 0).float()
        correct += (pred == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def fit(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    seed: int = SEED,
) -> dict:
    """Train one model end-to-end. Returns per-epoch metrics + final test score.

    `seed` controls model init + training-shuffle noise only. The train/val/test
    split is held constant by `build_dataloaders` (SEED=42 there) so multi-seed
    runs compare models on the same data partition.
    """
    set_seed(seed)
    model = get_model(model_name).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, train=False)
        best_val_acc = max(best_val_acc, va_acc)
        history.append({
            "model": model_name, "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "seconds": time.time() - t0,
        })
        print(f"[{model_name}] epoch {epoch:>2}/{epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}  ({time.time()-t0:.1f}s)")

    te_loss, te_acc = run_epoch(model, test_loader, criterion, None, train=False)
    return {"history": history, "test_loss": te_loss, "test_acc": te_acc,
            "best_val_acc": best_val_acc, "model_name": model_name,
            "model": model}  # return model so the notebook can build a confusion matrix without retraining


def save_csv(history: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--patches-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    train_loader, val_loader, test_loader = build_dataloaders(
        args.patches_dir, batch_size=args.batch_size
    )
    print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}  device={DEVICE}")

    results = []
    for name in ("baseline", "experimental"):
        res = fit(name, train_loader, val_loader, test_loader, args.epochs, args.lr)
        save_csv(res["history"], args.out_dir / f"metrics_{name}.csv")
        results.append(res)

    print("\n=== final test results ===")
    for r in results:
        print(f"{r['model_name']:>13s}  test_acc={r['test_acc']:.4f}  "
              f"test_loss={r['test_loss']:.4f}  best_val_acc={r['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()