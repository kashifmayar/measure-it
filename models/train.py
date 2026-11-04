"""
Train U-Net pen segmentation.

PowerShell:
  .\\.venv\\Scripts\\python.exe models\\train.py
  .\\.venv\\Scripts\\python.exe models\\train.py --epochs 25 --image-size 384
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.data import PenSegDataset, load_split_ids
from models.metrics import evaluate_batch_metrics
from models.unet import UNet

DEFAULT_RUN = ROOT / "models" / "runs" / "unet_pen"


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + smooth
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
    return 1 - (num / den).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    preds_list, targets_list = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds_list.append(probs[:, 0])
        targets_list.append(y.cpu().numpy()[:, 0])
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    metrics = evaluate_batch_metrics(preds, targets)
    return avg_loss, metrics["mean_iou"], metrics


def plot_curves(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_iou"], label="val IoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Validation IoU")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net pen segmentation.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_ids("train")
    val_ids = load_split_ids("val")
    train_ds = PenSegDataset(train_ids, image_size=args.image_size, augment=True)
    val_ds = PenSegDataset(val_ids, image_size=args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": args.image_size,
        "device": str(device),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "architecture": "UNet",
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = -1.0
    best_path = run_dir / "best.pt"

    print(f"Training on {device}: {len(train_ids)} train, {len(val_ids)} val, size={args.image_size}")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, bce, device)
        va_loss, va_iou, va_metrics = validate(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_iou"].append(va_iou)
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_iou={va_iou:.4f}",
            flush=True,
        )
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_iou": va_iou,
                },
                best_path,
            )

    elapsed = time.perf_counter() - t0
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_curves(history, run_dir)

    config["best_val_iou"] = best_iou
    config["training_seconds"] = round(elapsed, 2)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Best val IoU: {best_iou:.4f}  -> {best_path}")
    print(f"Curves -> {run_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
