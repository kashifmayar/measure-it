"""
Evaluate trained U-Net on the test split.

PowerShell:
  .\\.venv\\Scripts\\python.exe models\\evaluate.py --checkpoint models\\runs\\unet_pen\\best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.data import PenSegDataset, load_split_ids
from models.metrics import evaluate_batch_metrics
from models.unet import UNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "runs" / "unet_pen" / "best.pt")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "runs" / "unet_pen" / "test_metrics.json")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(ckpt.get("image_size", 384))
    model = UNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids = load_split_ids(args.split)
    ds = PenSegDataset(ids, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x)
            probs = torch.sigmoid(logits).numpy()[:, 0]
            preds_list.append(probs)
            targets_list.append(y.numpy()[:, 0])

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    metrics = evaluate_batch_metrics(preds, targets)
    metrics["split"] = args.split
    metrics["num_images"] = len(ids)
    metrics["checkpoint"] = str(args.checkpoint.resolve())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
