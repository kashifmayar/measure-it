"""Light tests for segmentation metrics and U-Net forward pass."""

from __future__ import annotations

import numpy as np
import torch

from models.metrics import evaluate_batch_metrics, iou_score
from models.unet import UNet


def test_iou_perfect_overlap() -> None:
    a = np.ones((32, 32), dtype=np.float32)
    assert iou_score(a, a) == 1.0


def test_unet_forward_shape() -> None:
    model = UNet()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 1, 128, 128)


def test_batch_metrics() -> None:
    preds = np.zeros((2, 16, 16), dtype=np.float32)
    targets = np.zeros((2, 16, 16), dtype=np.float32)
    preds[0, 4:12, 4:12] = 1.0
    targets[0, 4:12, 4:12] = 1.0
    m = evaluate_batch_metrics(preds, targets)
    assert m["mean_iou"] >= 0.99
