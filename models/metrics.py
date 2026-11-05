"""Segmentation metrics for binary pen class."""

from __future__ import annotations

import numpy as np


def _binarize(pred: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (pred >= thresh).astype(np.uint8)


def iou_score(pred: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> float:
    p = _binarize(pred, thresh)
    t = (target > 0.5).astype(np.uint8)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def precision_recall(pred: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> tuple[float, float]:
    p = _binarize(pred, thresh)
    t = (target > 0.5).astype(np.uint8)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, 1 - t).sum()
    fn = np.logical_and(1 - p, t).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall


def mask_ap_at_iou_thresholds(
    pred: np.ndarray, target: np.ndarray, thresholds: np.ndarray
) -> float:
    """Mean of per-threshold AP (image-level): 1 if IoU >= t else 0, averaged."""
    scores = []
    iou = iou_score(pred, target, 0.5)
    for t in thresholds:
        scores.append(1.0 if iou >= t else 0.0)
    return float(np.mean(scores))


def evaluate_batch_metrics(
    preds: np.ndarray, targets: np.ndarray, thresh: float = 0.5
) -> dict[str, float]:
    """preds/targets: (N, H, W) float."""
    ious = [iou_score(preds[i], targets[i], thresh) for i in range(preds.shape[0])]
    precs, recs = [], []
    for i in range(preds.shape[0]):
        p, r = precision_recall(preds[i], targets[i], thresh)
        precs.append(p)
        recs.append(r)
    thresholds = np.arange(0.5, 1.0, 0.05)
    iou_arr = np.asarray(ious, dtype=np.float64)
    map50 = float(np.mean(iou_arr >= 0.5)) if iou_arr.size else 0.0
    map5095 = float(
        np.mean([mask_ap_at_iou_thresholds(preds[i], targets[i], thresholds) for i in range(preds.shape[0])])
    )
    return {
        "mean_iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "map_at_0.5": map50,
        "map_0.5_0.95": map5095,
    }
