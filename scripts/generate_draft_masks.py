"""
Generate DRAFT segmentation masks (GrabCut) for quick labelling bootstrap.

These are NOT ground truth. Review and fix in Label Studio / CVAT, or re-draw.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGES = ROOT / "dataset" / "images"
OUT = ROOT / "dataset" / "masks_draft"


def draft_mask_grabcut(bgr: np.ndarray, margin_frac: float = 0.08) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = int(min(h, w) * margin_frac)
    rect = (m, m, max(1, w - 2 * m), max(1, h - 2 * m))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return binary


def run(limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(IMAGES.glob("*.jpg")) + sorted(IMAGES.glob("*.png"))
    if limit:
        paths = paths[:limit]
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        mask = draft_mask_grabcut(bgr)
        out = OUT / f"{path.stem}.png"
        cv2.imwrite(str(out), mask)
        logger.info("Draft mask: %s", out.name)
    logger.info("Done. Review masks in %s then copy/fix into dataset/masks/", OUT)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate draft GrabCut masks (review required).")
    p.add_argument("--limit", type=int, default=None, help="Process only first N images (try 5)")
    args = p.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
