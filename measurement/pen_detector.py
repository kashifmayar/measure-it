"""
Preliminary classical-CV pen segmentation baseline (not a trained U-Net).

Inspects rectified images and separates elongated foreground contours while
masking out the ArUco reference region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from measurement.config import MeasurementConfig

logger = logging.getLogger(__name__)

BASELINE_METHOD = "classical_cv_baseline"
UNET_METHOD = "unet_segmentation"


@dataclass
class PenDetectionResult:
    contour: np.ndarray
    mask: np.ndarray
    length_mm: float
    max_width_mm: float
    length_px: float
    max_width_px: float
    center: Tuple[float, float]
    confidence: float
    method: str = BASELINE_METHOD


def _marker_mask_rectified(
    rectified_bgr: np.ndarray,
    marker_size_mm: float,
    ppm: float,
    origin_margin_mm: float,
    inflate_px: int,
) -> np.ndarray:
    """Mask the known marker square at the rectified plane origin."""
    h, w = rectified_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = int(origin_margin_mm * ppm) - inflate_px
    y0 = int(origin_margin_mm * ppm) - inflate_px
    side = int(marker_size_mm * ppm) + 2 * inflate_px
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    x0 = max(0, x0)
    y0 = max(0, y0)
    mask[y0:y1, x0:x1] = 255
    return mask


def _foreground_mask(gray: np.ndarray, aruco_mask: np.ndarray) -> np.ndarray:
    """Adaptive background/foreground separation for varied pen colours."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Pick Otsu polarity with more elongated large component (pen-like)
    best = None
    best_score = -1.0
    for candidate in (otsu, otsu_inv, edges):
        cand = cv2.bitwise_and(candidate, cv2.bitwise_not(aruco_mask))
        cand = cv2.morphologyEx(
            cand, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
        )
        cand = cv2.morphologyEx(
            cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
        )
        score = float(np.count_nonzero(cand))
        if score > best_score:
            best_score = score
            best = cand
    return best if best is not None else np.zeros_like(gray)


def _contour_width_along_perpendicular(contour: np.ndarray, angle_deg: float) -> float:
    """Max thickness perpendicular to the principal axis (approx max width)."""
    rect = cv2.minAreaRect(contour)
    return float(min(rect[1]))


def _pen_from_binary_mask(
    mask: np.ndarray,
    config: MeasurementConfig,
    *,
    method: str,
    model_confidence: float = 0.0,
) -> Tuple[Optional[PenDetectionResult], str]:
    """Shared geometry from a binary pen mask on the rectified plane."""
    ppm = config.pixels_per_mm
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "Pen not detected (empty segmentation mask)"

    img_area = h * w
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_pen_area_mm2 * ppm * ppm:
            continue
        if area / img_area > config.max_pen_area_fraction:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / short_side
        if aspect < config.min_pen_aspect_ratio:
            continue
        hull = cv2.convexHull(cnt)
        solidity = float(area) / max(cv2.contourArea(hull), 1.0)
        if solidity < config.min_contour_solidity:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (
            x <= config.pen_margin_from_border_px
            or y <= config.pen_margin_from_border_px
            or x + bw >= w - config.pen_margin_from_border_px
            or y + bh >= h - config.pen_margin_from_border_px
        ):
            continue
        candidates.append((area, cnt, rect, aspect, solidity))

    if not candidates:
        return None, "Pen not detected (segmentation mask failed shape filters)"

    if len(candidates) > 1:
        candidates.sort(key=lambda t: t[0], reverse=True)
        if candidates[0][0] < candidates[1][0] * 1.25:
            return None, "Pen segmentation ambiguous (multiple similar regions)"

    _, contour, rect, aspect, solidity = candidates[0]
    length_px = max(rect[1])
    width_px = min(rect[1])
    confidence = model_confidence if method == UNET_METHOD else min(
        1.0, 0.4 + 0.2 * min(aspect / 8.0, 1.0) + 0.4 * solidity
    )
    out_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return PenDetectionResult(
        contour=contour,
        mask=out_mask,
        length_mm=length_px / ppm,
        max_width_mm=width_px / ppm,
        length_px=length_px,
        max_width_px=width_px,
        center=(float(rect[0][0]), float(rect[0][1])),
        confidence=confidence,
        method=method,
    ), ""


_UNET_CACHE: dict[str, object] = {}


def detect_pen_unet(
    rectified_bgr: np.ndarray,
    config: MeasurementConfig,
    checkpoint: Path,
    *,
    origin_margin_mm: float = 5.0,
    seg_threshold: float = 0.5,
) -> Tuple[Optional[PenDetectionResult], str]:
    from models.inference import PenUNet

    ck = Path(checkpoint)
    if not ck.is_file():
        return None, f"Missing U-Net checkpoint: {checkpoint}"

    key = str(ck.resolve())
    if key not in _UNET_CACHE:
        _UNET_CACHE[key] = PenUNet(ck)
    segmenter = _UNET_CACHE[key]
    mask, conf = segmenter.predict_mask(rectified_bgr, threshold=seg_threshold)

    ppm = config.pixels_per_mm
    aruco_mask = _marker_mask_rectified(
        rectified_bgr,
        config.marker_size_mm,
        ppm,
        origin_margin_mm,
        config.aruco_mask_inflate_px,
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(aruco_mask))
    return _pen_from_binary_mask(
        mask, config, method=UNET_METHOD, model_confidence=conf
    )


def detect_pen_baseline(
    rectified_bgr: np.ndarray,
    config: MeasurementConfig,
    *,
    origin_margin_mm: float = 5.0,
) -> Tuple[Optional[PenDetectionResult], str]:
    """
    Detect pen on a metric-rectified image. Returns failure reason if ambiguous.
    """
    if config.marker_size_mm <= 0:
        return None, "Marker physical size (mm) is not configured"

    ppm = config.pixels_per_mm
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    aruco_mask = _marker_mask_rectified(
        rectified_bgr,
        config.marker_size_mm,
        ppm,
        origin_margin_mm,
        config.aruco_mask_inflate_px,
    )
    fg = _foreground_mask(gray, aruco_mask)
    return _pen_from_binary_mask(fg, config, method=BASELINE_METHOD)


def draw_pen_overlay(
    image_bgr: np.ndarray,
    result: PenDetectionResult,
) -> np.ndarray:
    out = image_bgr.copy()
    cv2.drawContours(out, [result.contour], -1, (255, 0, 0), 2)
    text = (
        f"L={result.length_mm:.1f}mm W={result.max_width_mm:.1f}mm "
        f"({result.method})"
    )
    cv2.putText(out, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return out
