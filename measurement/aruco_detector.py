"""
ArUco marker detection (DICT_4X4_50) with geometry and quality metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from calibration.image_io import laplacian_sharpness

logger = logging.getLogger(__name__)

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray  # shape (4, 2) ordered TL, TR, BR, BL in image coords
    center: Tuple[float, float]
    side_lengths_px: Tuple[float, float, float, float]
    mean_side_px: float
    area_px: float
    sharpness: float
    aspect_ratio: float
    quality_score: float


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _side_lengths(corners: np.ndarray) -> Tuple[float, float, float, float]:
    c = corners
    lengths = []
    for i in range(4):
        j = (i + 1) % 4
        lengths.append(float(np.linalg.norm(c[j] - c[i])))
    return tuple(lengths)  # type: ignore[return-value]


def detect_markers(gray: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    h, w = gray.shape[:2]
    scales = [1.0]
    if max(h, w) > 1600:
        scales.append(1600.0 / max(h, w))

    for scale in scales:
        detect_gray = gray
        if scale != 1.0:
            detect_gray = cv2.resize(
                gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        corners_list, ids, _rejected = detector.detectMarkers(detect_gray)
        if ids is not None and len(ids) > 0:
            if scale != 1.0:
                scaled = []
                for c in corners_list:
                    scaled.append(c / scale)
                corners_list = scaled
            return corners_list, ids
    return [], None


def select_marker(
    gray: np.ndarray,
    expected_id: int,
    min_sharpness: float = 30.0,
) -> Tuple[Optional[MarkerDetection], str]:
    """
    Detect markers and return the expected ID, or None with reason.
    """
    corners_list, ids = detect_markers(gray)
    if ids is None or len(ids) == 0:
        return None, "ArUco marker not detected"

    id_list = [int(x) for x in ids.flatten()]
    matches = [i for i, mid in enumerate(id_list) if mid == expected_id]
    if not matches:
        return None, f"Expected marker ID {expected_id} not found (found IDs: {id_list})"
    if len(matches) > 1:
        return None, f"Multiple markers with ID {expected_id} detected ({len(matches)})"

    idx = matches[0]
    raw_corners = corners_list[idx][0]
    ordered = _order_corners(raw_corners)
    sides = _side_lengths(ordered)
    mean_side = float(np.mean(sides))
    area = float(cv2.contourArea(ordered.reshape(-1, 1, 2)))
    aspect = max(sides) / max(min(sides), 1e-6)
    center = tuple(np.mean(ordered, axis=0))
    sharp = laplacian_sharpness(gray)

    if sharp < min_sharpness:
        return None, f"Image sharpness too low for reliable marker detection (score={sharp:.1f})"

    # Geometric consistency: square marker should have similar side lengths
    side_std = float(np.std(sides))
    rel_std = side_std / max(mean_side, 1e-6)
    quality = max(0.0, 1.0 - rel_std * 4.0) * min(1.0, sharp / 100.0)

    if rel_std > 0.25:
        return None, f"Marker geometry severely distorted (side length variation={rel_std:.2f})"

    if aspect > 1.35:
        return None, f"Marker aspect ratio unrealistic ({aspect:.2f})"

    det = MarkerDetection(
        marker_id=expected_id,
        corners=ordered,
        center=(float(center[0]), float(center[1])),
        side_lengths_px=sides,
        mean_side_px=mean_side,
        area_px=area,
        sharpness=sharp,
        aspect_ratio=aspect,
        quality_score=quality,
    )
    return det, ""


def draw_marker(
    image_bgr: np.ndarray,
    detection: MarkerDetection,
) -> np.ndarray:
    out = image_bgr.copy()
    pts = detection.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    cx, cy = int(detection.center[0]), int(detection.center[1])
    cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
    label = f"ID {detection.marker_id} q={detection.quality_score:.2f}"
    cv2.putText(out, label, (pts[0][0][0], max(20, pts[0][0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
