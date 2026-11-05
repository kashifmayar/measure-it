"""Tests for ArUco detection, homography, rejection, and JSON schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from measurement.aruco_detector import ARUCO_DICT, detect_markers, select_marker
from measurement.config import MeasurementConfig
from measurement.measure_pen import MeasurementResult, measure_pen_image
from measurement.perspective import rectify_plane


def _synthetic_aruco_bgr(size_px: int = 200, marker_id: int = 0) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    canvas = np.ones((800, 800), dtype=np.uint8) * 255
    y0 = (800 - size_px) // 2
    x0 = (800 - size_px) // 2
    canvas[y0 : y0 + size_px, x0 : x0 + size_px] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_aruco_detects_synthetic_marker():
    img = _synthetic_aruco_bgr()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_markers(gray)
    assert ids is not None
    assert int(ids[0]) == 0


def test_wrong_marker_id_rejected():
    img = _synthetic_aruco_bgr(marker_id=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det, reason = select_marker(gray, expected_id=0, min_sharpness=1.0)
    assert det is None
    assert reason  # e.g. marker not detected or wrong ID


def test_homography_rectify_marker_size():
    img = _synthetic_aruco_bgr(240)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det, _ = select_marker(gray, 0, min_sharpness=1.0)
    assert det is not None
    ppm = 10.0
    marker_mm = 30.0
    plane = rectify_plane(img, det, marker_mm, ppm)
    assert plane.image.shape[0] > 0
    side_px = marker_mm * ppm
    roi = plane.image[
        int(5 * ppm) : int(5 * ppm + side_px),
        int(5 * ppm) : int(5 * ppm + side_px),
    ]
    assert roi.size > 0


def test_missing_calibration_rejected(tmp_path: Path):
    img_path = tmp_path / "x.png"
    cv2.imwrite(str(img_path), _synthetic_aruco_bgr())
    cfg = MeasurementConfig(marker_size_mm=25.0)
    result, _, _, _ = measure_pen_image(
        img_path, tmp_path / "missing.npz", cfg
    )
    assert result.status == "rejected"
    assert "calibration" in (result.rejection_reason or "").lower()


def test_missing_marker_size_rejected(tmp_path: Path):
    img_path = tmp_path / "x.png"
    cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))
    cfg = MeasurementConfig(marker_size_mm=0.0)
    result, _, _, _ = measure_pen_image(img_path, tmp_path / "missing.npz", cfg)
    assert result.rejection_reason is not None


def test_json_schema_fields():
    r = MeasurementResult(
        image="a.jpg",
        status="rejected",
        pen_length_mm=None,
        pen_max_width_mm=None,
        marker_detected=False,
        marker_id=0,
        marker_size_mm=25.0,
        measurement_plane_valid=False,
        measurement_method="aruco_planar_homography",
        segmentation_method="classical_cv_baseline",
        rejection_reason="test",
    )
    d = r.to_json_dict()
    json.dumps(d)
    assert d["pen_length_mm"] is None
