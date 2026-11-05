"""Tests for calibration helpers and undistortion."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.calibrate import build_object_points, detect_corners, MIN_IMAGES
from calibration.camera_params import CameraCalibration, load_camera_params, save_camera_params
from calibration.image_io import laplacian_sharpness


def test_object_points_shape_and_scale():
    obj = build_object_points(9, 7, 20.0)
    assert obj.shape == (9 * 7, 1, 3)
    assert float(obj[1, 0, 0]) == pytest.approx(20.0)


def test_synthetic_checkerboard_corners(tmp_path: Path):
    cols, rows = 9, 7
    square_px = 40
    w = (cols + 1) * square_px + 80
    h = (rows + 1) * square_px + 80
    img = np.ones((h, w), dtype=np.uint8) * 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = 40 + r * square_px, 40 + c * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    path = tmp_path / "board.png"
    cv2.imwrite(str(path), img)
    ok, corners, size = detect_corners(
        path, cols, rows, show=False, normalize_orientation=False
    )
    assert ok
    assert corners is not None
    assert size == (w, h)


def test_camera_params_roundtrip(tmp_path: Path):
    calib = CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((1, 5)),
        image_width=100,
        image_height=200,
        square_size_mm=20.0,
    )
    p = tmp_path / "params.npz"
    save_camera_params(p, calib)
    loaded = load_camera_params(p)
    assert loaded.image_width == 100
    assert loaded.square_size_mm == 20.0


def test_min_images_constant():
    assert MIN_IMAGES == 20


def test_blur_metric():
    sharp = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    score = laplacian_sharpness(sharp)
    assert score >= 0
