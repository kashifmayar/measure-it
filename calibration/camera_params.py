"""
Load and save camera calibration parameters (NPZ).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int
    square_size_mm: float

    @property
    def image_size_wh(self) -> tuple[int, int]:
        return self.image_width, self.image_height


def save_camera_params(path: Path, calib: CameraCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        camera_matrix=calib.camera_matrix,
        dist_coeffs=calib.dist_coeffs,
        image_width=np.int32(calib.image_width),
        image_height=np.int32(calib.image_height),
        square_size_mm=np.float64(calib.square_size_mm),
        # Legacy key used by earlier undistort.py
        image_size=np.array([calib.image_width, calib.image_height], dtype=np.int32),
    )


def load_camera_params(path: Path) -> CameraCalibration:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration parameters not found: {path.resolve()}\n"
            "Run calibrate.py first."
        )

    data = np.load(str(path))
    if "camera_matrix" not in data.files or "dist_coeffs" not in data.files:
        raise KeyError(f"Invalid NPZ (missing matrix/dist): {path}")

    if "image_width" in data.files and "image_height" in data.files:
        w = int(data["image_width"])
        h = int(data["image_height"])
    elif "image_size" in data.files:
        size = data["image_size"]
        w, h = int(size[0]), int(size[1])
    else:
        raise KeyError("NPZ must contain image_width/image_height or image_size.")

    square = float(data["square_size_mm"]) if "square_size_mm" in data.files else float("nan")

    return CameraCalibration(
        camera_matrix=data["camera_matrix"],
        dist_coeffs=data["dist_coeffs"],
        image_width=w,
        image_height=h,
        square_size_mm=square,
    )
