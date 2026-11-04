"""
Planar perspective rectification using a detected ArUco marker as metric reference.

Distortion correction must be applied *before* calling these functions.

Geometric notes
---------------
- **Lens distortion correction** removes radial/tangential lens warping using
  calibrated intrinsics; it does not establish millimetre units.
- **Perspective rectification** (homography) maps a flat plane in the scene to a
  fronto-parallel view so that the marker's black square matches ``marker_size_mm``.
- **Pixel-to-millimetre conversion** on the rectified plane uses ``pixels_per_mm``:
  1 mm in the plane equals ``pixels_per_mm`` image pixels after warping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from measurement.aruco_detector import MarkerDetection


@dataclass
class RectifiedPlane:
    image: np.ndarray
    homography: np.ndarray
    pixels_per_mm: float
    marker_size_mm: float
    origin_offset_mm: Tuple[float, float]


def _destination_corners(marker_size_mm: float, ppm: float) -> np.ndarray:
    s = marker_size_mm * ppm
    return np.array(
        [[0, 0], [s, 0], [s, s], [0, s]],
        dtype=np.float32,
    )


def compute_rectified_canvas_size(
    image_shape: Tuple[int, int],
    homography: np.ndarray,
    ppm: float,
    margin_mm: float = 15.0,
) -> Tuple[int, int]:
    h, w = image_shape
    corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography)
    xs = warped[:, 0, 0]
    ys = warped[:, 0, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    margin = margin_mm * ppm
    width = int(np.ceil(max_x - min_x + 2 * margin))
    height = int(np.ceil(max_y - min_y + 2 * margin))
    return max(width, 32), max(height, 32)


def rectify_plane(
    image_bgr: np.ndarray,
    marker: MarkerDetection,
    marker_size_mm: float,
    pixels_per_mm: float,
) -> RectifiedPlane:
    """
    Warp ``image_bgr`` so the marker's physical black square maps to
    ``marker_size_mm`` × ``marker_size_mm`` millimetres in the output plane.
    """
    src = marker.corners.astype(np.float32)
    dst = _destination_corners(marker_size_mm, pixels_per_mm)
    homography = cv2.getPerspectiveTransform(src, dst)

    out_w, out_h = compute_rectified_canvas_size(
        image_bgr.shape[:2], homography, pixels_per_mm
    )

    # Shift so content fits in positive coordinates
    h_tmp, w_tmp = image_bgr.shape[:2]
    corners = np.array(
        [[0, 0], [w_tmp, 0], [w_tmp, h_tmp], [0, h_tmp]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    warped_c = cv2.perspectiveTransform(corners, homography)
    min_x = float(warped_c[:, 0, 0].min())
    min_y = float(warped_c[:, 0, 1].min())
    translate = np.array([[1, 0, -min_x + pixels_per_mm * 5],
                          [0, 1, -min_y + pixels_per_mm * 5],
                          [0, 0, 1]], dtype=np.float64)
    homography_adj = translate @ homography

    rectified = cv2.warpPerspective(
        image_bgr,
        homography_adj,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return RectifiedPlane(
        image=rectified,
        homography=homography_adj,
        pixels_per_mm=pixels_per_mm,
        marker_size_mm=marker_size_mm,
        origin_offset_mm=(5.0, 5.0),
    )
