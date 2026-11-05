"""
End-to-end pen measurement: undistort → ArUco → rectify → baseline pen detect → mm.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_REPORT_PATH = Path("calibration") / "calibration_report.json"
MAX_CALIB_REPROJECTION_PX = 2.0

import cv2
import numpy as np

from calibration.camera_params import CameraCalibration, load_camera_params
from calibration.image_io import read_image_bgr
from measurement.aruco_detector import draw_marker, select_marker
from measurement.config import MeasurementConfig
from measurement.pen_detector import (
    BASELINE_METHOD,
    UNET_METHOD,
    detect_pen_baseline,
    detect_pen_unet,
    draw_pen_overlay,
)
from measurement.perspective import rectify_plane

logger = logging.getLogger(__name__)

MEASUREMENT_METHOD = "aruco_planar_homography"


@dataclass
class MeasurementResult:
    image: str
    status: str
    pen_length_mm: Optional[float]
    pen_max_width_mm: Optional[float]
    marker_detected: bool
    marker_id: Optional[int]
    marker_size_mm: Optional[float]
    measurement_plane_valid: Optional[bool]
    measurement_method: str
    segmentation_method: str
    rejection_reason: Optional[str]
    marker_quality: Optional[float] = None
    pen_confidence: Optional[float] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _undistort_bgr(
    image_bgr: np.ndarray,
    calib: CameraCalibration,
) -> tuple[np.ndarray, Optional[str]]:
    h, w = image_bgr.shape[:2]
    calib_w, calib_h = calib.image_width, calib.image_height
    camera_matrix = calib.camera_matrix.copy()
    if (w, h) != (calib_w, calib_h):
        if abs(w / calib_w - h / calib_h) > 0.02:
            return image_bgr, (
                f"Image resolution {w}x{h} incompatible with calibration {calib_w}x{calib_h}"
            )
        sx, sy = w / calib_w, h / calib_h
        scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        camera_matrix = scale @ camera_matrix

    new_cam, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, calib.dist_coeffs, (w, h), 1.0, (w, h)
    )
    undist = cv2.undistort(image_bgr, camera_matrix, calib.dist_coeffs, None, new_cam)
    return undist, None


def _calibration_quality_ok(report_path: Path) -> tuple[bool, str]:
    if not report_path.is_file():
        return True, ""
    try:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("calibration_summary", {})
        mean_err = summary.get("mean_reprojection_error_px")
        if mean_err is None:
            return True, ""
        if float(mean_err) > MAX_CALIB_REPROJECTION_PX:
            return False, (
                f"Calibration mean reprojection error {float(mean_err):.2f} px exceeds "
                f"{MAX_CALIB_REPROJECTION_PX} px; remeasure squares and recalibrate before trusting mm"
            )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return True, ""
    return True, ""


def measure_pen_image(
    image_path: Path,
    calib_path: Path,
    config: MeasurementConfig,
    *,
    allow_poor_calibration: bool = False,
    checkpoint: Optional[Path] = None,
    fallback_baseline: bool = True,
    seg_threshold: float = 0.5,
    skip_undistort: bool = False,
) -> tuple[MeasurementResult, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (result, annotated_bgr, rectified_bgr, mask).
    """
    name = image_path.name
    base_fail = MeasurementResult(
        image=name,
        status="rejected",
        pen_length_mm=None,
        pen_max_width_mm=None,
        marker_detected=False,
        marker_id=config.marker_id,
        marker_size_mm=config.marker_size_mm if config.marker_size_mm > 0 else None,
        measurement_plane_valid=False,
        measurement_method=MEASUREMENT_METHOD,
        segmentation_method=BASELINE_METHOD,
        rejection_reason=None,
    )

    if config.marker_size_mm <= 0:
        base_fail.rejection_reason = "Marker physical size (--marker-size-mm) is required"
        return base_fail, None, None, None

    if not calib_path.is_file():
        base_fail.rejection_reason = "Missing calibration file; run calibrate.py first"
        return base_fail, None, None, None

    bgr = read_image_bgr(image_path, normalize_orientation=True)
    if bgr is None:
        base_fail.rejection_reason = "Could not read input image"
        return base_fail, None, None, None

    try:
        calib = load_camera_params(calib_path)
    except (FileNotFoundError, KeyError) as exc:
        base_fail.rejection_reason = str(exc)
        return base_fail, None, None, None

    if not allow_poor_calibration:
        calib_ok, calib_msg = _calibration_quality_ok(DEFAULT_REPORT_PATH)
        if not calib_ok:
            base_fail.rejection_reason = calib_msg
            return base_fail, None, None, None

    if skip_undistort:
        undist, res_err = bgr, None
    else:
        undist, res_err = _undistort_bgr(bgr, calib)
        if res_err:
            base_fail.rejection_reason = res_err
            return base_fail, None, None, None

    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    marker, reason = select_marker(
        gray, config.marker_id, min_sharpness=config.min_aruco_sharpness
    )
    if marker is None:
        base_fail.rejection_reason = reason
        return base_fail, None, None, None

    base_fail.marker_detected = True
    if marker.mean_side_px < config.min_marker_side_px:
        base_fail.rejection_reason = (
            f"Marker too small in image ({marker.mean_side_px:.1f} px); "
            "move closer or use higher resolution"
        )
        return base_fail, None, None, None

    plane = rectify_plane(
        undist, marker, config.marker_size_mm, config.pixels_per_mm
    )

    seg_method = BASELINE_METHOD
    pen, pen_reason = None, ""
    if checkpoint is not None:
        pen, pen_reason = detect_pen_unet(
            plane.image, config, checkpoint, seg_threshold=seg_threshold
        )
        seg_method = UNET_METHOD
        if pen is None and fallback_baseline:
            pen, pen_reason = detect_pen_baseline(plane.image, config)
            seg_method = BASELINE_METHOD
    else:
        pen, pen_reason = detect_pen_baseline(plane.image, config)

    base_fail.segmentation_method = seg_method
    if pen is None:
        base_fail.rejection_reason = pen_reason
        annotated = draw_marker(undist, marker)
        return base_fail, annotated, plane.image, None

    if pen.confidence < config.min_pen_confidence:
        base_fail.rejection_reason = (
            f"Pen segmentation confidence {pen.confidence:.3f} below "
            f"threshold {config.min_pen_confidence}"
        )
        annotated = draw_marker(undist, marker)
        return base_fail, annotated, plane.image, pen.mask

    annotated_rect = draw_pen_overlay(plane.image, pen)
    annotated = draw_marker(undist, marker)

    success = MeasurementResult(
        image=name,
        status="success",
        pen_length_mm=round(pen.length_mm, 2),
        pen_max_width_mm=round(pen.max_width_mm, 2),
        marker_detected=True,
        marker_id=marker.marker_id,
        marker_size_mm=config.marker_size_mm,
        measurement_plane_valid=True,
        measurement_method=MEASUREMENT_METHOD,
        segmentation_method=pen.method,
        rejection_reason=None,
        marker_quality=round(marker.quality_score, 3),
        pen_confidence=round(pen.confidence, 3),
    )
    return success, annotated_rect, plane.image, pen.mask


def save_measurement_outputs(
    stem: str,
    output_dir: Path,
    result: MeasurementResult,
    annotated: Optional[np.ndarray],
    rectified: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_result.json"
    payload = result.to_json_dict()
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if annotated is not None:
        cv2.imwrite(str(output_dir / f"{stem}_annotated.jpg"), annotated)
    if rectified is not None:
        cv2.imwrite(str(output_dir / f"{stem}_rectified.jpg"), rectified)
    if mask is not None:
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask)
    return json_path
