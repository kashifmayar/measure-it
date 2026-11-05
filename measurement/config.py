"""Configurable thresholds for measurement rejection and detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MeasurementConfig:
    marker_id: int = 0
    marker_size_mm: float = 0.0  # must be set by CLI
    pixels_per_mm: float = 8.0
    min_marker_side_px: float = 20.0
    max_marker_aspect_ratio: float = 1.35
    min_aruco_sharpness: float = 30.0
    min_pen_area_mm2: float = 20.0
    max_pen_area_fraction: float = 0.85
    min_pen_aspect_ratio: float = 2.0
    pen_margin_from_border_px: int = 8
    aruco_mask_inflate_px: int = 12
    min_contour_solidity: float = 0.35
    min_pen_confidence: float = 0.35
