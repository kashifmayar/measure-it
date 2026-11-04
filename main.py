"""
Measure It — CLI for ArUco-based planar pen measurement (classical CV baseline).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from measurement.config import MeasurementConfig
from measurement.measure_pen import measure_pen_image, save_measurement_outputs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CALIB = Path("calibration") / "camera_params.npz"
DEFAULT_OUT = Path("measurement") / "output"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Undistort, detect ArUco, rectify plane, and measure pen (baseline CV).\n"
            "Marker physical size (--marker-size-mm) is required."
        )
    )
    p.add_argument("--image", type=Path, required=True, help="Input pen+ArUco image.")
    p.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_CALIB,
        help="camera_params.npz from calibrate.py",
    )
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument(
        "--marker-size-mm",
        type=float,
        required=True,
        help="Measured outer black-square side of printed ArUco marker (mm).",
    )
    p.add_argument(
        "--pixels-per-mm",
        type=float,
        default=8.0,
        help="Rectified plane scale (pixels per millimetre).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for annotated/rectified/mask/JSON outputs.",
    )
    p.add_argument(
        "--show-debug",
        action="store_true",
        help="Log extra detection details.",
    )
    p.add_argument(
        "--allow-poor-calibration",
        action="store_true",
        help="Run even if calibration reprojection error is high (not for production).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.marker_size_mm <= 0:
        print("[ERROR] --marker-size-mm must be a positive measured value.")
        sys.exit(1)

    if args.show_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = MeasurementConfig(
        marker_id=args.marker_id,
        marker_size_mm=args.marker_size_mm,
        pixels_per_mm=args.pixels_per_mm,
    )

    result, annotated, rectified, mask = measure_pen_image(
        args.image.resolve(),
        args.params.resolve(),
        config,
        allow_poor_calibration=args.allow_poor_calibration,
    )
    stem = args.image.stem
    json_path = save_measurement_outputs(
        stem, args.output_dir, result, annotated, rectified, mask
    )

    if result.status == "success":
        print(
            f"OK: length={result.pen_length_mm} mm, "
            f"max_width={result.pen_max_width_mm} mm "
            f"(baseline CV, not trained segmentation)"
        )
        print(f"Results -> {json_path}")
        sys.exit(0)

    print(f"REJECTED: {result.rejection_reason}")
    print(f"Partial outputs -> {json_path}")
    sys.exit(2)


if __name__ == "__main__":
    main()
