"""
calibration/calibrate.py
=========================
Performs camera calibration using a set of checkerboard images.

Workflow
--------
1. Load calibration images from a directory (default: calibration/images/).
2. Detect checkerboard inner corners in each image.
3. Refine detected corners to sub-pixel accuracy with cornerSubPix.
4. Reject images where detection failed; require at least 20 successful detections.
5. Run cv2.calibrateCamera to compute the camera matrix and distortion coefficients.
6. Calculate and report the mean reprojection error.
7. Save camera_params.npz and calibration_report.json.

SQUARE SIZE
-----------
You MUST physically measure the printed checkerboard squares and pass the
actual measured value via --square-size. Do NOT use the default blindly.

PowerShell usage examples
--------------------------
# Minimal — uses defaults (images in calibration/images/, 9×7 inner corners):
    .\.venv\Scripts\python.exe calibration\calibrate.py --square-size 24.7

# Custom image directory and inner-corner dimensions:
    .\.venv\Scripts\python.exe calibration\calibrate.py `
        --images-dir calibration\images `
        --cols 9 --rows 7 `
        --square-size 24.7

# Save outputs to a custom location:
    .\.venv\Scripts\python.exe calibration\calibrate.py `
        --square-size 24.7 `
        --out-params calibration\camera_params.npz `
        --out-report calibration\calibration_report.json

# Show detected corners visually (opens a window per image):
    .\.venv\Scripts\python.exe calibration\calibrate.py --square-size 24.7 --show
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from calibration.camera_params import CameraCalibration, save_camera_params
from calibration.image_io import (
    collect_image_paths,
    laplacian_sharpness,
    read_image_bgr,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLS: int = 9       # inner corners along X
DEFAULT_ROWS: int = 7       # inner corners along Y
MIN_IMAGES: int = 20        # minimum successful detections required

# Sub-pixel refinement parameters
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    30,       # max iterations
    1e-6,     # epsilon
)
SUBPIX_WIN_SIZE: Tuple[int, int] = (11, 11)
SUBPIX_ZERO_ZONE: Tuple[int, int] = (-1, -1)

DEFAULT_BLUR_THRESHOLD: float = 35.0  # Laplacian variance; lower = blurrier

# Default output paths
DEFAULT_IMAGES_DIR = Path("calibration") / "images"
DEFAULT_PARAMS_PATH = Path("calibration") / "camera_params.npz"
DEFAULT_REPORT_PATH = Path("calibration") / "calibration_report.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """
    Build the 3-D world coordinates of checkerboard inner corners.

    The board lies in the Z=0 plane. Coordinates are in millimetres.

    Parameters
    ----------
    cols : int
        Inner corner count along X.
    rows : int
        Inner corner count along Y.
    square_size_mm : float
        Physical side length of each printed square in millimetres.

    Returns
    -------
    np.ndarray, shape (cols*rows, 1, 3), dtype float32
    """
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp.reshape(-1, 1, 3)


def detect_corners(
    image_path: Path,
    cols: int,
    rows: int,
    show: bool = False,
    normalize_orientation: bool = True,
) -> Tuple[bool, np.ndarray | None, Tuple[int, int] | None]:
    """
    Load an image and detect checkerboard corners.

    Parameters
    ----------
    image_path : Path
        Path to the calibration image.
    cols : int
        Expected inner corner count along X.
    rows : int
        Expected inner corner count along Y.
    show : bool
        If True, display the image with drawn corners (press any key to continue).

    Returns
    -------
    success : bool
    corners : np.ndarray or None
        Refined corner locations, shape (cols*rows, 1, 2), dtype float32.
    image_size : (width, height) or None
    """
    img_bgr = read_image_bgr(image_path, normalize_orientation=normalize_orientation)
    if img_bgr is None:
        print(f"  [WARN] Could not read image: {image_path.name}")
        return False, None, None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_size = (w, h)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    detect_gray = gray
    scale_back = 1.0
    max_dim = max(h, w)
    if max_dim > 2000:
        scale_back = 2000.0 / max_dim
        detect_gray = cv2.resize(
            gray,
            (int(w * scale_back), int(h * scale_back)),
            interpolation=cv2.INTER_AREA,
        )

    found, corners = cv2.findChessboardCorners(
        detect_gray, (cols, rows), flags=flags
    )

    if found and scale_back != 1.0:
        corners = corners / scale_back

    # Prefer subpixel board on full-resolution grayscale when available (OpenCV 4.x+)
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        sb_found, sb_corners = cv2.findChessboardCornersSB(
            gray, (cols, rows), flags=sb_flags
        )
        if sb_found:
            found = True
            corners = sb_corners.reshape(-1, 1, 2).astype(np.float32)

    if not found:
        if show:
            cv2.imshow(f"FAILED: {image_path.name}", img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return False, None, image_size

    # Sub-pixel refinement
    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        SUBPIX_WIN_SIZE,
        SUBPIX_ZERO_ZONE,
        SUBPIX_CRITERIA,
    )

    if show:
        display = img_bgr.copy()
        cv2.drawChessboardCorners(display, (cols, rows), corners_refined, found)
        # Resize for display if image is very large
        max_dim = 900
        h_d, w_d = display.shape[:2]
        scale = min(max_dim / w_d, max_dim / h_d, 1.0)
        if scale < 1.0:
            display = cv2.resize(
                display,
                (int(w_d * scale), int(h_d * scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(f"OK: {image_path.name}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True, corners_refined, image_size


def compute_reprojection_errors(
    obj_points_all: List[np.ndarray],
    img_points_all: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[float, List[float]]:
    """
    Compute the mean reprojection error across all calibration images.

    Returns
    -------
    mean_error : float
        Overall mean reprojection error in pixels.
    per_image_errors : list of float
        Per-image mean reprojection errors.
    """
    per_image_errors: List[float] = []
    for obj_pts, img_pts, rvec, tvec in zip(
        obj_points_all, img_points_all, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs
        )
        observed = img_pts.reshape(-1, 2).astype(np.float64)
        pred = projected.reshape(-1, 2).astype(np.float64)
        err = float(np.sqrt(np.mean(np.sum((observed - pred) ** 2, axis=1))))
        per_image_errors.append(err)

    mean_error = float(np.mean(per_image_errors))
    return mean_error, per_image_errors


# ---------------------------------------------------------------------------
# Core calibration routine
# ---------------------------------------------------------------------------

def run_calibration(
    images_dir: Path,
    cols: int,
    rows: int,
    square_size_mm: float,
    out_params: Path,
    out_report: Path,
    show: bool = False,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    recursive: bool = False,
) -> None:
    """
    Main calibration pipeline.

    Parameters
    ----------
    images_dir : Path
        Directory containing calibration images.
    cols, rows : int
        Inner corner count along X and Y respectively.
    square_size_mm : float
        Actual measured physical square side length in millimetres.
    out_params : Path
        Output path for camera_params.npz.
    out_report : Path
        Output path for calibration_report.json.
    show : bool
        If True, display corner detection results interactively.
    """
    print(f"\n{'='*60}")
    print("CAMERA CALIBRATION")
    print(f"{'='*60}")
    print(f"  Images directory : {images_dir.resolve()}")
    print(f"  Inner corners    : {cols} × {rows}")
    print(f"  Square size      : {square_size_mm} mm  (physically measured)")
    print()

    # 1. Collect images
    image_paths = collect_image_paths(images_dir, recursive=recursive)
    total_images = len(image_paths)
    if total_images == 0:
        print(
            "[ERROR] No images found in the images directory.\n"
            f"        Place JPEG/PNG calibration photos in: {images_dir.resolve()}"
        )
        sys.exit(1)
    print(f"Found {total_images} image(s) to process.\n")

    # 2. Detect corners in each image
    objp_template = build_object_points(cols, rows, square_size_mm)

    obj_points_all: List[np.ndarray] = []
    img_points_all: List[np.ndarray] = []

    # Reference resolution — set from the first readable image; all subsequent
    # readable images MUST match this exactly. cv2.calibrateCamera requires a
    # single consistent image size across all inputs.
    reference_size: Tuple[int, int] | None = None

    successful_names:  List[str] = []
    failed_names:      List[str] = []   # unreadable or no corners detected
    rejected_names:    List[str] = []   # readable but wrong resolution / blur
    rejected_details:  List[Dict[str, str]] = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"  [{idx:>3}/{total_images}] {img_path.name} ... ", end="", flush=True)

        img_bgr = read_image_bgr(img_path, normalize_orientation=True)
        if img_bgr is None:
            failed_names.append(img_path.name)
            rejected_details.append({"image": img_path.name, "reason": "unreadable"})
            print("UNREADABLE")
            continue

        gray_probe = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray_probe.shape[:2]
        img_sz: Tuple[int, int] = (img_w, img_h)

        if reference_size is None:
            reference_size = img_sz
        elif img_sz != reference_size:
            rejected_names.append(img_path.name)
            rejected_details.append(
                {
                    "image": img_path.name,
                    "reason": (
                        f"resolution mismatch {img_w}x{img_h} "
                        f"expected {reference_size[0]}x{reference_size[1]}"
                    ),
                }
            )
            print(
                f"RESOLUTION MISMATCH  "
                f"(this={img_w}x{img_h},  expected={reference_size[0]}x{reference_size[1]})"
            )
            continue

        sharp = laplacian_sharpness(gray_probe)
        if sharp < blur_threshold:
            rejected_names.append(img_path.name)
            rejected_details.append(
                {
                    "image": img_path.name,
                    "reason": f"too blurry (Laplacian variance={sharp:.1f} < {blur_threshold})",
                }
            )
            print(f"BLURRY (sharpness={sharp:.1f})")
            continue

        # --- Corner detection ---
        success, corners, _ = detect_corners(img_path, cols, rows, show=show)

        if success:
            # Append a COPY so every image has its own independent object-point
            # array. Sharing a single reference is safe with the current OpenCV
            # implementation, but would silently corrupt inputs if OpenCV ever
            # mutated the array in place.
            obj_points_all.append(objp_template.copy())
            img_points_all.append(corners)
            successful_names.append(img_path.name)
            print("OK")
        else:
            failed_names.append(img_path.name)
            rejected_details.append(
                {"image": img_path.name, "reason": "checkerboard corners not detected"}
            )
            print("NOT DETECTED")

    # The image_size fed to calibrateCamera is the common resolution.
    image_size: Tuple[int, int] | None = reference_size

    # 3. Report detection summary
    print()
    print(
        f"Detection summary : {len(successful_names)} succeeded  |  "
        f"{len(failed_names)} failed  |  "
        f"{len(rejected_names)} resolution-rejected  "
        f"(out of {total_images} total)"
    )

    if rejected_names:
        ref_w, ref_h = reference_size if reference_size else (0, 0)
        print(
            f"\nImages REJECTED due to resolution mismatch "
            f"(expected {ref_w}x{ref_h}) ({len(rejected_names)}):"
        )
        for name in rejected_names:
            print(f"    {name}")

    if failed_names:
        print(f"\nImages where corners were NOT detected ({len(failed_names)}):")
        for name in failed_names:
            print(f"    {name}")

    if len(successful_names) < MIN_IMAGES:
        print(
            f"\n[ERROR] Only {len(successful_names)} image(s) succeeded; "
            f"at least {MIN_IMAGES} are required.\n"
            "Tips to improve detection:\n"
            "  Ensure the full checkerboard is visible and well-lit.\n"
            "  Avoid motion blur and overexposure.\n"
            "  Cover a wide range of angles and distances.\n"
            "  Increase the number of calibration photos."
        )
        sys.exit(1)

    # 4. Camera calibration
    print(f"\nRunning cv2.calibrateCamera with {len(successful_names)} images …")
    t0 = time.perf_counter()

    def _run_cal(obj_pts, img_pts):
        return cv2.calibrateCamera(
            obj_pts,
            img_pts,
            image_size,  # type: ignore[arg-type]
            None,
            None,
        )

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = _run_cal(
        obj_points_all, img_points_all
    )

    mean_err, per_image_errs = compute_reprojection_errors(
        obj_points_all, img_points_all, rvecs, tvecs, camera_matrix, dist_coeffs
    )
    median_err = float(np.median(per_image_errs))
    if median_err > 0.5 and len(successful_names) > MIN_IMAGES:
        keep = [
            i
            for i, e in enumerate(per_image_errs)
            if e <= max(2.5 * median_err, 1.0)
        ]
        if len(keep) >= MIN_IMAGES and len(keep) < len(successful_names):
            logger.info(
                "Removing %d high reprojection outliers and recalibrating.",
                len(successful_names) - len(keep),
            )
            successful_names = [successful_names[i] for i in keep]
            obj_points_all = [obj_points_all[i] for i in keep]
            img_points_all = [img_points_all[i] for i in keep]
            rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = _run_cal(
                obj_points_all, img_points_all
            )

    elapsed = time.perf_counter() - t0
    print(f"Calibration completed in {elapsed:.2f} s.")
    print(f"  RMS reprojection error (OpenCV): {rms_error:.6f} px")

    # 5. Detailed reprojection error (final)
    mean_err, per_image_errs = compute_reprojection_errors(
        obj_points_all, img_points_all, rvecs, tvecs, camera_matrix, dist_coeffs
    )
    print(f"  Mean reprojection error (manual): {mean_err:.6f} px")

    # 6. Display results
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist_coeffs.ravel())

    if mean_err > 1.0:
        print(
            "\n[WARNING] Mean reprojection error > 1 px. Calibration quality is poor.\n"
            "  Suggestions:\n"
            "  • Add more images covering more angles.\n"
            "  • Remove images with the highest per-image error.\n"
            "  • Verify the physically measured square size is correct.\n"
            "  • Ensure images are sharp and the board is flat."
        )
    elif mean_err > 0.5:
        print(
            "\n[WARNING] Mean reprojection error is moderate (0.5–1.0 px).\n"
            "  Consider adding more varied calibration images."
        )
    else:
        print("\n[OK] Mean reprojection error is acceptable (< 0.5 px).")

    # 7. Save parameters
    calib = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],  # type: ignore[index]
        image_height=image_size[1],  # type: ignore[index]
        square_size_mm=square_size_mm,
    )
    save_camera_params(out_params, calib)
    print(f"\nCamera parameters saved -> {out_params.resolve()}")

    # 8. Save JSON report
    per_image_report = [
        {"image": name, "reprojection_error_px": round(err, 6)}
        for name, err in zip(successful_names, per_image_errs)
    ]
    num_squares_x = cols + 1
    num_squares_y = rows + 1
    report = {
        "calibration_summary": {
            "calibration_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "opencv_version": cv2.__version__,
            "inner_corners": {"cols": cols, "rows": rows},
            "squares": {"cols": num_squares_x, "rows": num_squares_y},
            "square_size_mm": square_size_mm,
            "image_resolution": {"width": image_size[0], "height": image_size[1]},  # type: ignore[index]
            "image_size_wh": list(image_size),  # type: ignore[arg-type]
            "total_images_discovered": total_images,
            "successful_detections": len(successful_names),
            "rejected_count": len(rejected_names),
            "failed_count": len(failed_names),
            "total_images_found": total_images,
            "images_used": len(successful_names),
            "images_failed": len(failed_names),
            "images_resolution_rejected": len(rejected_names),
            "blur_threshold": blur_threshold,
            "rms_reprojection_error_px": round(float(rms_error), 6),
            "mean_reprojection_error_px": round(mean_err, 6),
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.ravel().tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "successful_images": per_image_report,
        "failed_images": failed_names,
        "resolution_rejected_images": rejected_names,
        "rejected_images": rejected_details,
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Calibration report saved  -> {out_report.resolve()}")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Camera calibration using checkerboard images.\n\n"
            "IMPORTANT: Pass the ACTUAL MEASURED square size (--square-size),\n"
            "not the target size from generate_patterns.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing calibration images (default: {DEFAULT_IMAGES_DIR}).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_COLS,
        help=f"Inner corners along X (default: {DEFAULT_COLS}).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Inner corners along Y (default: {DEFAULT_ROWS}).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help=(
            "REQUIRED. Physically measured side length of each printed square "
            "in millimetres. Measure with a ruler after printing."
        ),
    )
    parser.add_argument(
        "--out-params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help=f"Output path for camera_params.npz (default: {DEFAULT_PARAMS_PATH}).",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Output path for calibration_report.json (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each image with detected corners (requires a display).",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=DEFAULT_BLUR_THRESHOLD,
        help=f"Reject images with Laplacian variance below this (default: {DEFAULT_BLUR_THRESHOLD}).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories of --images-dir for images.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cols < 2 or args.rows < 2:
        parser.error("--cols and --rows must each be at least 2.")
    if args.square_size <= 0:
        parser.error("--square-size must be a positive number in millimetres.")

    run_calibration(
        images_dir=args.images_dir,
        cols=args.cols,
        rows=args.rows,
        square_size_mm=args.square_size,
        out_params=args.out_params,
        out_report=args.out_report,
        show=args.show,
        blur_threshold=args.blur_threshold,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
