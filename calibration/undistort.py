r"""
calibration/undistort.py
=========================
Corrects lens distortion in an image using saved calibration parameters.

Requires a camera_params.npz file produced by calibrate.py.

PowerShell usage examples
--------------------------
# Basic usage -- saves next to the input file with '_undistorted' suffix:
    .\.venv\Scripts\python.exe calibration\undistort.py --input photos\photo.jpg

# Specify an explicit output path:
    .\.venv\Scripts\python.exe calibration\undistort.py `
        --input photos\photo.jpg `
        --output photos\photo_corrected.jpg

# Use a custom calibration params file:
    .\.venv\Scripts\python.exe calibration\undistort.py `
        --input photos\photo.jpg `
        --params calibration\camera_params.npz

# Crop to the valid undistorted region (removes black border artefacts):
    .\.venv\Scripts\python.exe calibration\undistort.py `
        --input photos\photo.jpg `
        --crop

# Show before/after comparison:
    .\.venv\Scripts\python.exe calibration\undistort.py `
        --input photos\photo.jpg `
        --show

# Save a side-by-side comparison image (original | undistorted):
    .\.venv\Scripts\python.exe calibration\undistort.py `
        --input photos\photo.jpg `
        --save-comparison photos\photo_comparison.jpg
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from calibration.camera_params import load_camera_params
from calibration.image_io import read_image_bgr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARAMS_PATH = Path("calibration") / "camera_params.npz"

# Alpha parameter for cv2.getOptimalNewCameraMatrix:
#   0.0  ->  crop to fully valid pixels (no black borders)
#   1.0  ->  retain all original pixels (may have black borders)
ALPHA_FULL: float = 1.0
ALPHA_CROP: float = 0.0


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def load_calibration_params(
    params_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load NPZ; returns (camera_matrix, dist_coeffs, (width, height))."""
    calib = load_camera_params(params_path)
    return calib.camera_matrix, calib.dist_coeffs, calib.image_size_wh


# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------

def undistort_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    crop: bool = False,
) -> np.ndarray:
    """
    Remove lens distortion from an image.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image (as loaded by cv2.imread).
    camera_matrix : np.ndarray
        3x3 intrinsic camera matrix.
    dist_coeffs : np.ndarray
        Distortion coefficient array.
    crop : bool
        If True, crop the result to the region containing only valid pixels
        (removes black border artefacts). If False, the full image canvas is
        kept (may show black corners near the edges).

    Returns
    -------
    np.ndarray
        Undistorted image.
    """
    h, w = img.shape[:2]
    alpha = ALPHA_CROP if crop else ALPHA_FULL

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    if crop:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y : y + rh, x : x + rw]
        else:
            print(
                "[WARN] ROI from getOptimalNewCameraMatrix is empty; "
                "skipping crop. The calibration parameters may be unusual."
            )

    return undistorted


def build_output_path(input_path: Path) -> Path:
    """Derive a default output path by inserting '_undistorted' before the extension."""
    return input_path.with_stem(input_path.stem + "_undistorted")


# ---------------------------------------------------------------------------
# Comparison image (build / show / save)
# ---------------------------------------------------------------------------

def build_comparison_image(
    original: np.ndarray,
    corrected: np.ndarray,
) -> np.ndarray:
    """
    Build a labelled side-by-side comparison image.

    The corrected image is resized to match the original height if they differ
    (this happens when --crop is used). Both panels carry a text label.
    The combined image is capped at 1600 px wide for readability.

    Parameters
    ----------
    original  : np.ndarray  BGR image before undistortion.
    corrected : np.ndarray  BGR image after undistortion.

    Returns
    -------
    np.ndarray
        Side-by-side BGR image with labels.
    """
    h1, w1 = original.shape[:2]
    h2, w2 = corrected.shape[:2]

    if h1 != h2:
        scale = h1 / h2
        corrected_disp = cv2.resize(
            corrected, (int(w2 * scale), h1), interpolation=cv2.INTER_AREA
        )
    else:
        corrected_disp = corrected

    # Add labels
    orig_labelled = original.copy()
    corr_labelled = corrected_disp.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_labelled, "Original",    (20, 50), font, 1.5, (0, 0, 255), 3)
    cv2.putText(corr_labelled, "Undistorted", (20, 50), font, 1.5, (0, 255, 0), 3)

    combined = np.hstack([orig_labelled, corr_labelled])

    # Cap width for readability
    max_width = 1600
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        combined = cv2.resize(
            combined,
            (max_width, int(combined.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return combined


def show_comparison(
    original: np.ndarray,
    corrected: np.ndarray,
    title: str = "",
) -> None:
    """Display the side-by-side comparison in an OpenCV window."""
    combined = build_comparison_image(original, corrected)
    window_title = f"Comparison -- {title}" if title else "Comparison"
    cv2.imshow(window_title, combined)
    print("Press any key in the image window to close ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_comparison(
    original: np.ndarray,
    corrected: np.ndarray,
    out_path: Path,
) -> None:
    """
    Build the side-by-side comparison image and write it to *out_path*.

    Parameters
    ----------
    original  : np.ndarray  BGR image before undistortion.
    corrected : np.ndarray  BGR image after undistortion.
    out_path  : Path        Destination file; parent directories are created
                            if they do not already exist.
    """
    combined = build_comparison_image(original, corrected)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), combined)
    if ok:
        print(f"Comparison image saved -> {out_path.resolve()}")
        print(f"  Size: {combined.shape[1]} x {combined.shape[0]} px  "
              f"(original | undistorted side-by-side)")
    else:
        print(f"[WARN] Could not write comparison image: {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_undistort(
    input_path: Path,
    params_path: Path,
    output_path: "Path | None",
    crop: bool = False,
    show: bool = False,
    save_comparison_path: "Path | None" = None,
) -> None:
    """
    Full undistortion pipeline: load -> undistort -> save.

    Parameters
    ----------
    input_path          : Path       -- input image file.
    params_path         : Path       -- camera_params.npz from calibrate.py.
    output_path         : Path|None  -- if None, auto-derived from input_path.
    crop                : bool       -- crop to valid region if True.
    show                : bool       -- display before/after comparison if True.
    save_comparison_path: Path|None  -- if set, write a side-by-side PNG here.
    """
    # Resolve output path
    if output_path is None:
        output_path = build_output_path(input_path)

    print(f"\n{'='*60}")
    print("IMAGE UNDISTORTION")
    print(f"{'='*60}")
    print(f"  Input  : {input_path.resolve()}")
    print(f"  Params : {params_path.resolve()}")
    print(f"  Output : {output_path.resolve()}")
    print(f"  Crop   : {crop}")
    print()

    # Load calibration parameters
    try:
        camera_matrix, dist_coeffs, calib_image_size = load_calibration_params(params_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("Calibration parameters loaded.")
    print(f"  Camera matrix :\n{camera_matrix}")
    print(f"  Dist coeffs   : {dist_coeffs.ravel()}")
    print(f"  Calibrated at : {calib_image_size[0]} x {calib_image_size[1]} px")
    print()

    # Load input image
    if not input_path.is_file():
        print(f"[ERROR] Input image not found: {input_path.resolve()}")
        sys.exit(1)

    img = read_image_bgr(input_path, normalize_orientation=True)
    if img is None:
        print(f"[ERROR] Could not read image: {input_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Input image size: {w} x {h} px")

    # Warn if the image resolution differs significantly from calibration
    calib_w, calib_h = calib_image_size
    if (w, h) != (calib_w, calib_h):
        ratio_w = w / calib_w
        ratio_h = h / calib_h
        if abs(ratio_w - ratio_h) > 0.01:
            print(
                f"[WARN] Image aspect ratio ({w}x{h}) differs from calibration "
                f"({calib_w}x{calib_h}). Results may be inaccurate."
            )
        else:
            # Same aspect ratio but different resolution -- scale the focal lengths
            print(
                f"[INFO] Input resolution differs from calibration resolution "
                f"({calib_w}x{calib_h}). Scaling camera matrix by {ratio_w:.4f}."
            )
            scale_matrix = np.array(
                [
                    [ratio_w, 0,       0],
                    [0,       ratio_h, 0],
                    [0,       0,       1],
                ],
                dtype=np.float64,
            )
            camera_matrix = scale_matrix @ camera_matrix

    # Undistort
    print("Undistorting ...", end=" ", flush=True)
    corrected = undistort_image(img, camera_matrix, dist_coeffs, crop=crop)
    print("done.")
    print(f"Output image size: {corrected.shape[1]} x {corrected.shape[0]} px")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), corrected)
    if not success:
        print(f"[ERROR] Failed to write output image: {output_path}")
        sys.exit(1)
    print(f"\nUndistorted image saved -> {output_path.resolve()}")

    # Optional display
    if show:
        show_comparison(img, corrected, title=input_path.name)

    # Optional saved comparison
    if save_comparison_path is not None:
        if save_comparison_path.resolve() == input_path.resolve():
            print(
                "[ERROR] --save-comparison path is the same as the input image. "
                "Refusing to overwrite the original."
            )
        else:
            save_comparison(img, corrected, save_comparison_path)

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Undistort an image using camera calibration parameters.\n\n"
            "Run calibrate.py first to produce camera_params.npz."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input image to undistort.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the undistorted output image. "
            "If omitted, '_undistorted' is appended to the input filename."
        ),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help=f"Path to camera_params.npz (default: {DEFAULT_PARAMS_PATH}).",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help=(
            "Crop the output to the valid undistorted region. "
            "Removes black border artefacts but reduces image dimensions."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a side-by-side before/after comparison (requires a display).",
    )
    parser.add_argument(
        "--save-comparison",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save a labelled side-by-side comparison PNG (original | undistorted) "
            "to PATH. The original image is never overwritten."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_undistort(
        input_path=args.input,
        params_path=args.params,
        output_path=args.output,
        crop=args.crop,
        show=args.show,
        save_comparison_path=args.save_comparison,
    )


if __name__ == "__main__":
    main()
