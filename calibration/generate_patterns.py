"""
calibration/generate_patterns.py
=================================
Generates printable calibration patterns for camera calibration:
  1. A checkerboard with 9×7 inner corners (10×8 squares).
  2. An ArUco marker from the DICT_4X4_50 dictionary.

PHYSICAL SQUARE SIZE
--------------------
The square size below is a TARGET for printing. After printing at 100% scale
(no "fit to page", no scaling), you MUST physically measure several squares
with a ruler/calipers and record the actual mean value. Use that measured
value — not this constant — when running calibrate.py.

    TARGET_SQUARE_SIZE_MM = 25.0  mm  (≈ 1 inch)

PRINTING INSTRUCTIONS
---------------------
1. Open the generated PNG in an image viewer (Windows Photos, IrfanView, etc.).
2. Print at EXACTLY 100% scale / "Actual size". Disable any "fit to page" or
   "shrink to fit" option in the print dialog.
3. Verify the first square measures close to 25 mm on each side with a ruler.
4. If the measured size differs, record the ACTUAL measured average square size
   and pass it to calibrate.py via --square-size.

PowerShell usage examples
--------------------------
# Generate both patterns with defaults:
    .\.venv\Scripts\python.exe calibration\generate_patterns.py

# Custom output directory:
    .\.venv\Scripts\python.exe calibration\generate_patterns.py --out-dir calibration\patterns

# Custom checkerboard inner-corner dimensions and square size:
    .\.venv\Scripts\python.exe calibration\generate_patterns.py --cols 9 --rows 7 --square-size 25.0

# Custom ArUco marker ID and size:
    .\.venv\Scripts\python.exe calibration\generate_patterns.py --aruco-id 0 --aruco-size-px 600
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Inner corners (not squares). A 10x8 grid of squares has 9x7 inner corners.
DEFAULT_COLS: int = 9       # inner corners along the width
DEFAULT_ROWS: int = 7       # inner corners along the height

# Target printed square side length in millimetres.
# IMPORTANT: This drives the pixel-per-square calculation. After printing at
# 100% scale you MUST physically measure several squares and pass the actual
# measured value to calibrate.py via --square-size.
DEFAULT_SQUARE_SIZE_MM: float = 25.0  # mm

# Print resolution used to compute pixels-per-square.
# Formula: pixels_per_square = round(square_size_mm * dpi / 25.4)
# Example: round(25.0 * 300 / 25.4) = 295 px  -> prints at ~25 mm at 300 DPI.
DEFAULT_DPI: int = 300

# Border width around the checkerboard, in squares.
BORDER_SQUARES: int = 1

# ArUco defaults
DEFAULT_ARUCO_ID: int = 0
DEFAULT_ARUCO_SIZE_PX: int = 600       # pixels for the marker image
ARUCO_BORDER_BITS: int = 1            # quiet zone in modules
ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50


def compute_pixels_per_square(square_size_mm: float, dpi: int) -> int:
    """
    Compute the number of pixels per checkerboard square.

    Uses the standard formula::

        pixels_per_square = round(square_size_mm * dpi / 25.4)

    At the default values (25.0 mm, 300 DPI) this gives 295 pixels, which
    prints each square at exactly 25.0 mm on a 300-DPI printer.

    Parameters
    ----------
    square_size_mm : float
        Desired physical side length of each square in millimetres.
    dpi : int
        Printer resolution in dots per inch.

    Returns
    -------
    int
        Pixel side length of each square in the generated PNG.
    """
    return round(square_size_mm * dpi / 25.4)


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------

def generate_checkerboard(
    cols: int,
    rows: int,
    pixels_per_square: int = 295,   # = round(25.0 mm * 300 DPI / 25.4)
    border_squares: int = BORDER_SQUARES,
) -> np.ndarray:
    """
    Generate a checkerboard image.

    Parameters
    ----------
    cols : int
        Number of *inner* corners along the x-axis (= number of squares - 1).
    rows : int
        Number of *inner* corners along the y-axis (= number of squares - 1).
    pixels_per_square : int
        Side length of each square in pixels.
    border_squares : int
        Width of the white border around the board, measured in squares.

    Returns
    -------
    np.ndarray
        Grayscale (uint8) image of the checkerboard.
    """
    # Number of actual squares = inner corners + 1
    num_cols_sq = cols + 1
    num_rows_sq = rows + 1

    # Canvas includes border on all sides
    total_cols = num_cols_sq + 2 * border_squares
    total_rows = num_rows_sq + 2 * border_squares

    img_width  = total_cols * pixels_per_square
    img_height = total_rows * pixels_per_square

    board = np.ones((img_height, img_width), dtype=np.uint8) * 255  # white background

    for r in range(total_rows):
        for c in range(total_cols):
            # Determine whether this square is interior to the border
            is_inner_col = border_squares <= c < border_squares + num_cols_sq
            is_inner_row = border_squares <= r < border_squares + num_rows_sq
            if not (is_inner_col and is_inner_row):
                continue  # keep border white

            # Checkerboard colouring: offset by border so top-left interior square is black
            inner_c = c - border_squares
            inner_r = r - border_squares
            if (inner_r + inner_c) % 2 == 0:
                color = 0  # black
            else:
                color = 255  # white

            y0 = r * pixels_per_square
            y1 = y0 + pixels_per_square
            x0 = c * pixels_per_square
            x1 = x0 + pixels_per_square
            board[y0:y1, x0:x1] = color

    return board


def save_checkerboard(
    out_path: Path,
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    dpi: int = DEFAULT_DPI,
    border_squares: int = BORDER_SQUARES,
) -> None:
    """
    Compute pixels-per-square from physical dimensions, generate and save
    the checkerboard PNG with correct DPI metadata.

    Parameters
    ----------
    out_path : Path
        Destination file path.
    cols, rows : int
        Inner corner counts (squares = corners + 1 in each axis).
    square_size_mm : float
        Target physical side length of each square at 100% print scale.
    dpi : int
        Target print resolution; used to compute pixels_per_square.
    border_squares : int
        Width of the white border around the board, in squares.
    """
    pixels_per_square = compute_pixels_per_square(square_size_mm, dpi)
    img = generate_checkerboard(cols, rows, pixels_per_square, border_squares)

    # Write DPI metadata that matches the computed pixel density so that
    # print drivers that honour pHYs will reproduce the correct physical size.
    try:
        from PIL import Image  # type: ignore

        pil_img = Image.fromarray(img, mode="L")
        ppm = round(dpi / 25.4 * 1000)  # pixels per metre (for pHYs chunk)
        pil_img.save(str(out_path), dpi=(dpi, dpi), pHYs=(ppm, ppm, 1))
        print(f"[checkerboard] Saved with {dpi} DPI metadata: {out_path}")
    except ImportError:
        cv2.imwrite(str(out_path), img)
        print(f"[checkerboard] Saved (no DPI metadata -- Pillow not installed): {out_path}")

    num_cols_sq = cols + 1
    num_rows_sq = rows + 1
    print(
        f"  Inner corners  : {cols} x {rows}  |  "
        f"Squares         : {num_cols_sq} x {num_rows_sq}\n"
        f"  Target square  : {square_size_mm} mm  |  "
        f"Pixels/square   : {pixels_per_square} px  |  "
        f"DPI             : {dpi}\n"
        f"  Image size     : {img.shape[1]} x {img.shape[0]} px"
    )


# ---------------------------------------------------------------------------
# ArUco marker
# ---------------------------------------------------------------------------

def generate_aruco_marker(
    marker_id: int = DEFAULT_ARUCO_ID,
    size_px: int = DEFAULT_ARUCO_SIZE_PX,
    border_bits: int = ARUCO_BORDER_BITS,
    dict_id: int = ARUCO_DICT_ID,
) -> np.ndarray:
    """
    Generate a single ArUco marker image.

    Parameters
    ----------
    marker_id : int
        ID of the marker within the dictionary (0–49 for DICT_4X4_50).
    size_px : int
        Side length of the output image in pixels.
    border_bits : int
        Width of the black quiet-zone border in bit-modules.
    dict_id : int
        OpenCV ArUco dictionary constant.

    Returns
    -------
    np.ndarray
        Grayscale (uint8) marker image.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    return marker_img


def save_aruco_marker(
    out_path: Path,
    marker_id: int = DEFAULT_ARUCO_ID,
    size_px: int = DEFAULT_ARUCO_SIZE_PX,
    border_bits: int = ARUCO_BORDER_BITS,
) -> None:
    """Generate and save the ArUco marker PNG."""
    marker_img = generate_aruco_marker(marker_id, size_px, border_bits)
    cv2.imwrite(str(out_path), marker_img)
    print(
        f"[aruco] Saved marker ID={marker_id} "
        f"({size_px}×{size_px} px, DICT_4X4_50): {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate printable calibration patterns: checkerboard and ArUco marker.\n\n"
            "Pixels per square are computed from --square-size and --dpi using:\n"
            "  pixels_per_square = round(square_size_mm * dpi / 25.4)\n"
            "The PNG DPI metadata is set to match, so printing at 100%% / actual\n"
            "size should reproduce the intended physical square size.\n\n"
            "IMPORTANT: After printing, always physically verify the square size\n"
            "with a ruler and pass the MEASURED value to calibrate.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("calibration") / "patterns",
        help="Directory to save generated PNG files (default: calibration/patterns).",
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
        default=DEFAULT_SQUARE_SIZE_MM,
        help=(
            f"Target printed square size in mm (default: {DEFAULT_SQUARE_SIZE_MM}). "
            "Drives the pixels-per-square calculation together with --dpi. "
            "Always physically verify the printed size with a ruler."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=(
            f"Target printer resolution in DPI (default: {DEFAULT_DPI}). "
            "Used to compute pixels_per_square = round(square_size_mm * dpi / 25.4). "
            "The PNG DPI metadata is set to this value."
        ),
    )
    parser.add_argument(
        "--aruco-id",
        type=int,
        default=DEFAULT_ARUCO_ID,
        help=f"ArUco marker ID, 0-49 for DICT_4X4_50 (default: {DEFAULT_ARUCO_ID}).",
    )
    parser.add_argument(
        "--aruco-size-px",
        type=int,
        default=DEFAULT_ARUCO_SIZE_PX,
        help=f"ArUco marker image size in pixels (default: {DEFAULT_ARUCO_SIZE_PX}).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.cols < 2 or args.rows < 2:
        parser.error("--cols and --rows must each be at least 2 (inner corners).")
    if not (0 <= args.aruco_id <= 49):
        parser.error("--aruco-id must be in range 0-49 for DICT_4X4_50.")
    if args.aruco_size_px < 50:
        parser.error("--aruco-size-px must be at least 50 pixels.")
    if args.square_size <= 0:
        parser.error("--square-size must be a positive number.")
    if args.dpi <= 0:
        parser.error("--dpi must be a positive integer.")

    # Compute and display the pixel-per-square value that will be used
    pps = compute_pixels_per_square(args.square_size, args.dpi)
    print(
        f"Pixel calculation: round({args.square_size} mm x {args.dpi} DPI / 25.4) "
        f"= {pps} px/square"
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkerboard ---
    cb_path = out_dir / f"checkerboard_{args.cols}x{args.rows}.png"
    save_checkerboard(
        out_path=cb_path,
        cols=args.cols,
        rows=args.rows,
        square_size_mm=args.square_size,
        dpi=args.dpi,
    )

    # --- ArUco marker ---
    aruco_path = out_dir / f"aruco_4x4_50_id{args.aruco_id}.png"
    save_aruco_marker(
        out_path=aruco_path,
        marker_id=args.aruco_id,
        size_px=args.aruco_size_px,
    )

    # --- Print reminder ---
    print()
    print("=" * 60)
    print("PRINTING REMINDER")
    print("=" * 60)
    print(f"  Target square size : {args.square_size} mm at {args.dpi} DPI")
    print(f"  Pixels per square  : {pps} px")
    print("  1. Open the checkerboard PNG in Windows Photos or IrfanView.")
    print("  2. Print at EXACTLY 100% / 'Actual size'.")
    print("     -> Disable 'Fit to page' or 'Shrink to printable area'.")
    print("  3. After printing, measure several squares with a ruler.")
    print("  4. Record the ACTUAL measured average square size.")
    print("  5. Pass that measured value to calibrate.py via --square-size.")
    print("     (Do not assume the printer reproduced the target size exactly.)")
    print("=" * 60)


if __name__ == "__main__":
    main()

