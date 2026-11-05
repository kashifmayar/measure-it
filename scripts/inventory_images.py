"""
Build an image inventory with OpenCV checkerboard / ArUco heuristics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from calibration.image_io import IMAGE_EXTENSIONS, laplacian_sharpness, read_image_gray
from measurement.aruco_detector import detect_markers

DEFAULT_COLS = 9
DEFAULT_ROWS = 7


def classify_pen_heuristic(gray) -> tuple[bool, str]:
    """Rough heuristic: elongated dark/colour blob away from centre — not definitive."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=w // 6, maxLineGap=20)
    if lines is not None and len(lines) >= 3:
        return True, "possible elongated structure (Hough lines)"
    return False, "no strong pen-like structure detected"


def run_inventory(roots: list[Path], out_csv: Path, out_json: Path) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.suffix.lower() in {".mov"}:
                continue

            record: dict = {
                "filename": path.name,
                "directory": str(
                    path.parent.relative_to(ROOT)
                    if ROOT in path.parent.parents or path.parent == ROOT
                    else path.parent
                ),
                "format": path.suffix.lower(),
                "width": None,
                "height": None,
                "checkerboard_detected": False,
                "aruco_detected": False,
                "aruco_ids": "",
                "pen_heuristic": False,
                "pen_notes": "",
                "sharpness": None,
                "quality_concerns": "",
            }

            gray = read_image_gray(path)
            if gray is None:
                record["quality_concerns"] = "unreadable"
                rows.append(record)
                continue

            h, w = gray.shape
            record["width"] = w
            record["height"] = h
            sharp = laplacian_sharpness(gray)
            record["sharpness"] = round(sharp, 2)
            concerns = []
            if sharp < 40:
                concerns.append("possibly blurry")

            detect = gray
            mh, mw = gray.shape
            if max(mh, mw) > 1600:
                s = 1600.0 / max(mh, mw)
                detect = cv2.resize(
                    gray, (int(mw * s), int(mh * s)), interpolation=cv2.INTER_AREA
                )

            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            )
            found, _ = cv2.findChessboardCorners(
                detect, (DEFAULT_COLS, DEFAULT_ROWS), flags=flags
            )
            record["checkerboard_detected"] = bool(found)

            corners_list, ids = detect_markers(detect)
            if ids is not None and len(ids) > 0:
                record["aruco_detected"] = True
                record["aruco_ids"] = ",".join(str(int(x)) for x in ids.flatten())

            pen_guess, pen_note = classify_pen_heuristic(gray)
            record["pen_heuristic"] = pen_guess
            record["pen_notes"] = pen_note

            if record["checkerboard_detected"] and record["aruco_detected"]:
                concerns.append("contains both checkerboard and ArUco")
            if not record["checkerboard_detected"] and not record["aruco_detected"]:
                concerns.append("neither checkerboard nor ArUco detected")

            record["quality_concerns"] = "; ".join(concerns)
            rows.append(record)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Image inventory for Measure It.")
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("calibration/images"),
            Path("measurement/images"),
        ],
        help="Directories to scan recursively.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("calibration/output/image_inventory.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("calibration/output/image_inventory.json"),
    )
    args = parser.parse_args()
    roots = [(ROOT / r).resolve() if not r.is_absolute() else r for r in args.roots]
    rows = run_inventory(roots, args.out_csv, args.out_json)
    print(f"Inventory: {len(rows)} image(s) -> {args.out_csv}")


if __name__ == "__main__":
    main()
