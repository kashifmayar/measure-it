"""
Compare manual ruler/calliper measurements to system output.

Create a CSV with columns:
  image,reference_length_mm,reference_width_mm

Then run:
  .\\.venv\\Scripts\\python.exe scripts\\validate_ruler.py --csv validation\\ruler_refs.csv --marker-size-mm 25.0 --checkpoint models\\runs\\unet_pen\\best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measurement.config import MeasurementConfig
from measurement.measure_pen import measure_pen_image


def _resolve_image(stem: str, image_dir: Path, raw_dir: Path) -> Path | None:
    for base, ext in ((raw_dir, ".HEIC"), (raw_dir, ".heic"), (image_dir, ".jpg"), (image_dir, ".jpeg")):
        p = base / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ruler/calliper vs system mm comparison.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--params", type=Path, default=ROOT / "calibration" / "camera_params.npz")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "dataset" / "images")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "dataset" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "ruler_comparison.json")
    parser.add_argument("--allow-poor-calibration", action="store_true")
    parser.add_argument(
        "--skip-undistort",
        action="store_true",
        help="Use raw photos without undistort (recommended until calibration is fixed).",
    )
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["image"].strip()
            ref_l = float(row["reference_length_mm"])
            ref_w = float(row["reference_width_mm"])
            img_path = _resolve_image(stem, args.image_dir, args.raw_dir)
            if img_path is None:
                rows.append({"image": stem, "status": "missing_image"})
                continue
            result, _, _, _ = measure_pen_image(
                img_path,
                args.params,
                config,
                allow_poor_calibration=args.allow_poor_calibration,
                checkpoint=args.checkpoint,
                skip_undistort=args.skip_undistort,
            )
            entry = {
                "image": stem,
                "reference_length_mm": ref_l,
                "reference_width_mm": ref_w,
                "status": result.status,
                "system_length_mm": result.pen_length_mm,
                "system_width_mm": result.pen_max_width_mm,
                "segmentation": result.segmentation_method,
            }
            if result.status == "success" and result.pen_length_mm is not None:
                entry["length_abs_error_mm"] = abs(result.pen_length_mm - ref_l)
                entry["width_abs_error_mm"] = abs(result.pen_max_width_mm - ref_w)
                entry["length_pct_error"] = 100.0 * entry["length_abs_error_mm"] / ref_l
                entry["width_pct_error"] = 100.0 * entry["width_abs_error_mm"] / ref_w
            rows.append(entry)

    ok = [r for r in rows if r.get("length_abs_error_mm") is not None]
    summary = {"count": len(rows), "successful": len(ok)}
    if ok:
        summary["length_mae_mm"] = sum(r["length_abs_error_mm"] for r in ok) / len(ok)
        summary["width_mae_mm"] = sum(r["width_abs_error_mm"] for r in ok) / len(ok)
        summary["length_mean_pct_error"] = sum(r["length_pct_error"] for r in ok) / len(ok)
        summary["width_mean_pct_error"] = sum(r["width_pct_error"] for r in ok) / len(ok)

    payload = {"summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
