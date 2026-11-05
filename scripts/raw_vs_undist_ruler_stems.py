"""
Raw vs undistorted for stems listed in ruler_refs.csv (measurement evidence).

  .\\.venv\\Scripts\\python.exe scripts\\raw_vs_undist_ruler_stems.py --marker-size-mm 34.0
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
from scripts.validate_ruler import _resolve_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "validation" / "ruler_refs.csv")
    parser.add_argument("--marker-size-mm", type=float, default=34.0)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "runs" / "unet_pen" / "best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "raw_vs_undistorted_ruler_stems.json")
    parser.add_argument("--max-rows", type=int, default=12)
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    params = ROOT / "calibration" / "camera_params.npz"
    ckpt = args.checkpoint if args.checkpoint.is_file() else None
    kw = dict(allow_poor_calibration=True, checkpoint=ckpt)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        stems = [r["image"].strip() for r in csv.DictReader(f) if r.get("image")]

    for stem in stems[: args.max_rows]:
        img = _resolve_image(stem, ROOT / "dataset" / "images", ROOT / "dataset" / "raw")
        if img is None:
            continue
        raw, _, _, _ = measure_pen_image(img, params, config, skip_undistort=True, **kw)
        und, _, _, _ = measure_pen_image(img, params, config, skip_undistort=False, **kw)
        rows.append(
            {
                "stem": stem,
                "file": img.name,
                "raw_status": raw.status,
                "raw_length_mm": raw.pen_length_mm,
                "raw_width_mm": raw.pen_max_width_mm,
                "undist_status": und.status,
                "undist_length_mm": und.pen_length_mm,
                "undist_width_mm": und.pen_max_width_mm,
            }
        )

    raw_ok = sum(1 for r in rows if r["raw_status"] == "success")
    und_ok = sum(1 for r in rows if r["undist_status"] == "success")
    summary = {
        "stems_compared": len(rows),
        "raw_success": raw_ok,
        "undist_success": und_ok,
        "note": "Low undist_success expected until calibration reprojection <= 2 px.",
    }
    payload = {"summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
