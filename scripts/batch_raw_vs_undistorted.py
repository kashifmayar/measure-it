"""
Raw vs undistorted comparison for all raw HEIC where marker is found (skip-undistort vs full).

  .\\.venv\\Scripts\\python.exe scripts\\batch_raw_vs_undistorted.py --marker-size-mm 34.0 --limit 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measurement.config import MeasurementConfig
from measurement.measure_pen import measure_pen_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-size-mm", type=float, default=34.0)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "dataset" / "raw")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "runs" / "unet_pen" / "best.pt")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "raw_vs_undistorted_batch.json")
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    params = ROOT / "calibration" / "camera_params.npz"
    ckpt = args.checkpoint if args.checkpoint.is_file() else None
    kw = dict(allow_poor_calibration=True, checkpoint=ckpt)
    rows = []
    for path in sorted(args.raw_dir.glob("*.HEIC"))[: args.limit]:
        raw, _, _, _ = measure_pen_image(
            path, params, config, skip_undistort=True, **kw
        )
        und, _, _, _ = measure_pen_image(
            path, params, config, skip_undistort=False, **kw
        )
        rows.append(
            {
                "image": path.name,
                "raw_status": raw.status,
                "raw_length_mm": raw.pen_length_mm,
                "raw_width_mm": raw.pen_max_width_mm,
                "undist_status": und.status,
                "undist_length_mm": und.pen_length_mm,
                "undist_width_mm": und.pen_max_width_mm,
                "raw_rejection": raw.rejection_reason,
                "undist_rejection": und.rejection_reason,
            }
        )

    payload = {"marker_size_mm": args.marker_size_mm, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"count": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
