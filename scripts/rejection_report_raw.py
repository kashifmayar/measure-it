"""
Rejection stats on dataset/raw HEIC (where ArUco exists for measurement).

  .\\.venv\\Scripts\\python.exe scripts\\rejection_report_raw.py --marker-size-mm 34.0 --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "rejection_report_raw.json")
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    params = ROOT / "calibration" / "camera_params.npz"
    ckpt = args.checkpoint if args.checkpoint.is_file() else None
    paths = sorted(args.raw_dir.glob("*.HEIC"))[: args.limit]

    rows = []
    reasons = Counter()
    for path in paths:
        result, _, _, _ = measure_pen_image(
            path,
            params,
            config,
            allow_poor_calibration=True,
            checkpoint=ckpt,
            skip_undistort=True,
        )
        rows.append(result.to_json_dict())
        if result.status != "success":
            reasons[result.rejection_reason or "unknown"] += 1

    accepted = sum(1 for r in rows if r["status"] == "success")
    summary = {
        "source": "dataset/raw HEIC, skip_undistort=True",
        "total": len(rows),
        "accepted": accepted,
        "rejected": len(rows) - accepted,
        "accept_rate": round(accepted / len(rows), 4) if rows else 0.0,
        "rejection_reasons": dict(reasons),
    }
    payload = {"summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
