"""
Run measurement on a split and summarize accept vs reject reasons.

  .\\.venv\\Scripts\\python.exe scripts\\rejection_report.py --split test --marker-size-mm 25.0 --checkpoint models\\runs\\unet_pen\\best.pt --allow-poor-calibration
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

from models.data import load_split_ids
from measurement.config import MeasurementConfig
from measurement.measure_pen import measure_pen_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--params", type=Path, default=ROOT / "calibration" / "camera_params.npz")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "dataset" / "images")
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "rejection_report.json")
    parser.add_argument("--allow-poor-calibration", action="store_true")
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    ids = load_split_ids(args.split)
    rows = []
    reasons = Counter()
    for stem in ids:
        img = args.image_dir / f"{stem}.jpg"
        result, _, _, _ = measure_pen_image(
            img,
            args.params,
            config,
            allow_poor_calibration=args.allow_poor_calibration,
            checkpoint=args.checkpoint,
        )
        rows.append(result.to_json_dict())
        if result.status != "success":
            reasons[result.rejection_reason or "unknown"] += 1

    accepted = sum(1 for r in rows if r["status"] == "success")
    summary = {
        "split": args.split,
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
