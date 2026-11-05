"""
List raw HEIC images where measurement succeeds; write ruler_refs.csv template.

User fills reference_length_mm and reference_width_mm with calliper/ruler values.

  .\\.venv\\Scripts\\python.exe scripts\\bootstrap_ruler_csv.py --marker-size-mm 34.0 --min-success 10
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "ruler_refs.csv")
    parser.add_argument("--min-success", type=int, default=10)
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    ckpt = args.checkpoint if args.checkpoint.is_file() else None
    successes = []
    for path in sorted(args.raw_dir.glob("*.HEIC")):
        result, _, _, _ = measure_pen_image(
            path,
            ROOT / "calibration" / "camera_params.npz",
            config,
            allow_poor_calibration=True,
            checkpoint=ckpt,
            skip_undistort=True,
        )
        if result.status == "success":
            stem = path.stem
            successes.append(
                {
                    "image": stem,
                    "reference_length_mm": "",
                    "reference_width_mm": "",
                    "system_length_mm": result.pen_length_mm,
                    "system_width_mm": result.pen_max_width_mm,
                }
            )

    if len(successes) < args.min_success:
        print(
            f"Warning: only {len(successes)} successful measurements "
            f"(wanted {args.min_success}). Fill CSV anyway or fix detection."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "reference_length_mm",
                "reference_width_mm",
                "system_length_mm",
                "system_width_mm",
            ],
        )
        w.writeheader()
        for row in successes[: max(args.min_success, len(successes))]:
            w.writerow(row)

    print(f"Wrote {len(successes)} success row(s) -> {args.out}")
    print("Fill reference_* columns with ruler/calliper, then run validate_ruler.py")


if __name__ == "__main__":
    main()
