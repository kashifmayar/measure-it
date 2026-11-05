"""
One-shot: test-set segmentation eval + measurement attempts on raw HEIC (ArUco visible).

  .\\.venv\\Scripts\\python.exe scripts\\run_test_eval_and_reports.py --marker-size-mm 34.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.data import load_split_ids
from measurement.config import MeasurementConfig
from measurement.measure_pen import measure_pen_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-size-mm", type=float, default=34.0)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "runs" / "unet_pen" / "best.pt")
    parser.add_argument("--params", type=Path, default=ROOT / "calibration" / "camera_params.npz")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint
    if ckpt.is_file():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "evaluate.py"),
                "--checkpoint",
                str(ckpt),
                "--split",
                "test",
                "--out",
                str(args.out_dir / "test_metrics.json"),
            ],
            check=False,
            cwd=ROOT,
        )

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    raw_dir = ROOT / "dataset" / "raw"
    stems = load_split_ids("test")
    measure_rows = []
    for stem in stems:
        heic = raw_dir / f"{stem}.HEIC"
        jpg = raw_dir / f"{stem}.jpg"
        path = heic if heic.is_file() else jpg
        if not path.is_file():
            measure_rows.append({"stem": stem, "status": "missing_raw"})
            continue
        result, _, _, _ = measure_pen_image(
            path,
            args.params,
            config,
            allow_poor_calibration=True,
            checkpoint=ckpt if ckpt.is_file() else None,
            skip_undistort=True,
        )
        measure_rows.append(result.to_json_dict())

    accepted = sum(1 for r in measure_rows if r.get("status") == "success")
    summary = {
        "test_stems": len(stems),
        "measurement_attempted": len(measure_rows),
        "accepted": accepted,
        "rejected": len(measure_rows) - accepted,
        "note": "Uses dataset/raw HEIC with --skip-undistort until calibration is improved.",
    }
    out_path = args.out_dir / "test_split_measurement_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": measure_rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
