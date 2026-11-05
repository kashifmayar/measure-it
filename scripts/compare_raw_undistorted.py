"""
Compare pen mm on the same image with and without lens undistortion.

  .\\.venv\\Scripts\\python.exe scripts\\compare_raw_undistorted.py --image dataset\\images\\pen_0001.jpg --marker-size-mm 25.0 --checkpoint models\\runs\\unet_pen\\best.pt
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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--params", type=Path, default=ROOT / "calibration" / "camera_params.npz")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "raw_vs_undistorted.json")
    parser.add_argument("--allow-poor-calibration", action="store_true")
    args = parser.parse_args()

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    kw = dict(
        allow_poor_calibration=args.allow_poor_calibration,
        checkpoint=args.checkpoint,
    )
    raw, _, _, _ = measure_pen_image(
        args.image, args.params, config, skip_undistort=True, **kw
    )
    und, _, _, _ = measure_pen_image(
        args.image, args.params, config, skip_undistort=False, **kw
    )
    payload = {
        "image": args.image.name,
        "raw": raw.to_json_dict(),
        "undistorted": und.to_json_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
