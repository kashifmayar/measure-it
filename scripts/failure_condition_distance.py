"""
Failure-condition demo: error vs simulated distance (downscale marker/pen in frame).

  .\\.venv\\Scripts\\python.exe scripts\\failure_condition_distance.py --image dataset\\raw\\pen_0042.HEIC --marker-size-mm 34.0 --checkpoint models\\runs\\unet_pen\\best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from calibration.image_io import read_image_bgr
from measurement.config import MeasurementConfig
from measurement.measure_pen import measure_pen_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--marker-size-mm", type=float, default=34.0)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "runs" / "unet_pen" / "best.pt")
    parser.add_argument("--params", type=Path, default=ROOT / "calibration" / "camera_params.npz")
    parser.add_argument("--scales", type=str, default="1.0,0.75,0.55,0.4")
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "failure_distance.json")
    args = parser.parse_args()

    bgr = read_image_bgr(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read {args.image}")

    config = MeasurementConfig(marker_size_mm=args.marker_size_mm)
    rows = []
    tmp = ROOT / "validation" / "_failure_tmp.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    for s_str in args.scales.split(","):
        scale = float(s_str.strip())
        h, w = bgr.shape[:2]
        nh, nw = max(64, int(h * scale)), max(64, int(w * scale))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(tmp), up)
        result, _, _, _ = measure_pen_image(
            tmp,
            args.params,
            config,
            allow_poor_calibration=True,
            checkpoint=args.checkpoint if args.checkpoint.is_file() else None,
            skip_undistort=True,
        )
        row = result.to_json_dict()
        row["simulated_distance_scale"] = scale
        rows.append(row)

    payload = {
        "condition": "distance_proxy_via_downscale_upscale",
        "image": str(args.image),
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
