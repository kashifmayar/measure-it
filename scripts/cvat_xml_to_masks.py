"""
Convert CVAT 1.1 annotations.xml (polygons) to binary PNG masks in dataset/masks/.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

IMAGES = ROOT / "dataset" / "images"
MASKS = ROOT / "dataset" / "masks"


def polygon_points(attr: str) -> np.ndarray:
    pts = []
    for pair in attr.split(";"):
        if not pair.strip():
            continue
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.int32)


def convert(xml_path: Path, images_dir: Path, out_dir: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "polygons": 0, "written": 0, "skipped": []}

    for image_el in root.findall("image"):
        stats["images"] += 1
        name = image_el.get("name", "")
        stem = Path(name).name
        stem = Path(stem).stem
        w = int(image_el.get("width", 0))
        h = int(image_el.get("height", 0))

        ref = images_dir / f"{stem}.jpg"
        if ref.is_file():
            im = cv2.imread(str(ref))
            if im is not None:
                h, w = im.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        polys = image_el.findall("polygon")
        for poly in polys:
            if poly.get("label") != "pen":
                continue
            pts = polygon_points(poly.get("points", ""))
            if len(pts) < 3:
                continue
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
            stats["polygons"] += 1

        if not polys:
            stats["skipped"].append(stem)
            continue

        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        stats["written"] += 1

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="CVAT XML polygons -> dataset/masks PNG")
    p.add_argument(
        "--xml",
        type=Path,
        default=ROOT / "pen dataset" / "annotations.xml",
    )
    p.add_argument("--images-dir", type=Path, default=IMAGES)
    p.add_argument("--out-dir", type=Path, default=MASKS)
    args = p.parse_args()
    stats = convert(args.xml, args.images_dir, args.out_dir)
    print(f"XML images: {stats['images']}, polygons drawn: {stats['polygons']}")
    print(f"Masks written: {stats['written']} -> {args.out_dir}")
    if stats["skipped"]:
        print("Skipped (no polygon):", stats["skipped"])


if __name__ == "__main__":
    main()
