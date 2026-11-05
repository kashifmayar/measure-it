"""
Move images/masks into dataset/images and dataset/masks after a bad zip extract.

Run from repo root (Colab: /content/measure-it):

  python scripts/fix_dataset_layout.py
  python scripts/fix_dataset_layout.py --search-root /content/measure-it
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGES = ROOT / "dataset" / "images"
MASKS = ROOT / "dataset" / "masks"


def _move_glob(src_dir: Path, dest_dir: Path, pattern: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if not src_dir.is_dir():
        return 0
    for p in src_dir.glob(pattern):
        if not p.is_file():
            continue
        target = dest_dir / p.name
        if target.is_file():
            continue
        shutil.copy2(p, target)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix dataset folder layout after Colab zip extract.")
    parser.add_argument("--search-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.search_root.resolve()

    IMAGES.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)

    candidates_images = [
        root / "dataset" / "images",
        root / "images",
        root / "dataset" / "dataset" / "images",
    ]
    candidates_masks = [
        root / "dataset" / "masks",
        root / "masks",
        root / "dataset" / "dataset" / "masks",
    ]

    # Also search one level deep for pen_*.jpg
    for d in [root, root / "dataset"]:
        if d.is_dir():
            candidates_images.append(d)
            for sub in d.iterdir():
                if sub.is_dir() and sub.name in ("images", "Images"):
                    candidates_images.append(sub)

    img_moved = 0
    for c in candidates_images:
        img_moved += _move_glob(c, IMAGES, "pen_*.jpg")
        img_moved += _move_glob(c, IMAGES, "pen_*.jpeg")

    msk_moved = 0
    for c in candidates_masks:
        msk_moved += _move_glob(c, MASKS, "pen_*.png")

    # Flat search under root (max depth 4) if still empty
    if img_moved == 0:
        for p in root.rglob("pen_*.jpg"):
            if "dataset" in p.parts and p.parent.name == "images" and p.parent.parent.name == "dataset":
                if p.parent.resolve() == IMAGES.resolve():
                    continue
            t = IMAGES / p.name
            if not t.is_file():
                shutil.copy2(p, t)
                img_moved += 1
    if msk_moved == 0:
        for p in root.rglob("pen_*.png"):
            if p.parent.resolve() == MASKS.resolve():
                continue
            t = MASKS / p.name
            if not t.is_file():
                shutil.copy2(p, t)
                msk_moved += 1

    n_img = len(list(IMAGES.glob("pen_*.jpg")))
    n_msk = len(list(MASKS.glob("pen_*.png")))
    print(f"Copied/moved this run: {img_moved} images, {msk_moved} masks")
    print(f"Now in dataset/images: {n_img}, dataset/masks: {n_msk}")
    if n_img < 70 or n_msk < 70:
        print("ERROR: still missing files. Zip should contain dataset/images and dataset/masks")
        raise SystemExit(1)
    print("OK — run: python scripts/prepare_dataset.py validate")


if __name__ == "__main__":
    main()
