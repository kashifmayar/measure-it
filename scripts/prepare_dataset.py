"""
Organize, undistort, split, validate, and import masks for the pen dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from calibration.camera_params import load_camera_params
from calibration.image_io import IMAGE_EXTENSIONS, collect_image_paths, read_image_bgr
from calibration.undistort import undistort_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET = ROOT / "dataset"
RAW = DATASET / "raw"
IMAGES = DATASET / "images"
MASKS = DATASET / "masks"
SPLITS = DATASET / "splits"
META = DATASET / "meta"

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
MIN_IMAGES_ASSESSMENT = 70


def _ensure_dirs() -> None:
    for d in (RAW, IMAGES, MASKS, SPLITS, META):
        d.mkdir(parents=True, exist_ok=True)


def cmd_organize(include_measurement: bool) -> None:
    _ensure_dirs()
    moved = 0
    sources = [DATASET]
    if include_measurement:
        sources.append(ROOT / "measurement" / "images")

    for src in sources:
        if not src.is_dir():
            continue
        for path in sorted(src.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.parent.resolve() == RAW.resolve():
                continue
            dest = RAW / path.name
            if dest.exists():
                logger.info("Skip (already in raw): %s", path.name)
                continue
            shutil.move(str(path), str(dest))
            moved += 1
            logger.info("Moved -> raw/: %s", path.name)

    logger.info("Organize complete: %d file(s) moved to %s", moved, RAW)


def cmd_undistort(params_path: Path) -> None:
    _ensure_dirs()
    if not params_path.is_file():
        raise FileNotFoundError(
            f"Missing {params_path}. Run calibrate.py first."
        )
    calib = load_camera_params(params_path)
    camera_matrix = calib.camera_matrix
    dist_coeffs = calib.dist_coeffs

    paths = collect_image_paths(RAW, recursive=False)
    if not paths:
        logger.warning("No images in %s", RAW)
        return

    ok_count = 0
    for path in paths:
        bgr = read_image_bgr(path, normalize_orientation=True)
        if bgr is None:
            logger.warning("Unreadable: %s", path.name)
            continue
        h, w = bgr.shape[:2]
        cm = camera_matrix.copy()
        if (w, h) != (calib.image_width, calib.image_height):
            sx = w / calib.image_width
            sy = h / calib.image_height
            if abs(sx - sy) > 0.02:
                logger.warning("Skip aspect mismatch: %s", path.name)
                continue
            scale = __import__("numpy").array(
                [[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float
            )
            cm = scale @ cm

        out = undistort_image(bgr, cm, dist_coeffs, crop=False)
        out_path = IMAGES / f"{path.stem}.jpg"
        cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ok_count += 1
        logger.info("Undistorted: %s -> %s", path.name, out_path.name)

    logger.info("Undistort complete: %d image(s) in %s", ok_count, IMAGES)


def _list_image_stems() -> list[str]:
    stems = sorted(
        p.stem for p in IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return stems


def cmd_split() -> None:
    _ensure_dirs()
    stems = _list_image_stems()
    if not stems:
        raise SystemExit(f"No images in {IMAGES}. Run undistort first.")

    rng = random.Random(SPLIT_SEED)
    shuffled = stems.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_test = 1
        n_train = n - n_val - n_test

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    for name, items in ("train", train), ("val", val), ("test", test):
        path = SPLITS / f"{name}.txt"
        path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
        logger.info("Wrote %s (%d)", path.name, len(items))

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SPLIT_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "counts": {"total": n, "train": len(train), "val": len(val), "test": len(test)},
        "meets_minimum_70": n >= MIN_IMAGES_ASSESSMENT,
    }
    (META / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def cmd_validate() -> None:
    _ensure_dirs()
    stems = _list_image_stems()
    mask_stems = {
        p.stem for p in MASKS.iterdir() if p.suffix.lower() in {".png", ".bmp"}
    }

    missing_masks = [s for s in stems if s not in mask_stems]
    orphan_masks = sorted(mask_stems - set(stems))

    split_counts = {}
    for split in ("train", "val", "test"):
        p = SPLITS / f"{split}.txt"
        if p.is_file():
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            split_counts[split] = len(lines)
            missing_in_split = [s for s in lines if s not in mask_stems]
            if missing_in_split:
                logger.warning(
                    "%s: %d entries missing masks (first: %s)",
                    split,
                    len(missing_in_split),
                    missing_in_split[0],
                )

    ready = (
        len(stems) >= MIN_IMAGES_ASSESSMENT
        and len(missing_masks) == 0
        and split_counts.get("test", 0) > 0
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(stems),
        "mask_count": len(mask_stems),
        "missing_masks_count": len(missing_masks),
        "missing_masks_sample": missing_masks[:10],
        "orphan_masks_sample": orphan_masks[:10],
        "split_counts": split_counts,
        "assessment_minimum_images": MIN_IMAGES_ASSESSMENT,
        "meets_image_count": len(stems) >= MIN_IMAGES_ASSESSMENT,
        "ready_for_training": ready,
    }
    (META / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    if not ready:
        raise SystemExit(1)


def cmd_import_masks(from_dir: Path) -> None:
    _ensure_dirs()
    if not from_dir.is_dir():
        raise FileNotFoundError(from_dir)
    imported = 0
    for path in from_dir.iterdir():
        if path.suffix.lower() not in {".png", ".bmp", ".jpg"}:
            continue
        stem = path.stem
        for suffix in ("-mask", "_mask", "-segmentation"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        dest = MASKS / f"{stem}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(dest), binary)
        imported += 1
    logger.info("Imported %d mask(s) into %s", imported, MASKS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare pen dataset for segmentation training.")
    sub = p.add_subparsers(dest="command", required=True)

    o = sub.add_parser("organize", help="Move images into dataset/raw/")
    o.add_argument(
        "--include-measurement",
        action="store_true",
        help="Also move measurement/images/* into raw/",
    )

    u = sub.add_parser("undistort", help="Undistort raw/ -> images/")
    u.add_argument(
        "--params",
        type=Path,
        default=ROOT / "calibration" / "camera_params.npz",
    )

    sub.add_parser("split", help="Write splits/ train,val,test (seed 42)")
    sub.add_parser("validate", help="Check masks and write meta/dataset_summary.json")

    im = sub.add_parser("import-masks", help="Import exported masks into dataset/masks/")
    im.add_argument("--from-dir", type=Path, required=True)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "organize":
        cmd_organize(include_measurement=args.include_measurement)
    elif args.command == "undistort":
        cmd_undistort(args.params)
    elif args.command == "split":
        cmd_split()
    elif args.command == "validate":
        cmd_validate()
    elif args.command == "import-masks":
        cmd_import_masks(args.from_dir)


if __name__ == "__main__":
    main()
