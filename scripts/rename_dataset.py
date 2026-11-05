"""
Rename dataset files to stable IDs: pen_0001, pen_0002, ...

Updates raw/, images/, masks/, splits/, and writes meta/filename_map.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET = ROOT / "dataset"
RAW = DATASET / "raw"
IMAGES = DATASET / "images"
MASKS = DATASET / "masks"
MASKS_DRAFT = DATASET / "masks_draft"
SPLITS = DATASET / "splits"
META = DATASET / "meta"


def _image_stems() -> list[str]:
    stems = sorted(
        p.stem for p in IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return stems


def _find_raw(old_stem: str) -> Path | None:
    for ext in (".HEIC", ".heic", ".jpg", ".jpeg", ".png"):
        p = RAW / f"{old_stem}{ext}"
        if p.is_file():
            return p
    return None


def rename_dataset(prefix: str = "pen", start_index: int = 1, dry_run: bool = False) -> None:
    META.mkdir(parents=True, exist_ok=True)
    old_stems = _image_stems()
    if not old_stems:
        raise SystemExit(f"No images in {IMAGES}")

    mapping: dict[str, str] = {}
    renames: list[tuple[Path, Path]] = []

    for i, old in enumerate(old_stems, start=start_index):
        new_id = f"{prefix}_{i:04d}"
        mapping[new_id] = old

        img_old = next(
            p for p in IMAGES.iterdir() if p.stem == old and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        renames.append((img_old, IMAGES / f"{new_id}.jpg"))

        raw_old = _find_raw(old)
        if raw_old:
            renames.append((raw_old, RAW / f"{new_id}{raw_old.suffix}"))

        for folder in (MASKS, MASKS_DRAFT):
            for ext in (".png", ".bmp"):
                m = folder / f"{old}{ext}"
                if m.is_file():
                    renames.append((m, folder / f"{new_id}{ext}"))

    split_updates: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        sp = SPLITS / f"{split}.txt"
        if not sp.is_file():
            continue
        old_to_new = {old: f"{prefix}_{i+start_index:04d}" for i, old in enumerate(old_stems)}
        lines = [ln.strip() for ln in sp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        split_updates[split] = [old_to_new.get(ln, ln) for ln in lines]

    if dry_run:
        print(json.dumps({"would_rename": len(renames), "sample_map": dict(list(mapping.items())[:5])}, indent=2))
        return

    for src, dst in renames:
        if src.resolve() == dst.resolve():
            continue
        if dst.exists():
            raise FileExistsError(f"Target exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        logger.info("%s -> %s", src.name, dst.name)

    for split, ids in split_updates.items():
        (SPLITS / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        logger.info("Updated %s.txt (%d ids)", split, len(ids))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "id_to_original_stem": mapping,
        "original_stem_to_id": {v: k for k, v in mapping.items()},
        "prefix": prefix,
        "count": len(mapping),
    }
    (META / "filename_map.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote meta/filename_map.json (%d entries)", len(mapping))


def main() -> None:
    p = argparse.ArgumentParser(description="Rename dataset to pen_0001 style IDs.")
    p.add_argument("--prefix", default="pen", help="ID prefix (default: pen)")
    p.add_argument("--start", type=int, default=1, help="Starting index (default: 1)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rename_dataset(prefix=args.prefix, start_index=args.start, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
