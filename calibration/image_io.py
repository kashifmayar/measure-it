"""
Load calibration and measurement images, including HEIC from iPhone captures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".heif"}

_heif_registered = False


def _register_heif() -> None:
    global _heif_registered
    if _heif_registered:
        return
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        _heif_registered = True
    except ImportError:
        logger.debug("pillow_heif not installed; HEIC requires pillow-heif or pre-converted JPEG.")


def normalize_portrait_upright(bgr: np.ndarray) -> np.ndarray:
    """Rotate to portrait (height >= width) for consistent calibration geometry."""
    h, w = bgr.shape[:2]
    if w > h:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    return bgr


def read_image_bgr(path: Path, *, normalize_orientation: bool = False) -> np.ndarray | None:
    """
    Read an image as BGR uint8. Supports HEIC via pillow-heif when installed.
    """
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        _register_heif()
        try:
            from PIL import Image

            pil = Image.open(path)
            rgb = np.array(pil.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if normalize_orientation:
                bgr = normalize_portrait_upright(bgr)
            return bgr
        except Exception as exc:
            logger.warning("Failed to read HEIC %s: %s", path.name, exc)
            return None

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None and normalize_orientation:
        img = normalize_portrait_upright(img)
    return img


def read_image_gray(path: Path) -> np.ndarray | None:
    bgr = read_image_bgr(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def image_dimensions(path: Path) -> Tuple[int, int] | None:
    """Return (width, height) without full decode when possible."""
    gray = read_image_gray(path)
    if gray is None:
        return None
    h, w = gray.shape[:2]
    return w, h


def laplacian_sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher usually means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def collect_image_paths(
    directory: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Sorted list of supported image files under ``directory``."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths: list[Path] = []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for p in iterator:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)
