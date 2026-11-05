"""Pen segmentation dataset loader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
IMAGES = DATASET / "images"
MASKS = DATASET / "masks"
SPLITS = DATASET / "splits"


def load_split_ids(split: str) -> list[str]:
    path = SPLITS / f"{split}.txt"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


class PenSegDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        image_size: int = 384,
        augment: bool = False,
    ) -> None:
        self.ids = ids
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        stem = self.ids[idx]
        img_path = IMAGES / f"{stem}.jpg"
        mask_path = MASKS / f"{stem}.png"
        bgr = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise FileNotFoundError(f"Missing pair for {stem}")

        if self.augment:
            if np.random.rand() > 0.5:
                bgr = cv2.flip(bgr, 1)
                mask = cv2.flip(mask, 1)
            if np.random.rand() > 0.5:
                bgr = cv2.flip(bgr, 0)
                mask = cv2.flip(mask, 0)

        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        x = torch.from_numpy(rgb)
        y = torch.from_numpy(mask).unsqueeze(0)
        return x, y, stem
