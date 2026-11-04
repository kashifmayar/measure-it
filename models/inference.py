"""U-Net inference helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.unet import UNet


class PenUNet:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.image_size = int(ckpt.get("image_size", 384))
        self.model = UNet()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict_mask(self, bgr: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(self.device)
        logits = self.model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        confidence = float(prob.max())
        mask_small = (prob >= threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask, confidence
