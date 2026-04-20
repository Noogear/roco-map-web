from __future__ import annotations

import cv2
import numpy as np


class PlaceholderFeature2D:
    def detectAndCompute(self, _image_gray: np.ndarray, _mask: np.ndarray | None):
        return [], None


def create_orb_beblid_feature2d() -> PlaceholderFeature2D:
    return PlaceholderFeature2D()


def make_circular_mask(h: int, w: int, inner_ratio: float = 0.0, *, center: tuple[float, float] | None = None, radius: float | None = None) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = float(center[0]) if center is not None else w / 2.0
    cy = float(center[1]) if center is not None else h / 2.0
    r = float(radius) if radius is not None else min(cx, cy, max(0.0, w - cx), max(0.0, h - cy))
    r_outer = max(1, int(round(r)))
    cv2.circle(mask, (int(round(cx)), int(round(cy))), r_outer, 255, -1)
    if inner_ratio > 0:
        r_inner = max(1, int(round(r_outer * float(inner_ratio))))
        if r_inner < r_outer:
            cv2.circle(mask, (int(round(cx)), int(round(cy))), r_inner, 0, -1)
    return mask


class CircularMaskCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def get(self, h: int, w: int, inner_ratio: float = 0.0, *, center: tuple[float, float] | None = None, radius: float | None = None) -> np.ndarray:
        if center is not None or radius is not None:
            return make_circular_mask(h, w, inner_ratio, center=center, radius=radius)
        key = (h, w, int(round(max(0.0, min(0.95, float(inner_ratio))) * 1000)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        mask = make_circular_mask(h, w, inner_ratio)
        self._cache[key] = mask
        return mask
