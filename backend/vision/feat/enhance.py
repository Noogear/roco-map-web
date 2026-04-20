from __future__ import annotations

import numpy as np


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    return np.asarray(gray, dtype=np.uint8)


def process_minimap(gray_raw: np.ndarray) -> np.ndarray:
    return np.asarray(gray_raw, dtype=np.uint8)


def correct_color_temperature(bgr: np.ndarray) -> np.ndarray:
    return bgr


def classify_scene_by_color(_bgr: np.ndarray, prior_scene: str) -> str:
    return prior_scene


def make_scene_boosted_gray(_bgr: np.ndarray, _scene_detail: str) -> np.ndarray | None:
    return None
