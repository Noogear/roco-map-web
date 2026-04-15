"""
core/ecc.py - ECC 像素级对齐（低纹理场景兜底）

提供单函数接口，无可变状态。
"""

from __future__ import annotations

import numpy as np
import cv2


def ecc_align(
    minimap_gray: np.ndarray,
    logic_map_gray: np.ndarray,
    cx_hint: int,
    cy_hint: int,
    last_sift_scale: float,
    map_width: int,
    map_height: int,
    jump_threshold: int,
    min_cc: float = 0.15,
) -> tuple[int, int] | None:
    """
    ECC（增强相关系数）像素级匹配。
    在 (cx_hint, cy_hint) 附近、1.5 倍 crop 范围内寻找当前小地图的位置。

    Returns:
        (map_x, map_y) 大地图坐标，或 None（失败/超跳变阈值）。
    """
    s = last_sift_scale
    h_mm, w_mm = minimap_gray.shape[:2]
    crop_w = int(w_mm * s * 1.5)
    crop_h = int(h_mm * s * 1.5)
    x1 = max(0, cx_hint - crop_w // 2)
    y1 = max(0, cy_hint - crop_h // 2)
    x2 = min(map_width, x1 + crop_w)
    y2 = min(map_height, y1 + crop_h)
    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w < 20 or actual_h < 20:
        return None

    map_crop = logic_map_gray[y1:y2, x1:x2]
    map_resized = cv2.resize(map_crop, (w_mm, h_mm))

    ref = map_resized.astype(np.float32)
    tmpl = minimap_gray.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            ref, tmpl, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3))
    except cv2.error:
        return None

    if cc < min_cc:
        return None

    tx_px, ty_px = float(warp[0, 2]), float(warp[1, 2])
    scale_x = actual_w / w_mm
    scale_y = actual_h / h_mm
    center_ref_x = w_mm / 2.0 - tx_px
    center_ref_y = h_mm / 2.0 - ty_px
    map_x = int(x1 + center_ref_x * scale_x)
    map_y = int(y1 + center_ref_y * scale_y)

    if not (0 <= map_x < map_width and 0 <= map_y < map_height):
        return None
    if abs(map_x - cx_hint) + abs(map_y - cy_hint) < jump_threshold:
        return map_x, map_y
    return None
