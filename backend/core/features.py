"""
core/features.py - SIFT/ORB 特征提取与匹配（无状态纯函数 + 缓存工具）

所有返回值均基本数据类型或 numpy array，无 Web 框架依赖。
"""

from __future__ import annotations

import numpy as np
import cv2
from backend import config


# ---------------------------------------------------------------------------
# 圆形掩码（带 LRU 缓存）
# ---------------------------------------------------------------------------

def make_circular_mask(h: int, w: int) -> np.ndarray:
    """生成大小为 (h, w) 的圆形掩码。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.circle(mask, (cx, cy), min(cx, cy) - 2, 255, -1)
    return mask


class CircularMaskCache:
    """按 (h, w) 缓存圆形掩码，避免重复生成。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def get(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._cache:
            self._cache[key] = make_circular_mask(h, w)
        return self._cache[key]


# ---------------------------------------------------------------------------
# FLANN 索引工厂
# ---------------------------------------------------------------------------

def create_flann(descriptors: np.ndarray) -> cv2.FlannBasedMatcher:
    """
    从 des 数组构造 FLANN L2 索引。
    des 需为 float32，每行一个描述子。
    """
    des = descriptors.astype(np.float32)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([des])
    flann.train()
    return flann


# ---------------------------------------------------------------------------
# 小地图特征提取
# ---------------------------------------------------------------------------

def extract_minimap_features(
    minimap_gray: np.ndarray,
    sift: cv2.SIFT,
    mask_cache: CircularMaskCache,
) -> tuple[list | None, np.ndarray | None]:
    """提取小地图特征（带圆形掩码）。返回 (kp, des) 或 (None, None)。"""
    h, w = minimap_gray.shape[:2]
    mask = mask_cache.get(h, w)
    kp, des = sift.detectAndCompute(minimap_gray, mask)
    if des is None or len(kp) < 2:
        return None, None
    return kp, des


# ---------------------------------------------------------------------------
# SIFT 区域匹配
# ---------------------------------------------------------------------------

def sift_match_region(
    kp_mini: list,
    des_mini: np.ndarray,
    mm_shape: tuple[int, int],
    region_kp: list,
    region_flann: cv2.FlannBasedMatcher,
    ratio: float,
    min_match: int,
    map_width: int,
    map_height: int,
) -> tuple[int, int, int, float, float] | None:
    """
    通用 SIFT 区域匹配。
    返回 (tx, ty, inlier_count, quality, avg_scale) 或 None。
    """
    try:
        matches = region_flann.knnMatch(des_mini, k=2)
    except cv2.error:
        return None

    good = [m for m_n in matches if len(m_n) == 2
            for m, n in [m_n] if m.distance < ratio * n.distance]

    if len(good) < min_match:
        return None

    src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([region_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)
    if M is None:
        return None

    inlier_count = int(mask.sum()) if mask is not None else 0
    if inlier_count < min_match:
        return None

    sx = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    sy = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
    avg_scale = (sx + sy) / 2
    max_scale = getattr(config, 'SIFT_MAX_HOMOGRAPHY_SCALE', 8.0)
    if avg_scale > max_scale or avg_scale < 1.0 / max_scale:
        return None

    h, w = mm_shape[:2]
    center_pt = np.float32([[[w / 2.0, h / 2.0]]])
    dst_center = cv2.perspectiveTransform(center_pt, M)
    tx, ty = int(dst_center[0][0][0]), int(dst_center[0][0][1])

    if not (0 <= tx < map_width and 0 <= ty < map_height):
        return None

    inlier_ratio = inlier_count / max(len(good), 1)
    count_conf = min(1.0, inlier_count / 12.0)
    quality = min(1.0, inlier_ratio * count_conf)
    return tx, ty, inlier_count, quality, avg_scale
