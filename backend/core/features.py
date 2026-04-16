"""
core/features.py - SIFT 特征提取与匹配（含 RootSIFT 后处理）

所有返回值均基本数据类型或 numpy array，无 Web 框架依赖。
"""

from __future__ import annotations

import numpy as np
import cv2
from backend import config


# ---------------------------------------------------------------------------
# 圆形掩码（带 LRU 缓存）
# ---------------------------------------------------------------------------

def make_circular_mask(h: int, w: int, inner_ratio: float = 0.0) -> np.ndarray:
    """生成大小为 (h, w) 的圆形掩码；可选中心挖空（用于排除玩家箭头）。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r_outer = min(cx, cy) - 2
    cv2.circle(mask, (cx, cy), r_outer, 255, -1)
    if inner_ratio > 0:
        r_inner = max(1, int(round(r_outer * float(inner_ratio))))
        if r_inner < r_outer:
            cv2.circle(mask, (cx, cy), r_inner, 0, -1)
    return mask


class CircularMaskCache:
    """按 (h, w, inner_px) 缓存圆形掩码，避免重复生成。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def get(self, h: int, w: int, inner_ratio: float = 0.0) -> np.ndarray:
        cx, cy = w // 2, h // 2
        r_outer = max(1, min(cx, cy) - 2)
        ratio = max(0.0, min(0.95, float(inner_ratio)))
        inner_px = int(round(r_outer * ratio)) if ratio > 0 else 0
        key = (h, w, inner_px)
        if key not in self._cache:
            self._cache[key] = make_circular_mask(h, w, inner_px / float(r_outer))
        return self._cache[key]


# ---------------------------------------------------------------------------
# FLANN 索引工厂
# ---------------------------------------------------------------------------

def create_flann(descriptors: np.ndarray) -> cv2.FlannBasedMatcher:
    """
    从 des 数组构造 FLANN L2 索引。
    des 需为 float32，每行一个描述子。
    """
    des = np.asarray(descriptors, dtype=np.float32)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([des])
    flann.train()
    return flann


# ---------------------------------------------------------------------------
# RootSIFT 后处理（Hellinger 核映射，L2 距离等价 Hellinger 距离）
# ---------------------------------------------------------------------------

def _apply_rootsift(des: np.ndarray) -> np.ndarray:
    """将 SIFT 描述符映射到 Hellinger 核空间（RootSIFT），就地修改并返回。"""
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    np.sqrt(des, out=des)
    return des


# ---------------------------------------------------------------------------
# 小地图特征提取
# ---------------------------------------------------------------------------

def extract_minimap_features(
    minimap_gray: np.ndarray,
    sift: cv2.SIFT,
    mask_cache: CircularMaskCache,
    *,
    texture_std: float | None = None,
    inner_ratio: float | None = None,
) -> tuple[list | None, np.ndarray | None]:
    """提取小地图特征（带圆形掩码）。返回 (kp, des) 或 (None, None)。

    texture_std:  按场景分类（ocean/low_texture/mixed/urban）自动选择挖空比例。
    inner_ratio:  直接指定挖空比例，覆盖 config 及 texture_std 逻辑（多用于测试）。
    """
    h, w = minimap_gray.shape[:2]
    if inner_ratio is None:
        r_urban = float(getattr(config, 'MINIMAP_CENTER_EXCLUDE_RATIO', 0.16))
        r_mixed = float(getattr(config, 'MINIMAP_CENTER_EXCLUDE_RATIO_MIXED', r_urban))
        r_hard  = float(getattr(config, 'MINIMAP_CENTER_EXCLUDE_RATIO_HARD', r_mixed))
        if texture_std is not None:
            if texture_std < 35:    # low_texture + ocean
                inner_ratio = r_hard
            elif texture_std < 55:  # mixed
                inner_ratio = r_mixed
            else:                   # urban
                inner_ratio = r_urban
        else:
            inner_ratio = r_urban

    mask = mask_cache.get(h, w, inner_ratio=inner_ratio)

    kp, des = sift.detectAndCompute(minimap_gray, mask)
    if des is None or len(kp) < 2:
        return None, None
    _apply_rootsift(des)
    return kp, des


def extract_map_features_tiled(
    map_gray: np.ndarray,
    sift: cv2.SIFT,
    *,
    tile_size: int,
    overlap: int = 0,
    max_features_per_tile: int = 0,
) -> tuple[list, np.ndarray | None]:
    """
    以分块方式提取整张大地图的 SIFT 特征，降低 detectAndCompute 的峰值内存。

    overlap 用于覆盖 tile 边界附近的特征；通过“归属区域”裁剪避免重复保留。
    """
    h, w = map_gray.shape[:2]
    tile_size = max(256, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    step = max(1, tile_size - overlap)
    half_overlap = overlap // 2

    all_keypoints: list = []
    descriptor_chunks: list[np.ndarray] = []

    for y0 in range(0, h, step):
        y1 = min(h, y0 + tile_size)
        own_top = y0 if y0 == 0 else y0 + half_overlap
        own_bottom = y1 if y1 == h else y1 - half_overlap

        for x0 in range(0, w, step):
            x1 = min(w, x0 + tile_size)
            own_left = x0 if x0 == 0 else x0 + half_overlap
            own_right = x1 if x1 == w else x1 - half_overlap

            tile = map_gray[y0:y1, x0:x1]
            kp_tile, des_tile = sift.detectAndCompute(tile, None)
            if des_tile is None or not kp_tile:
                continue

            kept: list[tuple[int, float, float]] = []
            for idx, kp in enumerate(kp_tile):
                gx = kp.pt[0] + x0
                gy = kp.pt[1] + y0
                if own_left <= gx < own_right and own_top <= gy < own_bottom:
                    kept.append((idx, gx, gy))

            if not kept:
                continue

            if max_features_per_tile > 0 and len(kept) > max_features_per_tile:
                responses = np.array([kp_tile[idx].response for idx, _, _ in kept], dtype=np.float32)
                top = np.argpartition(responses, -max_features_per_tile)[-max_features_per_tile:]
                kept = [kept[i] for i in sorted(top)]

            descriptor_chunks.append(des_tile[[idx for idx, _, _ in kept]])
            for idx, gx, gy in kept:
                kp = kp_tile[idx]
                all_keypoints.append(cv2.KeyPoint(
                    x=float(gx),
                    y=float(gy),
                    size=kp.size,
                    angle=kp.angle,
                    response=kp.response,
                    octave=kp.octave,
                    class_id=kp.class_id,
                ))

    if not descriptor_chunks:
        return [], None

    des_all = np.vstack(descriptor_chunks).astype(np.float32, copy=False)
    _apply_rootsift(des_all)
    return all_keypoints, des_all


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
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC,
        ransacReprojThreshold=config.SIFT_RANSAC_THRESHOLD)
    if M is None:
        return None

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if inlier_count < min_match:
        return None

    # AffinePartial2D 返回 2x3 矩阵: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
    avg_scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    max_scale = getattr(config, 'SIFT_MAX_HOMOGRAPHY_SCALE', 8.0)
    if avg_scale > max_scale or avg_scale < 1.0 / max_scale:
        return None

    h, w = mm_shape[:2]
    center_src = np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float64)
    dst_center = M @ center_src
    tx, ty = int(dst_center[0]), int(dst_center[1])

    if not (0 <= tx < map_width and 0 <= ty < map_height):
        return None

    inlier_ratio = inlier_count / max(len(good), 1)
    count_conf = min(1.0, inlier_count / 12.0)
    quality = min(1.0, inlier_ratio * count_conf)
    return tx, ty, inlier_count, quality, avg_scale
