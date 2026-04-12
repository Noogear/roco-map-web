"""
core/cold_start.py - 低纹理冷启动（海洋/大片裸地场景定位）

ColdStarter 封装所有冷启动状态，提供两个公开方法：
  build_candidates(map_gray)    启动时调用一次，扫描低纹理候选区
  locate(minimap_gray_raw, ...)  每帧在候选区做多尺度模板匹配，返回坐标或 None
"""

from __future__ import annotations

import numpy as np
import cv2

from backend import config
from backend.core.features import CircularMaskCache


class ColdStarter:
    """
    低纹理冷启动定位器。

    候选区在初始化时用 build_candidates() 扫描大地图一次，
    之后每次 locate() 在候选区做多尺度模板匹配。
    """

    def __init__(
        self,
        map_width: int,
        map_height: int,
        logic_map_gray: np.ndarray,
        mask_cache: CircularMaskCache | None = None,
    ) -> None:
        self._map_width = map_width
        self._map_height = map_height
        self._logic_map_gray = logic_map_gray
        self._mask_cache = mask_cache or CircularMaskCache()

        self._region_tile: int = getattr(config, 'OCEAN_REGION_TILE', 100)
        self._std_thresh: float = getattr(config, 'OCEAN_STD_THRESHOLD', 35)

        self._candidates: list[tuple[int, int, float]] = []
        self._last_scale: float = 4.0
        self.cooldown: int = 0  # 外部每帧递减

    # ------------------------------------------------------------------
    def build_candidates(self) -> None:
        """扫描大地图，找出所有低纹理候选区（需在引擎 __init__ 末尾调用一次）。"""
        tile = self._region_tile
        h, w = self._logic_map_gray.shape[:2]
        half = tile // 2
        candidates = []
        for cy in range(half, h - half, tile):
            for cx in range(half, w - half, tile):
                patch = self._logic_map_gray[cy - half:cy + half, cx - half:cx + half]
                mean_val = float(np.mean(patch))
                if float(np.std(patch)) < self._std_thresh and 10 < mean_val < 200:
                    candidates.append((cx, cy, mean_val))
        self._candidates = candidates

    # ------------------------------------------------------------------
    def update_scale(self, scale: float) -> None:
        """SIFT/ECC 成功后同步最新 scale。"""
        self._last_scale = scale

    # ------------------------------------------------------------------
    def locate(
        self,
        minimap_gray_raw: np.ndarray,
        last_x: int | None,
        last_y: int | None,
        lost_frames: int,
    ) -> tuple[int, int] | None:
        """
        多尺度模板匹配，返回 (map_x, map_y) 或 None。

        Args:
            minimap_gray_raw: 未增强的原始灰度小地图。
            last_x/y        : 上次有效大地图坐标（None 表示完全丢失）。
            lost_frames     : 连续丢失帧数（用于动态扩大搜索半径）。
        """
        if not self._candidates:
            return None

        h_mm, w_mm = minimap_gray_raw.shape[:2]
        mini_mean = float(np.mean(minimap_gray_raw))
        color_thresh = getattr(config, 'OCEAN_COLOR_THRESH', 50)
        margin = self._region_tile // 2
        min_cc = getattr(config, 'OCEAN_COLD_START_MIN_CC', 0.20)

        _semi_lost_radius = min(2400, 1200 + max(0, lost_frames - 8) * 60)
        if last_x is not None:
            close_candidates = [
                (cx, cy, mm) for cx, cy, mm in self._candidates
                if (abs(mm - mini_mean) < color_thresh
                    and abs(cx - last_x) < _semi_lost_radius
                    and abs(cy - last_y) < _semi_lost_radius)
            ]
        else:
            close_candidates = [(cx, cy, mm) for cx, cy, mm in self._candidates
                                if abs(mm - mini_mean) < color_thresh]
        if not close_candidates:
            return None

        # 圆形掩码填角
        circ_mask = self._mask_cache.get(h_mm, w_mm)
        mini_filled = minimap_gray_raw.copy()
        mini_filled[circ_mask == 0] = int(mini_mean)

        base_s = self._last_scale
        scales = sorted({round(base_s * f, 2) for f in (0.75, 0.875, 1.0, 1.125, 1.25)}
                        | {3.5, 4.0, 4.5, 5.0})

        best_cc = -1.0
        best_pos: tuple[int, int] | None = None
        best_scale = base_s

        for s in scales:
            dst_w = max(1, int(w_mm * s))
            dst_h = max(1, int(h_mm * s))
            mini_scaled = cv2.resize(mini_filled, (dst_w, dst_h))

            for cx, cy, _ in close_candidates:
                x1 = max(0, cx - dst_w // 2 - margin)
                y1 = max(0, cy - dst_h // 2 - margin)
                x2 = min(self._map_width, x1 + dst_w + 2 * margin)
                y2 = min(self._map_height, y1 + dst_h + 2 * margin)
                region = self._logic_map_gray[y1:y2, x1:x2]
                if region.shape[0] <= dst_h or region.shape[1] <= dst_w:
                    continue
                try:
                    res = cv2.matchTemplate(region, mini_scaled, cv2.TM_CCOEFF_NORMED)
                except cv2.error:
                    continue
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_cc:
                    best_cc = max_val
                    match_x = x1 + max_loc[0] + dst_w // 2
                    match_y = y1 + max_loc[1] + dst_h // 2
                    best_pos = (match_x, match_y)
                    best_scale = s

        if best_pos is not None and best_cc >= min_cc:
            self._last_scale = best_scale
            return best_pos
        return None
