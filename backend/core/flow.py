"""
core/flow.py - LK 稀疏光流追踪器

封装 LK 所有可变状态，提供单帧更新接口。
无 Web 框架依赖。
"""

from __future__ import annotations

import numpy as np
import cv2

from backend.core.features import CircularMaskCache


_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class LKTracker:
    """
    LK 稀疏光流追踪器。

    状态：
      prev_gray   : 上一帧灰度图
      prev_pts    : 上一帧跟踪点 (N,1,2)
      map_scale   : 当前小地图→大地图的像素比例
      frame_num   : 已处理帧计数（供外部决定是否运行 SIFT）
    """

    def __init__(
        self,
        enabled: bool = True,
        sift_every: int = 4,
        min_conf: float = 0.5,
        mask_cache: CircularMaskCache | None = None,
    ) -> None:
        self.enabled = enabled
        self.sift_every = sift_every
        self.min_conf = min_conf
        self._mask_cache = mask_cache or CircularMaskCache()

        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None
        self.map_scale: float = 4.0
        self.frame_num: int = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None

    # ------------------------------------------------------------------
    def should_run_sift(self) -> bool:
        return self.frame_num % self.sift_every == 0

    # ------------------------------------------------------------------
    @staticmethod
    def _as_xy_points(pts: np.ndarray | None) -> np.ndarray:
        """将点集统一为 (N,2)；无法解析时返回空数组。"""
        if pts is None:
            return np.empty((0, 2), dtype=np.float32)
        arr = np.asarray(pts)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # OpenCV 常见输出形状：(N,1,2) 或 (N,2)
        if arr.ndim == 3 and arr.shape[-2:] == (1, 2):
            arr = arr.reshape(-1, 2)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        elif arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            return np.empty((0, 2), dtype=np.float32)

        return arr.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    def track(
        self,
        minimap_gray: np.ndarray,
        last_x: int | None,
    ) -> tuple[float, float, float] | None:
        """
        估计帧间位移，返回 (dx_map, dy_map, confidence) 或 None。

        Args:
            minimap_gray : 当前帧灰度图（已增强）
            last_x       : 上一有效大地图 X（None => 跳过）
        """
        if (not self.enabled
                or self.prev_gray is None
                or self.prev_pts is None
                or len(self.prev_pts) < 4
                or last_x is None):
            return None
        try:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, minimap_gray,
                self.prev_pts, None, **_LK_PARAMS)
            if curr_pts is None or status is None:
                return None
            ok = status.ravel() == 1
            good_curr = self._as_xy_points(curr_pts[ok])
            good_prev = self._as_xy_points(self.prev_pts[ok])
            if len(good_curr) < 4:
                return None
            if good_curr.shape != good_prev.shape or good_curr.shape[1] != 2:
                return None
            disp = good_curr - good_prev
            dx_mm = float(np.median(disp[:, 0]))
            dy_mm = float(np.median(disp[:, 1]))
            confidence = len(good_curr) / max(len(self.prev_pts), 1)
            self.prev_pts = good_curr.reshape(-1, 1, 2)
            s = self.map_scale
            return dx_mm * s, dy_mm * s, confidence
        except cv2.error:
            return None

    # ------------------------------------------------------------------
    def refresh_tracking_points(self, minimap_gray: np.ndarray) -> None:
        """绝对定位成功后，重新提取当前帧跟踪点供下一帧 LK 使用。"""
        h, w = minimap_gray.shape[:2]
        mask = self._mask_cache.get(h, w)
        pts = cv2.goodFeaturesToTrack(
            minimap_gray, maxCorners=60, qualityLevel=0.01,
            minDistance=7, mask=mask)
        self.prev_pts = pts if pts is not None and len(pts) >= 4 else None
