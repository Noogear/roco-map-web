"""
tracking/minimap.py - 小地图圆形检测与校准

_CircleCalibrator : 运行时自动校准圆心/半径（从 tracker_core 迁移）
detect_and_extract : 纯函数，单帧小地图提取（从 MapTrackerWeb._detect_and_extract_minimap 提取）
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from backend import config


class CircleCalibrator:
    """
    运行时自动校准小地图圆形参数（半径、圆心位置）。

    前 N 帧为校准期，所有检测到的圆都放行；校准收敛后通过半径和
    圆心偏移严格过滤异常检测（地图展开、战斗误检）。
    连续长时间未检测到圆 → 自动重置校准（适应分辨率/UI 变化）。
    """

    def __init__(self) -> None:
        self._history: deque[tuple[int, int, int]] = deque(maxlen=30)
        self._calibrated: bool = False
        self.expected_cx: int | None = None
        self.expected_cy: int | None = None
        self.expected_r: int | None = None
        self._consecutive_miss: int = 0

    def update(self, cx: int, cy: int, r: int) -> None:
        self._history.append((cx, cy, r))
        self._consecutive_miss = 0
        n_cal = getattr(config, 'MINIMAP_CIRCLE_CALIBRATION_FRAMES', 8)
        if len(self._history) >= n_cal:
            rs = [d[2] for d in self._history]
            cxs = [d[0] for d in self._history]
            cys = [d[1] for d in self._history]
            self.expected_r = int(np.median(rs))
            self.expected_cx = int(np.median(cxs))
            self.expected_cy = int(np.median(cys))
            if not self._calibrated:
                self._calibrated = True
                print(f"  [圆校准] 已收敛: center=({self.expected_cx},{self.expected_cy}), r={self.expected_r}")

    def is_valid(self, cx: int, cy: int, r: int) -> bool:
        if not self._calibrated:
            return True
        r_tol = getattr(config, 'MINIMAP_CIRCLE_R_TOLERANCE', 8)
        c_tol = getattr(config, 'MINIMAP_CIRCLE_CENTER_TOLERANCE', 15)
        return (abs(r - self.expected_r) <= r_tol
                and abs(cx - self.expected_cx) <= c_tol
                and abs(cy - self.expected_cy) <= c_tol)

    def record_miss(self) -> None:
        self._consecutive_miss += 1
        n_reset = getattr(config, 'MINIMAP_CIRCLE_RECALIBRATE_MISS', 30)
        if self._consecutive_miss >= n_reset and self._calibrated:
            self._history.clear()
            self._calibrated = False
            self._consecutive_miss = 0
            print("[圆校准] 连续未检测到小地图圆，重置校准（可能分辨率/UI 变化）")


def detect_and_extract(
    square_bgr: np.ndarray,
    calibrator: CircleCalibrator,
    engine_frozen: bool,
) -> np.ndarray | None:
    """
    从方形截取区域中检测圆形小地图并提取内容。

    Args:
        square_bgr  : 原始方形截取帧（BGR）
        calibrator  : 圆校准实例（有状态，外部持有）
        engine_frozen: 当前 SIFT 引擎是否处于冻结状态

    Returns:
        小地图 BGR 图像，或 None（未检测到有效小地图）
    """
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    min_r = getattr(config, 'MINIMAP_MIN_RADIUS', 60)
    max_r = getattr(config, 'MINIMAP_MAX_RADIUS', 160)
    param1 = getattr(config, 'HOUGH_PARAM1', 50)
    param2 = getattr(config, 'HOUGH_PARAM2', 30)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=w // 2,
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        if not engine_frozen:
            calibrator.record_miss()
        return None

    circles = np.round(circles[0, :]).astype(int)

    # 取最靠近图像中心的圆
    cx_img, cy_img = w / 2, h / 2
    best, best_dist = None, float('inf')
    for c in circles:
        d = math.sqrt((c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2)
        if d < best_dist:
            best_dist = d
            best = c

    det_cx, det_cy, det_r = int(best[0]), int(best[1]), int(best[2])

    if not calibrator.is_valid(det_cx, det_cy, det_r):
        if not engine_frozen:
            calibrator.record_miss()
        return None

    calibrator.update(det_cx, det_cy, det_r)

    x1, y1 = max(0, det_cx - det_r), max(0, det_cy - det_r)
    x2, y2 = min(w, det_cx + det_r), min(h, det_cy + det_r)
    minimap = square_bgr[y1:y2, x1:x2].copy()

    if minimap.size < 100:
        return None
    if np.var(cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)) < 20:
        return None

    return minimap
