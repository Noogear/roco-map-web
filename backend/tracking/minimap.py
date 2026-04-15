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

    def restore(self, cx: int, cy: int, r: int) -> None:
        """
        从持久化状态恢复校准，跳过 N 帧校准期。
        直接进入 calibrated 状态并填充历史以稳定后续校验。
        """
        n_cal = getattr(config, 'MINIMAP_CIRCLE_CALIBRATION_FRAMES', 8)
        self._history.clear()
        for _ in range(n_cal):
            self._history.append((cx, cy, r))
        self.expected_cx = cx
        self.expected_cy = cy
        self.expected_r = r
        self._calibrated = True
        self._consecutive_miss = 0

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

    def reseed(self, cx: int, cy: int, r: int) -> None:
        """
        冻结态恢复时用新检测到的圆重新播种。

        注意：这里故意不直接进入 calibrated 状态，避免加载场景/切分辨率后
        把单帧噪声固化成严格阈值；后续正常帧会重新积累历史并收敛。
        """
        self._history.clear()
        self._history.append((cx, cy, r))
        self._calibrated = False
        self.expected_cx = None
        self.expected_cy = None
        self.expected_r = None
        self._consecutive_miss = 0

    def to_dict(self) -> dict | None:
        """序列化校准状态，未校准时返回 None。"""
        if not self._calibrated:
            return None
        return {'cx': self.expected_cx, 'cy': self.expected_cy, 'r': self.expected_r}

    @classmethod
    def from_dict(cls, data: dict | None) -> 'CircleCalibrator':
        """从持久化数据恢复。"""
        cal = cls()
        if data and all(k in data for k in ('cx', 'cy', 'r')):
            cal.restore(data['cx'], data['cy'], data['r'])
        return cal


def _score_local_circle(gray: np.ndarray, edges: np.ndarray, cx: int, cy: int, r: int) -> float:
    """对局部方形截取中的候选圆打分（中心贴近度 + 边缘环强度 + 半径合理性）。"""
    h, w = gray.shape[:2]
    base = float(min(h, w))
    if base <= 0:
        return 0.0

    # 1) 越靠近方形中心越好
    center_dist = math.sqrt((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2)
    center_score = max(0.0, 1.0 - center_dist / max(base * 0.30, 1.0))

    # 2) 半径应与截取 margin 匹配：r / base ≈ 1 / (2 * capture_margin)
    capture_margin = max(1.0, float(getattr(config, 'MINIMAP_CAPTURE_MARGIN', 1.4)))
    expected_ratio = 1.0 / (2.0 * capture_margin)
    r_ratio = r / base
    size_score = max(0.0, 1.0 - abs(r_ratio - expected_ratio) / max(expected_ratio * 0.65, 1e-6))

    # 3) 圆环边缘强度：候选半径附近应有明显边缘响应
    pad = max(2, int(r * 1.25))
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(w, cx + pad)
    y2 = min(h, cy + pad)
    roi = edges[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    band = max(2.0, r * 0.12)
    ring_mask = (dist >= (r - band)) & (dist <= (r + band))
    if not np.any(ring_mask):
        edge_score = 0.0
    else:
        edge_score = float(np.mean(roi[ring_mask]) / 255.0)

    return 0.45 * center_score + 0.35 * edge_score + 0.20 * size_score


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

    base = min(h, w)
    min_ratio = float(getattr(config, 'MINIMAP_LOCAL_MIN_RADIUS_RATIO', 0.22))
    max_ratio = float(getattr(config, 'MINIMAP_LOCAL_MAX_RADIUS_RATIO', 0.48))
    min_r = max(10, int(base * min_ratio))
    max_r = min(int(base * 0.49), max(min_r + 6, int(base * max_ratio)))
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
        if calibrator._calibrated and calibrator.expected_cx is not None:
            det_cx, det_cy, det_r = calibrator.expected_cx, calibrator.expected_cy, calibrator.expected_r
            x1, y1 = max(0, det_cx - det_r), max(0, det_cy - det_r)
            x2, y2 = min(w, det_cx + det_r), min(h, det_cy + det_r)
            minimap = square_bgr[y1:y2, x1:x2].copy()
            if minimap.size >= 100:
                return minimap
        if not engine_frozen:
            calibrator.record_miss()
        return None

    circles = np.round(circles[0, :]).astype(int)
    edges = cv2.Canny(blurred, 60, 160)

    # 综合评分选最优圆：平时更稳，冻结态也能在跳过旧校准时尽量避免误判
    best, best_score = None, -1.0
    for c in circles:
        score = _score_local_circle(gray, edges, int(c[0]), int(c[1]), int(c[2]))
        if score > best_score:
            best_score = score
            best = c

    if best is None or best_score < float(getattr(config, 'MINIMAP_LOCAL_MIN_SCORE', 0.22)):
        if calibrator._calibrated and calibrator.expected_cx is not None:
            det_cx, det_cy, det_r = calibrator.expected_cx, calibrator.expected_cy, calibrator.expected_r
            x1, y1 = max(0, det_cx - det_r), max(0, det_cy - det_r)
            x2, y2 = min(w, det_cx + det_r), min(h, det_cy + det_r)
            minimap = square_bgr[y1:y2, x1:x2].copy()
            if minimap.size >= 100:
                return minimap
        if not engine_frozen:
            calibrator.record_miss()
        return None

    det_cx, det_cy, det_r = int(best[0]), int(best[1]), int(best[2])

    if not calibrator.is_valid(det_cx, det_cy, det_r):
        if engine_frozen:
            # 冻结态下旧校准可能已经陈旧：接受当前检测结果并重新播种，
            # 让 thaw 后的后续帧重新积累历史，而不是继续抱着旧圆心不放。
            calibrator.reseed(det_cx, det_cy, det_r)
        elif calibrator._calibrated and calibrator.expected_cx is not None:
            det_cx, det_cy, det_r = calibrator.expected_cx, calibrator.expected_cy, calibrator.expected_r
        else:
            calibrator.record_miss()
            return None
    else:
        calibrator.update(det_cx, det_cy, det_r)

    x1, y1 = max(0, det_cx - det_r), max(0, det_cy - det_r)
    x2, y2 = min(w, det_cx + det_r), min(h, det_cy + det_r)
    minimap = square_bgr[y1:y2, x1:x2].copy()

    if minimap.size < 100:
        return None

    return minimap
