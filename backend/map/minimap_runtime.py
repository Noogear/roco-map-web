from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from backend import config


@dataclass(frozen=True)
class MinimapCrop:
    image: np.ndarray
    center_xy: tuple[float, float]   # 中心在 minimap_img 内的坐标
    radius: int
    orig_cx: int = 0                  # 圆心在原始帧中的 x
    orig_cy: int = 0                  # 圆心在原始帧中的 y


class CircleCalibrator:
    def __init__(self) -> None:
        self._history: deque[tuple[int, int, int]] = deque(maxlen=30)
        self._calibrated: bool = False
        self.expected_cx: int | None = None
        self.expected_cy: int | None = None
        self.expected_r: int | None = None
        self._consecutive_miss: int = 0

    def restore(self, cx: int, cy: int, r: int) -> None:
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
        return abs(r - self.expected_r) <= r_tol and abs(cx - self.expected_cx) <= c_tol and abs(cy - self.expected_cy) <= c_tol

    def record_miss(self) -> None:
        self._consecutive_miss += 1
        n_reset = getattr(config, 'MINIMAP_CIRCLE_RECALIBRATE_MISS', 30)
        if self._consecutive_miss >= n_reset and self._calibrated:
            self._history.clear()
            self._calibrated = False
            self._consecutive_miss = 0
            print('[圆校准] 连续未检测到小地图圆，重置校准（可能分辨率/UI 变化）')

    def reseed(self, cx: int, cy: int, r: int) -> None:
        self._history.clear()
        self._history.append((cx, cy, r))
        self._calibrated = False
        self.expected_cx = None
        self.expected_cy = None
        self.expected_r = None
        self._consecutive_miss = 0

    def to_dict(self) -> dict | None:
        if not self._calibrated:
            return None
        return {'cx': self.expected_cx, 'cy': self.expected_cy, 'r': self.expected_r}

    @classmethod
    def from_dict(cls, data: dict | None) -> 'CircleCalibrator':
        cal = cls()
        if data and all(k in data for k in ('cx', 'cy', 'r')):
            cal.restore(data['cx'], data['cy'], data['r'])
        return cal


def _score_local_circle(gray: np.ndarray, edges: np.ndarray, cx: int, cy: int, r: int,
                        max_r_in_candidates: int = 0) -> float:
    """对候选圆打分。
    - 不再奖励"靠近图像中心"（小地图通常在角落）
    - 奖励边缘强度（真实圆形边界）
    - 奖励更大的半径（小地图是屏幕上最大的圆形 UI 元素）
    - 在合理的尺寸范围内不惩罚大圆
    """
    h, w = gray.shape[:2]
    base = float(min(h, w))
    if base <= 0:
        return 0.0

    # 尺寸评分：在 min_ratio~max_ratio 范围内越大越好
    capture_margin = max(1.0, float(getattr(config, 'MINIMAP_CAPTURE_MARGIN', 1.4)))
    min_ratio = float(getattr(config, 'MINIMAP_LOCAL_MIN_RADIUS_RATIO', 0.22))
    max_ratio = float(getattr(config, 'MINIMAP_LOCAL_MAX_RADIUS_RATIO', 0.48))
    r_ratio = r / base
    # 线性归一化：r_ratio 越接近 max_ratio 分数越高
    size_score = max(0.0, (r_ratio - min_ratio) / max(max_ratio - min_ratio, 1e-6))
    size_score = min(1.0, size_score)

    # 相对最大候选圆的大小比（保证多候选时倾向最大圆）
    if max_r_in_candidates > 0:
        rel_size_score = r / float(max_r_in_candidates)
    else:
        rel_size_score = size_score

    # 边缘响应：圆环上的 Canny 边缘密度
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

    # 权重：边缘强度 + 相对最大圆 + 绝对尺寸（不再用 center_score）
    return 0.40 * edge_score + 0.40 * rel_size_score + 0.20 * size_score


def _extract_circle_crop(square_bgr: np.ndarray, cx: int, cy: int, r: int) -> MinimapCrop | None:
    h, w = square_bgr.shape[:2]
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    minimap = square_bgr[y1:y2, x1:x2].copy()
    if minimap.size < 100:
        return None
    return MinimapCrop(image=minimap, center_xy=(float(cx - x1), float(cy - y1)),
                       radius=int(r), orig_cx=int(cx), orig_cy=int(cy))


def _extract_expected_circle_crop(square_bgr: np.ndarray, calibrator: CircleCalibrator) -> MinimapCrop | None:
    if not (calibrator._calibrated and calibrator.expected_cx is not None and calibrator.expected_r is not None):
        return None
    det_cx, det_cy, det_r = calibrator.expected_cx, calibrator.expected_cy, calibrator.expected_r
    return _extract_circle_crop(square_bgr, int(det_cx), int(det_cy), int(det_r))


def detect_and_extract_with_meta(square_bgr: np.ndarray, calibrator: CircleCalibrator, engine_frozen: bool) -> MinimapCrop | None:
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
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=w // 2, param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        fallback = _extract_expected_circle_crop(square_bgr, calibrator)
        if fallback is not None:
            return fallback
        if not engine_frozen:
            calibrator.record_miss()
        return None
    circles = np.round(circles[0, :]).astype(int)
    edges = cv2.Canny(blurred, 60, 160)
    best, best_score = None, -1.0
    max_r_in_candidates = int(np.max(circles[:, 2])) if len(circles) > 0 else 0
    # 过滤掉明显比最大候选圆小的圆（任务图标等），避免误选
    min_r_threshold = max_r_in_candidates * float(getattr(config, 'MINIMAP_CANDIDATE_MIN_R_RATIO', 0.50))
    for c in circles:
        if int(c[2]) < min_r_threshold:
            continue
        score = _score_local_circle(gray, edges, int(c[0]), int(c[1]), int(c[2]),
                                    max_r_in_candidates=max_r_in_candidates)
        if score > best_score:
            best_score = score
            best = c
    if best is None or best_score < float(getattr(config, 'MINIMAP_LOCAL_MIN_SCORE', 0.22)):
        fallback = _extract_expected_circle_crop(square_bgr, calibrator)
        if fallback is not None:
            return fallback
        if not engine_frozen:
            calibrator.record_miss()
        return None
    det_cx, det_cy, det_r = int(best[0]), int(best[1]), int(best[2])
    if not calibrator.is_valid(det_cx, det_cy, det_r):
        if engine_frozen:
            calibrator.reseed(det_cx, det_cy, det_r)
        elif calibrator._calibrated and calibrator.expected_cx is not None and calibrator.expected_r is not None:
            det_cx, det_cy, det_r = calibrator.expected_cx, calibrator.expected_cy, calibrator.expected_r
        else:
            calibrator.record_miss()
            return None
    else:
        calibrator.update(det_cx, det_cy, det_r)
        # 收敛后用中值平滑坐标，消除逐帧亚像素抖动
        if calibrator._calibrated and calibrator.expected_cx is not None:
            det_cx, det_cy, det_r = calibrator.expected_cx, calibrator.expected_cy, calibrator.expected_r
    return _extract_circle_crop(square_bgr, int(det_cx), int(det_cy), int(det_r))
