"""
tracking/direction.py - 方向系统

_angular_diff 工具函数 + ArrowDirectionSystem 类（原 _ArrowDirectionSystem）。
无 Web 依赖，纯坐标计算。
"""

from __future__ import annotations

import math
from collections import deque


def _angular_diff(a: float, b: float) -> float:
    """两角度之差，结果在 (-180, 180]。"""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


class ArrowDirectionSystem:
    """
    基于位置历史估计移动方向，返回 (angle_deg, is_stopped)。

    EMA 平滑角度、停止状态 debounce，防止微抖动导致箭头频繁翻转。
    """

    def __init__(
        self,
        history_size: int = 8,
        ema_alpha: float = 0.3,
        stop_speed_px: float = 2.0,
        stop_debounce: int = 6,
        snap_threshold: float = 30.0,
    ) -> None:
        self._history: deque[tuple[int, int]] = deque(maxlen=history_size)
        self._ema_alpha = ema_alpha
        self._stop_speed_px = stop_speed_px
        self._stop_debounce = stop_debounce
        self._snap_threshold = snap_threshold

        self._last_angle: float = 0.0
        self._is_stopped: bool = True
        self._stop_streak: int = 0

    # ------------------------------------------------------------------
    def update(
        self,
        map_x: int | None,
        map_y: int | None,
    ) -> tuple[float, bool]:
        """
        更新位置，返回 (angle_deg, is_stopped)。
        传入 None 时视为惯性帧（保持上次状态）。
        """
        if map_x is None or map_y is None:
            return self._last_angle, self._is_stopped

        self._history.append((map_x, map_y))

        if len(self._history) < 2:
            return self._last_angle, self._is_stopped

        # 用最近两点估计即时速度
        (px, py), (cx, cy) = self._history[-2], self._history[-1]
        dx, dy = cx - px, cy - py
        speed = math.sqrt(dx * dx + dy * dy)

        if speed < self._stop_speed_px:
            self._stop_streak += 1
        else:
            self._stop_streak = 0

        is_stopped = self._stop_streak >= self._stop_debounce
        if not is_stopped:
            raw_angle = math.degrees(math.atan2(dx, -dy)) % 360
            diff = abs(_angular_diff(raw_angle, self._last_angle))
            if diff > self._snap_threshold:
                self._last_angle = raw_angle
            else:
                d = _angular_diff(raw_angle, self._last_angle)
                self._last_angle = (self._last_angle + self._ema_alpha * d) % 360

        self._is_stopped = is_stopped
        return self._last_angle, is_stopped
