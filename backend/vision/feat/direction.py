from __future__ import annotations

import math
from collections import deque


def _angular_diff(a: float, b: float) -> float:
	d = (a - b) % 360
	return d - 360 if d > 180 else d


class ArrowDirectionSystem:
	def __init__(self, history_size: int = 4, ema_alpha: float = 0.35, stop_speed_px: float = 6.0, stop_debounce: int = 20, small_change_threshold: float = 12.0, big_change_threshold: float = 90.0) -> None:
		_ = small_change_threshold
		self._history: deque[tuple[int, int]] = deque(maxlen=history_size)
		self._ema_alpha = ema_alpha
		self._move_threshold = stop_speed_px
		self._stop_debounce = stop_debounce
		self._snap_threshold = big_change_threshold
		self._last_angle: float = 0.0
		self._ema_angle: float | None = None
		self._is_stopped: bool = True
		self._low_move_streak: int = 0

	def update(self, map_x: int | None, map_y: int | None) -> tuple[float, bool]:
		if map_x is None or map_y is None:
			return self._last_angle, self._is_stopped
		self._history.append((map_x, map_y))
		if len(self._history) < 2:
			return self._last_angle, self._is_stopped
		old_x, old_y = self._history[0]
		cum_dx = map_x - old_x
		cum_dy = map_y - old_y
		cum_dist = math.sqrt(cum_dx * cum_dx + cum_dy * cum_dy)
		if cum_dist < self._move_threshold:
			self._low_move_streak += 1
			if self._low_move_streak >= self._stop_debounce:
				self._is_stopped = True
			return self._last_angle, self._is_stopped
		self._low_move_streak = 0
		self._is_stopped = False
		raw_angle = math.degrees(math.atan2(cum_dx, -cum_dy)) % 360
		if self._ema_angle is None:
			self._ema_angle = raw_angle
		else:
			diff = _angular_diff(raw_angle, self._ema_angle)
			if abs(diff) >= self._snap_threshold:
				self._ema_angle = raw_angle
			else:
				self._ema_angle = (self._ema_angle + self._ema_alpha * diff) % 360
		self._last_angle = self._ema_angle
		return self._last_angle, self._is_stopped
