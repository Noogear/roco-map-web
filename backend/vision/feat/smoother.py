from __future__ import annotations

from collections import deque


class CoordSmoother:
	def __init__(self) -> None:
		self.pos_history: deque[tuple[int, int]] = deque(maxlen=20)
		self._last_x: int | None = None
		self._last_y: int | None = None

	def update(self, cx: int | None, cy: int | None, found: bool, *_args, **_kwargs) -> tuple[int, int, bool, bool]:
		if not found or cx is None or cy is None:
			return 0, 0, False, False
		self._last_x = int(cx)
		self._last_y = int(cy)
		self.pos_history.append((self._last_x, self._last_y))
		return self._last_x, self._last_y, False, False

	def reset_to(self, x: int, y: int) -> None:
		self._last_x = int(x)
		self._last_y = int(y)
		self.pos_history.clear()
		self.pos_history.append((self._last_x, self._last_y))

	def clear_state(self) -> None:
		self.pos_history.clear()
		self._last_x = None
		self._last_y = None

	def clear_position_history(self) -> None:
		self.pos_history.clear()

	def clear_runtime_state(self, clear_persisted: bool = False) -> None:
		_ = clear_persisted
		self.clear_state()

	def predict_position(self) -> tuple[int, int] | None:
		if self._last_x is None or self._last_y is None:
			return None
		return self._last_x, self._last_y
