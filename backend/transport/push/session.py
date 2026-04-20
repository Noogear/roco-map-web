from __future__ import annotations

from backend import config


class PushSession:
    def __init__(self, sid: str) -> None:
        self.sid = sid
        self._last_jpeg_x: float = -99999.0
        self._last_jpeg_y: float = -99999.0

    def needs_jpeg(self, cx: int, cy: int, margin: int = 8) -> bool:
        pad = int(getattr(config, 'JPEG_PAD', 40))
        budget = max(4, pad - margin)
        dx = cx - self._last_jpeg_x
        dy = cy - self._last_jpeg_y
        return (dx * dx + dy * dy) >= (budget * budget)

    def mark_jpeg_sent(self, cx: int, cy: int) -> None:
        self._last_jpeg_x = float(cx)
        self._last_jpeg_y = float(cy)

    def force_next_jpeg(self) -> None:
        self._last_jpeg_x = -99999.0
        self._last_jpeg_y = -99999.0
