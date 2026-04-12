"""
push/session.py - 每个 WebSocket 客户端的推送状态隔离

PushSession 持有每个 SID 独立的 JPEG 节流锚点，
避免多客户端共享 _last_jpeg_x/y 导致相互干扰。
"""

from __future__ import annotations

import math

from backend import config


class PushSession:
    """单个 WebSocket 客户端的推送状态。"""

    def __init__(self, sid: str) -> None:
        self.sid = sid
        self._last_jpeg_x: float = -99999.0
        self._last_jpeg_y: float = -99999.0

    # ------------------------------------------------------------------
    def needs_jpeg(self, cx: int, cy: int, margin: int = 8) -> bool:
        """
        是否需要发送新 JPEG（当前坐标超出 pan 余量时触发）。

        Args:
            cx, cy : 当前地图坐标（像素）
            margin : JPEG_PAD 中保留的缓冲像素数
        """
        pad = getattr(config, 'JPEG_PAD', 40)
        budget = pad - margin
        dist = math.sqrt(
            (cx - self._last_jpeg_x) ** 2 + (cy - self._last_jpeg_y) ** 2
        )
        return dist >= budget

    def mark_jpeg_sent(self, cx: int, cy: int) -> None:
        """记录本次 JPEG 发送时的坐标锚点。"""
        self._last_jpeg_x = float(cx)
        self._last_jpeg_y = float(cy)

    def force_next_jpeg(self) -> None:
        """强制下次必须发送 JPEG（request_jpeg 事件时调用）。"""
        self._last_jpeg_x = -99999.0
        self._last_jpeg_y = -99999.0
