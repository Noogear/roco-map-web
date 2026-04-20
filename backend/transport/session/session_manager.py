"""
session_manager.py - WebSocket 会话注册表

职责：
  - sid ↔ token 双向映射
  - token → MapTrackerWeb 懒创建与销毁
  - 整合 JpegPushManager 注册/注销
"""
from __future__ import annotations

from threading import RLock

from backend.transport.push.manager import JpegPushManager
from backend.vision.engine.map_tracker_web import MapTrackerWeb
from backend.vision.engine.shared_feature_factory import get_shared_feature


class SessionManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, MapTrackerWeb] = {}
        self._sid_token: dict[str, str] = {}
        self.push_mgr = JpegPushManager()

    def on_connect(self, sid: str) -> None:
        self.push_mgr.register_frame_client(sid, token='')

    def on_disconnect(self, sid: str) -> None:
        with self._lock:
            token = self._sid_token.pop(sid, None)
            if token:
                self._sessions.pop(token, None)
                # 清理 frame_processor 里的 per-token minimap 状态，防内存泄漏
                try:
                    from backend.transport.session.frame_processor import _minimap_states, _minimap_lock
                    with _minimap_lock:
                        _minimap_states.pop(token, None)
                except Exception:
                    pass
        self.push_mgr.unregister_client(sid)

    def bind(self, sid: str, token: str) -> None:
        with self._lock:
            self._sid_token[sid] = token
        self.push_mgr.register_frame_client(sid, token=token)
        self.get_or_create(token)

    def get_token(self, sid: str, fallback: str | None = None) -> str:
        with self._lock:
            return self._sid_token.get(sid, fallback if fallback is not None else sid)

    def get_or_create(self, token: str) -> MapTrackerWeb:
        with self._lock:
            if token not in self._sessions:
                shared = get_shared_feature()
                self._sessions[token] = MapTrackerWeb(session_id=token, shared=shared)
            return self._sessions[token]

    def get(self, token: str) -> MapTrackerWeb | None:
        with self._lock:
            return self._sessions.get(token)

    def reset(self, token: str) -> None:
        tracker = self.get(token)
        if tracker is not None:
            tracker.feature_engine.reset()
