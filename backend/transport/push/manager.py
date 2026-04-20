from __future__ import annotations

from threading import RLock

from backend.transport.push.session import PushSession


class JpegPushManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, PushSession] = {}
        self._push_target: dict[str, str] = {}
        self._sid_token: dict[str, str] = {}
        self._token_clients: dict[str, set[str]] = {}

    def register_frame_client(self, sid: str, token: str = '') -> None:
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = PushSession(sid)
            old_token = self._sid_token.get(sid, '')
            if old_token and old_token in self._token_clients:
                clients = self._token_clients[old_token]
                clients.discard(sid)
                if not clients:
                    del self._token_clients[old_token]
            self._sid_token[sid] = token
            if token:
                self._token_clients.setdefault(token, set()).add(sid)
                self._push_target[token] = sid

    def unregister_client(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)
            token = self._sid_token.pop(sid, '')
            if token and token in self._token_clients:
                clients = self._token_clients[token]
                clients.discard(sid)
                if not clients:
                    del self._token_clients[token]
            dead_tokens = [t for t, s in self._push_target.items() if s == sid]
            for t in dead_tokens:
                del self._push_target[t]

    def has_jpeg_clients(self, token: str = '') -> bool:
        with self._lock:
            if token:
                return bool(self._token_clients.get(token))
            return bool(self._sessions)

    def force_next_jpeg(self, token: str = '') -> None:
        with self._lock:
            sid = self._push_target.get(token, '')
            session = self._sessions.get(sid) if sid else None
        if session is not None:
            session.force_next_jpeg()

    def push_result(self, socketio, jpeg: bytes | None, cx: float, cy: float, header: bytes, status_json: bytes, token: str = '') -> None:
        with self._lock:
            sid = self._push_target.get(token, '')
            session = self._sessions.get(sid) if sid else None
        if not sid or session is None:
            return
        if jpeg and session.needs_jpeg(int(cx), int(cy)):
            session.mark_jpeg_sent(cx, cy)
            try:
                socketio.emit('result', header + status_json + jpeg, to=sid, namespace='/')
            except Exception as exc:
                print(f"[JpegPushManager] result → {sid} 失败: {exc}")
        else:
            try:
                socketio.emit('coords', header + status_json, to=sid, namespace='/')
            except Exception as exc:
                print(f"[JpegPushManager] coords → {sid} 失败: {exc}")
