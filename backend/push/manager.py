"""
push/manager.py — JPEG 客户端注册表与推送决策

职责:
  - 追踪哪些 WebSocket 客户端处于 JPEG 模式（plan B: 控制后台 JPEG 编码开关）
    - 记录 Plan A push 目标 SID（feature-worker 完成后定向推送）
  - 每客户端维护独立的 PushSession（节流锚点、force_next_jpeg）
  - 根据位移判断发送 result+JPEG 还是仅发 coords

全图模式客户端（fullmap）只接收 coords 广播，不注册进本管理器。
"""

from __future__ import annotations

from threading import RLock

from backend.push.session import PushSession


class JpegPushManager:
    """管理 JPEG 模式 WebSocket 推送的全部状态。"""

    def __init__(self) -> None:
        self._lock = RLock()
        # sid → PushSession，仅包含" plan B 消费者"（frame 事件的客户端）
        self._sessions: dict[str, PushSession] = {}
        # token → 最新 push 目标 SID（每个识别会话的活跃 frame 发送者）
        self._push_target: dict[str, str] = {}
        # sid -> token（用于按会话统计 JPEG 客户端）
        self._sid_token: dict[str, str] = {}
        # token -> sids
        self._token_clients: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ #
    # 客户端生命周期
    # ------------------------------------------------------------------ #

    def register_frame_client(self, sid: str, token: str = '') -> None:
        """
        客户端发送 'frame' 事件时调用。
        首次注册时创建 PushSession；同时更新对应 token 的 push 目标。
        """
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

            if token:
                self._push_target[token] = sid

    def unregister_client(self, sid: str) -> None:
        """客户端断开时清理，保证无资源泄漏。"""
        with self._lock:
            self._sessions.pop(sid, None)
            token = self._sid_token.pop(sid, '')
            if token and token in self._token_clients:
                clients = self._token_clients[token]
                clients.discard(sid)
                if not clients:
                    del self._token_clients[token]
            # 清理 push_target 中指向该 SID 的条目
            dead_tokens = [t for t, s in self._push_target.items() if s == sid]
            for t in dead_tokens:
                del self._push_target[t]

    # ------------------------------------------------------------------ #
    # 状态查询
    # ------------------------------------------------------------------ #

    def has_jpeg_clients(self, token: str = '') -> bool:
        """是否还有需要 JPEG 的客户端（可按 token 查询）。"""
        with self._lock:
            if token:
                return bool(self._token_clients.get(token))
            return bool(self._sessions)

    # ------------------------------------------------------------------ #
    # 节流控制
    # ------------------------------------------------------------------ #

    def force_next_jpeg(self, token: str = '') -> None:
        """
        强制 push 目标在下次 _on_result_ready 时必须发送 JPEG。
        对应前端 'request_jpeg' 事件（如切换到 JPEG 渲染模式）。
        """
        with self._lock:
            sid = self._push_target.get(token, '')
            session = self._sessions.get(sid) if sid else None
        if session is not None:
            session.force_next_jpeg()

    # ------------------------------------------------------------------ #
    # 推送决策（Plan A）
    # ------------------------------------------------------------------ #

    def push_result(
        self,
        socketio,
        jpeg: bytes | None,
        cx: float,
        cy: float,
        header: bytes,
        status_json: bytes,
        token: str = '',
    ) -> None:
        """
        向 Plan A push 目标推送一帧结果。

        - 位移超出 pan 余量且有 JPEG → 推 result（header + status_json + jpeg）
        - 否则 → 仅推 coords（header + status_json），节省带宽

        全图模式客户端不在本管理器内，不受影响。
        """
        with self._lock:
            sid = self._push_target.get(token, '')
            session = self._sessions.get(sid) if sid else None
        if not sid or session is None:
            return

        if jpeg and session.needs_jpeg(int(cx), int(cy)):
            session.mark_jpeg_sent(cx, cy)
            try:
                socketio.emit('result', header + status_json + jpeg,
                              to=sid, namespace='/')
            except Exception as exc:
                print(f"[JpegPushManager] result → {sid} 失败: {exc}")
        else:
            try:
                socketio.emit('coords', header + status_json,
                              to=sid, namespace='/')
            except Exception as exc:
                print(f"[JpegPushManager] coords → {sid} 失败: {exc}")
