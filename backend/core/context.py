"""
core/context.py - 统一上下文与会话注册器

定义三层数据域中的 SessionScoped 层：
  - SessionContext: 绑定 session_token 的会话状态容器
  - SessionRegistry: 线程安全地管理 token -> SessionContext
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Generic, TypeVar


T = TypeVar('T')


@dataclass
class SessionContext(Generic[T]):
    """单个会话上下文（SessionScoped）。"""

    token: str
    tracker: T
    created_ts: float = field(default_factory=time.time)
    last_active_ts: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active_ts = time.time()


class SessionRegistry(Generic[T]):
    """线程安全会话注册器：token -> SessionContext。"""

    def __init__(self, tracker_factory: Callable[[str], T]) -> None:
        self._tracker_factory = tracker_factory
        self._contexts: dict[str, SessionContext[T]] = {}
        self._lock = Lock()

    def get_or_create(self, token: str) -> tuple[SessionContext[T], bool]:
        """返回 (context, created)。"""
        with self._lock:
            ctx = self._contexts.get(token)
            if ctx is not None:
                ctx.touch()
                return ctx, False

            tracker = self._tracker_factory(token)
            ctx = SessionContext(token=token, tracker=tracker)
            self._contexts[token] = ctx
            return ctx, True

    def get_context(self, token: str) -> SessionContext[T]:
        ctx, _ = self.get_or_create(token)
        return ctx

    def get_tracker(self, token: str) -> T:
        return self.get_context(token).tracker

    def snapshot_contexts(self) -> list[SessionContext[T]]:
        """获取上下文快照（用于遍历，避免长时间持锁）。"""
        with self._lock:
            return list(self._contexts.values())

    def has_token(self, token: str) -> bool:
        with self._lock:
            return token in self._contexts
