from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Generic, TypeVar

T = TypeVar('T')


@dataclass
class SessionContext(Generic[T]):
    token: str
    tracker: T
    created_ts: float = field(default_factory=time.time)
    last_active_ts: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active_ts = time.time()


class SessionRegistry(Generic[T]):
    def __init__(self, tracker_factory: Callable[[str], T]) -> None:
        self._tracker_factory = tracker_factory
        self._contexts: dict[str, SessionContext[T]] = {}
        self._lock = Lock()

    def get_or_create(self, token: str) -> tuple[SessionContext[T], bool]:
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
        with self._lock:
            return list(self._contexts.values())

    def has_token(self, token: str) -> bool:
        with self._lock:
            return token in self._contexts
