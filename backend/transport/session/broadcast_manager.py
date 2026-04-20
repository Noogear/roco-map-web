"""
broadcast_manager.py - 广播频道状态管理
"""
from __future__ import annotations

from threading import RLock


class BroadcastManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._presenter: dict[str, str] = {}
        self._viewers: dict[str, set[str]] = {}
        self._sid_name: dict[str, str] = {}

    def viewer_join(self, sid: str, name: str) -> tuple[int, str | None]:
        with self._lock:
            self._viewers.setdefault(name, set()).add(sid)
            self._sid_name[sid] = name
            count = len(self._viewers[name])
            presenter_sid = self._presenter.get(name)
        return count, presenter_sid

    def viewer_leave(self, sid: str, name: str) -> None:
        with self._lock:
            if name in self._viewers:
                self._viewers[name].discard(sid)
            self._sid_name.pop(sid, None)

    def presenter_start(self, sid: str, name: str) -> None:
        with self._lock:
            self._presenter[name] = sid
            self._viewers.setdefault(name, set())

    def presenter_stop(self, sid: str) -> list[str]:
        with self._lock:
            dead = [n for n, s in self._presenter.items() if s == sid]
            for n in dead:
                del self._presenter[n]
        return dead

    def get_presenter_name(self, sid: str) -> str | None:
        with self._lock:
            return next((n for n, s in self._presenter.items() if s == sid), None)

    def on_disconnect(self, sid: str) -> None:
        with self._lock:
            name = self._sid_name.pop(sid, None)
            if name and name in self._viewers:
                self._viewers[name].discard(sid)
            dead = [n for n, s in self._presenter.items() if s == sid]
            for n in dead:
                del self._presenter[n]
                self._viewers.pop(n, None)
