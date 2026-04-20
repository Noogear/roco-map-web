"""
store.py - 持久化辅助（圆形状态 + 路线文件）
"""

from __future__ import annotations

import json
import os
import threading
import time

_ROUTE_CACHE_LOCK = threading.Lock()
_ROUTE_CACHE: dict[str, tuple[int, float, dict | None]] = {}
_JSON_CACHE_FALLBACK_TTL_SEC = 1.2


def _route_cache_key(routes_dir: str, filename: str) -> str:
    return os.path.abspath(os.path.join(routes_dir, filename))


def _copy_dict_payload(payload: dict | None) -> dict | None:
    return dict(payload) if isinstance(payload, dict) else None


def _load_json_dict_cached(filepath: str, cache_key: str, cache: dict[str, tuple[int, float, dict | None]], lock: threading.Lock) -> dict | None:
    now = time.time()
    try:
        stat = os.stat(filepath)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        with lock:
            cached = cache.get(cache_key)
            if cached is not None and (now - float(cached[1])) <= _JSON_CACHE_FALLBACK_TTL_SEC:
                return _copy_dict_payload(cached[2])
            cache.pop(cache_key, None)
        return None
    with lock:
        cached = cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return _copy_dict_payload(cached[2])
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        payload = data if isinstance(data, dict) else None
        with lock:
            cache[cache_key] = (mtime_ns, now, payload)
        return _copy_dict_payload(payload)
    except Exception:
        with lock:
            cached = cache.get(cache_key)
            if cached is not None and cached[0] == mtime_ns:
                return _copy_dict_payload(cached[2])
        return None


def get_route_files(routes_dir: str) -> list[str]:
    if not os.path.isdir(routes_dir):
        return []
    return sorted(f for f in os.listdir(routes_dir) if f.lower().endswith('.json'))


def load_route_data(routes_dir: str, filename: str) -> dict | None:
    filepath = os.path.join(routes_dir, filename)
    cache_key = _route_cache_key(routes_dir, filename)
    return _load_json_dict_cached(filepath, cache_key, _ROUTE_CACHE, _ROUTE_CACHE_LOCK)
