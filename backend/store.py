"""
store.py - 持久化辅助（圆形状态 + 路线文件）

从 server.py 中提取，集中管理所有文件 I/O，
server.py 只需从此模块调用函数，不直接操作文件。
"""

from __future__ import annotations

import json
import os
import time
import threading


# ---------------------------------------------------------------------------
# 路线文件
# ---------------------------------------------------------------------------

_ROUTE_CACHE_LOCK = threading.Lock()
_ROUTE_CACHE: dict[str, tuple[int, float, dict | None]] = {}


def _route_cache_key(routes_dir: str, filename: str) -> str:
    return os.path.abspath(os.path.join(routes_dir, filename))

def get_route_files(routes_dir: str) -> list[str]:
    """扫描 routes 目录，返回 .json 文件名列表（已排序）。"""
    if not os.path.isdir(routes_dir):
        return []
    return sorted(f for f in os.listdir(routes_dir) if f.lower().endswith('.json'))


def load_route_data(routes_dir: str, filename: str) -> dict | None:
    """加载单个路线 JSON 文件。filename 已经过安全校验（basename only）。"""
    filepath = os.path.join(routes_dir, filename)
    if not os.path.isfile(filepath):
        with _ROUTE_CACHE_LOCK:
            _ROUTE_CACHE.pop(_route_cache_key(routes_dir, filename), None)
        return None

    cache_key = _route_cache_key(routes_dir, filename)
    try:
        stat = os.stat(filepath)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        return None

    with _ROUTE_CACHE_LOCK:
        cached = _ROUTE_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            # 返回浅拷贝，避免调用方误改缓存对象
            data = cached[2]
            return dict(data) if isinstance(data, dict) else None

    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        payload = data if isinstance(data, dict) else None
        with _ROUTE_CACHE_LOCK:
            _ROUTE_CACHE[cache_key] = (mtime_ns, time.time(), payload)
        return dict(payload) if isinstance(payload, dict) else None
    except Exception:
        return None


