"""
fastjson.py - JSON 快速通道

目标：
  - 在不改接口协议的前提下，优先使用 orjson 提升序列化吞吐。
  - 兼容 Flask jsonify（通过 JSONProvider）与手动 bytes 序列化。
"""

from __future__ import annotations

import json as _json
from typing import Any

from flask.json.provider import DefaultJSONProvider

try:
    import orjson as _orjson
except Exception:  # pragma: no cover - fallback path
    _orjson = None


def dumps_bytes(obj: Any) -> bytes:
    """返回 UTF-8 JSON bytes（紧凑格式）。"""
    if _orjson is not None:
        return _orjson.dumps(obj)
    return _json.dumps(obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def dumps_text(obj: Any) -> str:
    """返回 JSON 文本（供 Flask JSONProvider 使用）。"""
    if _orjson is not None:
        return _orjson.dumps(obj).decode('utf-8')
    return _json.dumps(obj, ensure_ascii=False)


def loads_text(s: str | bytes | bytearray) -> Any:
    if _orjson is not None:
        return _orjson.loads(s)
    return _json.loads(s)


class OrjsonProvider(DefaultJSONProvider):
    """替换 Flask 默认 JSON provider，加速 jsonify/response JSON。"""

    def dumps(self, obj: Any, **kwargs) -> str:
        return dumps_text(obj)

    def loads(self, s: str | bytes, **kwargs) -> Any:
        return loads_text(s)
