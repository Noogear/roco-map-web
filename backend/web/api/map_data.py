"""地图页资源数据索引与分块加载辅助。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

# 导入根目录的 path_config
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from path_config import (
    MARKER_CATEGORIES_JSON_PATH,
    MARKER_DETAIL_JSON_PATH,
    MARKER_LITE_JSON_PATH,
)

_MARKERS_LITE_PATH = str(MARKER_LITE_JSON_PATH)
_MARKERS_DETAIL_PATH = str(MARKER_DETAIL_JSON_PATH)
_CATEGORIES_PATH = str(MARKER_CATEGORIES_JSON_PATH)
_CHUNK_SIZE = 768
_MAX_CHUNK_KEYS = 128
_MAX_DETAIL_IDS = 128
_VERSION_PROBE_INTERVAL_SEC = 0.8
_DATA_LOCK = threading.Lock()
_DATA_CACHE = None
_VERSION_CACHE = {'value': None, 'checked_at': 0.0}


def _load_json(path: str):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _compute_version() -> str:
    digest = hashlib.sha1()
    for path_obj in [MARKER_LITE_JSON_PATH, MARKER_DETAIL_JSON_PATH, MARKER_CATEGORIES_JSON_PATH]:
        path = str(path_obj)
        stat = os.stat(path)
        digest.update(os.path.basename(path).encode('utf-8'))
        digest.update(b':')
        digest.update(str(stat.st_size).encode('ascii'))
        digest.update(b':')
        digest.update(str(stat.st_mtime_ns).encode('ascii'))
        digest.update(b'|')
    return digest.hexdigest()[:12]


def _get_current_version(force_refresh: bool = False) -> str:
    now = time.monotonic()
    cached_value = _VERSION_CACHE['value']
    if not force_refresh and cached_value is not None and (now - float(_VERSION_CACHE['checked_at'])) < _VERSION_PROBE_INTERVAL_SEC:
        return cached_value
    version = _compute_version()
    _VERSION_CACHE['value'] = version
    _VERSION_CACHE['checked_at'] = now
    return version


def _normalize_chunk_key(raw_key: str) -> str | None:
    if raw_key is None:
        return None
    key = str(raw_key).strip()
    if not key or ':' not in key:
        return None
    lhs, rhs = key.split(':', 1)
    try:
        cx = int(lhs)
        cy = int(rhs)
    except ValueError:
        return None
    if cx < 0 or cy < 0:
        return None
    return f'{cx}:{cy}'


def _normalize_chunk_keys(raw_keys) -> list[str]:
    normalized = []
    seen = set()
    for raw_key in list(raw_keys or [])[:_MAX_CHUNK_KEYS]:
        key = _normalize_chunk_key(raw_key)
        if key is None or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _normalize_marker_id(raw_marker_id) -> str | None:
    if raw_marker_id is None:
        return None
    marker_id = str(raw_marker_id).strip()
    return marker_id or None


def _normalize_marker_ids(raw_ids) -> list[str]:
    normalized = []
    seen = set()
    for raw_marker_id in list(raw_ids or [])[:_MAX_DETAIL_IDS]:
        marker_id = _normalize_marker_id(raw_marker_id)
        if marker_id is None or marker_id in seen:
            continue
        seen.add(marker_id)
        normalized.append(marker_id)
    return normalized


def _normalize_detail_entry(raw_detail, fallback_title: str) -> dict:
    raw_detail = raw_detail or {}
    title = str(raw_detail.get('title') or fallback_title or '')
    description = raw_detail.get('description') or ''
    return {'title': title, 'description': str(description) if description else ''}


def _marker_chunk_key(x: int, y: int) -> str:
    return f'{max(0, x) // _CHUNK_SIZE}:{max(0, y) // _CHUNK_SIZE}'


def _build_store() -> dict:
    categories_raw = _load_json(_CATEGORIES_PATH) or {}
    markers_raw = _load_json(_MARKERS_LITE_PATH) or []
    details_raw = _load_json(_MARKERS_DETAIL_PATH) or {}
    categories = {str(key): {'name': str((value or {}).get('name') or key), 'group': str((value or {}).get('group') or '未分组')} for key, value in categories_raw.items()}
    marker_type_counts = {}
    markers_by_id = {}
    details_by_id = {}
    chunk_index = defaultdict(list)
    search_index = []
    for raw_marker in markers_raw:
        marker_id = _normalize_marker_id(raw_marker.get('id'))
        if marker_id is None or marker_id in markers_by_id:
            continue
        try:
            x = int(round(float(raw_marker.get('x', 0))))
            y = int(round(float(raw_marker.get('y', 0))))
        except (TypeError, ValueError):
            continue
        mark_type = str(raw_marker.get('markType') or '')
        category = categories.get(mark_type, {})
        fallback_title = category.get('name') or f'资源点 #{marker_id}'
        detail = _normalize_detail_entry(details_raw.get(marker_id), fallback_title)
        marker = {'id': marker_id, 'markType': mark_type, 'x': x, 'y': y, 'title': detail['title'] or fallback_title}
        markers_by_id[marker_id] = marker
        details_by_id[marker_id] = detail
        marker_type_counts[mark_type] = marker_type_counts.get(mark_type, 0) + 1
        chunk_index[_marker_chunk_key(x, y)].append(marker)
        search_index.append({'id': marker_id, 'markType': mark_type, 'x': x, 'y': y, 'title': marker['title'], 'description': detail['description']})
    for chunk_items in chunk_index.values():
        chunk_items.sort(key=lambda item: (item['y'], item['x'], item['id']))
    populated_chunk_keys = sorted(chunk_index.keys(), key=lambda key: tuple(int(part) for part in key.split(':', 1)))
    search_index.sort(key=lambda item: (item['title'], item['id']))
    return {'version': _compute_version(), 'chunkSize': _CHUNK_SIZE, 'categories': categories, 'markerTypeCounts': marker_type_counts, 'markersById': markers_by_id, 'detailsById': details_by_id, 'chunks': dict(chunk_index), 'searchIndex': search_index, 'populatedChunkKeys': populated_chunk_keys, 'totalChunkCount': len(populated_chunk_keys), 'totalMarkers': len(markers_by_id)}


def get_map_data_store(force_reload: bool = False) -> dict:
    global _DATA_CACHE
    with _DATA_LOCK:
        current_version = _get_current_version(force_refresh=force_reload)
        if force_reload or _DATA_CACHE is None or _DATA_CACHE['version'] != current_version:
            _DATA_CACHE = _build_store()
            _VERSION_CACHE['value'] = _DATA_CACHE['version']
            _VERSION_CACHE['checked_at'] = time.monotonic()
        return _DATA_CACHE


def get_marker_manifest(force_reload: bool = False) -> dict:
    store = get_map_data_store(force_reload=force_reload)
    return {'success': True, 'version': store['version'], 'chunkSize': store['chunkSize'], 'categories': store['categories'], 'markerTypeCounts': store['markerTypeCounts'], 'totalMarkers': store['totalMarkers'], 'totalChunkCount': store['totalChunkCount'], 'populatedChunkKeys': store['populatedChunkKeys']}


def get_marker_chunks(raw_keys) -> dict:
    store = get_map_data_store()
    requested_keys = _normalize_chunk_keys(raw_keys)
    chunks = {key: store['chunks'].get(key, []) for key in requested_keys}
    return {'success': True, 'version': store['version'], 'requestedKeys': requested_keys, 'chunks': chunks, 'totalReturned': sum(len(items) for items in chunks.values())}


def get_marker_details(raw_ids) -> dict:
    store = get_map_data_store()
    requested_ids = _normalize_marker_ids(raw_ids)
    items = {}
    for marker_id in requested_ids:
        detail = store['detailsById'].get(marker_id)
        if detail is not None:
            items[marker_id] = detail
    return {'success': True, 'version': store['version'], 'requestedIds': requested_ids, 'items': items}


def get_marker_search_index() -> dict:
    store = get_map_data_store()
    return {'success': True, 'version': store['version'], 'items': store['searchIndex'], 'total': len(store['searchIndex'])}
