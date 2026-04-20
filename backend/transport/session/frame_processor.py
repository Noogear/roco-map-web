"""
frame_processor.py - 帧处理：解码、识别、编码推送包
"""
from __future__ import annotations

import struct
from threading import Lock

import cv2
import numpy as np

from backend import config
from backend.common.runtime.fastjson import dumps_bytes
from backend.transport.push.manager import JpegPushManager
from backend.web.api.recognize import extract_minimap_with_autodetect
from backend.vision.engine.map_tracker_web import MapTrackerWeb

# ── 小地图提取稳定锁定 ───────────────────────────────────────────────
# 连续 _LOCK_AFTER_N 帧位置一致后，固定裁剪区域，不再逐帧 Hough 检测
_LOCK_AFTER_N = 10
_RESET_AFTER_MISS = 30   # 连续 N 帧 miss 后重置
_POS_TOLERANCE = 8       # 位置变化容差（像素）

_minimap_states: dict[str, dict] = {}   # token → state dict
_minimap_lock = Lock()


def _get_state(token: str) -> dict:
    with _minimap_lock:
        if token not in _minimap_states:
            from backend.map.minimap import CircleCalibrator
            _minimap_states[token] = {
                'stable_count': 0,
                'miss_count': 0,
                'locked': False,
                'last_cx': 0, 'last_cy': 0,
                'orig_cx': 0, 'orig_cy': 0, 'orig_r': 0,
                'calibrator': CircleCalibrator(),   # 复用，积累历史
            }
        return _minimap_states[token]


def _reset_state(state: dict) -> None:
    state['stable_count'] = 0
    state['miss_count'] = 0
    state['locked'] = False
    state['orig_cx'] = 0
    state['orig_cy'] = 0
    state['orig_r'] = 0
    state['calibrator'].reseed(0, 0, 0)   # 重置校准历史但保留对象


def extract_minimap_stable(img, token):
    """
    提取小地图：调用一次 extract_minimap_with_autodetect（传入复用校准器），
    从其返回的 raw_crop 取 orig_cx/cy 构建 crop_box，稳定后锁定。
    """
    state = _get_state(token)

    # 1. 一次调用，传入复用的 calibrator，历史累积后检测更稳定
    minimap_img, mm_center, mm_radius, source_tag, raw_crop = extract_minimap_with_autodetect(
        img, calibrator=state['calibrator']
    )

    if state['locked']:
        cx, cy, r = state['orig_cx'], state['orig_cy'], state['orig_r']
        ih, iw = img.shape[:2]
        crop_box = (max(0, cx - r), max(0, cy - r), min(iw, cx + r), min(ih, cy + r))
        return minimap_img, mm_center, mm_radius, source_tag, crop_box

    # 非锁定：从 raw_crop 取原帧圆坐标
    crop_box = None
    if raw_crop is not None:
        orig_cx, orig_cy, r = raw_crop.orig_cx, raw_crop.orig_cy, raw_crop.radius
        ih, iw = img.shape[:2]
        crop_box = (max(0, orig_cx - r), max(0, orig_cy - r),
                    min(iw, orig_cx + r), min(ih, orig_cy + r))
        dist = abs(orig_cx - state['last_cx']) + abs(orig_cy - state['last_cy'])
        if dist <= _POS_TOLERANCE:
            state['stable_count'] += 1
        else:
            state['stable_count'] = 1
        state['last_cx'] = orig_cx
        state['last_cy'] = orig_cy
        state['miss_count'] = 0
        if state['stable_count'] >= _LOCK_AFTER_N:
            state['locked'] = True
            state['orig_cx'] = orig_cx
            state['orig_cy'] = orig_cy
            state['orig_r'] = r
    else:
        state['miss_count'] += 1
        state['stable_count'] = max(0, state['stable_count'] - 1)
        if state['miss_count'] >= _RESET_AFTER_MISS:
            _reset_state(state)

    return minimap_img, mm_center, mm_radius, source_tag, crop_box


# ── 解码 ─────────────────────────────────────────────────────────────

def decode_blob(data) -> np.ndarray | None:
    try:
        if not isinstance(data, (bytes, bytearray)):
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ── 状态打包 ─────────────────────────────────────────────────────────

def make_status(
    result: dict,
    *,
    mode: str = 'orb',
    state: str | None = None,
    frozen: bool = False,
    is_teleport: bool = False,
) -> dict:
    found = bool(result.get('found', False))
    cx = int(result.get('center_x') or 0)
    cy = int(result.get('center_y') or 0)
    return {
        'm': mode,
        's': state or ('FOUND' if found else 'LOST'),
        'x': cx,
        'y': cy,
        'f': 1 if found else 0,
        'c': int(result.get('matches', 0)),
        'q': float(result.get('match_quality', 0.0)),
        'a': float(result.get('arrow_angle', 0.0)),
        'as': 0,
        'l': 0,
        'src': str(result.get('source', '')),
        'fz': 1 if frozen else 0,
        'tp': 1 if is_teleport else 0,
    }


def build_binary_result(status_dict: dict, jpeg: bytes | None) -> bytes:
    jbytes = dumps_bytes(status_dict)
    header = struct.pack('>I', len(jbytes))
    if jpeg:
        return header + jbytes + jpeg
    return header + jbytes


def encode_jpeg_b64(img: np.ndarray, quality: int = 80) -> str | None:
    import base64
    try:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('ascii')
    except Exception:
        return None


# ── 主流程 ────────────────────────────────────────────────────────────

def process_frame(
    tracker: MapTrackerWeb,
    img: np.ndarray,
    push_mgr: JpegPushManager,
    token: str,
) -> tuple[bytes, bool, dict | None]:
    """
    Returns: (payload, has_jpeg, debug_info)
    debug_info 含 _dm（小地图 b64）、_dms、_dc（裁剪区域 b64）、_dcs
    """
    minimap_img, mm_center, mm_radius, source_tag, crop_box = extract_minimap_stable(img, token)
    result = tracker.feature_engine.match(minimap_img, minimap_center=mm_center, minimap_radius=mm_radius)
    result.setdefault('matches', 0)

    found = bool(result.get('found', False))
    cx = int(result.get('center_x') or 0)
    cy = int(result.get('center_y') or 0)
    status = make_status(result)

    jpeg_bytes: bytes | None = None
    if found and push_mgr.has_jpeg_clients(token):
        target_sid = push_mgr._push_target.get(token, '')
        session_obj = push_mgr._sessions.get(target_sid)
        needs_jpeg = session_obj is None or session_obj.needs_jpeg(cx, cy)
        if needs_jpeg:
            h = w = config.VIEW_SIZE
            out = np.zeros((h, w, 3), dtype=np.uint8)
            ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                jpeg_bytes = jpg.tobytes()
                if session_obj:
                    session_obj.mark_jpeg_sent(cx, cy)

    # 诊断 debug_info：小地图缩略图 + 裁剪原始区域
    debug_info: dict | None = None
    mm_b64 = encode_jpeg_b64(minimap_img, quality=75)
    if mm_b64:
        debug_info = {
            '_dm': mm_b64,
            '_dms': f'{minimap_img.shape[1]}x{minimap_img.shape[0]}',
        }
        # 裁剪区域（原帧中小地图所在区域）
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            h_img, w_img = img.shape[:2]
            crop_region = img[max(0,y1):min(y2,h_img), max(0,x1):min(x2,w_img)]
            dc_b64 = encode_jpeg_b64(crop_region, quality=70)
            if dc_b64:
                debug_info['_dc'] = dc_b64
                debug_info['_dcs'] = f'{crop_region.shape[1]}x{crop_region.shape[0]}'

    payload = build_binary_result(status, jpeg_bytes)
    return payload, jpeg_bytes is not None, debug_info


def process_frame_coords(
    tracker: MapTrackerWeb,
    img: np.ndarray,
) -> bytes:
    minimap_img, mm_center, mm_radius, _, __ = extract_minimap_with_autodetect(img)
    result = tracker.feature_engine.match(minimap_img, minimap_center=mm_center, minimap_radius=mm_radius)
    result.setdefault('matches', 0)
    status = make_status(result)
    return build_binary_result(status, None)

