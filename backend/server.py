"""
server.py - Web 控制层（Flask + SocketIO 路由）

职责:
  - Flask / SocketIO 应用初始化
  - HTTP API 路由（上传/状态/控制）
  - WebSocket 消息处理
  - 圆形选区持久化

识别引擎和编排逻辑已拆分到:
  backend/tracker_engines.py   SIFTMapTracker (纯识别)
  backend/tracker_core.py      MapTrackerWeb (平滑/过滤/渲染编排)
"""

import sys
import os
import time
import struct
import threading
import importlib.util
from importlib import metadata as importlib_metadata
from collections import deque

import cv2
import numpy as np
import base64
from io import BytesIO

from backend import config
from flask import Flask, jsonify, redirect, request, send_file, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_compress import Compress
from backend.tracker_core import MapTrackerWeb
from backend.tracker_engines import get_shared_sift
from backend.core.context import SessionRegistry
from backend.core.data_standards import DataScope, audit_tracker_scope
from backend.core.fastjson import OrjsonProvider, dumps_bytes
from backend.map_data import (
    get_marker_chunks,
    get_marker_details,
    get_marker_manifest,
    get_marker_search_index,
)
from backend.store import get_route_files, load_route_data

# 项目目录路径（backend/ 的上一级）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(_BASE_DIR, 'frontend')
ASSETS_DIR   = os.path.join(_BASE_DIR, 'assets')


# ==================== 路线文件管理 ====================

_ROUTES_DIR = os.path.join(_BASE_DIR, 'routes')


# 固定使用 SIFT 引擎

# OpenCV 内部线程数配置：
#   单进程部署（python main_web.py）→ 让 OpenCV 使用全部核心做 SIFT 内部并行（默认行为）
#   多进程部署（gunicorn -w N）→ 每个 worker 设为 cpu_count//N，避免线程过多竞争
#   可通过环境变量 CV2_NUM_THREADS 覆盖，例如: CV2_NUM_THREADS=4 python main_web.py cpu
_cv2_threads = int(os.environ.get('CV2_NUM_THREADS', 0))  # 0 = OpenCV 自动选择
if _cv2_threads > 0:
    cv2.setNumThreads(_cv2_threads)
    print(f"[cv2] 已设置 OpenCV 线程数: {_cv2_threads}")

# Flask + SocketIO
_SOCKETIO_ASYNC_MODE = os.environ.get('SOCKETIO_ASYNC_MODE', 'threading').lower()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


# 兼容性兜底：threading + werkzeug 场景默认关闭升级，避免 websocket 握手触发 500
_SOCKETIO_ALLOW_UPGRADES = _env_flag(
    'SOCKETIO_ALLOW_UPGRADES',
    _SOCKETIO_ASYNC_MODE in ('gevent', 'eventlet')
)


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return 'not-installed'


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _log_socketio_runtime_diagnostics():
    effective_mode = getattr(socketio, 'async_mode', 'unknown')
    print('[diag] Socket.IO runtime check:')
    print(f'  - configured async_mode: {_SOCKETIO_ASYNC_MODE}')
    print(f'  - effective async_mode : {effective_mode}')
    print(f'  - allow_upgrades      : {_SOCKETIO_ALLOW_UPGRADES}')
    print(f"  - flask-socketio      : {_package_version('flask-socketio')}")
    print(f"  - python-socketio     : {_package_version('python-socketio')}")
    print(f"  - python-engineio     : {_package_version('python-engineio')}")
    print(f"  - gevent              : {_package_version('gevent')} (module={_module_available('gevent')})")
    print(f"  - gevent-websocket    : {_package_version('gevent-websocket')} (module={_module_available('geventwebsocket')})")

    if _SOCKETIO_ASYNC_MODE == 'gevent' and effective_mode != 'gevent':
        print('  ! WARNING: 已配置 gevent，但运行时并未启用 gevent，请检查依赖或启动方式。')
    if _SOCKETIO_ALLOW_UPGRADES and effective_mode not in ('gevent', 'eventlet'):
        print('  ! WARNING: 当前运行模式不建议启用 websocket upgrade，可能导致握手失败。')

app = Flask(__name__)
# 高性能 JSON provider（orjson）
app.json = OrjsonProvider(app)

# 网络吞吐优化：HTTP 压缩 + 长连接传输参数
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # HTTP 上传最大 8MB
app.config['COMPRESS_ALGORITHM'] = ['br', 'gzip']
app.config['COMPRESS_LEVEL'] = int(os.environ.get('COMPRESS_LEVEL', '5'))
app.config['COMPRESS_MIN_SIZE'] = int(os.environ.get('COMPRESS_MIN_SIZE', '700'))
Compress(app)

_WS_FRAME_MAX_BYTES = 4 * 1024 * 1024               # WS 帧最大 4MB
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    async_mode=_SOCKETIO_ASYNC_MODE,
    allow_upgrades=_SOCKETIO_ALLOW_UPGRADES,
    http_compression=True,
    compression_threshold=1024,
    max_http_buffer_size=_WS_FRAME_MAX_BYTES,
)

# 会话隔离：token -> tracker
_DEFAULT_SESSION_TOKEN = 'default'


def _create_tracker_for_token(token: str) -> MapTrackerWeb:
    tracker_obj = MapTrackerWeb(session_id=token)
    audit_tracker_scope(tracker_obj)
    return tracker_obj


_session_registry: SessionRegistry[MapTrackerWeb] = SessionRegistry(_create_tracker_for_token)
_TRACKER_SCOPE = DataScope.SESSION_SCOPED

# JPEG 推送管理器（per-client 节流锚点 + Plan A/B 状态）
# 全图模式客户端只接收 coords 广播，不注册进管理器。
from backend.push.manager import JpegPushManager
_jpeg_mgr = JpegPushManager()

# 会话隔离：sid <-> token 绑定，并按 token 房间定向推送 coords/result
_SESSION_ROOM_PREFIX = 'session:'
_sid_token: dict[str, str] = {}
_sid_token_lock = threading.Lock()
_hash_query_lock = threading.Lock()


def _normalize_session_token(token) -> str:
    token = str(token or '').strip()
    if len(token) < 8:
        return ''
    # 限制长度并移除空白，避免异常 token 影响房间命名
    token = ''.join(token.split())
    return token[:128]


def _coerce_session_token(token) -> str:
    normalized = _normalize_session_token(token)
    return normalized or _DEFAULT_SESSION_TOKEN


def _extract_http_session_token() -> str:
    token = request.args.get('token') or request.headers.get('X-Session-Token')
    if not token and request.is_json:
        data = request.get_json(silent=True) or {}
        token = data.get('token')
    if not token:
        token = request.form.get('token') if hasattr(request, 'form') else ''
    return _coerce_session_token(token)


def _get_tracker_for_token(token: str) -> MapTrackerWeb:
    ctx, created = _session_registry.get_or_create(_coerce_session_token(token))
    tracker_obj = ctx.tracker
    if created or tracker_obj.result_callback is None:
        tracker_obj.result_callback = (
            lambda _token=ctx.token, _tracker=tracker_obj: _on_result_ready(_token, _tracker)
        )
    tracker_obj.touch_active()
    return tracker_obj


def _get_tracker_for_http_request() -> MapTrackerWeb:
    return _get_tracker_for_token(_extract_http_session_token())


def _get_tracker_for_sid(sid: str) -> MapTrackerWeb:
    return _get_tracker_for_token(_get_sid_token(sid) or _DEFAULT_SESSION_TOKEN)


def _session_room(token: str) -> str:
    return _SESSION_ROOM_PREFIX + token


def _bind_sid_token(sid: str, token: str) -> str:
    """绑定 sid -> token，返回旧 token（不存在则空字符串）。"""
    with _sid_token_lock:
        old = _sid_token.get(sid, '')
        _sid_token[sid] = token
    return old


def _unbind_sid_token(sid: str) -> str:
    with _sid_token_lock:
        return _sid_token.pop(sid, '')


def _get_sid_token(sid: str) -> str:
    with _sid_token_lock:
        return _sid_token.get(sid, '')

# ── 展示室（Broadcast Room）──────────────────────────────────────
# 每位展示者有独立房间，观众订阅特定展示者。
# 节流：每位展示者服务器端以 ~10fps 广播，避免观众数量放大流量。
#   _bcast_rooms[name] = {
#       'sid': str,               # 展示者 socket.sid
#       'viewers': set[str],      # 正在订阅的观众 sids
#       'last_ts': float,         # 上次广播时间戳
#       'last_frame': bytes|None, # 最新帧
#   }
_bcast_rooms: dict[str, dict] = {}
_bcast_lock = threading.Lock()
_BCAST_MIN_INTERVAL = 0.10          # 10 fps 硬上限（10fps 已足够观看）


def _make_status_json(tracker_obj: MapTrackerWeb) -> bytes:
    """构建 WS 推送用的紧凑 JSON bytes（避免 ws_receive_frame 与 _on_result_ready 重复编码）。"""
    status = tracker_obj.latest_status
    return dumps_bytes({
        'm': tracker_obj.current_mode,
        's': status['state'],
        'x': status['position']['x'],
        'y': status['position']['y'],
        'f': int(status['found']),
        'c': status['matches'],
        'q': round(status.get('match_quality', 0), 2),
        'a': round(status.get('arrow_angle', 0), 1),
        'as': int(status.get('arrow_stopped', True)),
        'tp': int(status.get('is_teleport', False)),
    })


def _on_result_ready(token: str, tracker_obj: MapTrackerWeb):
    """
    Plan A 推送回调：sift-worker 处理完帧后立即调用，主动推送新结果。
    比 pull 模式（等下一帧 ws_receive_frame）延迟低 ~80ms。

    JPEG 模式：由 _jpeg_mgr.push_result() 负责节流决策，定向推送 result/coords。
    全图模式：客户端未注册进 _jpeg_mgr，依赖下方广播的 coords 更新位置点。
    每个 emit 独立 try/except，单条失败不影响其他。
    """
    if tracker_obj.latest_status.get('state') == '--':
        return  # 首帧尚未完成，不推送

    status_json = _make_status_json(tracker_obj)
    header = struct.pack('>I', len(status_json))
    cx = tracker_obj.latest_status['position']['x']
    cy = tracker_obj.latest_status['position']['y']

    token = _coerce_session_token(_normalize_session_token(getattr(tracker_obj, 'latest_result_token', '')) or token)

    # JPEG 模式：按 token 定向推送（含节流）
    _jpeg_mgr.push_result(socketio, tracker_obj.latest_result_jpeg, cx, cy, header, status_json, token=token)

    # 全图模式 / 地图页：仅向同会话房间推送，避免跨会话串流
    if token:
        try:
            socketio.emit('coords', header + status_json, to=_session_room(token), namespace='/')
        except Exception as e:
            print(f"[push] coords room push 失败: {e}")


_get_tracker_for_token(_DEFAULT_SESSION_TOKEN)

# (路径常量 _BASE_DIR / FRONTEND_DIR / ASSETS_DIR 已在文件顶部定义)


# ==================== 配置元数据 / 运行时热更新 ====================

_CONFIG_RUNTIME_RULES = {
    'JPEG_PAD': {
        'type': 'int', 'min': 0, 'max': 160,
        'group': '界面与渲染', 'label': 'JPEG 扩边像素',
        'description': '控制 JPEG 模式额外扩边，便于前端平滑位移。',
        'editable': True,
    },
    'SEARCH_RADIUS': {
        'type': 'int', 'min': 120, 'max': 2000,
        'group': '识别引擎', 'label': '局部搜索半径',
        'description': '影响局部匹配范围，越大越稳但也更慢。',
        'editable': True,
    },
    'NEARBY_SEARCH_RADIUS': {
        'type': 'int', 'min': 120, 'max': 2400,
        'group': '识别引擎', 'label': '附近搜索半径',
        'description': '局部失败时的附近补救搜索半径。',
        'editable': True,
    },
    'LOCAL_FAIL_LIMIT': {
        'type': 'int', 'min': 1, 'max': 20,
        'group': '识别引擎', 'label': '局部失败阈值',
        'description': '局部匹配连续失败多少次后退回全局搜索。',
        'editable': True,
    },
    'SIFT_JUMP_THRESHOLD': {
        'type': 'int', 'min': 80, 'max': 2000,
        'group': '识别引擎', 'label': '跳变阈值',
        'description': '识别结果与上一帧距离过大时，用于判定异常跳点。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_INTERVAL': {
        'type': 'int', 'min': 1, 'max': 30,
        'group': '识别引擎', 'label': '局部复核周期',
        'description': '局部命中多少次后强制做一次全局复核。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_MIN_QUALITY': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '识别引擎', 'label': '局部复核质量阈值',
        'description': '局部质量低于该值时提前触发全局复核。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_MARGIN': {
        'type': 'float', 'min': 0.0, 'max': 0.5,
        'group': '识别引擎', 'label': '复核覆盖边际',
        'description': '全局质量至少高出该阈值才覆盖局部结果。',
        'editable': True,
    },
    'LOCAL_REVALIDATE_DIFF': {
        'type': 'int', 'min': 50, 'max': 1000,
        'group': '识别引擎', 'label': '复核冲突距离',
        'description': '局部与全局结果差异超过该值时视为冲突。',
        'editable': True,
    },
    'MAX_LOST_FRAMES': {
        'type': 'int', 'min': 0, 'max': 300,
        'group': '识别引擎', 'label': '最大惯性帧数',
        'description': '完全丢失前允许惯性维持的最大帧数。',
        'editable': True,
    },
    'LK_ENABLED': {
        'type': 'bool', 'group': '识别引擎', 'label': '启用 LK 光流',
        'description': '开启后可降低部分帧的 SIFT 压力。',
        'editable': True,
    },
    'LK_SIFT_INTERVAL': {
        'type': 'int', 'min': 1, 'max': 30,
        'group': '识别引擎', 'label': 'LK 强制 SIFT 周期',
        'description': '每隔多少帧强制做一次 SIFT 纠偏。',
        'editable': True,
    },
    'LK_MIN_CONFIDENCE': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '识别引擎', 'label': 'LK 最低置信度',
        'description': '低于该值时放弃光流结果。',
        'editable': True,
    },
    'ECC_ENABLED': {
        'type': 'bool', 'group': '识别引擎', 'label': '启用 ECC 兜底',
        'description': '低纹理附近搜索失败时允许 ECC 对齐兜底。',
        'editable': True,
    },
    'ECC_MIN_CORRELATION': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '识别引擎', 'label': 'ECC 最低相关度',
        'description': 'ECC 结果低于该值将被忽略。',
        'editable': True,
    },
    'ARROW_ANGLE_SMOOTH_ALPHA': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '箭头', 'label': '箭头平滑系数',
        'description': '数值越高，箭头越灵敏。',
        'editable': True,
    },
    'ARROW_MOVE_MIN_DISPLACEMENT': {
        'type': 'float', 'min': 0.1, 'max': 20.0,
        'group': '箭头', 'label': '最小移动判定',
        'description': '小于该位移时认为角色接近静止。',
        'editable': True,
    },
    'ARROW_POS_HISTORY_LEN': {
        'type': 'int', 'min': 2, 'max': 20,
        'group': '箭头', 'label': '箭头历史长度',
        'description': '箭头方向计算所保留的坐标历史数。',
        'editable': True,
    },
    'ARROW_STOPPED_DEBOUNCE': {
        'type': 'int', 'min': 1, 'max': 60,
        'group': '箭头', 'label': '静止防抖帧数',
        'description': '连续多少帧接近静止后才视为停下。',
        'editable': True,
    },
    'ARROW_SNAP_THRESHOLD': {
        'type': 'float', 'min': 5.0, 'max': 180.0,
        'group': '箭头', 'label': '箭头吸附阈值',
        'description': '方向突变超过该角度时直接吸附到新方向。',
        'editable': True,
    },
    'LINEAR_FILTER_ENABLED': {
        'type': 'bool', 'group': '渲染平滑', 'label': '启用线性过滤',
        'description': '对连续异常跳点做线性过滤。',
        'editable': True,
    },
    'LINEAR_FILTER_WINDOW': {
        'type': 'int', 'min': 1, 'max': 40,
        'group': '渲染平滑', 'label': '线性过滤窗口',
        'description': '异常值过滤时参考的历史窗口大小。',
        'editable': True,
    },
    'LINEAR_FILTER_MAX_DEVIATION': {
        'type': 'int', 'min': 20, 'max': 500,
        'group': '渲染平滑', 'label': '最大偏差',
        'description': '超出该偏差的点更容易被判为异常。',
        'editable': True,
    },
    'LINEAR_FILTER_MAX_CONSECUTIVE': {
        'type': 'int', 'min': 1, 'max': 30,
        'group': '渲染平滑', 'label': '最大连续过滤数',
        'description': '避免长时间连续过滤导致无法恢复。',
        'editable': True,
    },
    'RENDER_STILL_THRESHOLD': {
        'type': 'int', 'min': 0, 'max': 50,
        'group': '渲染平滑', 'label': '静止死区',
        'description': '平滑显示时的小位移忽略阈值。',
        'editable': True,
    },
    'RENDER_EMA_ALPHA': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '渲染平滑', 'label': '最低 EMA Alpha',
        'description': '慢速移动时的最低平滑系数。',
        'editable': True,
    },
    'RENDER_EMA_ALPHA_MAX': {
        'type': 'float', 'min': 0.0, 'max': 1.0,
        'group': '渲染平滑', 'label': '最高 EMA Alpha',
        'description': '快速移动时的最高平滑系数。',
        'editable': True,
    },
    'RENDER_EMA_SLOW_DIST': {
        'type': 'int', 'min': 1, 'max': 200,
        'group': '渲染平滑', 'label': '慢速距离阈值',
        'description': '低于该距离时使用最低 EMA alpha。',
        'editable': True,
    },
    'RENDER_EMA_FAST_DIST': {
        'type': 'int', 'min': 1, 'max': 500,
        'group': '渲染平滑', 'label': '快速距离阈值',
        'description': '高于该距离时使用最高 EMA alpha。',
        'editable': True,
    },
    'RENDER_OFFSET_X': {
        'type': 'int', 'min': -200, 'max': 200,
        'group': '渲染平滑', 'label': '渲染偏移 X',
        'description': '用于显示侧的微调偏移。',
        'editable': True,
    },
    'RENDER_OFFSET_Y': {
        'type': 'int', 'min': -200, 'max': 200,
        'group': '渲染平滑', 'label': '渲染偏移 Y',
        'description': '用于显示侧的微调偏移。',
        'editable': True,
    },
    'TP_JUMP_THRESHOLD': {
        'type': 'int', 'min': 50, 'max': 2000,
        'group': '渲染平滑', 'label': '传送跳变阈值',
        'description': '超过该阈值时会进入传送候选确认。',
        'editable': True,
    },
    'TP_CONFIRM_FRAMES': {
        'type': 'int', 'min': 1, 'max': 10,
        'group': '渲染平滑', 'label': '传送确认帧数',
        'description': '连续多少帧聚类命中后确认传送。',
        'editable': True,
    },
    'TP_CLUSTER_RADIUS': {
        'type': 'int', 'min': 20, 'max': 500,
        'group': '渲染平滑', 'label': '传送聚类半径',
        'description': '传送候选聚类时允许的最大散布半径。',
        'editable': True,
    },
}

_CONFIG_RESTART_REQUIRED = {
    'PORT', 'WINDOW_GEOMETRY', 'VIEW_SIZE', 'LOGIC_MAP_PATH', 'DISPLAY_MAP_PATH',
    'SIFT_CONTRAST_THRESHOLD', 'MINIMAP', 'MINIMAP_CAPTURE_MARGIN',
    'MINIMAP_CIRCLE_CALIBRATION_FRAMES', 'MINIMAP_CIRCLE_R_TOLERANCE',
    'MINIMAP_CIRCLE_CENTER_TOLERANCE', 'MINIMAP_CIRCLE_RECALIBRATE_MISS',
}


def _iter_config_keys() -> list[str]:
    return sorted(
        key for key, value in vars(config).items()
        if key.isupper() and not key.startswith('_') and not callable(value)
    )


def _infer_config_group(key: str) -> str:
    if key in ('PORT', 'MINIMAP', 'WINDOW_GEOMETRY', 'VIEW_SIZE', 'JPEG_PAD'):
        return '基础服务'
    if key.endswith('_PATH'):
        return '地图文件'
    if key.startswith('AUTO_DETECT') or key.startswith('MINIMAP_'):
        return '自动圆检测'
    if key.startswith('ARROW_'):
        return '箭头'
    if key.startswith('RENDER_') or key.startswith('LINEAR_FILTER_') or key.startswith('TP_'):
        return '渲染平滑'
    return '识别引擎'


def _infer_config_type(value) -> str:
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int) and not isinstance(value, bool):
        return 'int'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, dict):
        return 'json'
    return 'string'


def _serialize_config_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value)


def _build_config_meta(key: str, value):
    rule = _CONFIG_RUNTIME_RULES.get(key, {})
    editable = bool(rule.get('editable', False))
    restart_required = key in _CONFIG_RESTART_REQUIRED or (not editable and key not in _CONFIG_RUNTIME_RULES)
    return {
        'key': key,
        'label': rule.get('label', key.replace('_', ' ')),
        'group': rule.get('group', _infer_config_group(key)),
        'type': rule.get('type', _infer_config_type(value)),
        'editable': editable,
        'restartRequired': restart_required,
        'min': rule.get('min'),
        'max': rule.get('max'),
        'step': rule.get('step'),
        'description': rule.get('description', ''),
        'reason': rule.get('reason') or (
            '该配置需要重启相关引擎或服务后才能安全生效。' if restart_required else '当前版本仅开放只读查看。'
        ),
    }


def _build_config_payload():
    values = {}
    meta = {}
    groups = set()
    for key in _iter_config_keys():
        value = getattr(config, key)
        values[key] = _serialize_config_value(value)
        meta[key] = _build_config_meta(key, value)
        groups.add(meta[key]['group'])
    return {
        'success': True,
        'values': values,
        'meta': meta,
        'groups': sorted(groups),
        'editableKeys': sorted(key for key, item in meta.items() if item['editable']),
        'readonlyKeys': sorted(key for key, item in meta.items() if not item['editable']),
    }


def _coerce_runtime_value(meta: dict, raw_value):
    value_type = meta['type']
    if value_type == 'bool':
        if isinstance(raw_value, bool):
            value = raw_value
        elif isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in ('1', 'true', 'yes', 'on'):
                value = True
            elif lowered in ('0', 'false', 'no', 'off'):
                value = False
            else:
                raise ValueError('必须为 true / false')
        else:
            raise ValueError('必须为布尔值')
        return value

    if value_type == 'int':
        if isinstance(raw_value, bool):
            raise ValueError('不能为布尔值')
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if raw_value == '':
                raise ValueError('不能为空')
        try:
            value = int(float(raw_value))
        except (TypeError, ValueError):
            raise ValueError('必须为整数')
        if float(value) != float(raw_value):
            raise ValueError('必须为整数')
        return value

    if value_type == 'float':
        if isinstance(raw_value, bool):
            raise ValueError('不能为布尔值')
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            raise ValueError('必须为数字')

    if value_type == 'string':
        if raw_value is None:
            raise ValueError('不能为空')
        return str(raw_value)

    raise ValueError('当前类型暂不支持在线修改')


def _validate_runtime_config_updates(updates: dict):
    approved = {}
    rejected = {}

    for key, raw_value in updates.items():
        current_value = getattr(config, key, None)
        meta = _build_config_meta(key, current_value)
        if current_value is None or key not in _iter_config_keys():
            rejected[key] = '配置项不存在'
            continue
        if not meta['editable']:
            rejected[key] = meta['reason']
            continue
        try:
            value = _coerce_runtime_value(meta, raw_value)
        except ValueError as exc:
            rejected[key] = str(exc)
            continue

        min_value = meta.get('min')
        max_value = meta.get('max')
        if min_value is not None and value < min_value:
            rejected[key] = f'不能小于 {min_value}'
            continue
        if max_value is not None and value > max_value:
            rejected[key] = f'不能大于 {max_value}'
            continue
        approved[key] = value

    return approved, rejected


def _apply_runtime_config_updates(updates: dict):
    for key, value in updates.items():
        setattr(config, key, value)

    def _apply_to_tracker(tracker_obj: MapTrackerWeb):
        engine = tracker_obj.sift_engine
        process_lock = getattr(tracker_obj, '_process_lock', None)
        # 使用带超时的 acquire() 避免无限期阻塞导致 UI 卡顿
        # 如果后台处理线程正在运行，至多等待 2 秒；超时则跳过此 tracker
        lock_acquired = False
        if process_lock is not None:
            lock_acquired = process_lock.acquire(timeout=2.0)
            if not lock_acquired:
                print(f"[config] 无法获得后台处理锁（超时），跳过 tracker 配置更新")
                return  # 跳过此 tracker，避免竞态条件
        try:
            with engine._lock:
                if 'SEARCH_RADIUS' in updates:
                    engine.SEARCH_RADIUS = config.SEARCH_RADIUS
                if 'NEARBY_SEARCH_RADIUS' in updates:
                    engine.NEARBY_SEARCH_RADIUS = config.NEARBY_SEARCH_RADIUS
                if 'LOCAL_FAIL_LIMIT' in updates:
                    engine.LOCAL_FAIL_LIMIT = config.LOCAL_FAIL_LIMIT
                if 'SIFT_JUMP_THRESHOLD' in updates:
                    engine.JUMP_THRESHOLD = config.SIFT_JUMP_THRESHOLD
                if 'LOCAL_REVALIDATE_INTERVAL' in updates:
                    engine._local_revalidate_interval = config.LOCAL_REVALIDATE_INTERVAL
                if 'LOCAL_REVALIDATE_MIN_QUALITY' in updates:
                    engine._local_revalidate_min_quality = config.LOCAL_REVALIDATE_MIN_QUALITY
                if 'LOCAL_REVALIDATE_MARGIN' in updates:
                    engine._local_revalidate_margin = config.LOCAL_REVALIDATE_MARGIN
                if 'LOCAL_REVALIDATE_DIFF' in updates:
                    engine._local_revalidate_diff = config.LOCAL_REVALIDATE_DIFF

                if hasattr(engine, '_lk'):
                    if 'LK_ENABLED' in updates:
                        engine._lk.enabled = config.LK_ENABLED
                    if 'LK_SIFT_INTERVAL' in updates:
                        engine._lk.sift_every = config.LK_SIFT_INTERVAL
                    if 'LK_MIN_CONFIDENCE' in updates:
                        engine._lk.min_conf = config.LK_MIN_CONFIDENCE

                if 'ECC_ENABLED' in updates:
                    engine._ecc_enabled = config.ECC_ENABLED
                if 'ECC_MIN_CORRELATION' in updates:
                    engine._ecc_min_cc = config.ECC_MIN_CORRELATION

                arrow_dir = getattr(engine, '_arrow_dir', None)
                if arrow_dir is not None:
                    if 'ARROW_ANGLE_SMOOTH_ALPHA' in updates:
                        arrow_dir._ema_alpha = config.ARROW_ANGLE_SMOOTH_ALPHA
                    if 'ARROW_MOVE_MIN_DISPLACEMENT' in updates:
                        arrow_dir._stop_speed_px = config.ARROW_MOVE_MIN_DISPLACEMENT
                    if 'ARROW_STOPPED_DEBOUNCE' in updates:
                        arrow_dir._stop_debounce = config.ARROW_STOPPED_DEBOUNCE
                    if 'ARROW_SNAP_THRESHOLD' in updates:
                        arrow_dir._snap_threshold = config.ARROW_SNAP_THRESHOLD
                    if 'ARROW_POS_HISTORY_LEN' in updates:
                        history = list(arrow_dir._history)
                        arrow_dir._history = deque(history[-config.ARROW_POS_HISTORY_LEN:], maxlen=config.ARROW_POS_HISTORY_LEN)
        finally:
            if lock_acquired and process_lock is not None:
                process_lock.release()

    for ctx in _session_registry.snapshot_contexts():
        _apply_to_tracker(ctx.tracker)


# ==================== HTTP 路由 ====================

@app.route('/')
def root():
    return redirect('/recognize')

@app.route('/recognize')
def index():
    return send_file(os.path.join(FRONTEND_DIR, 'recognize.html'))


@app.route('/map')
def serve_map():
    return send_file(os.path.join(FRONTEND_DIR, 'map.html'))


@app.route('/settings')
def serve_settings():
    return send_file(os.path.join(FRONTEND_DIR, 'settings.html'))


@app.route('/<path:filename>')
def serve_static(filename):
    """提供静态资源：先尝试 frontend/，不存在则找 assets/"""
    frontend_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(frontend_path):
        return send_from_directory(FRONTEND_DIR, filename)
    return send_from_directory(ASSETS_DIR, filename)


@app.route('/api/test_images')
def list_test_images():
    """列出 frontend/img/ 下可用的测试小地图"""
    img_dir = os.path.join(FRONTEND_DIR, 'img')
    images = []
    if os.path.isdir(img_dir):
        for f in sorted(os.listdir(img_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append(f'/img/{f}')
    return jsonify(images)


def _decode_base64_image(b64_data: str):
    """解码单张 base64 图片，失败返回 None。"""
    if not b64_data:
        return None
    if ',' in b64_data:
        b64_data = b64_data.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _decode_image_from_request():
    """从当前请求中解析图片（支持 FormData 与 JSON base64），失败返回 None。"""
    # 方式1: FormData 文件上传
    if 'image' in request.files:
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 方式2: JSON base64 上传
    if request.is_json:
        data = request.get_json(silent=True) or {}
        return _decode_base64_image(data.get('image', ''))

    return None


def _decode_images_from_request(max_count: int = 8) -> list[np.ndarray]:
    """从请求中解析多张图片（JSON images 数组优先），失败返回空列表。"""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        images = data.get('images')
        if isinstance(images, list) and images:
            decoded = []
            for item in images[:max(1, max_count)]:
                img = _decode_base64_image(item)
                if img is not None:
                    decoded.append(img)
            if decoded:
                return decoded

    single = _decode_image_from_request()
    return [single] if single is not None else []


@app.route('/api/upload_minimap', methods=['POST'])
def upload_minimap():
    """接收前端上传的小地图图片（支持 FormData 和 JSON base64 两种格式）。
    同步处理并返回结果，不唤醒后台 SIFT 工作线程。"""
    tracker_obj = _get_tracker_for_http_request()
    img = _decode_image_from_request()

    if img is not None:
        # 直接设置当前帧（不调用 set_minimap 以避免唤醒后台线程导致双重处理）
        with tracker_obj.lock:
            tracker_obj.current_frame_bgr = img.copy()
        result = tracker_obj.process_frame(need_base64=True, need_jpeg=False)
        if result and result[0]:
            return jsonify({
                'success': True,
                'image': result[0],
                'status': tracker_obj.latest_status,
            })
    return jsonify({'error': 'Invalid image'}), 400


@app.route('/api/recognize_single', methods=['POST'])
def recognize_single():
    """无状态单次识别：直接做全局 SIFT 匹配，不操作/不依赖共享 tracker 状态。
    适用于上传截图测试场景。"""
    from backend.core.features import extract_minimap_features, sift_match_region, CircularMaskCache
    from backend.core.enhance import enhance_minimap, make_clahe_pair, correct_color_temperature
    from backend.core.ecc import ecc_align
    from backend.tracking.minimap import detect_and_extract

    img = _decode_image_from_request()
    if img is None:
        return jsonify({'error': 'No image data'}), 400

    shared = get_shared_sift()
    input_h, input_w = img.shape[:2]
    source_tag = 'RAW'

    # 1. 圆形小地图检测 + 提取（使用临时校准器，不影响共享状态）
    from backend.tracking.minimap import CircleCalibrator
    temp_cal = CircleCalibrator()
    extracted = detect_and_extract(img, temp_cal, engine_frozen=False)
    if extracted is not None:
        img = extracted
        source_tag = 'LOCAL_CIRCLE'
    else:
        # 1.1 当输入是整张截图时，尝试自动定位小地图圆后再裁剪
        # 说明：上传测试常见失败原因是前端固定裁剪参数与截图布局不一致。
        if min(input_h, input_w) >= 320:
            from backend.tracking.autodetect import detect_minimap_circle

            det = detect_minimap_circle(img)
            if det is not None:
                h, w = img.shape[:2]
                bs = min(w, h)
                cx = float(det['cx']) * w
                cy = float(det['cy']) * h
                r = float(det['r']) * bs
                margin = max(1.0, float(getattr(config, 'MINIMAP_CAPTURE_MARGIN', 1.4)))

                sz = max(10, int(round(r * 2 * margin)))
                rx = max(0, min(int(round(cx - sz / 2)), w - sz))
                ry = max(0, min(int(round(cy - sz / 2)), h - sz))
                crop = img[ry:ry + sz, rx:rx + sz].copy()

                # 再跑一次局部圆提取，确保进入与常规路径一致的小地图域
                extracted2 = detect_and_extract(crop, CircleCalibrator(), engine_frozen=False)
                img = extracted2 if extracted2 is not None else crop
                source_tag = f"AUTO_DETECT_{det.get('layout', 'full').upper()}"

    # 2. 预处理：色温补偿 + CLAHE 增强
    img = correct_color_temperature(img)
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_std = float(np.std(gray_raw))
    clahe_normal, clahe_low = make_clahe_pair()
    low_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
    gray = enhance_minimap(gray_raw, texture_std, clahe_normal, clahe_low, low_thresh)

    # 3. 提取小地图特征点
    mask_cache = CircularMaskCache()
    kp_mini, des_mini = extract_minimap_features(gray, shared.sift, mask_cache, texture_std=texture_std)
    if des_mini is None or len(kp_mini) < 3:
        return jsonify({
            'success': False,
            'status': {
                'state': 'NO_FEATURES', 'found': False,
                'position': {'x': 0, 'y': 0}, 'matches': 0,
                'match_quality': 0, 'mode': 'sift',
            }
        })

    hash_index = getattr(shared, '_hash_index', None)

    def _locate_hash_candidates_adaptive(max_results: int = 3):
        """无状态场景下自适应放宽 hash 阈值，提升低纹理召回率。"""
        if hash_index is None:
            return []

        # 先严格后宽松，避免常规场景误召回。
        thresholds = [None, 16, 20]
        with _hash_query_lock:
            original = int(getattr(hash_index, '_hamming_thresh', 12))
            try:
                for th in thresholds:
                    if th is not None:
                        hash_index._hamming_thresh = int(th)
                    candidates = hash_index.locate(gray_raw, last_x=None, last_y=None, radius=0, max_results=max_results)
                    if candidates:
                        return candidates
                return []
            finally:
                hash_index._hamming_thresh = original

    def _hash_ecc_or_hash_fallback(source_suffix: str):
        """无状态兜底：先 hash 粗定位，再用 ECC 在候选附近细化。"""
        if hash_index is None:
            return None

        candidates = _locate_hash_candidates_adaptive(max_results=3)
        if not candidates:
            return None

        # ECC 两侧保持在增强域，降低亮度差带来的失败率
        logic_map_for_ecc = getattr(shared, '_logic_map_gray_clahe', shared._logic_map_gray)
        ecc_min_cc = float(getattr(config, 'SINGLE_ECC_MIN_CORRELATION', 0.30))
        ecc_jump = int(getattr(config, 'SINGLE_ECC_JUMP_THRESHOLD', 360))
        ecc_scale_hint = float(getattr(config, 'HASH_INDEX_PATCH_SCALE', 4.0))

        for hx, hy, hdist in candidates:
            refined = ecc_align(
                gray, logic_map_for_ecc,
                int(hx), int(hy),
                ecc_scale_hint,
                shared.map_width, shared.map_height,
                ecc_jump,
                min_cc=ecc_min_cc,
            )
            if refined is not None:
                ex, ey = refined
                consistent = True
                try:
                    consistent = bool(hash_index.check_consistency(gray_raw, int(ex), int(ey)))
                except Exception:
                    consistent = True
                # 兜底质量分：保证前端有可读性，同时不虚高
                base_q = max(0.18, 0.42 - float(hdist) * 0.015)
                if consistent:
                    base_q = max(base_q, 0.35)
                return jsonify({
                    'success': True,
                    'status': {
                        'state': 'FOUND', 'found': True, 'source': f'HASH_ECC_{source_suffix}',
                        'position': {'x': int(ex), 'y': int(ey)}, 'matches': 0,
                        'match_quality': float(min(base_q, 0.65)), 'mode': 'sift',
                    }
                })

        # ECC 未细化成功，退回 hash 粗定位（至少给可用近似坐标）
        hx, hy, hdist = candidates[0]
        return jsonify({
            'success': True,
            'status': {
                'state': 'HASH_FOUND', 'found': True, 'source': f'HASH_INDEX_{source_suffix}',
                'position': {'x': int(hx), 'y': int(hy)}, 'matches': 0,
                'match_quality': max(0.15, 0.35 - float(hdist) * 0.02), 'mode': 'sift',
            }
        })

    # 4. 全局 SIFT 匹配（使用共享只读 FLANN 索引，线程安全）
    match_ratio = getattr(config, 'SIFT_MATCH_RATIO', 0.82)
    min_match = getattr(config, 'SIFT_MIN_MATCH_COUNT', 5)
    result = sift_match_region(
        kp_mini, des_mini, gray.shape,
        shared.kp_big_all, shared.flann_global,
        match_ratio, min_match,
        shared.map_width, shared.map_height)

    if result is None:
        # 尝试宽松参数
        result = sift_match_region(
            kp_mini, des_mini, gray.shape,
            shared.kp_big_all, shared.flann_global,
            0.88, 3,
            shared.map_width, shared.map_height)

    if result is None:
        # SIFT 失败：走 hash+ECC 兜底链，提高低纹理召回率
        fallback_resp = _hash_ecc_or_hash_fallback(source_tag)
        if fallback_resp is not None:
            return fallback_resp
        return jsonify({
            'success': False,
            'status': {
                'state': 'NOT_FOUND', 'found': False,
                'position': {'x': 0, 'y': 0}, 'matches': 0,
                'match_quality': 0, 'mode': 'sift',
            }
        })

    tx, ty, inlier_count, quality, avg_scale = result
    # 5. 无状态识别安全闸门：低置信或视觉不一致时不直接返回 FOUND
    # 目的：避免出现“识别到了，但坐标错”的误报。
    min_inliers = int(getattr(config, 'SIFT_MIN_MATCH_COUNT', 5))
    min_quality = float(getattr(config, 'SINGLE_RECOGNIZE_MIN_QUALITY', 0.12))
    sift_confident = (int(inlier_count) >= max(3, min_inliers)) and (float(quality) >= min_quality)
    # 哈希索引是粗校验，不宜压过“非常强”的 SIFT 命中（避免误拒真阳性）。
    sift_very_strong = (int(inlier_count) >= 8) and (float(quality) >= 0.55)

    hash_consistent = True
    if hash_index is not None:
        try:
            hash_consistent = bool(hash_index.check_consistency(gray_raw, int(tx), int(ty)))
        except Exception:
            hash_consistent = True

    if (not sift_confident) or ((not hash_consistent) and (not sift_very_strong)):
        fallback_resp = _hash_ecc_or_hash_fallback(source_tag)
        if fallback_resp is not None:
            return fallback_resp

        return jsonify({
            'success': False,
            'status': {
                'state': 'LOW_CONFIDENCE', 'found': False,
                'position': {'x': 0, 'y': 0}, 'matches': int(inlier_count),
                'match_quality': float(quality), 'mode': 'sift',
                'source': f'SIFT_GLOBAL_{source_tag}',
            }
        })

    # 5. 渲染结果图片（使用灰度大地图）
    gray_map = shared._logic_map_gray
    half_view = config.VIEW_SIZE // 2
    pad = getattr(config, 'JPEG_PAD', 0)
    y1 = max(0, ty - half_view - pad)
    y2 = min(shared.map_height, ty + half_view + pad)
    x1 = max(0, tx - half_view - pad)
    x2 = min(shared.map_width, tx + half_view + pad)
    crop = gray_map[y1:y2, x1:x2]
    full_size = config.VIEW_SIZE + 2 * pad
    canvas = np.full((full_size, full_size), 43, dtype=np.uint8)
    h, w = crop.shape[:2]
    yo = max(0, (full_size - h) // 2)
    xo = max(0, (full_size - w) // 2)
    ph = min(h, full_size - yo)
    pw = min(w, full_size - xo)
    canvas[yo:yo+ph, xo:xo+pw] = crop[:ph, :pw]
    _, jpeg_buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_buf.tobytes()).decode('utf-8')

    return jsonify({
        'success': True,
        'image': img_b64,
        'status': {
            'state': 'FOUND', 'found': True, 'source': f'SIFT_GLOBAL_{source_tag}',
            'position': {'x': int(tx), 'y': int(ty)}, 'matches': int(inlier_count),
            'match_quality': float(quality), 'mode': 'sift',
        }
    })


@app.route('/api/status')
def get_status():
    """获取当前追踪状态"""
    tracker_obj = _get_tracker_for_http_request()
    return jsonify(tracker_obj.latest_status)


@app.route('/api/reset', methods=['POST'])
def force_reset():
    """强制重置追踪器状态，清空引擎/平滑器/圆校准器所有状态，防止死锁和停滞。"""
    tracker_obj = _get_tracker_for_http_request()
    cleared = tracker_obj.full_reset()
    return jsonify({'ok': True, 'cleared': cleared})


@app.route('/api/map_info')
def get_map_info():
    """返回大地图尺寸。"""
    tracker_obj = _get_tracker_for_http_request()
    return jsonify({
        'map_width': tracker_obj.map_width,
        'map_height': tracker_obj.map_height,
    })


@app.route('/api/detect_minimap_circle', methods=['POST'])
def detect_minimap_circle_api():
    """
    从上传的全屏截图中自动检测小地图圆形位置，返回归一化坐标供前端更新 selCircle。
    前端"自动定位小地图"按钮调用此接口。
    """
    from backend.tracking.autodetect import detect_minimap_circle

    img = _decode_image_from_request()
    if img is None:
        return jsonify({'ok': False, 'error': '无法解析图片'}), 400

    h, w = img.shape[:2]
    if max(h, w) < 200:
        return jsonify({'ok': False, 'error': '图片太小，请上传全屏截图'}), 400

    det = detect_minimap_circle(img)
    if det is None:
        return jsonify({'ok': False, 'error': '未检测到小地图圆形，请确保游戏小地图可见'})

    return jsonify({
        'ok': True,
        'cx': round(det['cx'], 6),
        'cy': round(det['cy'], 6),
        'r':  round(det['r'],  6),
        'confidence': det['confidence'],
        'layout': det.get('layout', ''),
        'frame_width': det['frame_width'],
        'frame_height': det['frame_height'],
    })


# ==================== 地图资源数据 API ====================

_MAP_BUILDER_DIR = os.path.join(_BASE_DIR, 'map_builder')


def _json_with_cache(payload: dict, *, max_age: int = 0, immutable: bool = False):
    response = jsonify(payload)
    if max_age > 0:
        cache_control = f'public, max-age={max_age}'
        if immutable:
            cache_control += ', immutable'
        response.headers['Cache-Control'] = cache_control
    else:
        response.headers['Cache-Control'] = 'no-store'
    return response


def _has_version_query() -> bool:
    return bool((request.args.get('v') or '').strip())


@app.route('/api/markers/manifest')
def api_markers_manifest():
    """返回分块加载所需的标记点清单。"""
    force_reload = (request.args.get('refresh') or '').strip() == '1'
    return _json_with_cache(get_marker_manifest(force_reload=force_reload), max_age=30)


@app.route('/api/markers/chunks')
def api_marker_chunks():
    """按 chunk key 返回指定区块的轻量点位数据。"""
    raw_keys = (request.args.get('keys') or '').split(',')
    payload = get_marker_chunks(raw_keys)
    if not payload['requestedKeys']:
        return _json_with_cache({'success': False, 'error': 'keys 不能为空'}, max_age=0), 400
    return _json_with_cache(payload, max_age=(31536000 if _has_version_query() else 0), immutable=_has_version_query())


@app.route('/api/markers/details')
def api_marker_details():
    """按 id 批量返回标记点详情。"""
    raw_ids = (request.args.get('ids') or '').split(',')
    payload = get_marker_details(raw_ids)
    if not payload['requestedIds']:
        return _json_with_cache({'success': False, 'error': 'ids 不能为空'}, max_age=0), 400
    return _json_with_cache(payload, max_age=(31536000 if _has_version_query() else 0), immutable=_has_version_query())


@app.route('/api/markers/search_index')
def api_marker_search_index():
    """返回延迟加载的搜索索引（标题/描述/坐标）。"""
    return _json_with_cache(get_marker_search_index(), max_age=(31536000 if _has_version_query() else 0), immutable=_has_version_query())

@app.route('/api/markers')
def api_markers():
    """返回精简版标记点数据 (id, markType, x, y)"""
    return send_from_directory(_MAP_BUILDER_DIR, 'rocom_markers_lite.json')

@app.route('/api/markers/detail')
def api_markers_detail():
    """返回标记点详情 (title, description)，按 id 索引"""
    return send_from_directory(_MAP_BUILDER_DIR, 'rocom_markers_detail.json')

@app.route('/api/categories')
def api_categories():
    """返回分类字典 (id -> {name, group})"""
    return send_from_directory(_MAP_BUILDER_DIR, 'categories.json')


@app.route('/api/config', methods=['GET', 'POST'])
def api_config_runtime():
    """读取/更新后端配置。只开放热更新安全子集，其他配置只读展示。"""
    if request.method == 'GET':
        return jsonify(_build_config_payload())

    data = request.get_json(silent=True) or {}
    updates = data.get('updates') if isinstance(data, dict) else None
    if not isinstance(updates, dict) or not updates:
        return jsonify({'success': False, 'error': 'updates 必须是非空对象'}), 400

    approved, rejected = _validate_runtime_config_updates(updates)
    if approved:
        _apply_runtime_config_updates(approved)

    payload = _build_config_payload()
    payload.update({
        'success': not rejected,
        'partial': bool(approved and rejected),
        'applied': {key: _serialize_config_value(value) for key, value in approved.items()},
        'rejected': rejected,
    })
    return jsonify(payload), (200 if not rejected else 400)


@app.route('/api/result')
def get_result():
    """获取最新的结果图片"""
    tracker_obj = _get_tracker_for_http_request()
    image = tracker_obj.get_latest_result_base64()
    if image:
        return jsonify({
            'image': image,
            'status': tracker_obj.latest_status,
        })
    return jsonify({'error': 'No result yet'}), 404


@app.route('/api/latest_frame')
def get_latest_frame():
    """获取最新渲染的地图帧（JPEG 二进制）- 供外部悬浮窗使用"""
    tracker_obj = _get_tracker_for_http_request()
    jpeg_bytes = tracker_obj.get_latest_result_jpeg()
    if jpeg_bytes:
        return send_file(
            BytesIO(jpeg_bytes),
            mimetype='image/jpeg',
        )
    return jsonify({'error': 'No frame available yet'}), 404


@app.route('/api/process')
def process():
    """手动触发一次处理（用于文件模式）"""
    tracker_obj = _get_tracker_for_http_request()
    result = tracker_obj.process_frame(need_base64=True, need_jpeg=False)
    if result and result[0]:
        return jsonify({
            'image': result[0],
            'status': tracker_obj.latest_status,
        })
    return jsonify({'error': 'No minimap set'}), 400


# ==================== 路线 API ====================

@app.route('/api/routes')
def api_list_routes():
    """列出所有可用路线文件"""
    files = get_route_files(_ROUTES_DIR)
    routes = []
    for f in files:
        data = load_route_data(_ROUTES_DIR, f)
        if data:
            routes.append({
                'filename': f,
                'name': data.get('name', f),
                'point_count': len(data.get('points', [])),
            })
    return jsonify(routes)


@app.route('/api/routes/<path:filename>')
def api_get_route(filename):
    """获取单个路线的完整数据"""
    # 安全检查：防止路径穿越
    safe_name = os.path.basename(filename)
    if not safe_name.endswith('.json'):
        return jsonify({'error': 'Only JSON files allowed'}), 400
    data = load_route_data(_ROUTES_DIR, safe_name)
    if data is None:
        return jsonify({'error': 'Route not found'}), 404
    data['filename'] = safe_name
    return jsonify(data)


# ==================== WebSocket 端点 ====================

@socketio.on('connect')
def ws_connect():
    """客户端建立 WS 连接"""
    print(f"WebSocket 客户端已连接: {request.sid}")
    tracker_obj = _get_tracker_for_token(_DEFAULT_SESSION_TOKEN)
    emit('status', tracker_obj.latest_status)


@socketio.on('session_join')
def ws_session_join(data):
    """绑定识别会话 token，并将连接加入对应房间。"""
    token = _normalize_session_token((data or {}).get('token') if isinstance(data, dict) else '')
    if not token:
        return

    old = _bind_sid_token(request.sid, token)
    if old and old != token:
        try:
            leave_room(_session_room(old))
        except Exception:
            pass
    join_room(_session_room(token))
    tracker_obj = _get_tracker_for_token(token)
    emit('status', tracker_obj.latest_status)

    # 若客户端先发了 frame 再 session_join，补绑 push 目标
    _jpeg_mgr.register_frame_client(request.sid, token)


@socketio.on('disconnect')
def ws_disconnect():
    old_token = _coerce_session_token(_unbind_sid_token(request.sid))

    # JPEG 模式：从管理器清理 SID（Plan A/B 状态），若无消费者则关闭后台编码
    _jpeg_mgr.unregister_client(request.sid)
    tracker_obj = _get_tracker_for_token(old_token)
    tracker_obj._push_jpeg = _jpeg_mgr.has_jpeg_clients(old_token)

    # 展示室清理：展示者断线 → 删除房间并通知观众；观众断线 → 从订阅集合移除
    with _bcast_lock:
        # 展示者断线
        dead_rooms = [n for n, i in _bcast_rooms.items() if i['sid'] == request.sid]
        for name in dead_rooms:
            room = _bcast_rooms.pop(name)
            for vsid in list(room['viewers']):
                try:
                    socketio.emit('broadcast_ended', {'name': name}, to=vsid)
                except Exception:
                    pass
        # 观众断线
        for info in _bcast_rooms.values():
            info['viewers'].discard(request.sid)

    if dead_rooms:
        with _bcast_lock:
            rooms = [{'name': n, 'viewers': len(i['viewers'])} for n, i in _bcast_rooms.items()]
        socketio.emit('broadcast_list', {'rooms': rooms})

    print(f"WebSocket 客户端断开: {request.sid}")


@socketio.on('request_jpeg')
def ws_request_jpeg():
    """前端切换到 JPEG 渲染模式时触发，强制下次 _on_result_ready 必须推送 result+JPEG。"""
    _jpeg_mgr.force_next_jpeg(_coerce_session_token(_get_sid_token(request.sid)))


@socketio.on('frame')
def ws_receive_frame(raw_bytes):
    """
    JPEG 模式专用：接收二进制 JPEG 帧，存帧并立即返回缓存坐标（pull 兜底）。
    SIFT 运行在后台线程，完成后由 _on_result_ready 主动推送新结果（Plan A）。
    协议: 客户端发送原始 JPEG bytes → 服务端返回 [JSON头(4字节长度) + JSON]
    """
    if len(raw_bytes) > _WS_FRAME_MAX_BYTES:
        return  # 拒绝超大帧，防止内存耗尽

    token = _coerce_session_token(_get_sid_token(request.sid))
    tracker_obj = _get_tracker_for_token(token)

    # Plan B + Plan A：注册客户端，开启后台 JPEG 编码，并设为 push 目标
    _jpeg_mgr.register_frame_client(request.sid, token)
    tracker_obj._push_jpeg = True

    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker_obj.set_minimap(img, token=token)  # 存帧并唤醒后台 SIFT 线程，结果由 _on_result_ready 定向推送
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err)


@socketio.on('frame_coords')
def ws_frame_coords(raw_bytes):
    """
    接收帧并存帧（后台处理），立即返回上一帧缓存坐标。大幅减少带宽。
    Plan B: 无 'frame' 客户端时设 _push_jpeg=False，后台线程跳过 JPEG 编码。
    """
    if len(raw_bytes) > _WS_FRAME_MAX_BYTES:
        return  # 拒绝超大帧，防止内存耗尽

    token = _coerce_session_token(_get_sid_token(request.sid))
    tracker_obj = _get_tracker_for_token(token)

    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        # Plan B: 当前无 JPEG 消费者时关闭编码，节省 10-15ms/帧
        if not _jpeg_mgr.has_jpeg_clients(token):
            tracker_obj._push_jpeg = False

        tracker_obj.set_minimap(img, token=token)  # 存帧并唤醒后台 SIFT 线程，结果由 _on_result_ready 定向推送
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err)


@socketio.on('broadcast_list')
def ws_bcast_list():
    """返回当前活跃展示者列表给调用者（观众模式用）。"""
    with _bcast_lock:
        rooms = [
            {'name': name, 'viewers': len(info['viewers'])}
            for name, info in _bcast_rooms.items()
        ]
    emit('broadcast_list', {'rooms': rooms})


@socketio.on('broadcast_join')
def ws_bcast_join(data):
    """
    展示者加入展示室（data = {name: str}）。
    同名已存在且原展示者 sid 不同时拒绝（防止抢占）。
    """
    name = (data.get('name') or '').strip()[:32] if isinstance(data, dict) else ''
    if not name:
        emit('broadcast_joined', {'ok': False, 'error': '昵称不能为空'})
        return
    with _bcast_lock:
        existing = _bcast_rooms.get(name)
        if existing and existing['sid'] != request.sid:
            emit('broadcast_joined', {'ok': False, 'error': '该昵称已被他人使用'})
            return
        _bcast_rooms[name] = {
            'sid': request.sid,
            'viewers': existing['viewers'] if existing else set(),
            'last_ts': 0.0,
            'last_frame': None,
        }
    # 通知所有人列表更新
    with _bcast_lock:
        rooms = [{'name': n, 'viewers': len(i['viewers'])} for n, i in _bcast_rooms.items()]
    socketio.emit('broadcast_list', {'rooms': rooms})
    emit('broadcast_joined', {'ok': True, 'name': name})
    print(f"[bcast] {name}({request.sid}) 已加入展示室")


@socketio.on('broadcast_leave')
def ws_bcast_leave(data):
    """展示者离开展示室（或踢出）。"""
    name = (data.get('name') or '').strip() if isinstance(data, dict) else ''
    with _bcast_lock:
        room = _bcast_rooms.get(name)
        if room and room['sid'] == request.sid:
            # 通知所有观众展示者已离线
            for vsid in list(room['viewers']):
                try:
                    socketio.emit('broadcast_ended', {'name': name}, to=vsid)
                except Exception:
                    pass
            del _bcast_rooms[name]
    with _bcast_lock:
        rooms = [{'name': n, 'viewers': len(i['viewers'])} for n, i in _bcast_rooms.items()]
    socketio.emit('broadcast_list', {'rooms': rooms})


@socketio.on('broadcast_watch')
def ws_bcast_watch(data):
    """观众订阅展示者（data = {name: str}）。"""
    name = (data.get('name') or '').strip() if isinstance(data, dict) else ''
    with _bcast_lock:
        room = _bcast_rooms.get(name)
        if not room:
            emit('broadcast_watch_ack', {'ok': False, 'error': '展示者不存在或已离线'})
            return
        room['viewers'].add(request.sid)
        rooms = [{'name': n, 'viewers': len(i['viewers'])} for n, i in _bcast_rooms.items()]
    socketio.emit('broadcast_list', {'rooms': rooms})
    emit('broadcast_watch_ack', {'ok': True, 'name': name})
    print(f"[bcast] 观众 {request.sid} 订阅 {name}")


@socketio.on('broadcast_unwatch')
def ws_bcast_unwatch(data):
    """观众取消订阅。"""
    name = (data.get('name') or '').strip() if isinstance(data, dict) else ''
    changed = False
    with _bcast_lock:
        room = _bcast_rooms.get(name)
        if room:
            room['viewers'].discard(request.sid)
            changed = True
            rooms = [{'name': n, 'viewers': len(i['viewers'])} for n, i in _bcast_rooms.items()]
    if changed:
        socketio.emit('broadcast_list', {'rooms': rooms})


@socketio.on('broadcast_frame')
def ws_bcast_frame(raw_bytes):
    """
    展示者推送 JPEG 帧。服务端节流（10fps），把帧广播给所有订阅观众。
    协议：{name, frame}，frame 为 JPEG 二进制；name 用于观众端精确路由到对应 tile。

    节流逻辑：通道内每 100ms 最多发一帧。节流期间收到的帧暂存为 last_frame，
    并调度一次"冲刷任务"，在间隔结束后把 last_frame 补发出去，
    保证展示者最后一帧一定能到达观众（防止永久丢帧）。
    """
    if not raw_bytes:
        return
    # 找出该 sid 对应的房间
    with _bcast_lock:
        room_name = None
        room = None
        for name, info in _bcast_rooms.items():
            if info['sid'] == request.sid:
                room_name = name
                room = info
                break
        if room is None:
            return  # 未注册为展示者，忽略

        now = time.monotonic()
        if now - room['last_ts'] < _BCAST_MIN_INTERVAL:
            room['last_frame'] = bytes(raw_bytes)  # 暂存最新帧
            # 若尚无冲刷任务在路上，调度一个在间隔结束后补发
            if not room.get('flush_scheduled'):
                room['flush_scheduled'] = True
                delay = _BCAST_MIN_INTERVAL - (now - room['last_ts'])

                def _flush(rname=room_name):
                    with _bcast_lock:
                        r = _bcast_rooms.get(rname)
                        if not r:
                            return
                        r['flush_scheduled'] = False
                        frame = r.get('last_frame')
                        if not frame or not r['viewers']:
                            return
                        r['last_ts'] = time.monotonic()
                        viewers_snap = list(r['viewers'])
                    dead = set()
                    for vsid in viewers_snap:
                        try:
                            socketio.emit('broadcast_frame', {'name': rname, 'frame': frame}, to=vsid)
                        except Exception:
                            dead.add(vsid)
                    if dead:
                        with _bcast_lock:
                            rv = _bcast_rooms.get(rname)
                            if rv:
                                rv['viewers'] -= dead

                socketio.start_background_task(lambda d=delay, f=_flush: (
                    __import__('time').sleep(d), f()
                ))
            return

        room['last_ts'] = now
        room['last_frame'] = bytes(raw_bytes)
        room['flush_scheduled'] = False
        viewers = list(room['viewers'])

    if not viewers:
        return

    # 向每个存活的观众 emit（失败则清理）
    dead = set()
    for vsid in viewers:
        try:
            socketio.emit('broadcast_frame', {'name': room_name, 'frame': raw_bytes}, to=vsid)
        except Exception:
            dead.add(vsid)

    if dead:
        with _bcast_lock:
            r = _bcast_rooms.get(room_name)
            if r:
                r['viewers'] -= dead


# ==================== 启动入口 ====================

# ---- 多用户/生产部署（gunicorn）说明 ----
# 单用户直接: python main_web.py cpu
# 多用户 6 核: MAP_TRACKER_MODE=sift SOCKETIO_ASYNC_MODE=gevent \
#              gunicorn -w 4 --threads 2 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
#              --preload "main_web:app"
# 多用户 16 核: MAP_TRACKER_MODE=sift SOCKETIO_ASYNC_MODE=gevent \
#               gunicorn -w 12 --threads 2 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
#              --preload "main_web:app"
# --preload 保证 SIFT 特征点/FLANN 索引只构建一次，fork 后各 worker 共享只读内存（Linux CoW）
# SocketIO 需要 geventwebsocket: pip install gevent gevent-websocket
# 多进程时建议通过环境变量限制每进程 cv2 线程数: CV2_NUM_THREADS=2 gunicorn ...


def main():
    """\u63a7\u5236\u53f0\u5165\u53e3\uff1a\u4e0e pyproject.toml \u7684 console_scripts \u5bf9\u9f50\u3002"""
    print("=" * 50)
    print("  \u5730\u56fe\u8ddf\u70b9 - \u7f51\u9875\u7248 [SIFT]")
    print(f"  SocketIO async_mode: {_SOCKETIO_ASYNC_MODE}")
    print(f"  SocketIO allow_upgrades: {_SOCKETIO_ALLOW_UPGRADES}")
    _log_socketio_runtime_diagnostics()
    print("  \u6253\u5f00\u6d4f\u89c8\u5668\u8bbf\u95ee: http://0.0.0.0:" + str(config.PORT))
    print("  WebSocket: ws://0.0.0.0:" + str(config.PORT) + "/socket.io/?transport=websocket")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=config.PORT, debug=False, allow_unsafe_werkzeug=True)
