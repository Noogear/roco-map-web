"""
server.py - Web 控制层（Flask + SocketIO 路由）

职责:
  - Flask / SocketIO 应用初始化
  - HTTP API 路由（上传/状态/控制）
  - WebSocket 消息处理
  - 圆形选区持久化

识别引擎和编排逻辑已拆分到:
  backend/tracker_engines.py   FeatureMapTracker (纯识别)
  backend/tracker_core.py      MapTrackerWeb (平滑/过滤/渲染编排)
"""

import sys
import os
import time
import struct
import threading
import importlib.util
from pathlib import Path
from importlib import metadata as importlib_metadata

import cv2
import numpy as np
import base64
from io import BytesIO

from backend import config
from flask import Flask, jsonify, redirect, request, send_file, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_compress import Compress
from backend.tracker_core import MapTrackerWeb
try:
    # 新版特征引擎入口
    from backend.tracker_engines import get_shared_feature
except ImportError:
    # 兼容测试替身/旧实现：仅暴露 get_shared_sift 的场景
    from backend.tracker_engines import get_shared_sift as get_shared_feature
from backend.core.context import SessionRegistry
from backend.core.data_standards import DataScope, audit_tracker_scope
from backend.core.fastjson import OrjsonProvider, dumps_bytes
from backend.core.config_runtime import (
    apply_runtime_config_command,
    build_config_payload,
)
from backend.map_data import (
    get_marker_chunks,
    get_marker_details,
    get_marker_manifest,
    get_marker_search_index,
)
from backend.store import get_route_files, load_route_data
from backend.frontend_build import load_active_manifest, resolve_preferred_static_root
from backend.core.recognize_single_fallback import build_hash_ecc_or_hash_fallback_status

# 项目目录路径（backend/ 的上一级）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(_BASE_DIR, 'frontend')
ASSETS_DIR   = os.path.join(_BASE_DIR, 'assets')
FRONTEND_BUILD_DIR = os.environ.get('FRONTEND_BUILD_DIR', os.path.join(_BASE_DIR, 'frontend_build'))


# ==================== 路线文件管理 ====================

_ROUTES_DIR = os.path.join(_BASE_DIR, 'routes')


def _get_frontend_static_roots() -> list[str]:
    roots = []
    if _FRONTEND_STATIC_ROOT is not None:
        roots.append(str(_FRONTEND_STATIC_ROOT))
    roots.append(FRONTEND_DIR)
    return roots


def _resolve_frontend_static_path(filename: str):
    safe_filename = str(filename or '').lstrip('/\\')
    for root in _get_frontend_static_roots():
        candidate = os.path.join(root, safe_filename)
        if os.path.isfile(candidate):
            return candidate
    return None


_FRONTEND_BUILD_MANIFEST = load_active_manifest(Path(_BASE_DIR))
_FRONTEND_STATIC_ROOT = resolve_preferred_static_root(Path(_BASE_DIR))
if _FRONTEND_BUILD_MANIFEST and _FRONTEND_STATIC_ROOT is not None:
    print(f"[frontend-build] 已检测到可用构建产物目录: {_FRONTEND_STATIC_ROOT}")
else:
    print('[frontend-build] 未检测到可用 active 构建产物，静态资源将回退到 frontend/ 源码目录。')


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
_command_listener_started = False
_command_listener_lock = threading.Lock()
_CONFIG_HTTP_READONLY_ERROR = '出于安全与稳定性考虑，后端配置仅允许在服务端后台通过指令热更新，或直接修改 config.py。'


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


def _sleep_for_broadcast_flush(delay: float) -> None:
    """展示流冲刷延时：优先使用 SocketIO 的协作式 sleep，失败时回退到 time.sleep。"""
    delay = max(0.0, float(delay))
    try:
        socketio.sleep(delay)
    except Exception:
        time.sleep(delay)


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
    """提供静态资源：先尝试 active 构建产物，再回退 frontend/ 和 assets/。"""
    frontend_path = _resolve_frontend_static_path(filename)
    if frontend_path and os.path.isfile(frontend_path):
        return send_file(frontend_path)
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
    from backend.core.features import extract_minimap_features, match_region, CircularMaskCache
    from backend.core.enhance import (
        enhance_minimap, make_clahe_pair, correct_color_temperature,
        classify_scene_by_color, make_scene_boosted_gray,
    )
    from backend.tracking.minimap import detect_and_extract

    img = _decode_image_from_request()
    if img is None:
        return jsonify({'error': 'No image data'}), 400

    shared = get_shared_feature()
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

    # 2. 预处理：色温补偿 + 场景细化 + CLAHE 增强
    img = correct_color_temperature(img)
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_std = float(np.std(gray_raw))

    # 与主链一致的粗场景分类
    if texture_std < 15:
        scene = 'ocean'
    elif texture_std < 35:
        scene = 'low_texture'
    elif texture_std < 55:
        scene = 'mixed'
    else:
        scene = 'urban'

    scene_detail = classify_scene_by_color(img, scene)
    if (scene_detail == 'grassland'
            and bool(getattr(config, 'SCENE_COLOR_FORCE_LOW_TEXTURE_GRASSLAND', True))
            and scene in ('mixed', 'low_texture')
            and texture_std <= float(getattr(config, 'SCENE_COLOR_FORCE_LOW_TEXTURE_STD_MAX', 36.0))):
        scene = 'low_texture'
    elif scene_detail == 'snow' and scene in ('mixed', 'low_texture'):
        scene = 'low_texture'

    low_texture_like = (
        scene in ('ocean', 'low_texture')
        or scene_detail == 'grassland'
        or texture_std < float(getattr(config, 'LOW_TEXTURE_LIKE_STD_THRESHOLD', 40.0))
    )

    sift_input = gray_raw
    if bool(getattr(config, 'SCENE_BOOSTED_GRAY_ENABLED', True)) and scene_detail in ('ocean', 'grassland', 'snow'):
        boosted = make_scene_boosted_gray(img, scene_detail)
        if boosted is not None:
            sift_input = boosted

    clahe_normal, clahe_low = make_clahe_pair()
    low_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
    gray = enhance_minimap(sift_input, texture_std, clahe_normal, clahe_low, low_thresh)

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

    # 4. 全局匹配（与主链策略保持一致：高纹理可选 LSH/GMS）
    match_ratio = getattr(config, 'FEATURE_MATCH_RATIO', 0.82)
    min_match = getattr(config, 'FEATURE_MIN_MATCH_COUNT', 5)

    use_lsh = (
        bool(getattr(config, 'MATCHER_POLICY_ENABLE', True))
        and bool(getattr(config, 'MATCHER_GLOBAL_USE_LSH', True))
        and (not low_texture_like)
        and (getattr(shared, 'flann_global_lsh', None) is not None)
        and len(kp_mini) >= int(getattr(config, 'MATCHER_LSH_MIN_KP', 120))
    )
    use_gms = (
        bool(getattr(config, 'MATCHER_POLICY_ENABLE', True))
        and bool(getattr(config, 'MATCHER_GMS_ENABLE', True))
        and (not low_texture_like)
        and len(kp_mini) >= int(getattr(config, 'MATCHER_GMS_MIN_KP', 140))
    )

    global_matcher = shared.flann_global_lsh if use_lsh else shared.flann_global
    result = match_region(
        kp_mini, des_mini, gray.shape,
        shared.kp_big_all, global_matcher,
        match_ratio, min_match,
        shared.map_width, shared.map_height,
        use_gms=use_gms,
        gms_train_shape=(shared.map_height, shared.map_width),
        gms_min_matches=int(getattr(config, 'MATCHER_GMS_MIN_MATCHES', 20)),
        gms_with_rotation=bool(getattr(config, 'MATCHER_GMS_WITH_ROTATION', True)),
        gms_with_scale=bool(getattr(config, 'MATCHER_GMS_WITH_SCALE', False)))

    if result is None:
        # 尝试宽松参数
        result = match_region(
            kp_mini, des_mini, gray.shape,
            shared.kp_big_all, global_matcher,
            0.88, 3,
            shared.map_width, shared.map_height,
            use_gms=use_gms,
            gms_train_shape=(shared.map_height, shared.map_width),
            gms_min_matches=int(getattr(config, 'MATCHER_GMS_MIN_MATCHES', 20)),
            gms_with_rotation=bool(getattr(config, 'MATCHER_GMS_WITH_ROTATION', True)),
            gms_with_scale=bool(getattr(config, 'MATCHER_GMS_WITH_SCALE', False)))

    if result is None:
        fallback_status = build_hash_ecc_or_hash_fallback_status(
            shared=shared,
            hash_index=hash_index,
            gray=gray,
            gray_raw=gray_raw,
            source_suffix=source_tag,
        )
        if fallback_status is not None:
            return jsonify({'success': True, 'status': fallback_status})
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
    min_inliers = int(getattr(config, 'FEATURE_MIN_MATCH_COUNT', 5))
    min_quality = float(getattr(config, 'SINGLE_RECOGNIZE_MIN_QUALITY', 0.12))
    feature_confident = (int(inlier_count) >= max(3, min_inliers)) and (float(quality) >= min_quality)
    # 哈希索引是粗校验，不宜压过“非常强”的特征匹配命中（避免误拒真阳性）。
    feature_very_strong = (int(inlier_count) >= 8) and (float(quality) >= 0.55)

    hash_consistent = True
    if hash_index is not None:
        try:
            hash_consistent = bool(hash_index.check_consistency(gray_raw, int(tx), int(ty)))
        except Exception:
            hash_consistent = True

    if (not feature_confident) or ((not hash_consistent) and (not feature_very_strong)):
        return jsonify({
            'success': False,
            'status': {
                'state': 'LOW_CONFIDENCE', 'found': False,
                'position': {'x': 0, 'y': 0}, 'matches': int(inlier_count),
                'match_quality': float(quality), 'mode': 'sift',
                'source': f'ORB_GLOBAL_{source_tag}',
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

    source_parts = ['ORB_GLOBAL']
    if use_lsh:
        source_parts.append('LSH')
    if use_gms:
        source_parts.append('GMS')
    source_name = '_'.join(source_parts) + f'_{source_tag}'

    return jsonify({
        'success': True,
        'image': img_b64,
        'status': {
            'state': 'FOUND', 'found': True, 'source': source_name,
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
    from backend.tracking.autodetect import detect_minimap_circle, detect_minimap_circle_batch

    images = _decode_images_from_request(max_count=8)
    if not images:
        return jsonify({'ok': False, 'error': '无法解析图片'}), 400

    img0 = images[0]
    h, w = img0.shape[:2]
    if max(h, w) < 200:
        return jsonify({'ok': False, 'error': '图片太小，请上传全屏截图'}), 400

    det = detect_minimap_circle_batch(images) if len(images) > 1 else detect_minimap_circle(img0)
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
    """读取后端配置（前端只读）。"""
    if request.method == 'GET':
        return jsonify(build_config_payload())

    return jsonify({
        'success': False,
        'error': _CONFIG_HTTP_READONLY_ERROR,
    }), 403


def _run_server_command_listener():
    """服务端后台命令监听：仅允许在服务器控制台输入配置热更新指令。"""
    print('[config-cmd] 已启用服务端后台配置指令。输入 help 查看用法。')
    while True:
        try:
            line = sys.stdin.readline()
        except Exception as exc:
            print(f'[config-cmd] 读取命令失败: {exc}')
            return

        if line == '':
            # stdin 关闭/不可用，结束监听线程
            print('[config-cmd] stdin 不可用，已停止命令监听。')
            return

        command = line.strip()
        if not command:
            continue

        if command.lower() in ('quit', 'exit'):
            print('[config-cmd] 已忽略 quit/exit（不会停止服务）。')
            continue

        result = apply_runtime_config_command(command, _session_registry)
        if result.get('success'):
            applied = result.get('applied') or {}
            if applied:
                print(f"[config-cmd] 已应用: {applied}")
            else:
                msg = result.get('message') or '执行成功'
                print(f"[config-cmd] {msg}")
            rejected = result.get('rejected') or {}
            if rejected:
                print(f"[config-cmd] 已拒绝: {rejected}")
        else:
            print(f"[config-cmd] 失败: {result.get('error', result)}")


def _start_server_command_listener_if_possible():
    global _command_listener_started
    with _command_listener_lock:
        if _command_listener_started:
            return
        stdin_obj = getattr(sys, 'stdin', None)
        if stdin_obj is None or not hasattr(stdin_obj, 'readline'):
            print('[config-cmd] 未检测到可读 stdin，跳过后台命令监听。')
            _command_listener_started = True
            return
        t = threading.Thread(
            target=_run_server_command_listener,
            daemon=True,
            name='config-command-listener',
        )
        t.start()
        _command_listener_started = True


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
    frame_bytes = bytes(raw_bytes)
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
            room['last_frame'] = frame_bytes  # 暂存最新帧
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

                def _delayed_flush(d=delay, f=_flush):
                    _sleep_for_broadcast_flush(d)
                    f()

                socketio.start_background_task(_delayed_flush)
            return

        room['last_ts'] = now
        room['last_frame'] = frame_bytes
        room['flush_scheduled'] = False
        viewers = list(room['viewers'])

    if not viewers:
        return

    # 向每个存活的观众 emit（失败则清理）
    dead = set()
    for vsid in viewers:
        try:
            socketio.emit('broadcast_frame', {'name': room_name, 'frame': frame_bytes}, to=vsid)
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
    _start_server_command_listener_if_possible()
    print("  \u6253\u5f00\u6d4f\u89c8\u5668\u8bbf\u95ee: http://0.0.0.0:" + str(config.PORT))
    print("  WebSocket: ws://0.0.0.0:" + str(config.PORT) + "/socket.io/?transport=websocket")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=config.PORT, debug=False, allow_unsafe_werkzeug=True)


