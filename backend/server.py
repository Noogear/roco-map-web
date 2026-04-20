"""
server.py - Flask + Flask-SocketIO 入口

只负责：HTTP/WS 路由注册、事件绑定、静态文件服务。
业务逻辑由 services 层承担：
  - SessionManager   会话注册表
  - BroadcastManager 广播频道
  - frame_processor  帧解码 / 匹配 / 编包
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_compress import Compress
from flask_socketio import SocketIO, emit, join_room, leave_room

from backend.common.runtime.fastjson import OrjsonProvider
from backend.transport.session.broadcast_manager import BroadcastManager
from backend.transport.session.frame_processor import (
    build_binary_result,
    decode_blob,
    make_status,
    process_frame,
    process_frame_coords,
)
from backend.web.io.input import decode_image_from_request
from backend.web.api.recognize import analyze_image as _analyze_image
from backend.transport.session.session_manager import SessionManager
from backend.map.autodetect import detect_minimap_circle

# ── 异步模式 ──────────────────────────────────────────────────────────
_ASYNC_MODE = os.environ.get('SOCKETIO_ASYNC_MODE', 'gevent')
_ALLOW_UPGRADES = os.environ.get('SOCKETIO_ALLOW_UPGRADES', 'true').lower() != 'false'

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.json = OrjsonProvider(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
Compress(app)

socketio = SocketIO(
    app,
    async_mode=_ASYNC_MODE,
    cors_allowed_origins='*',
    allow_upgrades=_ALLOW_UPGRADES,
    binary=True,
    ping_timeout=60,
    ping_interval=25,
)

# ── 静态文件根目录 ────────────────────────────────────────────────────
_BUILD_DIR = FRONTEND_BUILD_ACTIVE_DIR
_FRONT_DIR = FRONTEND_DIR
_ASSETS_DIR = ASSETS_ROOT_DIR

# ── 服务单例 ──────────────────────────────────────────────────────────
_sess = SessionManager()
_bcast = BroadcastManager()

# ── debug_crop 节流计数器（按 token 计帧，每 DEBUG_CROP_INTERVAL 帧推一次）──
_DEBUG_CROP_INTERVAL = 5
_debug_crop_counters: dict[str, int] = {}
_debug_crop_lock = __import__('threading').Lock()


# ══════════════════════════════════════════════════════════════════════
#  静态文件
# ══════════════════════════════════════════════════════════════════════

def _serve_page(html_filename: str):
    """按优先级查找并返回 HTML 页面：frontend_build/active → frontend/。"""
    if (_BUILD_DIR / html_filename).exists():
        return send_from_directory(str(_BUILD_DIR), html_filename)
    return send_from_directory(str(_FRONT_DIR), html_filename)


@app.route('/')
@app.route('/recognize')
def recognize_page():
    return _serve_page('recognize.html')


@app.route('/map')
def map_page():
    return _serve_page('map.html')


@app.route('/settings')
def settings_page():
    return _serve_page('settings.html')


@app.route('/assets/<path:filename>')
def assets_files(filename: str):
    """assets 目录静态资源（地图/缓存等）。"""
    return send_from_directory(str(_ASSETS_DIR), filename)


@app.route('/<path:filename>')
def static_files(filename: str):
    """静态资源：先从 build 目录，再回退到 frontend/ 源目录。"""
    candidate = _BUILD_DIR / filename
    if candidate.exists():
        return send_from_directory(str(_BUILD_DIR), filename)
    return send_from_directory(str(_FRONT_DIR), filename)


# ══════════════════════════════════════════════════════════════════════
#  REST API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """简单无状态单帧识别（向后兼容）。"""
    img = decode_image_from_request(request)
    if img is None:
        return jsonify({'error': 'No image data'}), 400
    return jsonify(_analyze_image(img))


@app.route('/api/recognize_single', methods=['POST'])
def api_recognize_single():
    """单帧识别（截图工具用），返回 {status: {...}} 格式。"""
    img = decode_image_from_request(request)
    if img is None:
        return jsonify({'error': 'No image data'}), 400
    result = _analyze_image(img)
    found = bool(result.get('success', False))
    pos = result.get('position', {'x': 0, 'y': 0})
    return jsonify({
        'status': {
            'mode': 'orb',
            'state': 'FOUND' if found else 'LOST',
            'position': pos,
            'found': found,
            'matches': int(result.get('matches', 0)),
            'match_quality': float(result.get('match_quality', 0.0)),
            'arrow_angle': 0.0,
            'arrow_stopped': False,
            'coord_lock': False,
            'source': str(result.get('source', '')),
            'frozen': False,
        }
    })


@app.route('/api/upload_minimap', methods=['POST'])
def api_upload_minimap():
    """HTTP 上传小地图帧（Socket.IO 不可用时的回退）。"""
    img = decode_image_from_request(request)
    if img is None:
        return jsonify({'error': 'No image data'}), 400
    token = ''
    if request.is_json:
        token = (request.get_json(silent=True) or {}).get('token', '')
    tracker = _sess.get_or_create(token or 'http_default')
    result = tracker.feature_engine.match(img)
    found = bool(result.get('found', False))
    pos = {'x': int(result.get('center_x') or 0), 'y': int(result.get('center_y') or 0)}
    return jsonify({
        'status': {
            'mode': 'orb',
            'state': 'FOUND' if found else 'LOST',
            'position': pos,
            'found': found,
            'matches': int(result.get('matches', 0)),
            'match_quality': float(result.get('match_quality', 0.0)),
            'arrow_angle': 0.0,
            'arrow_stopped': False,
            'coord_lock': False,
            'source': str(result.get('source', '')),
            'frozen': False,
        }
    })


@app.route('/api/detect_minimap_circle', methods=['POST'])
def api_detect_minimap_circle():
    """全帧自动检测小地图圆形位置，用于自动校准。"""
    img = decode_image_from_request(request)
    if img is None:
        return jsonify({'ok': False, 'error': 'No image data'}), 400
    det = detect_minimap_circle(img)
    if det is None:
        return jsonify({'ok': False, 'error': '未检测到小地图圆形'})
    return jsonify({
        'ok': True,
        'cx': float(det['cx']),
        'cy': float(det['cy']),
        'r': float(det['r']),
        'confidence': float(det.get('confidence', 0.0)),
        'layout': det.get('layout', 'full'),
    })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置指定 session 的追踪状态。"""
    token = ''
    if request.is_json:
        token = (request.get_json(silent=True) or {}).get('token', '')
    _sess.reset(token or 'http_default')
    return jsonify({'ok': True})


@app.route('/api/test_images', methods=['GET'])
def api_test_images():
    """列出 test_file/ 目录中的图片文件，供测试工具用。"""
    test_dir = Path(__file__).resolve().parent.parent / 'test_file'
    images = []
    if test_dir.exists():
        for f in sorted(test_dir.iterdir()):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                images.append(f'/test_file/{f.name}')
    return jsonify(images)


@app.route('/test_file/<path:filename>')
def serve_test_file(filename: str):
    test_dir = Path(__file__).resolve().parent.parent / 'test_file'
    return send_from_directory(str(test_dir), filename)


# ══════════════════════════════════════════════════════════════════════
#  Socket.IO 事件
# ══════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    _sess.on_connect(request.sid)
    print(f'[WS] connect sid={request.sid}')


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    _sess.on_disconnect(sid)
    _bcast.on_disconnect(sid)
    print(f'[WS] disconnect sid={sid}')


@socketio.on('session_join')
def on_session_join(data):
    """客户端报告 token，绑定会话并预热 tracker。"""
    token = (data or {}).get('token', '') if isinstance(data, dict) else ''
    if token:
        _sess.bind(request.sid, token)


@socketio.on('frame')
def on_frame(data):
    """接收实时截图帧，处理后推送 result（含 JPEG）或 coords 事件。
    每 DEBUG_CROP_INTERVAL 帧推一次 debug_crop（节流，避免缩略图抖动）。
    """
    token = _sess.get_token(request.sid)
    img = decode_blob(data)
    if img is None:
        return
    tracker = _sess.get_or_create(token)
    payload, has_jpeg, debug_info = process_frame(tracker, img, _sess.push_mgr, token)
    emit('result' if has_jpeg else 'coords', payload)
    if debug_info:
        with _debug_crop_lock:
            cnt = _debug_crop_counters.get(token, 0)
            _debug_crop_counters[token] = cnt + 1
        if cnt % _DEBUG_CROP_INTERVAL == 0:
            emit('debug_crop', debug_info)


@socketio.on('frame_coords')
def on_frame_coords(data):
    """接收帧，只返回坐标（大地图页面，节省带宽）。"""
    token = _sess.get_token(request.sid)
    img = decode_blob(data)
    if img is None:
        return
    tracker = _sess.get_or_create(token)
    payload = process_frame_coords(tracker, img)
    emit('coords', payload)


@socketio.on('fullframe_for_calibration')
def on_fullframe_for_calibration(data):
    """接收全帧，自动校准小地图圆形位置。"""
    img = decode_blob(data)
    if img is None:
        return
    det = detect_minimap_circle(img)
    if det is not None:
        emit('calibration_update', {
            'ok': True,
            'cx': float(det['cx']),
            'cy': float(det['cy']),
            'r': float(det['r']),
            'reason': 'auto_detected',
        })
    else:
        emit('calibration_update', {'ok': False, 'reason': 'not_found'})


@socketio.on('request_jpeg')
def on_request_jpeg():
    """强制下一帧发送 JPEG。"""
    token = _sess.get_token(request.sid)
    _sess.push_mgr.force_next_jpeg(token)


# ── 广播频道 ─────────────────────────────────────────────────────────

@socketio.on('broadcast_join')
def on_broadcast_join(data):
    name = (data or {}).get('name', '') if isinstance(data, dict) else ''
    if not name:
        emit('broadcast_joined', {'ok': False, 'error': '无效频道名'})
        return
    count, presenter_sid = _bcast.viewer_join(request.sid, name)
    join_room(f'bcast_{name}')
    emit('broadcast_joined', {'ok': True, 'viewerCount': count})
    if presenter_sid:
        socketio.emit('broadcast_viewer_count', {'count': count}, to=presenter_sid)


@socketio.on('broadcast_leave')
def on_broadcast_leave(data):
    name = (data or {}).get('name', '') if isinstance(data, dict) else ''
    if name:
        _bcast.viewer_leave(request.sid, name)
        leave_room(f'bcast_{name}')


@socketio.on('broadcast_start')
def on_broadcast_start(data):
    name = (data or {}).get('name', '') if isinstance(data, dict) else ''
    if not name:
        emit('broadcast_started', {'ok': False, 'error': '无效频道名'})
        return
    _bcast.presenter_start(request.sid, name)
    emit('broadcast_started', {'ok': True, 'name': name})


@socketio.on('broadcast_stop')
def on_broadcast_stop(data):
    _bcast.presenter_stop(request.sid)
    emit('broadcast_stopped', {'ok': True})


@socketio.on('broadcast_frame')
def on_broadcast_frame(data):
    """主播将当前渲染帧转发给所有观众。"""
    name = _bcast.get_presenter_name(request.sid)
    if name:
        socketio.emit('broadcast_frame', data, to=f'bcast_{name}', skip_sid=request.sid)


# ══════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════

def main():
    port = int(os.environ.get('PORT', 8686))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
