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
import json
import time
import struct

import cv2
import numpy as np
import base64
from io import BytesIO

from backend import config
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from backend.tracker_core import MapTrackerWeb
from backend.store import get_route_files, load_route_data, load_circle_state, save_circle_state

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
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_SOCKETIO_ASYNC_MODE)

# 全局追踪器实例
tracker = MapTrackerWeb()

# Plan B: 记录当前有哪些客户端在使用 'frame' 事件（需要 JPEG）
# 无此类客户端时 _push_jpeg=False，后台线程跳过 cv2.imencode 节省 10-15ms/帧。
_frame_clients: set = set()

# JPEG 节流：记录上次发送 JPEG 时的地图坐标。
# 当新坐标偏移 < JPEG_PAD - JPEG_THROTTLE_MARGIN 时只发 coords，节省 80-200KB/次。
import math as _math
_last_jpeg_x: float = 0.0
_last_jpeg_y: float = 0.0
_JPEG_THROTTLE_MARGIN = 8   # 保留 8px 缓冲，pan 超出前 8px 就发新图

# Plan A (push model): 最新 'frame' 客户端的 SID，供 _on_result_ready 定向推送。
# 叠加式：push 失败时 pull 兜底（ws_receive_frame 仍发 coords）。
_push_frame_sid: str = ''


def _make_status_json() -> bytes:
    """构建 WS 推送用的紧凑 JSON bytes（避免 ws_receive_frame 与 _on_result_ready 重复编码）。"""
    status = tracker.latest_status
    return json.dumps({
        'm': tracker.current_mode,
        's': status['state'],
        'x': status['position']['x'],
        'y': status['position']['y'],
        'f': int(status['found']),
        'c': status['matches'],
        'q': round(status.get('match_quality', 0), 2),
        'a': round(status.get('arrow_angle', 0), 1),
        'as': int(status.get('arrow_stopped', True)),
        'l': int(tracker.sift_engine.coord_lock_enabled),
        'tp': int(status.get('is_teleport', False)),
    }, separators=(',', ':')).encode('utf-8')


def _on_result_ready():
    """
    Plan A 推送回调：sift-worker 处理完帧后立即调用，主动推送新结果。
    比 pull 模式（等下一帧 ws_receive_frame）延迟低 ~80ms。
    JPEG 节流逻辑在此执行（dist < budget → 只推 coords，节省带宽）。
    每个 emit 独立 try/except，单条失败不影响其他。
    """
    global _last_jpeg_x, _last_jpeg_y

    if tracker.latest_status.get('state') == '--':
        return  # 首帧尚未完成，不推送

    status_json = _make_status_json()
    header = struct.pack('>I', len(status_json))

    # JPEG 节流：超出 pan 余量才发新 JPEG
    pad = getattr(config, 'JPEG_PAD', 40)
    budget = pad - _JPEG_THROTTLE_MARGIN
    cx = tracker.latest_status['position']['x']
    cy = tracker.latest_status['position']['y']
    dist = _math.sqrt((cx - _last_jpeg_x) ** 2 + (cy - _last_jpeg_y) ** 2)

    sid = _push_frame_sid
    if sid:
        jpeg = tracker.latest_result_jpeg
        if jpeg and dist >= budget:
            # 位移超出余量：推 result（含 JPEG），更新锚点
            _last_jpeg_x, _last_jpeg_y = cx, cy
            try:
                socketio.emit('result', header + status_json + jpeg,
                              to=sid, binary=True, namespace='/')
            except Exception as e:
                print(f"[push] result → {sid} 失败: {e}")
        else:
            # 位移在余量内：只推轻量 coords，前端 pan 消化
            try:
                socketio.emit('coords', header + status_json,
                              to=sid, binary=True, namespace='/')
            except Exception as e:
                print(f"[push] coords → {sid} 失败: {e}")

    # 向所有客户端广播坐标（bigmap.html 等；frame 客户端收到重复 coords 无害）
    try:
        socketio.emit('coords', header + status_json,
                      broadcast=True, binary=True, namespace='/')
    except Exception as e:
        print(f"[push] coords broadcast 失败: {e}")


tracker.result_callback = _on_result_ready

# (路径常量 _BASE_DIR / FRONTEND_DIR / ASSETS_DIR 已在文件顶部定义)



# ==================== 圆形选区持久化 ====================

_CIRCLE_STATE_FILE = os.path.join(_BASE_DIR, '.circle_state.json')


def _load_circle_state():
    return load_circle_state(_CIRCLE_STATE_FILE)


def _save_circle_state(cx, cy, r):
    return save_circle_state(_CIRCLE_STATE_FILE, cx, cy, r)


# 启动时加载已有状态
_saved_circle = _load_circle_state()
if _saved_circle:
    print(f"📍 已恢复圆形选区: cx={_saved_circle.get('cx')}, cy={_saved_circle.get('cy')}, r={_saved_circle.get('r')}")


# ==================== HTTP 路由 ====================

@app.route('/')
def index():
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))


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


@app.route('/api/upload_minimap', methods=['POST'])
def upload_minimap():
    """接收前端上传的小地图图片（支持 FormData 和 JSON base64 两种格式）"""
    img = None

    # 方式1: FormData 文件上传
    if 'image' in request.files:
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 方式2: JSON base64 上传
    elif request.is_json:
        data = request.get_json()
        b64_data = data.get('image', '')
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        result = tracker.process_frame(need_base64=True, need_jpeg=False)
        if result and result[0]:
            return jsonify({
                'success': True,
                'image': result[0],
                'status': tracker.latest_status,
            })
    return jsonify({'error': 'Invalid image'}), 400


@app.route('/api/status')
def get_status():
    """获取当前追踪状态"""
    return jsonify(tracker.latest_status)


@app.route('/api/map_info')
def get_map_info():
    """大地图模式：返回地图尺寸和图片路径"""
    return jsonify({
        'map_width': tracker.map_width,
        'map_height': tracker.map_height,
        'display_map_url': '/big_map-1.png',
    })


@app.route('/bigmap')
def bigmap_page():
    """大地图独立页面"""
    return send_file(os.path.join(FRONTEND_DIR, 'bigmap.html'))


@app.route('/api/coord_lock', methods=['POST'])
def api_coord_lock():
    """坐标锁定模式: 开/关/查询"""
    data = request.get_json(silent=True) or {}
    action = data.get('action', 'toggle').lower()
    engine = tracker.sift_engine

    if action == 'query':
        return jsonify({
            'enabled': engine.coord_lock_enabled,
            'history_count': len(tracker.pos_history),
            'can_activate': len(tracker.pos_history) >= engine._lock_min_to_activate,
        })

    if action == 'on':
        if len(tracker.pos_history) < engine._lock_min_to_activate:
            return jsonify({'error': f'历史坐标不足 (需要{engine._lock_min_to_activate}个，当前{len(tracker.pos_history)}个)'}), 400
        ok = engine.set_coord_lock(True)
        return jsonify({'success': ok, 'enabled': True})

    elif action == 'off':
        ok = engine.set_coord_lock(False)
        return jsonify({'success': ok, 'enabled': False})

    else:  # toggle
        if engine.coord_lock_enabled:
            ok = engine.set_coord_lock(False)
            return jsonify({'success': ok, 'enabled': False})
        else:
            if len(tracker.pos_history) < engine._lock_min_to_activate:
                return jsonify({'error': f'历史坐标不足 (需要{engine._lock_min_to_activate}个，当前{len(tracker.pos_history)}个)'}), 400
            ok = engine.set_coord_lock(True)
            return jsonify({'success': ok, 'enabled': True})


@app.route('/api/reset_history', methods=['POST'])
def api_reset_history():
    """清空坐标历史 + 关闭锁定 + 重置线性过滤器"""
    engine = tracker.sift_engine
    cleared = len(tracker.pos_history)

    tracker.pos_history.clear()

    was_locked = engine.coord_lock_enabled
    engine.set_coord_lock(False)

    tracker._linear_filter_consecutive = 0

    print(f"🗑 历史已重置: 清除{cleared}条记录, 锁定{'已关闭' if was_locked else '未开启'}, 线性过滤器已重置")
    return jsonify({
        'success': True,
        'cleared_count': cleared,
        'was_locked': was_locked,
    })


@app.route('/api/circle_state', methods=['GET', 'POST'])
def api_circle_state():
    """获取/保存 圆形选区状态 (cx, cy, r) 到服务器本地"""
    if request.method == 'GET':
        state = _load_circle_state()
        if state:
            return jsonify({'success': True, **state})
        return jsonify({'success': False, 'error': 'No saved state'})

    # POST: 保存状态
    data = request.get_json(silent=True) or {}
    try:
        cx = float(data.get('cx', 0))
        cy = float(data.get('cy', 0))
        r = float(data.get('r', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid parameters'}), 400

    if _save_circle_state(cx, cy, r):
        return jsonify({'success': True, 'cx': cx, 'cy': cy, 'r': r})
    return jsonify({'error': 'Save failed'}), 500


@app.route('/api/result')
def get_result():
    """获取最新的结果图片"""
    image = tracker.get_latest_result_base64()
    if image:
        return jsonify({
            'image': image,
            'status': tracker.latest_status,
        })
    return jsonify({'error': 'No result yet'}), 404


@app.route('/api/latest_frame')
def get_latest_frame():
    """获取最新渲染的地图帧（JPEG 二进制）- 供外部悬浮窗使用"""
    jpeg_bytes = tracker.get_latest_result_jpeg()
    if jpeg_bytes:
        return send_file(
            BytesIO(jpeg_bytes),
            mimetype='image/jpeg',
        )
    return jsonify({'error': 'No frame available yet'}), 404


@app.route('/api/process')
def process():
    """手动触发一次处理（用于文件模式）"""
    result = tracker.process_frame(need_base64=True, need_jpeg=False)
    if result and result[0]:
        return jsonify({
            'image': result[0],
            'status': tracker.latest_status,
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
                'loop': data.get('loop', False),
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
    emit('status', tracker.latest_status)


@socketio.on('disconnect')
def ws_disconnect():
    global _push_frame_sid
    # Plan B: 客户端断开时从 frame_clients 移除，若没有 JPEG 消费者则关闭编码
    _frame_clients.discard(request.sid)
    tracker._push_jpeg = bool(_frame_clients)
    # Plan A: 若断开的是 push 目标，清除 SID 避免无效推送
    if request.sid == _push_frame_sid:
        _push_frame_sid = ''
    print(f"WebSocket 客户端断开: {request.sid}")


@socketio.on('coord_lock')
def ws_coord_lock(data):
    """通过 WS 切换坐标锁定模式"""
    action = (data.get('action', 'toggle') if isinstance(data, dict) else 'toggle').lower()
    engine = tracker.sift_engine

    if action == 'query':
        emit('lock_result', {
            'enabled': engine.coord_lock_enabled,
            'history_count': len(tracker.pos_history),
            'can_activate': len(tracker.pos_history) >= engine._lock_min_to_activate,
        })
        return

    if action == 'on' or (action == 'toggle' and not engine.coord_lock_enabled):
        need = engine._lock_min_to_activate
        have = len(tracker.pos_history)
        if have < need:
            emit('lock_result', {'success': False, 'error': f'历史不足({have}/{need})'})
            return
        ok = engine.set_coord_lock(True)
        emit('lock_result', {'success': ok, 'enabled': True})
    elif action == 'off' or (action == 'toggle' and engine.coord_lock_enabled):
        ok = engine.set_coord_lock(False)
        emit('lock_result', {'success': ok, 'enabled': False})


@socketio.on('reset_history')
def ws_reset_history():
    """通过 WS 重置坐标历史"""
    engine = tracker.sift_engine
    cleared = len(tracker.pos_history)

    tracker.pos_history.clear()
    was_locked = engine.coord_lock_enabled
    engine.set_coord_lock(False)
    tracker._linear_filter_consecutive = 0

    print(f"🗑 [WS] 历史已重置: 清除{cleared}条, 锁定{'已关闭' if was_locked else '未开启'}")
    emit('reset_result', {
        'success': True,
        'cleared_count': cleared,
        'was_locked': was_locked,
    })


@socketio.on('request_jpeg')
def ws_request_jpeg():
    """前端请求强制推送一张新 JPEG（如切换到 JPEG 渲染模式时触发）。
    重置节流锚点到不可能的坐标，确保下次 _on_result_ready 必定发送 result+JPEG。
    """
    global _last_jpeg_x, _last_jpeg_y
    _last_jpeg_x = -99999.0
    _last_jpeg_y = -99999.0


@socketio.on('frame')
def ws_receive_frame(raw_bytes):
    """
    接收二进制 JPEG 帧，存帧并立即返回上一帧缓存坐标（非阻塞，pull 兜底）。
    SIFT 匹配运行在后台线程，处理完成后由 _on_result_ready 主动推送新结果（Plan A）。
    协议: 客户端发送原始 JPEG bytes → 服务端返回 [JSON头(4字节长度) + JSON]
    Plan B: 注册该客户端为 JPEG 消费者，确保后台线程生成 JPEG。
    Plan A: 记录该 SID 为 push 目标，SIFT 完成后立即推送（含 JPEG 节流）。
    """
    global _push_frame_sid

    # Plan B: 记录该 sid 需要 JPEG，后台线程保持编码开启
    if request.sid not in _frame_clients:
        _frame_clients.add(request.sid)
        tracker._push_jpeg = True
    # Plan A: 记录最新 JPEG 客户端 SID，供 _on_result_ready 定向推送
    _push_frame_sid = request.sid

    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)  # 存帧并唤醒后台 SIFT 线程，立即返回

        # Pull 兜底：立即返回上一帧缓存坐标，等待 push 推送新 JPEG（~20ms 后）
        status = tracker.latest_status
        if status.get('state') == '--':
            return  # 启动后尚无任何结果

        status_json = _make_status_json()
        header = struct.pack('>I', len(status_json))
        emit('coords', header + status_json, binary=True)

        # 向其他监听客户端（如 bigmap.html）广播仅坐标的轻量更新
        emit('coords', header + status_json,
             broadcast=True, include_self=False, binary=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


@socketio.on('frame_coords')
def ws_frame_coords(raw_bytes):
    """
    接收帧并存帧（后台处理），立即返回上一帧缓存坐标。大幅减少带宽。
    Plan B: 无 'frame' 客户端时设 _push_jpeg=False，后台线程跳过 JPEG 编码。
    """
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        # Plan B: 当前无 JPEG 消费者时关闭编码，节省 10-15ms/帧
        if not _frame_clients:
            tracker._push_jpeg = False

        tracker.set_minimap(img)  # 存帧并唤醒后台 SIFT 线程

        if tracker.latest_status.get('state') == '--':
            return  # 尚无任何处理结果（首帧），静默等待

        status = tracker.latest_status
        status_json = json.dumps({
            'm': tracker.current_mode,
            's': status['state'],
            'x': status['position']['x'],
            'y': status['position']['y'],
            'f': int(status['found']),
            'c': status['matches'],
            'q': round(status.get('match_quality', 0), 2),
            'a': round(status.get('arrow_angle', 0), 1),
            'as': int(status.get('arrow_stopped', True)),
            'l': int(tracker.sift_engine.coord_lock_enabled),
            'h': int(status.get('hybrid_busy', False)),
            'hy': int(status.get('hybrid', False)),
        }, separators=(',', ':')).encode('utf-8')

        emit('coords',
             struct.pack('>I', len(status_json)) + status_json,
             binary=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


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
    print("  \u6253\u5f00\u6d4f\u89c8\u5668\u8bbf\u95ee: http://0.0.0.0:" + str(config.PORT))
    print("  WebSocket: ws://0.0.0.0:" + str(config.PORT) + "/socket.io/?transport=websocket")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=config.PORT, debug=False, allow_unsafe_werkzeug=True)
