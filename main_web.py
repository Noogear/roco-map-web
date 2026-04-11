"""
main_web.py - Web 控制层（Flask + SocketIO 路由入口）

职责:
  - Flask / SocketIO 应用初始化
  - HTTP API 路由（上传/状态/控制）
  - WebSocket 消息处理
  - 圆形选区持久化
  - 启动入口

识别引擎和编排逻辑已拆分到:
  tracker_engines.py   SIFTMapTracker + LoFTRMapTracker (纯识别)
  tracker_core.py      AIMapTrackerWeb (平滑/过滤/渲染编排)
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

import config
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from tracker_core import AIMapTrackerWeb


# ==================== 路线文件管理 ====================

_ROUTES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sift-map-tracker', 'routes'))


def _get_route_files():
    """扫描 routes 目录，返回可用的路线文件列表"""
    routes = []
    if not os.path.isdir(_ROUTES_DIR):
        return routes
    for f in sorted(os.listdir(_ROUTES_DIR)):
        if f.lower().endswith('.json'):
            routes.append(f)
    return routes


def _load_route_data(filename):
    """加载单个路线 JSON 文件"""
    filepath = os.path.join(_ROUTES_DIR, filename)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


# 解析启动参数: python main_web.py [cpu|sift|ai|loftr]
_START_MODE = (sys.argv[1].lower() if len(sys.argv) > 1 else 'ai')
_SIFT_ONLY = _START_MODE in ('cpu', 'sift')

# Flask + SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局追踪器实例
tracker = AIMapTrackerWeb(sift_only=_SIFT_ONLY)

# Web 目录路径
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')


# ==================== 圆形选区持久化 ====================

_CIRCLE_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.circle_state.json')


def _load_circle_state():
    """启动时从本地文件恢复圆形选区状态"""
    if not os.path.isfile(_CIRCLE_STATE_FILE):
        return None
    try:
        with open(_CIRCLE_STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _save_circle_state(cx, cy, r):
    """将圆形选区状态写入本地 JSON 文件"""
    try:
        data = {'cx': cx, 'cy': cy, 'r': r, 'ts': time.time()}
        with open(_CIRCLE_STATE_FILE, 'w') as f:
            json.dump(data, f)
        print(f"💾 圆形选区已保存: cx={cx:.4f}, cy={cy:.4f}, r={r:.4f}")
        return True
    except Exception as e:
        print(f"[警告] 保存圆形选区失败: {e}")
        return False


# 启动时加载已有状态
_saved_circle = _load_circle_state()
if _saved_circle:
    print(f"📍 已恢复圆形选区: cx={_saved_circle.get('cx')}, cy={_saved_circle.get('cy')}, r={_saved_circle.get('r')}")


# ==================== HTTP 路由 ====================

@app.route('/')
def index():
    return send_file(os.path.join(WEB_DIR, 'index.html'))


@app.route('/<path:filename>')
def serve_static(filename):
    """提供 web/ 目录下的静态资源（图片等）"""
    return send_from_directory(WEB_DIR, filename)


@app.route('/api/test_images')
def list_test_images():
    """列出 web/img/ 下可用的测试小地图"""
    img_dir = os.path.join(WEB_DIR, 'img')
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
        result = tracker.process_frame()
        if result:
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
    return send_file(os.path.join(WEB_DIR, 'bigmap.html'))


@app.route('/api/mode', methods=['POST'])
def set_mode():
    """切换识别模式: sift 或 loftr"""
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'sift').lower()
    if tracker.set_mode(mode):
        return jsonify({'success': True, 'mode': tracker.current_mode})
    return jsonify({'error': f'Invalid mode: {mode}'}), 400


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
    if tracker.latest_result_image:
        return jsonify({
            'image': tracker.latest_result_image,
            'status': tracker.latest_status,
        })
    return jsonify({'error': 'No result yet'}), 404


@app.route('/api/latest_frame')
def get_latest_frame():
    """获取最新渲染的地图帧（JPEG 二进制）- 供外部悬浮窗使用"""
    # 优先返回缓存的 JPEG 字节
    if tracker.latest_result_jpeg:
        return send_file(
            BytesIO(tracker.latest_result_jpeg),
            mimetype='image/jpeg',
        )
    # 如果有 base64 结果，尝试解码
    if tracker.latest_result_image:
        try:
            b64 = tracker.latest_result_image
            if ',' in b64:
                b64 = b64.split(',', 1)[1]
            jpeg_bytes = base64.b64decode(b64)
            return send_file(
                BytesIO(jpeg_bytes),
                mimetype='image/jpeg',
            )
        except Exception:
            pass
    return jsonify({'error': 'No frame available yet'}), 404


@app.route('/api/process')
def process():
    """手动触发一次处理（用于文件模式）"""
    result = tracker.process_frame()
    if result:
        return jsonify({
            'image': result[0],
            'status': tracker.latest_status,
        })
    return jsonify({'error': 'No minimap set'}), 400


# ==================== 路线 API ====================

@app.route('/api/routes')
def api_list_routes():
    """列出所有可用路线文件"""
    files = _get_route_files()
    routes = []
    for f in files:
        data = _load_route_data(f)
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
    data = _load_route_data(safe_name)
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
    print(f"WebSocket 客户端断开: {request.sid}")


@socketio.on('mode')
def ws_set_mode(data):
    """通过 WS 切换模式"""
    mode = (data.get('mode', 'sift') if isinstance(data, dict) else 'sift').lower()
    ok = tracker.set_mode(mode)
    emit('mode_result', {'success': ok, 'mode': tracker.current_mode})


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


@socketio.on('frame')
def ws_receive_frame(raw_bytes):
    """
    接收二进制 JPEG 帧，处理并返回结果。
    协议: 客户端发送原始 JPEG bytes → 服务端返回 [JSON头(4字节长度) + JPEG图片]
    """
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        _, jpeg_result = tracker.process_frame()

        status_json = json.dumps({
            'm': tracker.current_mode,
            's': tracker.latest_status['state'],
            'x': tracker.latest_status['position']['x'],
            'y': tracker.latest_status['position']['y'],
            'f': int(tracker.latest_status['found']),
            'c': tracker.latest_status['matches'],
            'l': int(tracker.sift_engine.coord_lock_enabled),
        }, separators=(',', ':')).encode('utf-8')

        emit('result',
             struct.pack('>I', len(status_json)) + status_json + jpeg_result,
             binary=True, broadcast=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


@socketio.on('frame_coords')
def ws_frame_coords(raw_bytes):
    """接收帧但只返回坐标（不返回裁剪图片），大幅减少带宽"""
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        tracker.process_frame()

        status_json = json.dumps({
            'm': tracker.current_mode,
            's': tracker.latest_status['state'],
            'x': tracker.latest_status['position']['x'],
            'y': tracker.latest_status['position']['y'],
            'f': int(tracker.latest_status['found']),
            'c': tracker.latest_status['matches'],
            'l': int(tracker.sift_engine.coord_lock_enabled),
        }, separators=(',', ':')).encode('utf-8')

        emit('coords',
             struct.pack('>I', len(status_json)) + status_json,
             binary=True, broadcast=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


# ==================== 启动入口 ====================

if __name__ == "__main__":
    print("=" * 50)
    mode_label = "SIFT-only (快速模式)" if _SIFT_ONLY else "双引擎 (SIFT + LoFTR AI)"
    print(f"  地图跟点 - 网页版 [{mode_label}]")
    if _SIFT_ONLY:
        print("  用法: python main_web.py cpu   → 仅 SIFT (无需 torch)")
        print("       python main_web.py ai    → SIFT + LoFTR AI")
    print("  打开浏览器访问: http://0.0.0.0:" + str(config.PORT))
    print("  WebSocket: ws://0.0.0.0:" + str(config.PORT) + "/socket.io/?transport=websocket")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=config.PORT, debug=False, allow_unsafe_werkzeug=True)
