import sys
import cv2
import numpy as np
from PIL import Image as PILImage
import base64
import os
from io import BytesIO
import json
import struct
import config
from flask import Flask, render_template, Response, jsonify, request, send_file, send_from_directory, copy_current_request_context
from threading import Lock
from flask_socketio import SocketIO, emit, disconnect
from collections import deque

# 坐标过滤参数
POS_HISTORY_SIZE = 20      # 保留最近 N 次坐标
POS_OUTLIER_THRESHOLD = 200 # 超过此距离视为异常帧，丢弃

# 解析启动参数: python main_ai_web.py [cpu|sift|ai|loftr]
#   cpu / sift  -> 仅 SIFT 模式（不加载 torch/kornia，秒启动）
#   ai  / loftr -> 完整双引擎模式（需要 torch + kornia）
_START_MODE = (sys.argv[1].lower() if len(sys.argv) > 1 else 'ai')
_SIFT_ONLY = _START_MODE in ('cpu', 'sift')

app = Flask(__name__)
# 使用 eventlet 或 threading 模式支持 WebSocket
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


class SIFTMapTracker:
    """SIFT 传统特征匹配引擎（从 main_sift.py 移植）"""
    def __init__(self):
        print("正在初始化 SIFT 引擎...")
        self.clahe = cv2.createCLAHE(clipLimit=config.SIFT_CLAHE_LIMIT, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create()

        logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = logic_map_bgr.shape[:2]

        logic_map_gray = cv2.cvtColor(logic_map_bgr, cv2.COLOR_BGR2GRAY)
        logic_map_gray = self.clahe.apply(logic_map_gray)
        print("正在提取大地图 SIFT 特征点...")
        self.kp_big, self.des_big = self.sift.detectAndCompute(logic_map_gray, None)
        print(f"SIFT 初始化完成！共 {len(self.kp_big)} 个锚点。")

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.last_x = None
        self.last_y = None
        self.lost_frames = 0

    def match(self, minimap_bgr):
        found = False
        center_x, center_y = None, None
        is_inertial = False

        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = self.clahe.apply(minimap_gray)

        kp_mini, des_mini = self.sift.detectAndCompute(minimap_gray, None)

        if des_mini is not None and len(kp_mini) >= 2:
            matches = self.flann.knnMatch(des_mini, self.des_big, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= config.SIFT_MIN_MATCH_COUNT:
                src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.kp_big[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)
                if M is not None:
                    h, w = minimap_gray.shape
                    center_pt = np.float32([[[w / 2, h / 2]]])
                    dst_center = cv2.perspectiveTransform(center_pt, M)
                    temp_x = int(dst_center[0][0][0])
                    temp_y = int(dst_center[0][0][1])

                    if 0 <= temp_x < self.map_width and 0 <= temp_y < self.map_height:
                        found = True
                        center_x = temp_x
                        center_y = temp_y
                        self.last_x = center_x
                        self.last_y = center_y
                        self.lost_frames = 0

        if not found and self.last_x is not None:
            self.lost_frames += 1
            if self.lost_frames <= config.MAX_LOST_FRAMES:
                found = True
                center_x = self.last_x
                center_y = self.last_y
                is_inertial = True

        return {
            'found': found,
            'center_x': center_x,
            'center_y': center_y,
            'is_inertial': is_inertial,
            'match_count': 0,
            'map_width': self.map_width,
            'map_height': self.map_height,
        }


class LoFTRMapTracker:
    """LoFTR AI 深度学习匹配引擎（仅在需要时才 import torch/kornia）"""
    def __init__(self):
        # 延迟导入，SIFT 模式下完全不会触发
        import torch as _torch
        import kornia as _K
        from kornia.feature import LoFTR as _LoFTR

        print("正在加载 LoFTR AI 模型...")
        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        print(f"当前计算设备: {self.device}")

        # 优先从本地 web/ 目录加载模型权重，避免联网下载
        local_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'loftr_outdoor.ckpt')
        if os.path.exists(local_ckpt):
            print(f"从本地加载模型: {local_ckpt}")
            self.matcher = _LoFTR(pretrained=None).to(self.device)
            raw = _torch.load(local_ckpt, map_location=self.device, weights_only=False)
            sd = raw.get('state_dict', raw)
            self.matcher.load_state_dict(sd)
        else:
            print("本地未找到模型，尝试联网下载...")
            self.matcher = _LoFTR(pretrained='outdoor').to(self.device)

        self.matcher.eval()
        self.torch = _torch
        self.kornia = _K
        print("AI 模型加载完成！")

        self.logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if self.logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = self.logic_map_bgr.shape[:2]

        self.display_map_bgr = cv2.imread(config.DISPLAY_MAP_PATH)
        if self.display_map_bgr is None:
            raise FileNotFoundError(f"找不到显示地图: {config.DISPLAY_MAP_PATH}！")

        self.state = "GLOBAL_SCAN"
        self.last_x = 0
        self.last_y = 0
        self.scan_size = config.AI_SCAN_SIZE
        self.scan_step = config.AI_SCAN_STEP
        self.scan_x = 0
        self.scan_y = 0
        self.search_radius = config.AI_TRACK_RADIUS
        self.lost_frames = 0
        self.max_lost_frames = 3

    def preprocess_image(self, img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        new_h = h - (h % 8)
        new_w = w - (w % 8)
        img_gray = cv2.resize(img_gray, (new_w, new_h))
        tensor = self.kornia.image_to_tensor(img_gray, False).float() / 255.0
        return tensor.to(self.device)

    def match(self, minimap_bgr):
        found = False
        display_crop = None
        half_view = config.VIEW_SIZE // 2
        match_count = 0

        if self.state == "GLOBAL_SCAN":
            x1 = self.scan_x
            y1 = self.scan_y
            x2 = min(self.map_width, x1 + self.scan_size)
            y2 = min(self.map_height, y1 + self.scan_size)
            display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
            display_crop = cv2.resize(display_crop, (config.VIEW_SIZE, int(config.VIEW_SIZE * max(1, y2 - y1) / max(1, x2 - x1))))
            cv2.putText(display_crop, f"Global Scan: X:{x1} Y:{y1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        else:
            x1 = max(0, self.last_x - self.search_radius)
            y1 = max(0, self.last_y - self.search_radius)
            x2 = min(self.map_width, self.last_x + self.search_radius)
            y2 = min(self.map_height, self.last_y + self.search_radius)

        local_logic_map = self.logic_map_bgr[y1:y2, x1:x2]

        if local_logic_map.shape[0] >= 16 and local_logic_map.shape[1] >= 16:
            tensor_mini = self.preprocess_image(minimap_bgr)
            tensor_big_local = self.preprocess_image(local_logic_map)
            input_dict = {"image0": tensor_mini, "image1": tensor_big_local}

            with self.torch.no_grad():                correspondences = self.matcher(input_dict)

            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()

            valid_idx = confidence > config.AI_CONFIDENCE_THRESHOLD
            mkpts0 = mkpts0[valid_idx]
            mkpts1 = mkpts1[valid_idx]
            match_count = len(mkpts0)

            if len(mkpts0) >= config.AI_MIN_MATCH_COUNT:
                M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, config.AI_RANSAC_THRESHOLD)
                if M is not None:
                    h, w = minimap_bgr.shape[:2]
                    center_pt = np.float32([[[w / 2, h / 2]]])
                    dst_center_local = cv2.perspectiveTransform(center_pt, M)
                    center_x = int(dst_center_local[0][0][0] + x1)
                    center_y = int(dst_center_local[0][0][1] + y1)

                    if 0 <= center_x < self.map_width and 0 <= center_y < self.map_height:
                        found = True
                        self.last_x = center_x
                        self.last_y = center_y
                        self.state = "LOCAL_TRACK"
                        self.lost_frames = 0

                        vy1 = max(0, center_y - half_view)
                        vy2 = min(self.map_height, center_y + half_view)
                        vx1 = max(0, center_x - half_view)
                        vx2 = min(self.map_width, center_x + half_view)
                        display_crop = self.display_map_bgr[vy1:vy2, vx1:vx2].copy()
                        local_cx = center_x - vx1
                        local_cy = center_y - vy1
                        cv2.circle(display_crop, (local_cx, local_cy), radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.circle(display_crop, (local_cx, local_cy), radius=7, color=(255, 255, 255), thickness=1)

        if not found:
            if self.state == "LOCAL_TRACK":
                self.lost_frames += 1
                if self.lost_frames <= self.max_lost_frames:
                    vy1 = max(0, self.last_y - half_view)
                    vy2 = min(self.map_height, self.last_y + half_view)
                    vx1 = max(0, self.last_x - half_view)
                    vx2 = min(self.map_width, self.last_x + half_view)
                    display_crop = self.display_map_bgr[vy1:vy2, vx1:vx2].copy()
                    local_cx = self.last_x - vx1
                    local_cy = self.last_y - vy1
                    cv2.circle(display_crop, (local_cx, local_cy), radius=5, color=(0, 255, 255), thickness=-1)
                else:
                    print("彻底丢失目标，启动全局雷达扫描...")
                    self.state = "GLOBAL_SCAN"
                    self.scan_x = 0
                    self.scan_y = 0
                    display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                    cv2.putText(display_crop, "Radar Initializing...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
            elif self.state == "GLOBAL_SCAN":
                self.scan_x += self.scan_step
                if self.scan_x >= self.map_width:
                    self.scan_x = 0
                    self.scan_y += self.scan_step
                    if self.scan_y >= self.map_height:
                        self.scan_x = 0
                        self.scan_y = 0

        return {
            'found': found,
            'display_crop': display_crop,
            'state': self.state,
            'last_x': self.last_x,
            'last_y': self.last_y,
            'match_count': match_count,
            'map_width': self.map_width,
            'map_height': self.map_height,
        }


class AIMapTrackerWeb:
    def __init__(self):
        # --- 加载引擎 ---
        print("=" * 50)
        self.sift_engine = SIFTMapTracker()
        if _SIFT_ONLY:
            print("  [SIFT-only 模式] 已跳过 torch/kornia/LoFTR 加载")
            self.loftr_engine = None
            self.current_mode = 'sift'
        else:
            self.loftr_engine = LoFTRMapTracker()
            # 默认 SIFT 模式（用户可在前端切换）
            self.current_mode = 'sift'
        print("=" * 50)

        # 显示地图（两种模式共用）
        self.display_map_bgr = cv2.imread(config.DISPLAY_MAP_PATH)
        self.map_height = self.sift_engine.map_height
        self.map_width = self.sift_engine.map_width

        # 线程安全锁
        self.lock = Lock()

        # 当前帧数据
        self.current_frame_bgr = None
        self.latest_result_image = None
        self.latest_result_jpeg = None

        # 坐标历史（用于异常值过滤）
        self.pos_history = deque(maxlen=POS_HISTORY_SIZE)

        self.latest_status = {
            'mode': 'sift',
            'state': '--',
            'position': {'x': 0, 'y': 0},
            'found': False,
            'matches': 0,
        }

    def set_minimap(self, minimap_bgr):
        with self.lock:
            self.current_frame_bgr = minimap_bgr.copy()

    def set_mode(self, mode):
        """切换识别模式: 'sift' 或 'loftr'"""
        if mode not in ('sift', 'loftr'):
            return False
        if mode == 'loftr' and self.loftr_engine is None:
            print("当前为 SIFT-only 模式，无法切换到 LoFTR（请不带参数启动以启用 AI）")
            return False
        self.current_mode = mode
        # 切换模式时重置状态，避免残留
        if mode == 'loftr':
            self.loftr_engine.state = "GLOBAL_SCAN"
            self.loftr_engine.scan_x = 0
            self.loftr_engine.scan_y = 0
        return True

    def process_frame(self):
        with self.lock:
            if self.current_frame_bgr is None:
                return None
            minimap_bgr = self.current_frame_bgr.copy()

        half_view = config.VIEW_SIZE // 2

        if self.current_mode == 'sift':
            result = self.sift_engine.match(minimap_bgr)
            found = result['found']
            cx, cy = result['center_x'], result['center_y']
            is_inertial = result.get('is_inertial', False)

            if found and cx is not None:
                y1 = max(0, cy - half_view)
                y2 = min(self.map_height, cy + half_view)
                x1 = max(0, cx - half_view)
                x2 = min(self.map_width, cx + half_view)
                display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
                local_x = cx - x1
                local_y = cy - y1
                if not is_inertial:
                    cv2.circle(display_crop, (local_x, local_y), radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.circle(display_crop, (local_x, local_y), radius=7, color=(255, 255, 255), thickness=1)
                else:
                    cv2.circle(display_crop, (local_x, local_y), radius=5, color=(0, 255, 255), thickness=-1)
                    cv2.circle(display_crop, (local_x, local_y), radius=7, color=(0, 150, 150), thickness=1)
            else:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                cv2.putText(display_crop, "SIFT Searching...", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            status_state = 'INERTIAL' if is_inertial else ('FOUND' if found else 'SEARCHING')
            match_count = 0
            last_x = cx if cx else 0
            last_y = cy if cy else 0

        else:  # loftr
            if self.loftr_engine is None:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                cv2.putText(display_crop, "AI Engine Disabled", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                found = False
                status_state = 'DISABLED'
                match_count = 0
                last_x = last_y = 0
            else:
                result = self.loftr_engine.match(minimap_bgr)
            display_crop = result.get('display_crop')
            if display_crop is None:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
            found = result['found']
            status_state = result.get('state', 'GLOBAL_SCAN')
            match_count = result.get('match_count', 0)
            last_x = result.get('last_x', 0)
            last_y = result.get('last_y', 0)

        # ====== 坐标异常值过滤 ======
        if found and (last_x or last_y):
            is_outlier = False
            if len(self.pos_history) >= 3:
                # 取历史中位数作为参考
                hist = list(self.pos_history)
                ref_x = sorted(h[0] for h in hist)[len(hist)//2]
                ref_y = sorted(h[1] for h in hist)[len(hist)//2]
                if abs(last_x - ref_x) > POS_OUTLIER_THRESHOLD or abs(last_y - ref_y) > POS_OUTLIER_THRESHOLD:
                    is_outlier = True
            if not is_outlier:
                self.pos_history.append((last_x, last_y))
            else:
                # 异常帧：丢弃，用最近一次有效坐标
                last_x = self.pos_history[-1][0]
                last_y = self.pos_history[-1][1]
        elif found:
            self.pos_history.append((last_x, last_y))

        # 渲染最终输出
        display_rgb = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(display_rgb)
        final_img = PILImage.new('RGB', (config.VIEW_SIZE, config.VIEW_SIZE), (43, 43, 43))
        final_img.paste(pil_image,
                        (max(0, half_view - pil_image.width // 2), max(0, half_view - pil_image.height // 2)))

        buffer_png = BytesIO()
        final_img.save(buffer_png, format='PNG')
        img_base64 = base64.b64encode(buffer_png.getvalue()).decode('utf-8')

        # JPEG 二进制输出（给 WebSocket 用，更小更快）
        buffer_jpeg = BytesIO()
        final_img.save(buffer_jpeg, format='JPEG', quality=85, optimize=True)
        jpeg_bytes = buffer_jpeg.getvalue()

        self.latest_status = {
            'mode': self.current_mode,
            'state': status_state,
            'position': {'x': last_x, 'y': last_y},
            'found': found,
            'matches': match_count,
        }
        self.latest_result_image = img_base64
        self.latest_result_jpeg = jpeg_bytes

        return img_base64, jpeg_bytes


# 全局实例
tracker = AIMapTrackerWeb()

# 获取 web 目录绝对路径
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

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
            b64_data = b64_data.split(',')[1]  # 去掉 data:image/png;base64, 前缀
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
                'status': tracker.latest_status
            })
    return jsonify({'error': 'Invalid image'}), 400


@app.route('/api/status')
def get_status():
    """获取当前追踪状态"""
    return jsonify(tracker.latest_status)


@app.route('/api/mode', methods=['POST'])
def set_mode():
    """切换识别模式: sift 或 loftr"""
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'sift').lower()
    if tracker.set_mode(mode):
        return jsonify({'success': True, 'mode': tracker.current_mode})
    return jsonify({'error': f'Invalid mode: {mode}'}), 400


@app.route('/api/result')
def get_result():
    """获取最新的结果图片"""
    if tracker.latest_result_image:
        return jsonify({
            'image': tracker.latest_result_image,
            'status': tracker.latest_status
        })
    return jsonify({'error': 'No result yet'}), 404


@app.route('/api/process')
def process():
    """手动触发一次处理（用于文件模式）"""
    result = tracker.process_frame()
    if result:
        return jsonify({
            'image': result[0],
            'status': tracker.latest_status
        })
    return jsonify({'error': 'No minimap set'}), 400


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


@socketio.on('frame')
def ws_receive_frame(raw_bytes):
    """
    接收二进制 JPEG 帧，处理并返回结果。
    协议: 客户端发送原始 JPEG bytes → 服务端返回 [JSON头(4字节长度) + JPEG图片]
    """
    # 解码图像
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        _, jpeg_result = tracker.process_frame()

        # 构造状态 JSON（紧凑格式）
        status_json = json.dumps({
            'm': tracker.current_mode,
            's': tracker.latest_status['state'],
            'x': tracker.latest_status['position']['x'],
            'y': tracker.latest_status['position']['y'],
            'f': int(tracker.latest_status['found']),
            'c': tracker.latest_status['matches'],
        }, separators=(',', ':')).encode('utf-8')

        # 发送: [4字节大端长度][JSON状态][JPEG图片]
        emit('result',
             struct.pack('>I', len(status_json)) + status_json + jpeg_result,
             binary=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


if __name__ == "__main__":
    print("=" * 50)
    mode_label = "SIFT-only (快速模式)" if _SIFT_ONLY else "双引擎 (SIFT + LoFTR AI)"
    print(f"  地图跟点 - 网页版 [{mode_label}]")
    if _SIFT_ONLY:
        print("  用法: python main_ai_web.py cpu   → 仅 SIFT (无需 torch)")
        print("       python main_ai_web.py ai    → SIFT + LoFTR AI")
    print("  打开浏览器访问: http://0.0.0.0:8686")
    print("  WebSocket: ws://0.0.0.0:8686/socket.io/?transport=websocket")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=8686, debug=False)
