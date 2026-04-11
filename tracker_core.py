"""
tracker_core.py - 追踪器编排层（无 Web 框架依赖）

职责:
  - 管理引擎实例(SIFT/LoFTR)的生命周期
  - 坐标平滑/线性过滤/异常值过滤
  - 帧处理流程编排: set_minimap -> process_frame
  - 结果渲染(裁剪地图 + 画箭头 + 编码PNG/JPEG)
  - 坐标历史持久化

此模块仅依赖 cv2 / numpy / PIL / config / tracker_engines，
可被 Flask/Tornado/CLI 等任何前端复用。
"""

import math
import json
import os
import time

import cv2
import numpy as np
import base64
from threading import Lock, Thread, Event
from collections import deque

import config
from tracker_engines import SIFTMapTracker, LoFTRMapTracker


# 坐标过滤参数
POS_HISTORY_SIZE = 20      # 保留最近 N 次坐标
POS_OUTLIER_THRESHOLD = 200 # 超过此距离视为异常帧，丢弃


def _create_kalman_filter():
    """
    创建 4 状态 (x, y, vx, vy) / 2 观测 (x, y) 的卡尔曼滤波器。
    用于替代线性外推+中位数平滑，提供更准确的预测和更平滑的输出。
    """
    kf = cv2.KalmanFilter(4, 2)
    # 状态转移矩阵: x' = x + vx*dt, y' = y + vy*dt (dt=1 frame)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    # 观测矩阵: 只能观测 x, y
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    # 过程噪声 (模型不确定性)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
    kf.processNoiseCov[2, 2] = 2.0  # 速度变化更不确定
    kf.processNoiseCov[3, 3] = 2.0
    # 观测噪声 (默认中等信任)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
    # 初始协方差
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 500.0
    return kf


class AIMapTrackerWeb:
    """
    地图追踪编排器：封装引擎调用 + 坐标后处理 + 渲染输出。
    不包含任何 Web/Flask/SocketIO 代码。
    """

    def __init__(self, sift_only=False):
        # --- 加载引擎 ---
        print("=" * 50)
        self.sift_engine = SIFTMapTracker()
        if sift_only:
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

        # 线性过滤器：连续丢弃帧计数器（防死锁）
        self._linear_filter_consecutive = 0

        # === 卡尔曼滤波器（替代线性外推+中位数平滑）===
        self._kalman = _create_kalman_filter()
        self._kalman_initialized = False  # 第一次观测后才开始滤波

        # 坐标平滑缓冲池（60帧滑动窗口 + 中位数滤波，消除 SIFT 抖动）
        # 注：卡尔曼滤波作为主滤波器，smooth_buffer 仅用于持久化恢复
        self.SMOOTH_BUFFER_SIZE = 60
        self.smooth_buffer_x = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self.smooth_buffer_y = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self._smooth_median_window = 15  # 取最近15帧的中位数作为输出（抗抖动）

        # 持久化文件路径（与 main_web.py 同级）
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._smooth_file = os.path.join(_base_dir, '.smooth_coords.json')
        self._load_smooth_buffer()  # 启动时恢复历史坐标

        self.latest_status = {
            'mode': 'sift',
            'state': '--',
            'position': {'x': 0, 'y': 0},
            'found': False,
            'matches': 0,
        }

        # ========== 混合引擎（后台 LoFTR 重定位）==========
        self._hybrid_enabled = (
            not sift_only
            and self.loftr_engine is not None
            and getattr(config, 'HYBRID_ENABLED', False)
        )
        self._hybrid_thread = None
        self._hybrid_busy = False          # 后台线程是否正在运行
        self._hybrid_last_trigger = 0.0    # 上次触发时间戳
        self._hybrid_result = None         # (x, y) 或 None，由后台线程写入
        self._hybrid_result_lock = Lock()  # 保护 _hybrid_result 的读写
        self._hybrid_stop = Event()        # 停止信号

        if self._hybrid_enabled:
            print("  🔀 混合引擎已启用 (SIFT 主引擎 + LoFTR 后台重定位)")

    # ========== 坐标平滑 ==========

    def _smooth_coord(self, raw_x, raw_y):
        """
        坐标平滑：60帧滑动窗口 + 最近 N 帧中位数滤波。
        返回 (smooth_x, smooth_y)，消除 SIFT 单帧随机抖动。
        每60帧自动持久化到本地文件。
        """
        if raw_x is not None and raw_y is not None:
            self.smooth_buffer_x.append(raw_x)
            self.smooth_buffer_y.append(raw_y)
            if len(self.smooth_buffer_x) >= self.SMOOTH_BUFFER_SIZE:
                self._save_smooth_buffer()

        n = min(len(self.smooth_buffer_x), self._smooth_median_window)
        if n < 3:
            return raw_x or 0, raw_y or 0

        recent_x = list(self.smooth_buffer_x)[-n:]
        recent_y = list(self.smooth_buffer_y)[-n:]
        recent_x.sort()
        recent_y.sort()
        mid = n // 2
        if n % 2 == 0:
            sx = (recent_x[mid - 1] + recent_x[mid]) // 2
            sy = (recent_y[mid - 1] + recent_y[mid]) // 2
        else:
            sx = recent_x[mid]
            sy = recent_y[mid]
        return int(sx), int(sy)

    def _load_smooth_buffer(self):
        """启动时从本地文件恢复最近60个坐标点"""
        if not os.path.isfile(self._smooth_file):
            return
        try:
            with open(self._smooth_file, 'r') as f:
                data = json.load(f)
            xs = data.get('x', [])
            ys = data.get('y', [])
            if len(xs) == len(ys) and len(xs) > 0:
                self.smooth_buffer_x.extend(xs[-self.SMOOTH_BUFFER_SIZE:])
                self.smooth_buffer_y.extend(ys[-self.SMOOTH_BUFFER_SIZE:])
                # 用最后一个历史坐标初始化卡尔曼
                self._kalman_reset(xs[-1], ys[-1])
                print(f"📍 已从本地恢复 {len(xs)} 个历史坐标点 (最新: {xs[-1]}, {ys[-1]})")
        except Exception as e:
            print(f"[警告] 加载坐标历史失败: {e}")

    def _save_smooth_buffer(self):
        """将当前缓冲池的坐标写入本地 JSON 文件"""
        try:
            data = {
                'x': list(self.smooth_buffer_x),
                'y': list(self.smooth_buffer_y),
                'ts': time.time(),
            }
            with open(self._smooth_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            pass  # 静默失败，不影响主流程

    # ========== 卡尔曼滤波器 ==========

    def _kalman_update(self, cx, cy, quality=1.0):
        """
        卡尔曼滤波：用观测值 (cx, cy) 校正，返回平滑后的坐标。
        quality: 0-1，匹配质量越低 → 观测噪声越大 → 更信赖预测值。
        """
        if not self._kalman_initialized:
            # 首次初始化状态
            self._kalman.statePost = np.array(
                [[np.float32(cx)], [np.float32(cy)], [0], [0]], dtype=np.float32)
            self._kalman_initialized = True
            return cx, cy

        # 根据质量动态调整观测噪声: 高质量→小噪声(信赖观测), 低质量→大噪声(信赖预测)
        noise_scale = max(1.0, 50.0 * (1.0 - quality))
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise_scale

        # predict → correct
        self._kalman.predict()
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
        corrected = self._kalman.correct(measurement)
        return int(corrected[0, 0]), int(corrected[1, 0])

    def _kalman_predict(self):
        """
        无观测值时使用卡尔曼预测（比惯性重复上一坐标更准确）。
        返回 (predicted_x, predicted_y) 或 None（未初始化时）。
        """
        if not self._kalman_initialized:
            return None
        predicted = self._kalman.predict()
        return int(predicted[0, 0]), int(predicted[1, 0])

    def _kalman_reset(self, cx, cy):
        """重定位后重置卡尔曼状态（如混合引擎注入新坐标）"""
        self._kalman = _create_kalman_filter()
        self._kalman.statePost = np.array(
            [[np.float32(cx)], [np.float32(cy)], [0], [0]], dtype=np.float32)
        self._kalman_initialized = True

    # ========== 场景切换检测 ==========

    @staticmethod
    def _is_invalid_minimap(minimap_bgr):
        """
        检测无效小地图帧（黑屏、Loading、纯色UI覆盖等）。
        判断逻辑: 灰度方差极低 → 接近单色 → 非正常地图画面。
        """
        if minimap_bgr is None or minimap_bgr.size == 0:
            return True
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        mean_val = np.mean(gray)
        # 方差极低 = 近乎纯色（黑屏<20, 白屏, 纯色Loading）
        if variance < 80:
            return True
        # 极暗帧（均值<10 且无明显纹理）
        if mean_val < 10 and variance < 200:
            return True
        return False

    # ========== 线性速度一致性过滤 ==========

    def _linear_filter(self, cx, cy):
        window = getattr(config, 'LINEAR_FILTER_WINDOW', 10)
        max_dev = getattr(config, 'LINEAR_FILTER_MAX_DEVIATION', 300)
        max_consecutive = getattr(config, 'LINEAR_FILTER_MAX_CONSECUTIVE', 5)

        if len(self.pos_history) < window:
            self._linear_filter_consecutive = 0
            return cx, cy

        recent = list(self.pos_history)[-window:]
        n = len(recent)
        vx = sum(recent[i][0] - recent[i - 1][0] for i in range(1, n)) / (n - 1)
        vy = sum(recent[i][1] - recent[i - 1][1] for i in range(1, n)) / (n - 1)

        last_x, last_y = recent[-1]
        pred_x = last_x + vx
        pred_y = last_y + vy

        dev = math.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2)

        if dev > max_dev:
            self._linear_filter_consecutive += 1
            if self._linear_filter_consecutive >= max_consecutive:
                self._linear_filter_consecutive = 0
                print(f"[线性过滤] 连续{max_consecutive}帧超差(偏差{dev:.0f}px)，"
                      f"强制接受真实坐标({cx},{cy})，重置速度模型")
                return cx, cy
            print(f"[线性过滤] 偏差={dev:.0f}px > {max_dev}px → 丢弃 "
                  f"({cx},{cy}) → 用惯性 ({int(pred_x)},{int(pred_y)}) "
                  f"[第{self._linear_filter_consecutive}/{max_consecutive}次]")
            return int(pred_x), int(pred_y)

        self._linear_filter_consecutive = 0
        return cx, cy

    # ========== 公开 API ==========

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
        self.smooth_buffer_x.clear()
        self.smooth_buffer_y.clear()
        if mode == 'loftr':
            self.loftr_engine.state = "GLOBAL_SCAN"
            self.loftr_engine.scan_x = 0
            self.loftr_engine.scan_y = 0
        return True

    def process_frame(self):
        """处理当前帧，返回 (img_base64, jpeg_bytes) 或 None"""
        with self.lock:
            if self.current_frame_bgr is None:
                return None
            minimap_bgr = self.current_frame_bgr.copy()

        half_view = config.VIEW_SIZE // 2

        if self.current_mode == 'sift':
            result = self._process_sift(minimap_bgr, half_view)
        else:
            result = self._process_loftr(minimap_bgr, half_view)

        found, display_crop, status_state, match_count, last_x, last_y = result

        # ====== 坐标异常值过滤 ======
        if found and (last_x or last_y):
            is_outlier = False
            if len(self.pos_history) >= 3:
                hist = list(self.pos_history)
                ref_x = sorted(h[0] for h in hist)[len(hist)//2]
                ref_y = sorted(h[1] for h in hist)[len(hist)//2]
                if abs(last_x - ref_x) > POS_OUTLIER_THRESHOLD or abs(last_y - ref_y) > POS_OUTLIER_THRESHOLD:
                    is_outlier = True
            if not is_outlier:
                self.pos_history.append((last_x, last_y))
            else:
                last_x = self.pos_history[-1][0]
                last_y = self.pos_history[-1][1]
        elif found:
            self.pos_history.append((last_x, last_y))

        # 渲染最终输出
        img_base64, jpeg_bytes = self._render_output(display_crop, half_view)

        self.latest_status = {
            'mode': self.current_mode,
            'state': status_state,
            'position': {'x': last_x, 'y': last_y},
            'found': found,
            'matches': match_count,
            'match_quality': getattr(self, '_last_match_quality', 0),
            'coord_lock': self.sift_engine.coord_lock_enabled,
            'hybrid': self._hybrid_enabled,
            'hybrid_busy': self._hybrid_busy,
        }
        self.latest_result_image = img_base64
        self.latest_result_jpeg = jpeg_bytes

        return img_base64, jpeg_bytes

    # ========== 内部处理方法 ==========

    def _process_sift(self, minimap_bgr, half_view):
        """SIFT 模式的完整处理流程，返回 (found, crop, state, matches, x, y)"""

        # === 场景切换检测：黑屏/Loading/UI全覆盖 ===
        if self._is_invalid_minimap(minimap_bgr):
            # 无效帧：跳过匹配，使用卡尔曼预测维持惯性
            predicted = self._kalman_predict()
            if predicted is not None:
                sx, sy = predicted
                display_crop, status_state = self._render_sift_crop(
                    sx, sy, sx, sy, True, True, self.sift_engine._last_arrow_angle,
                    {'_locked_state': ''}, half_view)
                return True, display_crop, 'SCENE_CHANGE', 0, sx, sy
            display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
            cv2.putText(display_crop, "Scene Change...", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return False, display_crop, 'SCENE_CHANGE', 0, 0, 0

        # === 混合引擎：检查后台 LoFTR 是否有重定位结果 ===
        if self._hybrid_enabled:
            self._hybrid_try_inject()

        # === 坐标锁定模式检测 ===
        if self.sift_engine.coord_lock_enabled:
            anchor = SIFTMapTracker._compute_lock_anchor(
                self.pos_history, self.sift_engine._lock_history_size)
            if anchor is not None:
                result = self.sift_engine._match_locked(minimap_bgr, anchor[0], anchor[1])
                locked_state = result.get('_locked_state', '')
                if result['found'] and not result.get('is_inertial'):
                    self.sift_engine.last_x = result['center_x']
                    self.sift_engine.last_y = result['center_y']
                elif not result['found'] and self.sift_engine.last_x is not None:
                    result['center_x'] = self.sift_engine.last_x
                    result['center_y'] = self.sift_engine.last_y
                    result['found'] = True
                    result['is_inertial'] = True
            else:
                result = self.sift_engine.match(minimap_bgr)
        else:
            result = self.sift_engine.match(minimap_bgr)

        found = result['found']
        cx, cy = result['center_x'], result['center_y']
        arrow_angle = result.get('arrow_angle', 0) or 0
        is_inertial = result.get('is_inertial', False)
        match_quality = result.get('match_quality', 1.0 if found and not is_inertial else 0.0)
        self._last_match_quality = match_quality

        # === 卡尔曼滤波（替代线性外推+中位数平滑）===
        if found and cx is not None and not is_inertial:
            # 真实匹配: 用质量分调整卡尔曼观测噪声
            smooth_x, smooth_y = self._kalman_update(cx, cy, quality=match_quality)
        elif is_inertial:
            # 惯性帧: 用卡尔曼预测（比重复上一坐标准确）
            predicted = self._kalman_predict()
            if predicted is not None:
                smooth_x, smooth_y = predicted
            else:
                smooth_x, smooth_y = cx or 0, cy or 0
        else:
            # 完全丢失
            smooth_x, smooth_y = cx or 0, cy or 0

        # 持久化（保留 smooth_buffer 用于恢复）
        if smooth_x and smooth_y:
            self.smooth_buffer_x.append(smooth_x)
            self.smooth_buffer_y.append(smooth_y)
            if len(self.smooth_buffer_x) >= self.SMOOTH_BUFFER_SIZE:
                self._save_smooth_buffer()

        # 渲染裁剪区域 + 画点
        display_crop, status_state = self._render_sift_crop(
            smooth_x, smooth_y, cx, cy, found, is_inertial, arrow_angle,
            result, half_view)

        match_count = 0
        last_x = smooth_x
        last_y = smooth_y

        # === 混合引擎：SIFT 惯性/丢失/混乱时触发后台 LoFTR ===
        if self._hybrid_enabled:
            sift_confused = getattr(self.sift_engine, '_sift_confused', False)
            if is_inertial or sift_confused:
                self._hybrid_maybe_trigger(minimap_bgr, confused=sift_confused)

        return found, display_crop, status_state, match_count, last_x, last_y

    def _render_sift_crop(self, smooth_x, smooth_y, cx, cy, found, is_inertial, arrow_angle, result, half_view):
        """SIFT 模式的地图裁剪和标记绘制，返回 (display_crop, status_state)"""
        if found and cx is not None:
            ox = getattr(config, 'RENDER_OFFSET_X', 0)
            oy = getattr(config, 'RENDER_OFFSET_Y', 0)

            y1 = max(0, smooth_y + oy - half_view)
            y2 = min(self.map_height, smooth_y + oy + half_view)
            x1 = max(0, smooth_x + ox - half_view)
            x2 = min(self.map_width, smooth_x + ox + half_view)
            display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
            local_x = (smooth_x + ox) - x1
            local_y = (smooth_y + oy) - y1
            if not is_inertial:
                self.sift_engine._draw_arrow_marker(display_crop, local_x, local_y, angle=arrow_angle)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=4, color=(0, 255, 0), thickness=-1)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=7, color=(255, 255, 255), thickness=1)
            else:
                self.sift_engine._draw_arrow_marker(display_crop, local_x, local_y, angle=arrow_angle)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=4, color=(0, 255, 255), thickness=-1)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=7, color=(255, 255, 255), thickness=1)
        else:
            y1 = max(0, self.sift_engine.last_y - half_view)
            y2 = min(self.map_height, self.sift_engine.last_y + half_view)
            x1 = max(0, self.sift_engine.last_x - half_view)
            x2 = min(self.map_width, self.sift_engine.last_x + half_view)
            if x2 > x1 and y2 > y1:
                display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
                local_x = self.sift_engine.last_x - x1
                local_y = self.sift_engine.last_y - y1
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=8, color=(255, 255, 255), thickness=1)
            else:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                lock_label = "🔒 " if self.sift_engine.coord_lock_enabled else ""
                cv2.putText(display_crop, f"{lock_label}Lost...", (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 状态显示
        if self.sift_engine.coord_lock_enabled:
            locked_state = result.get('_locked_state', '')
            if locked_state == 'LOCKED':
                status_state = '🔒锁定'
            elif locked_state in ('LOCK_FAIL', 'NO_FEATURES', 'FEW_KPTS'):
                status_state = '🔒重试中'
            elif is_inertial:
                status_state = '🔒惯性'
            else:
                status_state = 'INERTIAL' if is_inertial else ('FOUND' if found else 'SEARCHING')
        else:
            status_state = 'INERTIAL' if is_inertial else ('FOUND' if found else 'SEARCHING')

        return display_crop, status_state

    def _process_loftr(self, minimap_bgr, half_view):
        """LoFTR 模式的完整处理流程"""
        if self.loftr_engine is None:
            display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
            cv2.putText(display_crop, "AI Engine Disabled", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            return False, display_crop, 'DISABLED', 0, 0, 0

        result = self.loftr_engine.match(minimap_bgr)
        display_crop = result.get('display_crop')
        if display_crop is None:
            display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
        found = result['found']
        status_state = result.get('state', 'GLOBAL_SCAN')
        match_count = result.get('match_count', 0)
        last_x = result.get('last_x', 0)
        last_y = result.get('last_y', 0)
        return found, display_crop, status_state, match_count, last_x, last_y

    def _render_output(self, display_crop, half_view):
        """最终渲染: OpenCV 直接 JPEG 编码（替代 PIL，减少 RGB 转换和内存拷贝）"""
        view_size = config.VIEW_SIZE
        # 创建 BGR 画布并居中粘贴
        final_bgr = np.full((view_size, view_size, 3), 43, dtype=np.uint8)
        h, w = display_crop.shape[:2]
        y_off = max(0, half_view - h // 2)
        x_off = max(0, half_view - w // 2)
        paste_h = min(h, view_size - y_off)
        paste_w = min(w, view_size - x_off)
        final_bgr[y_off:y_off + paste_h, x_off:x_off + paste_w] = display_crop[:paste_h, :paste_w]

        # 直接 JPEG 编码（跳过 BGR→RGB→PIL→JPEG 的多次拷贝）
        _, jpeg_buf = cv2.imencode('.jpg', final_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = jpeg_buf.tobytes()

        img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode('utf-8')

        return img_base64, jpeg_bytes

    # ========== 混合引擎：后台 LoFTR 重定位 ==========

    def _hybrid_maybe_trigger(self, minimap_bgr, confused=False):
        """
        检查是否应该触发后台 LoFTR 重定位。
        条件: (SIFT 连续失败 N 帧 OR SIFT 几何混乱) + 冷却时间已过 + 后台无任务运行。
        """
        lost = self.sift_engine.lost_frames
        trigger_threshold = getattr(config, 'HYBRID_TRIGGER_LOST_FRAMES', 5)
        cooldown = getattr(config, 'HYBRID_COOLDOWN', 3.0)
        confused_immediate = getattr(config, 'HYBRID_CONFUSED_IMMEDIATE', True)

        # 判断是否应触发
        should_trigger = False
        trigger_reason = ""
        if confused and confused_immediate:
            should_trigger = True
            trigger_reason = "SIFT 几何混乱(scale异常)"
        elif lost >= trigger_threshold:
            should_trigger = True
            trigger_reason = f"SIFT 连续丢失 {lost} 帧"

        if not should_trigger:
            return
        if self._hybrid_busy:
            return
        if (time.time() - self._hybrid_last_trigger) < cooldown:
            return

        # 取当前帧快照送入后台
        snapshot = minimap_bgr.copy()
        self._hybrid_last_trigger = time.time()
        self._hybrid_busy = True

        t = Thread(target=self._hybrid_worker, args=(snapshot,), daemon=True)
        t.name = "LoFTR-Relocate"
        t.start()
        self._hybrid_thread = t
        print(f"[混合引擎] 🔍 触发后台 LoFTR 重定位 ({trigger_reason})")

    def _hybrid_worker(self, minimap_bgr):
        """
        后台线程：调用 LoFTR.relocate() 执行金字塔重定位。
        结果写入 _hybrid_result，由主线程在下一帧读取注入。
        """
        try:
            if self._hybrid_stop.is_set():
                return
            found, rx, ry = self.loftr_engine.relocate(minimap_bgr)
            if found and rx is not None:
                with self._hybrid_result_lock:
                    self._hybrid_result = (rx, ry)
        except Exception as e:
            print(f"[混合引擎] LoFTR 重定位异常: {e}")
        finally:
            self._hybrid_busy = False

    def _hybrid_try_inject(self):
        """
        主线程每帧调用：检查后台 LoFTR 是否有结果。
        如果有，注入到 SIFT 引擎以恢复追踪。
        """
        result = None
        with self._hybrid_result_lock:
            if self._hybrid_result is not None:
                result = self._hybrid_result
                self._hybrid_result = None

        if result is None:
            return

        rx, ry = result
        with self.sift_engine._lock:
            self.sift_engine.last_x = rx
            self.sift_engine.last_y = ry
            self.sift_engine.lost_frames = 0
            self.sift_engine._switch_to_local(rx, ry)
        # 重置卡尔曼状态以匹配新位置
        self._kalman_reset(rx, ry)
        print(f"[混合引擎] ✅ 已注入 LoFTR 坐标 ({rx}, {ry}) → SIFT 局部搜索已重建")
