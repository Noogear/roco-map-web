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


class _CircleCalibrator:
    """
    运行时自动校准小地图圆形参数 (半径、圆心位置)。
    前 N 帧为校准期，所有检测到的圆都放行；
    校准收敛后，通过半径+圆心偏移严格过滤异常检测(地图展开、战斗误检)。
    连续长时间未检测到 → 自动重置校准(适应分辨率/UI变化)。
    """

    def __init__(self):
        self._history = deque(maxlen=30)
        self._calibrated = False
        self.expected_cx = None
        self.expected_cy = None
        self.expected_r = None
        self._consecutive_miss = 0

    def update(self, cx, cy, r):
        self._history.append((cx, cy, r))
        self._consecutive_miss = 0
        n_cal = getattr(config, 'MINIMAP_CIRCLE_CALIBRATION_FRAMES', 8)
        if len(self._history) >= n_cal:
            rs = [d[2] for d in self._history]
            cxs = [d[0] for d in self._history]
            cys = [d[1] for d in self._history]
            self.expected_r = int(np.median(rs))
            self.expected_cx = int(np.median(cxs))
            self.expected_cy = int(np.median(cys))
            if not self._calibrated:
                self._calibrated = True
                print(f"  [圆校准] 已收敛: center=({self.expected_cx},{self.expected_cy}), r={self.expected_r}")

    def is_valid(self, cx, cy, r):
        if not self._calibrated:
            return True
        r_tol = getattr(config, 'MINIMAP_CIRCLE_R_TOLERANCE', 8)
        c_tol = getattr(config, 'MINIMAP_CIRCLE_CENTER_TOLERANCE', 15)
        return (abs(r - self.expected_r) <= r_tol and
                abs(cx - self.expected_cx) <= c_tol and
                abs(cy - self.expected_cy) <= c_tol)

    def record_miss(self):
        self._consecutive_miss += 1
        n_reset = getattr(config, 'MINIMAP_CIRCLE_RECALIBRATE_MISS', 30)
        if self._consecutive_miss >= n_reset and self._calibrated:
            self._history.clear()
            self._calibrated = False
            self._consecutive_miss = 0
            print("[圆校准] 连续未检测到小地图圆，重置校准（可能分辨率/UI变化）")


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
        self.lock = Lock()          # 保护 current_frame_bgr 的读写
        self._process_lock = Lock() # 防止后台线程与 HTTP 路由并发跑 SIFT

        # 当前帧数据
        self.current_frame_bgr = None
        self.latest_result_image = None
        self.latest_result_jpeg = None
        self.latest_result_frame = None
        self._render_revision = 0
        self._latest_result_image_revision = -1
        self._latest_result_jpeg_revision = -1

        # 坐标历史（用于异常值过滤）
        self.pos_history = deque(maxlen=POS_HISTORY_SIZE)

        # 线性过滤器：连续丢弃帧计数器（防死锁）
        self._linear_filter_consecutive = 0

        # === 传送检测：候选聚类确认缓冲（防误判）===
        # 连续 TP_CONFIRM_FRAMES 帧 SIFT 稳定匹配到同一远处新位置才视为传送，
        # 随机噪声帧在空间上分散，不会通过聚类校验，因此不会误触。
        self._tp_candidate_buffer = deque(maxlen=getattr(config, 'TP_CONFIRM_FRAMES', 3))

        # === 圆形小地图检测器 (方形截取 + HoughCircles 自动校准) ===
        self._circle_cal = _CircleCalibrator()

        # === 卡尔曼滤波器（替代线性外推+中位数平滑）===
        self._kalman = _create_kalman_filter()
        self._kalman_initialized = False  # 第一次观测后才开始滤波

        # 坐标平滑缓冲池（60帧滑动窗口 + 中位数滤波，消除 SIFT 抖动）
        # 注：卡尔曼滤波作为主滤波器，smooth_buffer 仅用于持久化恢复
        self.SMOOTH_BUFFER_SIZE = 60
        self.smooth_buffer_x = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self.smooth_buffer_y = deque(maxlen=self.SMOOTH_BUFFER_SIZE)

        # 渲染静止死区：记录上一帧实际用于渲染的坐标，避免噪声引起地图抖动
        self._display_x = None
        self._display_y = None
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

        # ========== 后台帧处理线程（解耦 SIFT 与 WebSocket 热路径）==========
        # SIFT 匹配运行在独立线程，WebSocket handler 可立即返回上一帧缓存结果。
        # 自动跳帧：若处理跟不上帧率，Event 被多次 set() 后只唤醒一次，
        # 每次 wait() 后取到的是最新的 current_frame_bgr（旧帧自动丢弃）。
        # Plan B: _push_jpeg=False 时跳过 cv2.imencode，节省 10-15ms/帧。
        # 由 main_web.py 根据客户端类型（frame / frame_coords）动态切换。
        self._push_jpeg = True
        self._new_frame_event = Event()
        self._worker_thread = Thread(
            target=self._background_processor,
            daemon=True,
            name='sift-worker',
        )
        self._worker_thread.start()
        print("  ⚡ SIFT 后台处理线程已启动")

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

    def _detect_and_extract_minimap(self, square_bgr):
        """
        从方形截取区域中检测圆形小地图并提取。
        利用 HoughCircles 几何检测（替代启发式阈值）判定小地图是否存在：
          - 正常帧: 检测到圆 → 提取圆内方形区域供 SIFT/LoFTR 匹配
          - 战斗/暂停/UI: 无圆 → 返回 None
          - 地图展开: 检测到圆但半径异常 → 校准器拒绝 → 返回 None

        自动校准: 前 N 帧收集统计 → 自动收紧半径/圆心容差 → 适应任何分辨率

        Returns:
            numpy.ndarray — 提取的小地图 (tight square, 与原圆形裁剪格式一致),
            或 None (无有效小地图)
        """
        if square_bgr is None or square_bgr.size == 0:
            return None

        gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        min_dim = min(h, w)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1.2, min_dim // 2,
            param1=80, param2=35,
            minRadius=int(min_dim * 0.25),
            maxRadius=int(min_dim * 0.48)
        )

        if circles is None or len(circles[0]) == 0:
            if not self.sift_engine.frozen:
                self._circle_cal.record_miss()
            return None

        # 取最靠近中心的圆（小地图应在截取区域中央）
        cx_img, cy_img = w / 2, h / 2
        best, best_dist = None, float('inf')
        for c in circles[0]:
            d = math.sqrt((c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2)
            if d < best_dist:
                best_dist = d
                best = c

        det_cx, det_cy, det_r = int(best[0]), int(best[1]), int(best[2])

        # 校准验证（校准后拒绝半径/位置异常的圆，如地图展开 r 偏大）
        if not self._circle_cal.is_valid(det_cx, det_cy, det_r):
            if not self.sift_engine.frozen:
                self._circle_cal.record_miss()
            return None

        self._circle_cal.update(det_cx, det_cy, det_r)

        # 提取圆内方形区域（与之前引擎输入格式一致）
        x1, y1 = max(0, det_cx - det_r), max(0, det_cy - det_r)
        x2, y2 = min(w, det_cx + det_r), min(h, det_cy + det_r)
        minimap = square_bgr[y1:y2, x1:x2].copy()

        # 基础防护：过小或纯色（Loading 时的圆形 UI 元素）
        if minimap.size < 100:
            return None
        if np.var(cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)) < 80:
            return None

        return minimap

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
        self._new_frame_event.set()  # 唤醒后台 SIFT 工作线程

    def _background_processor(self):
        """
        后台 SIFT 工作线程：持续处理最新帧，不阻塞 WebSocket 接收。

        Event 语义保证自动跳帧：多帧对应同一个 Event.set()，
        clear() 后 current_frame_bgr 始终是最新帧（旧帧自动丢弃）。
        """
        while True:
            self._new_frame_event.wait()
            self._new_frame_event.clear()
            try:
                self.process_frame(need_base64=False, need_jpeg=self._push_jpeg)
            except Exception as e:
                print(f"[sift-worker] 处理异常: {e}")

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

    def process_frame(self, need_base64=True, need_jpeg=True):
        """
        处理当前帧，返回 (img_base64, jpeg_bytes) 或 None。

        need_base64:
            是否立即生成 base64 图片（HTTP JSON 返回使用）
        need_jpeg:
            是否立即生成 JPEG 字节（WebSocket / 最新帧接口使用）
        """
        # 防止后台线程与 HTTP 同步路由（upload_minimap / /api/process）并发跑 SIFT。
        # 后来的调用者等前一次完成后才进入，避免双重处理和结果互相覆盖。
        with self._process_lock:
            return self._process_frame_locked(need_base64, need_jpeg)

    def _process_frame_locked(self, need_base64, need_jpeg):
        """process_frame 的实际实现，调用前必须持有 _process_lock。"""
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

        # ====== 坐标异常值过滤（欧氏距离，避免对角线方向失衡） ======
        if found and (last_x or last_y):
            is_outlier = False
            if len(self.pos_history) >= 3:
                hist = list(self.pos_history)
                ref_x = sorted(h[0] for h in hist)[len(hist)//2]
                ref_y = sorted(h[1] for h in hist)[len(hist)//2]
                dist = math.sqrt((last_x - ref_x) ** 2 + (last_y - ref_y) ** 2)
                if dist > POS_OUTLIER_THRESHOLD:
                    is_outlier = True
            if not is_outlier:
                self.pos_history.append((last_x, last_y))
            else:
                last_x = self.pos_history[-1][0]
                last_y = self.pos_history[-1][1]
        elif found:
            self.pos_history.append((last_x, last_y))

        final_bgr = self._compose_output_frame(display_crop, half_view)

        img_base64 = None
        jpeg_bytes = None
        if need_base64 or need_jpeg:
            img_base64, jpeg_bytes = self._encode_output_frame(
                final_bgr,
                need_base64=need_base64,
                need_jpeg=True,
            )

        self.latest_status = {
            'mode': self.current_mode,
            'state': status_state,
            'position': {'x': last_x, 'y': last_y},
            'found': found,
            'matches': match_count,
            'match_quality': getattr(self, '_last_match_quality', 0),
            'arrow_angle': getattr(self, '_last_arrow_angle_out', 0),
            'arrow_stopped': getattr(self, '_last_arrow_stopped_out', True),
            'coord_lock': self.sift_engine.coord_lock_enabled,
            'hybrid': self._hybrid_enabled,
            'hybrid_busy': self._hybrid_busy,
        }

        self.latest_result_frame = final_bgr
        self._render_revision += 1
        self._latest_result_image_revision = -1
        self._latest_result_jpeg_revision = -1

        if jpeg_bytes is not None:
            self.latest_result_jpeg = jpeg_bytes
            self._latest_result_jpeg_revision = self._render_revision
        else:
            self.latest_result_jpeg = None

        if img_base64 is not None:
            self.latest_result_image = img_base64
            self._latest_result_image_revision = self._render_revision
        else:
            self.latest_result_image = None

        return img_base64, jpeg_bytes

    # ========== 内部处理方法 ==========

    def _process_sift(self, minimap_bgr, half_view):
        """SIFT 模式的完整处理流程，返回 (found, crop, state, matches, x, y)"""

        # === 圆形小地图检测：从方形截取中定位并提取 ===
        extracted = self._detect_and_extract_minimap(minimap_bgr)
        if extracted is None:
            # 无有效小地图(战斗/暂停/UI/地图展开)

            # 冻结 SIFT 引擎状态（仅首次进入 BLOCK 时触发）
            self.sift_engine._freeze_state()

            # 冻结期间：优先返回冻结坐标（避免 Kalman 发散）
            frozen_pos = self.sift_engine.frozen_position
            if frozen_pos is not None:
                sx, sy = frozen_pos
                display_crop, status_state = self._render_sift_crop(
                    sx, sy, sx, sy, True, True, self.sift_engine._last_arrow_angle,
                    self.sift_engine._last_arrow_stopped,
                    {'_locked_state': ''}, half_view)
                return True, display_crop, 'SCENE_CHANGE', 0, sx, sy

            # 无冻结坐标 → 尝试 Kalman 预测
            predicted = self._kalman_predict()
            if predicted is not None:
                sx, sy = predicted
                display_crop, status_state = self._render_sift_crop(
                    sx, sy, sx, sy, True, True, self.sift_engine._last_arrow_angle,
                    self.sift_engine._last_arrow_stopped,
                    {'_locked_state': ''}, half_view)
                return True, display_crop, 'SCENE_CHANGE', 0, sx, sy
            display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
            cv2.putText(display_crop, "Scene Change...", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return False, display_crop, 'SCENE_CHANGE', 0, 0, 0

        # 小地图重现：解冻引擎状态（恢复局部索引+坐标）
        self.sift_engine._thaw_state()
        minimap_bgr = extracted  # 提取的圆内区域，与之前引擎输入格式一致

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
        arrow_stopped = result.get('arrow_stopped', True)
        self._last_arrow_angle_out = arrow_angle
        self._last_arrow_stopped_out = arrow_stopped
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
                # 用预测值更新卡尔曼状态，保持速度估计活性
                self._kalman_update(smooth_x, smooth_y, quality=0.3)
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

        # === 传送检测：候选聚类确认 ===
        # 来源1: 真实匹配帧的卡尔曼输出超跳变阈值（原有逻辑）
        # 来源2: SIFT 引擎全局搜索发现远处高质量匹配但被跳变过滤丢弃（新增）
        _tp_thresh = getattr(config, 'TP_JUMP_THRESHOLD', 300)
        _tp_new_entry = False  # 本帧是否有新的传送候选进入缓冲

        # 来源2：引擎传回的远距离候选（绕过了 JUMP_THRESHOLD 但质量达标）
        _far_cand = result.get('_tp_far_candidate')
        if _far_cand is not None and self._display_x is not None:
            _fc_x, _fc_y, _fc_q = _far_cand
            _fc_jump = math.sqrt(
                (_fc_x - self._display_x) ** 2 + (_fc_y - self._display_y) ** 2
            )
            if _fc_jump > _tp_thresh:
                self._tp_candidate_buffer.append((_fc_x, _fc_y))
                _tp_new_entry = True

        # 来源1: 真实匹配帧的卡尔曼输出超跳变阈值
        if (found and not is_inertial and cx is not None
                and self._display_x is not None):
            _tp_jump = math.sqrt(
                (smooth_x - self._display_x) ** 2 + (smooth_y - self._display_y) ** 2
            )
            if _tp_jump > _tp_thresh:
                self._tp_candidate_buffer.append((smooth_x, smooth_y))
                _tp_new_entry = True
            else:
                # 跳变量正常 → 是正常移动，清除候选缓冲
                if self._tp_candidate_buffer:
                    self._tp_candidate_buffer.clear()

        # 聚类确认（不限制必须在 found+非惯性帧内，来源2的候选也能触发）
        _tp_confirm = getattr(config, 'TP_CONFIRM_FRAMES', 3)
        _tp_radius = getattr(config, 'TP_CLUSTER_RADIUS', 150)
        if _tp_new_entry and len(self._tp_candidate_buffer) >= _tp_confirm:
            _cands = list(self._tp_candidate_buffer)
            _med_x = sorted(c[0] for c in _cands)[len(_cands) // 2]
            _med_y = sorted(c[1] for c in _cands)[len(_cands) // 2]
            _scatter = max(
                math.sqrt((c[0] - _med_x) ** 2 + (c[1] - _med_y) ** 2)
                for c in _cands
            )
            if _scatter <= _tp_radius:
                # 聚类确认：立即重置所有过滤器到新位置，消除延迟
                print(f"[传送检测] 聚类确认 scatter={_scatter:.0f}px → "
                      f"立即重置到 ({_med_x},{_med_y})")
                self.pos_history.clear()
                self.smooth_buffer_x.clear()
                self.smooth_buffer_y.clear()
                self._kalman_reset(_med_x, _med_y)
                self._linear_filter_consecutive = 0
                self._display_x = _med_x
                self._display_y = _med_y
                self._tp_candidate_buffer.clear()
                smooth_x, smooth_y = _med_x, _med_y
                # 同步 SIFT 引擎位置，让后续帧立即在新位置局部搜索
                with self.sift_engine._lock:
                    self.sift_engine.last_x = _med_x
                    self.sift_engine.last_y = _med_y
                    self.sift_engine.lost_frames = 0
                    self.sift_engine._switch_to_local(_med_x, _med_y)

        # === 渲染平滑防抖：静止死区 + 速度自适应 EMA ===
        # alpha 随速度线性增大：慢速平滑抗抖，快速立即跟随避免视觉滞后
        #   dist ≤ still_threshold → 静止死区（不动）
        #   dist ≤ ema_slow_dist  → alpha = ema_alpha_min（最平滑）
        #   dist ≥ ema_fast_dist  → alpha = ema_alpha_max（立即跟随）
        #   中间线性插值
        still_threshold = getattr(config, 'RENDER_STILL_THRESHOLD', 2)
        ema_alpha_min = getattr(config, 'RENDER_EMA_ALPHA', 0.35)     # 慢速最低 alpha
        ema_alpha_max = getattr(config, 'RENDER_EMA_ALPHA_MAX', 0.92) # 快速最高 alpha
        ema_slow_dist = getattr(config, 'RENDER_EMA_SLOW_DIST', 6)    # 低于此速度用 min
        ema_fast_dist = getattr(config, 'RENDER_EMA_FAST_DIST', 45)   # 高于此速度用 max
        if found and smooth_x and smooth_y:
            if self._display_x is None:
                # 首帧直接初始化
                self._display_x = smooth_x
                self._display_y = smooth_y
            elif not is_inertial:
                ddx = smooth_x - self._display_x
                ddy = smooth_y - self._display_y
                dist = math.sqrt(ddx * ddx + ddy * ddy)
                if dist <= still_threshold:
                    # 静止死区：保持不动
                    pass
                else:
                    # 速度自适应 alpha：慢→平滑，快→立即跟随
                    t = max(0.0, min(1.0, (dist - ema_slow_dist) / (ema_fast_dist - ema_slow_dist)))
                    adaptive_alpha = ema_alpha_min + t * (ema_alpha_max - ema_alpha_min)
                    self._display_x = int(self._display_x + adaptive_alpha * ddx)
                    self._display_y = int(self._display_y + adaptive_alpha * ddy)
            else:
                # 惯性帧（卡尔曼预测）：同样速度自适应，预测值不应锁死
                ddx = smooth_x - self._display_x
                ddy = smooth_y - self._display_y
                dist = math.sqrt(ddx * ddx + ddy * ddy)
                t = max(0.0, min(1.0, (dist - ema_slow_dist) / (ema_fast_dist - ema_slow_dist)))
                adaptive_alpha = ema_alpha_min + t * (ema_alpha_max - ema_alpha_min)
                self._display_x = int(self._display_x + adaptive_alpha * ddx)
                self._display_y = int(self._display_y + adaptive_alpha * ddy)
            smooth_x, smooth_y = self._display_x, self._display_y

        # 渲染裁剪区域 + 画点
        display_crop, status_state = self._render_sift_crop(
            smooth_x, smooth_y, cx, cy, found, is_inertial, arrow_angle, arrow_stopped,
            result, half_view)

        match_count = 0
        last_x = smooth_x
        last_y = smooth_y

        # 取出 SIFT 引擎返回的实际匹配内点数
        if found and not is_inertial:
            match_count = result.get('match_count', 0)

        # === 混合引擎：SIFT 惯性/丢失/混乱时触发后台 LoFTR ===
        if self._hybrid_enabled:
            sift_confused = getattr(self.sift_engine, '_sift_confused', False)
            if is_inertial or sift_confused:
                self._hybrid_maybe_trigger(minimap_bgr, confused=sift_confused)

        return found, display_crop, status_state, match_count, last_x, last_y

    def _render_sift_crop(self, smooth_x, smooth_y, cx, cy, found, is_inertial, arrow_angle, arrow_stopped, result, half_view):
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
                self.sift_engine._draw_arrow_marker(display_crop, local_x, local_y, angle=arrow_angle, stopped=arrow_stopped)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=4, color=(0, 255, 0), thickness=-1)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=7, color=(255, 255, 255), thickness=1)
            else:
                self.sift_engine._draw_arrow_marker(display_crop, local_x, local_y, angle=arrow_angle, stopped=arrow_stopped)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=4, color=(0, 255, 255), thickness=-1)
                cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=7, color=(255, 255, 255), thickness=1)
        else:
            if self.sift_engine.last_x is not None and self.sift_engine.last_y is not None:
                y1 = max(0, self.sift_engine.last_y - half_view)
                y2 = min(self.map_height, self.sift_engine.last_y + half_view)
                x1 = max(0, self.sift_engine.last_x - half_view)
                x2 = min(self.map_width, self.sift_engine.last_x + half_view)
            else:
                x1 = x2 = y1 = y2 = 0

            if (self.sift_engine.last_x is not None and self.sift_engine.last_y is not None and
                    x2 > x1 and y2 > y1):
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

    def _compose_output_frame(self, display_crop, half_view):
        """将裁剪地图贴到固定大小画布上，返回最终 BGR 帧。"""
        view_size = config.VIEW_SIZE
        # 创建 BGR 画布并居中粘贴
        final_bgr = np.full((view_size, view_size, 3), 43, dtype=np.uint8)
        h, w = display_crop.shape[:2]
        y_off = max(0, half_view - h // 2)
        x_off = max(0, half_view - w // 2)
        paste_h = min(h, view_size - y_off)
        paste_w = min(w, view_size - x_off)
        final_bgr[y_off:y_off + paste_h, x_off:x_off + paste_w] = display_crop[:paste_h, :paste_w]

        return final_bgr

    def _encode_output_frame(self, final_bgr, need_base64=True, need_jpeg=True):
        """
        按需编码最终帧。

        注意：base64 本质上也是 JPEG 的文本包装，因此一旦请求 base64，
        会复用同一次 JPEG 编码结果，避免重复压缩。
        """
        if not need_base64 and not need_jpeg:
            return None, None

        # 直接 JPEG 编码（跳过 BGR→RGB→PIL→JPEG 的多次拷贝）
        _, jpeg_buf = cv2.imencode('.jpg', final_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = jpeg_buf.tobytes()

        img_base64 = None
        if need_base64:
            img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode('utf-8')

        return img_base64, jpeg_bytes

    def get_latest_result_jpeg(self):
        """返回最新 JPEG；若当前帧尚未编码则按需懒生成。"""
        if self.latest_result_frame is None:
            return None
        if (self._latest_result_jpeg_revision == self._render_revision and
                self.latest_result_jpeg is not None):
            return self.latest_result_jpeg

        _, jpeg_bytes = self._encode_output_frame(self.latest_result_frame, need_base64=False, need_jpeg=True)
        self.latest_result_jpeg = jpeg_bytes
        self._latest_result_jpeg_revision = self._render_revision
        return jpeg_bytes

    def get_latest_result_base64(self):
        """返回最新 base64；若当前帧尚未编码则按需懒生成。"""
        if self.latest_result_frame is None:
            return None
        if (self._latest_result_image_revision == self._render_revision and
                self.latest_result_image is not None):
            return self.latest_result_image

        jpeg_bytes = self.get_latest_result_jpeg()
        if not jpeg_bytes:
            return None

        img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode('utf-8')
        self.latest_result_image = img_base64
        self._latest_result_image_revision = self._render_revision
        return img_base64

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
