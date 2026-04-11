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
from PIL import Image as PILImage
from io import BytesIO
import base64
from threading import Lock
from collections import deque

import config
from tracker_engines import SIFTMapTracker, LoFTRMapTracker


# 坐标过滤参数
POS_HISTORY_SIZE = 20      # 保留最近 N 次坐标
POS_OUTLIER_THRESHOLD = 200 # 超过此距离视为异常帧，丢弃


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

        # 坐标平滑缓冲池（60帧滑动窗口 + 中位数滤波，消除 SIFT 抖动）
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
            'coord_lock': self.sift_engine.coord_lock_enabled,
        }
        self.latest_result_image = img_base64
        self.latest_result_jpeg = jpeg_bytes

        return img_base64, jpeg_bytes

    # ========== 内部处理方法 ==========

    def _process_sift(self, minimap_bgr, half_view):
        """SIFT 模式的完整处理流程，返回 (found, crop, state, matches, x, y)"""
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

        # === 线性速度一致性过滤 ===
        if (found and cx is not None and not is_inertial
                and getattr(config, 'LINEAR_FILTER_ENABLED', True)
                and self.sift_engine.coord_lock_enabled):
            filtered_cx, filtered_cy = self._linear_filter(cx, cy)
            if filtered_cx != cx or filtered_cy != cy:
                cx, cy = filtered_cx, filtered_cy
                is_inertial = True

        # 坐标平滑
        smooth_x, smooth_y = self._smooth_coord(cx, cy)

        # 渲染裁剪区域 + 画点
        display_crop, status_state = self._render_sift_crop(
            smooth_x, smooth_y, cx, cy, found, is_inertial, arrow_angle,
            result, half_view)

        match_count = 0
        last_x = smooth_x
        last_y = smooth_y
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
        """最终渲染: BGR->RGB -> PIL -> PNG base64 + JPEG bytes"""
        display_rgb = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(display_rgb)
        final_img = PILImage.new('RGB', (config.VIEW_SIZE, config.VIEW_SIZE), (43, 43, 43))
        final_img.paste(pil_image,
                        (max(0, half_view - pil_image.width // 2), max(0, half_view - pil_image.height // 2)))

        buffer_png = BytesIO()
        final_img.save(buffer_png, format='PNG')
        img_base64 = base64.b64encode(buffer_png.getvalue()).decode('utf-8')

        buffer_jpeg = BytesIO()
        final_img.save(buffer_jpeg, format='JPEG', quality=85, optimize=True)
        jpeg_bytes = buffer_jpeg.getvalue()

        return img_base64, jpeg_bytes
