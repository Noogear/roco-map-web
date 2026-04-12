"""
tracker_core.py - 追踪器编排层（无 Web 框架依赖）

职责:
  - 管理引擎实例(SIFT)的生命周期
  - 坐标平滑/线性过滤/异常值过滤
  - 帧处理流程编排: set_minimap -> process_frame
  - 结果渲染(裁剪地图 + 画箭头 + 编码PNG/JPEG)
  - 坐标历史持久化

此模块仅依赖 cv2 / numpy / PIL / config / tracker_engines，
可被 Flask/Tornado/CLI 等任何前端复用。
"""

import os

import cv2
import numpy as np
import base64
from threading import Lock, Thread, Event

from backend import config
from backend.tracker_engines import SIFTMapTracker
from backend.tracking.minimap import CircleCalibrator, detect_and_extract
from backend.tracking.smoother import CoordSmoother


class MapTrackerWeb:
    """
    地图追踪编排器：封装引擎调用 + 坐标后处理 + 渲染输出。
    不包含任何 Web/Flask/SocketIO 代码。
    """

    def __init__(self):
        # --- 加载引擎 ---
        print("=" * 50)
        self.sift_engine = SIFTMapTracker()
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

        # 线性过滤器连续丢弃帧计数器已迁至 CoordSmoother

        # 传送确认标记（由 CoordSmoother.update 返回）
        self._tp_just_confirmed = False  # 当帧传送聚类确认成功，通知前端直接跳位

        # === SCENE_CHANGE 防抖：挂机时单帧检测失败不立即进入冻结状态 ===
        # 加载画面(渐黑 fade)通常持续 >30 帧，2 帧 debounce 不影响真实场景切换
        # 但能吸收 HoughCircles 或方差检测的偶发单帧误判
        self._scene_change_streak = 0
        self._scene_change_debounce = getattr(config, 'SCENE_CHANGE_DEBOUNCE', 2)

        # === 圆形小地图检测器 (方形截取 + HoughCircles 自动校准) ===
        self._circle_cal = CircleCalibrator()

        # === 坐标平滑器（卡尔曼 + EMA 防抖 + TP 检测缓冲）===
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._smooth_file = os.path.join(_base_dir, '.smooth_coords.json')
        self._smoother = CoordSmoother(smooth_buffer_path=self._smooth_file)
        self.pos_history = self._smoother.pos_history  # 公开属性，server.py 等外部代码使用

        self.latest_status = {
            'mode': 'sift',
            'state': '--',
            'position': {'x': 0, 'y': 0},
            'found': False,
            'matches': 0,
        }

        # ========== 后台帧处理线程（解耦 SIFT 与 WebSocket 热路径）==========
        # SIFT 匹配运行在独立线程，WebSocket handler 可立即返回上一帧缓存结果。
        # 自动跳帧：若处理跟不上帧率，Event 被多次 set() 后只唤醒一次，
        # 每次 wait() 后取到的是最新的 current_frame_bgr（旧帧自动丢弃）。
        # Plan B: _push_jpeg=False 时跳过 cv2.imencode，节省 10-15ms/帧。
        # 由 main_web.py 根据客户端类型（frame / frame_coords）动态切换。
        self._push_jpeg = True
        # Plan A: result_callback 由 main_web.py 注册，处理完后立即推送新结果。
        # 叠加式：pull 逻辑仍保留作兜底，callback 失败不影响正常工作。
        self.result_callback = None
        self._new_frame_event = Event()
        self._worker_thread = Thread(
            target=self._background_processor,
            daemon=True,
            name='sift-worker',
        )
        self._worker_thread.start()
        print("  ⚡ SIFT 后台处理线程已启动")

    # ========== 场景切换检测 ==========

    def _detect_and_extract_minimap(self, square_bgr):
        return detect_and_extract(square_bgr, self._circle_cal, self.sift_engine.frozen)

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
                continue
            if self.result_callback is not None:
                try:
                    self.result_callback()
                except Exception as cb_e:
                    print(f"[sift-worker] result_callback 异常: {cb_e}")

    def set_mode(self, mode):
        """模式切换（仅支持 sift）"""
        return mode == 'sift'

    def reset_history(self):
        """清空平滑/历史状态并关闭坐标锁定，返回 (cleared_count, was_locked)。"""
        engine = self.sift_engine
        cleared = len(self.pos_history)
        was_locked = engine.coord_lock_enabled

        self._smoother.clear_state()
        engine.set_coord_lock(False)
        return cleared, was_locked

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

        result = self._process_sift(minimap_bgr, half_view)

        found, display_crop, status_state, match_count, last_x, last_y = result

        # ====== 坐标异常值过滤（欧氏距离，避免对角线方向失衡） ======
        # CoordSmoother.update() 已在内部完成异常值过滤并维护 pos_history，此处无需重复

        final_bgr = self._compose_output_frame(display_crop, half_view)

        img_base64 = None
        jpeg_bytes = None
        if need_base64 or need_jpeg:
            img_base64, jpeg_bytes = self._encode_output_frame(
                final_bgr,
                need_base64=need_base64,
                need_jpeg=True,
            )

        _is_tp = self._tp_just_confirmed
        self._tp_just_confirmed = False
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
            'is_teleport': _is_tp,
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

            # SCENE_CHANGE debounce：需连续 N 帧失败才真正进入冻结状态
            # 防止挂机时单帧 HoughCircles 抖动或低纹理方差误判引起"重定位中"闪烁
            self._scene_change_streak += 1
            _debounce = self._scene_change_debounce
            if self._scene_change_streak < _debounce and self.sift_engine.last_x is not None:
                # 未达 debounce 阈值 + 引擎有上一有效坐标：用惯性代替 SCENE_CHANGE
                sx, sy = self.sift_engine.last_x, self.sift_engine.last_y
                display_crop, _ = self._render_sift_crop(
                    sx, sy, sx, sy, True, True,
                    self.sift_engine._last_arrow_angle,
                    self.sift_engine._last_arrow_stopped,
                    {'_locked_state': ''}, half_view)
                return True, display_crop, 'INERTIAL', 0, sx, sy

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
            predicted = self._smoother._kalman_predict()
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
        self._scene_change_streak = 0   # 重置连续失帧计数
        minimap_bgr = extracted  # 提取的圆内区域，与之前引擎输入格式一致

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

        # === 坐标平滑 + 传送检测（统一由 CoordSmoother 处理）===
        smooth_x, smooth_y, tp_confirmed = self._smoother.update(
            cx, cy, found, is_inertial, match_quality, arrow_stopped,
            tp_far_candidate=result.get('_tp_far_candidate'),
        )
        if tp_confirmed:
            self._tp_just_confirmed = True
            # 同步 SIFT 引擎位置，让后续帧立即在新位置局部搜索
            with self.sift_engine._lock:
                self.sift_engine.last_x = smooth_x
                self.sift_engine.last_y = smooth_y
                self.sift_engine.lost_frames = 0
                self.sift_engine._switch_to_local(smooth_x, smooth_y)

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

        return found, display_crop, status_state, match_count, last_x, last_y

    def _render_sift_crop(self, smooth_x, smooth_y, cx, cy, found, is_inertial, arrow_angle, arrow_stopped, result, half_view):
        """SIFT 模式的地图裁剪，返回 (display_crop, status_state)。
        扩边 JPEG_PAD 像素供前端亚帧微平移；箭头/圆点由前端 60fps 叠加，此处不绘制。
        """
        pad = getattr(config, 'JPEG_PAD', 0)
        if found and cx is not None:
            ox = getattr(config, 'RENDER_OFFSET_X', 0)
            oy = getattr(config, 'RENDER_OFFSET_Y', 0)

            y1 = max(0, smooth_y + oy - half_view - pad)
            y2 = min(self.map_height, smooth_y + oy + half_view + pad)
            x1 = max(0, smooth_x + ox - half_view - pad)
            x2 = min(self.map_width, smooth_x + ox + half_view + pad)
            display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
            # 箭头与位置标记由前端在 canvas 上叠加，后端不再绘制
        else:
            if self.sift_engine.last_x is not None and self.sift_engine.last_y is not None:
                y1 = max(0, self.sift_engine.last_y - half_view - pad)
                y2 = min(self.map_height, self.sift_engine.last_y + half_view + pad)
                x1 = max(0, self.sift_engine.last_x - half_view - pad)
                x2 = min(self.map_width, self.sift_engine.last_x + half_view + pad)
            else:
                x1 = x2 = y1 = y2 = 0

            if (self.sift_engine.last_x is not None and self.sift_engine.last_y is not None and
                    x2 > x1 and y2 > y1):
                display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
            else:
                full_size = config.VIEW_SIZE + 2 * pad
                display_crop = np.zeros((full_size, full_size, 3), dtype=np.uint8)
                lock_label = "🔒 " if self.sift_engine.coord_lock_enabled else ""
                cv2.putText(display_crop, f"{lock_label}Lost...", (130 + pad, 200 + pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

    def _compose_output_frame(self, display_crop, half_view):
        """将裁剪地图居中贴到画布上，返回最终 BGR 帧。
        画布尺寸 = VIEW_SIZE + 2*JPEG_PAD，供前端亚帧微平移使用。
        """
        pad = getattr(config, 'JPEG_PAD', 0)
        full_size = config.VIEW_SIZE + 2 * pad
        # 创建 BGR 画布并居中粘贴
        final_bgr = np.full((full_size, full_size, 3), 43, dtype=np.uint8)
        h, w = display_crop.shape[:2]
        y_off = max(0, (full_size - h) // 2)
        x_off = max(0, (full_size - w) // 2)
        paste_h = min(h, full_size - y_off)
        paste_w = min(w, full_size - x_off)
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
