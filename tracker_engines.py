"""
tracker_engines.py - 纯识别引擎模块（无 Web 框架依赖）

包含:
  SIFTMapTracker    SIFT 传统特征匹配引擎
  LoFTRMapTracker   LoFTR AI 深度学习匹配引擎（延迟加载 torch/kornia）

此模块仅依赖 cv2 / numpy / PIL / config，可被任何前端（Web/CLI/TUI）复用。
"""

import math
import os
import cv2
import numpy as np
from collections import deque
from threading import Lock
import config


def _angular_diff(a, b):
    """最短角度差 (带符号, -180~180)"""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


class _ArrowDirectionSystem:
    """
    纯坐标驱动的方向系统（无视觉特征依赖）。

    核心思路:
      - 每帧接收地图坐标，根据坐标位移方向计算箭头朝向
      - 多帧累积位移 + EMA 平滑，过滤 SIFT 单帧抖动
      - 瞬间反向转向时快速跟随（角度差超阈值则跳过平滑直接赋值）
      - 静止不动时标记 is_stopped，前端渲染为圆点
    """

    def __init__(self):
        # 位置历史（用于累积位移判断移动/静止）
        _hist_len = getattr(config, 'ARROW_POS_HISTORY_LEN', 4)
        self._pos_history = deque(maxlen=_hist_len)

        # 移动判定阈值（累积位移 < 阈值 → 静止）
        self._move_threshold = getattr(config, 'ARROW_MOVE_MIN_DISPLACEMENT', 6)

        # 静止防抖：连续 N 帧低于阈值才判定为静止
        self._stopped_debounce = getattr(config, 'ARROW_STOPPED_DEBOUNCE', 3)
        self._low_move_streak = 0

        # EMA 平滑
        self._ema_alpha = getattr(config, 'ARROW_ANGLE_SMOOTH_ALPHA', 0.35)

        # 急转弯阈值：角度差超过此值时跳过 EMA 直接赋值（应付瞬间反向）
        self._snap_threshold = getattr(config, 'ARROW_SNAP_THRESHOLD', 90)

        # 输出状态
        self._last_angle = 0.0
        self._ema_angle = None
        self._is_stopped = True

    def update(self, map_x=None, map_y=None):
        """
        每帧调用：根据坐标位移更新方向。

        Args:
            map_x, map_y: 当前帧地图坐标 (None = 定位失败/惯性帧)

        Returns:
            (angle, is_stopped)
            angle: float — 箭头朝向角 (0=北, 顺时针)
            is_stopped: bool — 是否处于静止状态
        """
        if map_x is None or map_y is None:
            return self._last_angle, self._is_stopped

        self._pos_history.append((map_x, map_y))

        if len(self._pos_history) < 2:
            return self._last_angle, self._is_stopped

        # 累积位移：最旧帧 → 当前帧
        old_x, old_y = self._pos_history[0]
        cum_dx = map_x - old_x
        cum_dy = map_y - old_y
        cum_dist = math.sqrt(cum_dx * cum_dx + cum_dy * cum_dy)

        if cum_dist < self._move_threshold:
            # 低位移，累加静止计数
            self._low_move_streak += 1
            if self._low_move_streak >= self._stopped_debounce:
                self._is_stopped = True
            return self._last_angle, self._is_stopped

        # 有效移动 → 重置静止计数
        self._low_move_streak = 0
        self._is_stopped = False

        raw_angle = math.degrees(math.atan2(cum_dx, -cum_dy)) % 360

        # 平滑 or 急转快跳
        if self._ema_angle is None:
            self._ema_angle = raw_angle
        else:
            diff = _angular_diff(raw_angle, self._ema_angle)
            if abs(diff) >= self._snap_threshold:
                self._ema_angle = raw_angle
            else:
                self._ema_angle = (self._ema_angle + self._ema_alpha * diff) % 360

        self._last_angle = self._ema_angle
        return self._last_angle, self._is_stopped


class SIFTMapTracker:
    """SIFT 传统特征匹配引擎"""

    def __init__(self):
        print("正在初始化 SIFT 引擎...")
        # 自适应 CLAHE：双档 (标准 / 低纹理)，运行时按纹理 std 线性插值
        self._clahe_normal = cv2.createCLAHE(
            clipLimit=getattr(config, 'CLAHE_LIMIT_NORMAL', 3.0), tileGridSize=(8, 8))
        self._clahe_low = cv2.createCLAHE(
            clipLimit=getattr(config, 'CLAHE_LIMIT_LOW_TEXTURE', 6.0), tileGridSize=(8, 8))
        self._low_texture_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)

        # SIFT 检测器
        _sift_contrast = getattr(config, 'SIFT_CONTRAST_THRESHOLD', 0.02)
        self.sift = cv2.SIFT_create(contrastThreshold=_sift_contrast)

        # 上一帧箭头角度
        self._last_arrow_angle = 0.0
        self._last_arrow_stopped = True
        # 箭头方向系统（纯坐标驱动）
        self._arrow_dir = _ArrowDirectionSystem()

        # 线程锁（防止并发请求导致 kp_local / flann 竞态）
        self._lock = Lock()

        logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = logic_map_bgr.shape[:2]

        logic_map_gray = cv2.cvtColor(logic_map_bgr, cv2.COLOR_BGR2GRAY)
        self._logic_map_gray = logic_map_gray
        logic_map_enhanced = self._adaptive_clahe_map(logic_map_gray)
        print("正在提取大地图 SIFT 特征点...")
        self.kp_big_all, self.des_big_all = self.sift.detectAndCompute(logic_map_enhanced, None)
        print(f"✅ 全局特征点: {len(self.kp_big_all)} 个")

        self.kp_coords = np.array([kp.pt for kp in self.kp_big_all], dtype=np.float32)

        print("正在构建全局 FLANN 索引...")
        self.flann_global = self._create_flann(self.des_big_all)

        # 局部 FLANN（动态重建）
        self.kp_local = list(self.kp_big_all)
        self.des_local = self.des_big_all.copy()
        self.flann_local = self.flann_global
        self.using_local = False
        self.local_fail_count = 0

        self.SEARCH_RADIUS = getattr(config, 'SEARCH_RADIUS', 400)
        self.LOCAL_FAIL_LIMIT = getattr(config, 'LOCAL_FAIL_LIMIT', 5)
        self.JUMP_THRESHOLD = getattr(config, 'SIFT_JUMP_THRESHOLD', 500)
        self.NEARBY_SEARCH_RADIUS = getattr(config, 'NEARBY_SEARCH_RADIUS', 600)

        # ORB 备份引擎（邻近搜索时使用）
        self.orb = cv2.ORB_create(nfeatures=500)
        self.orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # LK 光流状态（帧间快速跟踪）
        self._lk_enabled = getattr(config, 'LK_ENABLED', True)
        self._lk_prev_gray = None      # 上一帧小地图灰度图
        self._lk_prev_pts = None       # 上一帧的跟踪点（小地图像素坐标）
        self._lk_map_scale = 4.0       # 小地图像素 → 大地图像素的比例（SIFT 成功后更新）
        self._lk_frame_num = 0
        self._lk_sift_every = getattr(config, 'LK_SIFT_INTERVAL', 4)
        self._lk_min_conf = getattr(config, 'LK_MIN_CONFIDENCE', 0.5)
        self._lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # ECC 低纹理兜底
        self._ecc_enabled = getattr(config, 'ECC_ENABLED', True)
        self._ecc_min_cc = getattr(config, 'ECC_MIN_CORRELATION', 0.25)
        self._last_sift_scale = self._lk_map_scale  # 默认与 LK scale 一致，进海洋后 ECC 可立即工作

        # 附近搜索 FLANN 缓存（按网格桶缓存，避免每帧重建）
        self._nearby_flann_cache = {}
        self._nearby_flann_bucket = 200  # 网格粒度（地图像素）

        # 惯性导航状态
        self.last_x = None
        self.last_y = None
        self.lost_frames = 0
        self._sift_confused = False

        # 状态冻结
        self._frozen = False
        self._frozen_last_x = None
        self._frozen_last_y = None
        self._frozen_using_local = False
        self._frozen_kp_local = None
        self._frozen_des_local = None
        self._frozen_flann_local = None
        import time as _time
        self._perf_time = _time
        self._frozen_at = 0.0
        self._frame_times = []

        # 坐标锁定模式
        self.coord_lock_enabled = False
        self._lock_history_size = getattr(config, 'COORD_LOCK_HISTORY_SIZE', 10)
        self._lock_search_radius = getattr(config, 'COORD_LOCK_SEARCH_RADIUS', 400)
        self._lock_max_retries = getattr(config, 'COORD_LOCK_MAX_RETRIES', 5)
        self._lock_min_to_activate = getattr(config, 'COORD_LOCK_MIN_HISTORY_TO_ACTIVATE', 15)

    # ---- 状态冻结 / 恢复 ----
    def _freeze_state(self):
        if self._frozen:
            return
        self._frozen = True
        self._frozen_last_x = self.last_x
        self._frozen_last_y = self.last_y
        self._frozen_using_local = self.using_local
        if self.using_local:
            self._frozen_kp_local = self.kp_local
            self._frozen_des_local = self.des_local
            self._frozen_flann_local = self.flann_local
        self._frozen_at = self._perf_time.time()

    def _thaw_state(self):
        if not self._frozen:
            return
        self._frozen = False
        elapsed = self._perf_time.time() - self._frozen_at
        timeout = getattr(config, 'FREEZE_TIMEOUT', 30.0)
        if elapsed > timeout:
            return
        if self._frozen_last_x is not None:
            self.last_x = self._frozen_last_x
            self.last_y = self._frozen_last_y
        if self._frozen_using_local and self._frozen_flann_local is not None:
            self.kp_local = self._frozen_kp_local
            self.des_local = self._frozen_des_local
            self.flann_local = self._frozen_flann_local
            self.using_local = True
        self.lost_frames = 0
        self.local_fail_count = 0

    @property
    def frozen(self):
        return self._frozen

    @property
    def frozen_position(self):
        """冻结期间的静态坐标（供 Kalman 暂停时使用）"""
        if self._frozen and self._frozen_last_x is not None:
            return self._frozen_last_x, self._frozen_last_y
        return None

    # ---- 自适应 CLAHE ----
    def _adaptive_clahe(self, gray):
        """低纹理用 _clahe_low，否则 _clahe_normal"""
        if np.std(gray) < self._low_texture_thresh:
            return self._clahe_low.apply(gray)
        return self._clahe_normal.apply(gray)

    def _adaptive_clahe_map(self, gray):
        """大地图分块自适应 CLAHE"""
        h, w = gray.shape[:2]
        tile = 256
        result = np.empty_like(gray)
        thresh = self._low_texture_thresh
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                patch = gray[y:y+tile, x:x+tile]
                clahe = self._clahe_low if np.std(patch) < thresh else self._clahe_normal
                result[y:y+tile, x:x+tile] = clahe.apply(patch)
        return result

    @staticmethod
    def _create_flann(descriptors):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flann.add([descriptors])
        flann.train()
        return flann

    # ---- 局部/全局搜索切换 ----
    def _switch_to_local(self, cx, cy):
        """以 (cx, cy) 为中心，提取半径内的特征点，重建局部 FLANN"""
        if self.using_local and hasattr(self, '_local_center'):
            dx = abs(cx - self._local_center[0])
            dy = abs(cy - self._local_center[1])
            if dx + dy < self.SEARCH_RADIUS * 0.3:
                return
        r = self.SEARCH_RADIUS
        dx = np.abs(self.kp_coords[:, 0] - cx)
        dy = np.abs(self.kp_coords[:, 1] - cy)
        mask = (dx < r) & (dy < r)
        indices = np.where(mask)[0]
        if len(indices) < 20:
            return
        self.kp_local = [self.kp_big_all[i] for i in indices]
        self.des_local = self.des_big_all[indices]
        self.flann_local = self._create_flann(self.des_local)
        self.using_local = True
        self.local_fail_count = 0
        self._local_center = (cx, cy)

    def _switch_to_global(self):
        self.using_local = False
        self.local_fail_count = 0
        self._local_center = None

    # ---- 坐标锁定 ----
    def set_coord_lock(self, enabled):
        self.coord_lock_enabled = enabled
        return True

    @staticmethod
    def _compute_lock_anchor(pos_deque, n=10):
        if pos_deque is None or len(pos_deque) < n:
            return None
        recent = list(pos_deque)[-n:]
        ax = sum(p[0] for p in recent) / len(recent)
        ay = sum(p[1] for p in recent) / len(recent)
        return int(ax), int(ay)

    def _match_locked(self, minimap_bgr, anchor_x, anchor_y):
        """锁定模式匹配：复用 _sift_match_region，逐步放宽阈值"""
        t_start = self._perf_time.perf_counter()
        r = self._lock_search_radius

        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = self._adaptive_clahe(minimap_gray)
        kp_mini, des_mini = self._extract_minimap_features(minimap_gray)
        if des_mini is None:
            return self._make_result(False, t_start=t_start, _locked_state="NO_FEATURES")

        # 筛选锚点范围内的特征点
        dx = np.abs(self.kp_coords[:, 0] - anchor_x)
        dy = np.abs(self.kp_coords[:, 1] - anchor_y)
        indices = np.where((dx < r) & (dy < r))[0]
        if len(indices) < 5:
            return self._make_result(False, t_start=t_start, _locked_state="FEW_KPTS")

        locked_kp = [self.kp_big_all[i] for i in indices]
        locked_des = self.des_big_all[indices]
        locked_flann = self._create_flann(locked_des)

        # 逐步放宽阈值重试
        for ratio, min_m in [(0.85, 5), (0.78, 4), (0.70, 3)]:
            result = self._sift_match_region(
                kp_mini, des_mini, minimap_gray.shape,
                locked_kp, locked_flann, ratio, min_m)
            if result is not None:
                tx, ty, inlier_count, quality = result
                if abs(tx - anchor_x) <= r * 1.5 and abs(ty - anchor_y) <= r * 1.5:
                    angle, stopped = self._arrow_dir.update(tx, ty)
                    self._last_arrow_angle = angle
                    self._last_arrow_stopped = stopped
                    return self._make_result(True, tx, ty, t_start=t_start,
                                             arrow_angle=angle, arrow_stopped=stopped,
                                             match_count=inlier_count, match_quality=quality,
                                             _locked_state="LOCKED")

        return self._make_result(False, t_start=t_start, _locked_state="LOCK_FAIL")

    def _make_result(self, found, cx=None, cy=None, t_start=None,
                     arrow_angle=None, arrow_stopped=True, is_inertial=False,
                     match_count=0, match_quality=0.0, _locked_state=''):
        """统一构造返回字典"""
        if t_start is not None:
            t_elapsed = (self._perf_time.perf_counter() - t_start) * 1000
            self._frame_times.append(t_elapsed)
            if len(self._frame_times) >= 60:
                avg = sum(self._frame_times) / len(self._frame_times)
                self._frame_times.clear()
        return {
            'found': found, 'center_x': cx, 'center_y': cy,
            'arrow_angle': arrow_angle, 'arrow_stopped': arrow_stopped,
            'is_inertial': is_inertial, 'match_count': match_count,
            'match_quality': match_quality,
            'map_width': self.map_width, 'map_height': self.map_height,
            '_locked_state': _locked_state,
        }

    @staticmethod
    def _draw_arrow_marker(img_bgr, cx, cy, size=None, angle=0, stopped=False):
        if size is None:
            size = 12
        overlay = img_bgr.copy()
        if stopped:
            r = int(size * 0.6)
            cv2.circle(overlay, (int(cx), int(cy)), r, (48, 182, 254), -1)
            cv2.circle(overlay, (int(cx), int(cy)), r, (255, 255, 255), 1)
        else:
            pts = np.array([
                [cx, cy - size],
                [cx - size * 0.6, cy + size * 0.7],
                [cx + size * 0.6, cy + size * 0.7],
            ], dtype=np.float64)
            if angle != 0:
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                centered = pts - np.array([cx, cy])
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                pts = ((rot_mat @ centered.T).T + np.array([cx, cy])).astype(np.int32)
            else:
                pts = pts.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (48, 182, 254))
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)

    @staticmethod
    def _paste_arrow_patch(img_bgr, cx, cy, arrow_patch, offset_x, offset_y, scale=1.0):
        if arrow_patch is None or arrow_patch.size == 0:
            return
        ph, pw = arrow_patch.shape[:2]
        if scale != 1.0:
            new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
            arrow_patch = cv2.resize(arrow_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            offset_x, offset_y = int(offset_x * scale), int(offset_y * scale)
            ph, pw = new_w, new_h
        paste_x, paste_y = cx - offset_x, cy - offset_y
        dh, dw = img_bgr.shape[:2]
        sx1, sy1 = max(0, -paste_x), max(0, -paste_y)
        dx1, dy1 = max(0, paste_x), max(0, paste_y)
        dx2, dy2 = min(dw, paste_x + pw), min(dh, paste_y + ph)
        sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
        if dx2 <= dx1 or dy2 <= dy1 or sx2 <= sx1 or sy2 <= sy1:
            return
        roi = img_bgr[dy1:dy2, dx1:dx2]
        patch_region = arrow_patch[sy1:sy2, sx1:sx2]
        bgr_part = patch_region[:, :, :3]
        alpha_part = patch_region[:, :, 3:4].astype(np.float32) / 255.0
        blended = (bgr_part.astype(np.float32) * alpha_part +
                   roi.astype(np.float32) * (1.0 - alpha_part))
        img_bgr[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)

    # ---- 工具方法 ----
    @staticmethod
    def _make_circular_mask(h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cv2.circle(mask, (cx, cy), min(cx, cy) - 2, 255, -1)
        return mask

    def _get_circular_mask(self, h, w):
        key = (h, w)
        if not hasattr(self, '_circ_mask_cache'):
            self._circ_mask_cache = {}
        cached = self._circ_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = self._make_circular_mask(h, w)
        self._circ_mask_cache[key] = mask
        return mask

    def _enhance_for_texture(self, gray, texture_std):
        """统一的纹理增强：根据 texture_std 连续决定增强强度"""
        if texture_std < 15:
            # 超低纹理(海面/天空)：双边保边 + 高CLAHE + 强锐化
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = self._clahe_low.apply(filtered)
            blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
            return cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)
        elif texture_std < 30:
            # 低纹理(雪地/草坪)：温和锐化
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
            return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        return gray

    def _extract_minimap_features(self, minimap_gray):
        """提取小地图特征点（带圆形掩码），返回 (kp, des)"""
        h_mm, w_mm = minimap_gray.shape[:2]
        circ_mask = self._get_circular_mask(h_mm, w_mm)
        kp, des = self.sift.detectAndCompute(minimap_gray, circ_mask)
        if des is None or len(kp) < 2:
            return None, None
        return kp, des

    def _sift_match_region(self, kp_mini, des_mini, mm_shape,
                           region_kp, region_flann, ratio, min_match):
        """
        通用 SIFT 区域匹配。
        返回 (tx, ty, inlier_count, quality) 或 None。
        """
        try:
            matches = region_flann.knnMatch(des_mini, k=2)
        except cv2.error:
            return None

        good = [m for m_n in matches if len(m_n) == 2
                for m, n in [m_n] if m.distance < ratio * n.distance]

        if len(good) < min_match:
            return None

        src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([region_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)
        if M is None:
            return None

        inlier_count = int(mask.sum()) if mask is not None else 0
        if inlier_count < min_match:
            return None

        # 缩放合理性
        sx = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        sy = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
        avg_scale = (sx + sy) / 2
        max_scale = getattr(config, 'SIFT_MAX_HOMOGRAPHY_SCALE', 8.0)
        if avg_scale > max_scale or avg_scale < 1.0 / max_scale:
            return None

        # 记录 scale 供 LK / ECC 使用（side-channel）
        self._last_sift_scale = avg_scale

        h, w = mm_shape[:2]
        center_pt = np.float32([[[w / 2.0, h / 2.0]]])
        dst_center = cv2.perspectiveTransform(center_pt, M)
        tx, ty = int(dst_center[0][0][0]), int(dst_center[0][0][1])

        if not (0 <= tx < self.map_width and 0 <= ty < self.map_height):
            return None

        inlier_ratio = inlier_count / max(len(good), 1)
        count_conf = min(1.0, inlier_count / 12.0)
        quality = min(1.0, inlier_ratio * count_conf)
        return tx, ty, inlier_count, quality

    def _orb_nearby_match(self, minimap_gray, cx_hint, cy_hint, radius):
        """ORB 附近搜索：在 (cx_hint, cy_hint) 周围 radius 内做 ORB 匹配"""
        if self.orb is None:
            return None
        h_mm, w_mm = minimap_gray.shape[:2]
        circ_mask = self._get_circular_mask(h_mm, w_mm)
        kp_orb_mini, des_orb_mini = self.orb.detectAndCompute(minimap_gray, circ_mask)
        if des_orb_mini is None or len(kp_orb_mini) < 3:
            return None

        x1 = max(0, cx_hint - radius)
        y1 = max(0, cy_hint - radius)
        x2 = min(self.map_width, cx_hint + radius)
        y2 = min(self.map_height, cy_hint + radius)

        local_gray = self._logic_map_gray[y1:y2, x1:x2]
        local_gray = self._adaptive_clahe(local_gray)
        kp_orb_map, des_orb_map = self.orb.detectAndCompute(local_gray, None)
        if des_orb_map is None or len(kp_orb_map) < 5:
            return None

        matches = self.orb_bf.knnMatch(des_orb_mini, des_orb_map, k=2)
        good = [m for m_n in matches if len(m_n) == 2
                for m, n in [m_n] if m.distance < 0.75 * n.distance]
        if len(good) < 5:
            return None

        src_pts = np.float32([kp_orb_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_orb_map[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        if M is None or (int(mask.sum()) if mask is not None else 0) < 4:
            return None

        sx_h = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        sy_h = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
        if not (0.2 < (sx_h + sy_h) / 2 < 5.0):
            return None

        center_pt = np.float32([[[w_mm / 2.0, h_mm / 2.0]]])
        dst_center = cv2.perspectiveTransform(center_pt, M)
        tx = int(dst_center[0][0][0]) + x1
        ty = int(dst_center[0][0][1]) + y1

        if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
            jump = abs(tx - cx_hint) + abs(ty - cy_hint)
            if jump < self.JUMP_THRESHOLD:
                return tx, ty
        return None

    def _lk_track(self, minimap_gray):
        """
        LK 稀疏光流估计帧间位移。
        把小地图像素位移乘以 scale 转换到大地图坐标系。
        返回 (dx_map, dy_map, confidence) 或 None。
        """
        if (not self._lk_enabled
                or self._lk_prev_gray is None
                or self._lk_prev_pts is None
                or len(self._lk_prev_pts) < 4
                or self.last_x is None):
            return None
        try:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._lk_prev_gray, minimap_gray,
                self._lk_prev_pts, None, **self._lk_params)
            if curr_pts is None or status is None:
                return None
            ok = status.ravel() == 1
            good_curr = curr_pts[ok]
            good_prev = self._lk_prev_pts[ok]
            if len(good_curr) < 4:
                return None
            disp = good_curr - good_prev
            dx_mm = float(np.median(disp[:, 0]))
            dy_mm = float(np.median(disp[:, 1]))
            confidence = len(good_curr) / max(len(self._lk_prev_pts), 1)
            # 更新跟踪点为当前帧（仅用良好点）
            self._lk_prev_pts = good_curr.reshape(-1, 1, 2)
            s = self._lk_map_scale
            return dx_mm * s, dy_mm * s, confidence
        except cv2.error:
            return None

    def _ecc_nearby_match(self, minimap_gray, cx_hint, cy_hint):
        """
        ECC（增强相关系数）像素级匹配，专用于低纹理场景（海洋/雪地）。
        利用上次 SIFT 确定的 scale 在大地图中提取等比例区域，
        用 ECC 找出当前小地图相对于该区域的微小平移。
        返回 (tx, ty) 或 None。
        """
        if not self._ecc_enabled or self._last_sift_scale is None:
            return None
        s = self._last_sift_scale
        h_mm, w_mm = minimap_gray.shape[:2]
        crop_w = int(w_mm * s)
        crop_h = int(h_mm * s)
        x1 = max(0, cx_hint - crop_w // 2)
        y1 = max(0, cy_hint - crop_h // 2)
        x2 = min(self.map_width, x1 + crop_w)
        y2 = min(self.map_height, y1 + crop_h)
        actual_w = x2 - x1
        actual_h = y2 - y1
        if actual_w < 20 or actual_h < 20:
            return None

        map_crop = self._logic_map_gray[y1:y2, x1:x2]
        map_resized = cv2.resize(map_crop, (w_mm, h_mm))

        ref = map_resized.astype(np.float32)
        tmpl = minimap_gray.astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                ref, tmpl, warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1e-3))
        except cv2.error:
            return None

        if cc < self._ecc_min_cc:
            return None

        # warp maps tmpl→ref: tmpl(x,y) ≈ ref(x+tx, y+ty)
        # 小地图中心在 ref 空间的位置 = (w_mm/2 - tx, h_mm/2 - ty)
        # 转换回大地图坐标（ref 像素 → map 坐标）
        tx_px, ty_px = float(warp[0, 2]), float(warp[1, 2])
        scale_x = actual_w / w_mm
        scale_y = actual_h / h_mm
        center_ref_x = w_mm / 2.0 - tx_px
        center_ref_y = h_mm / 2.0 - ty_px
        map_x = int(x1 + center_ref_x * scale_x)
        map_y = int(y1 + center_ref_y * scale_y)

        if not (0 <= map_x < self.map_width and 0 <= map_y < self.map_height):
            return None
        if abs(map_x - cx_hint) + abs(map_y - cy_hint) < self.JUMP_THRESHOLD:
            return map_x, map_y
        return None

    # ---- 核心匹配 ----
    def match(self, minimap_bgr):
        with self._lock:
            return self._match_impl(minimap_bgr)

    def _match_impl(self, minimap_bgr):
        t_start = self._perf_time.perf_counter()
        found = False
        center_x, center_y = None, None
        arrow_angle = None
        is_inertial = False
        match_quality = 0.0

        minimap_gray_raw = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        texture_std = float(np.std(minimap_gray_raw))

        # 自适应 CLAHE + 纹理增强
        minimap_gray = self._adaptive_clahe(minimap_gray_raw)
        minimap_gray = self._enhance_for_texture(minimap_gray, texture_std)

        self._lk_frame_num += 1
        run_sift = (self._lk_frame_num % self._lk_sift_every == 0)

        # ======================================================
        # 第一层：LK 光流快速跟踪（每帧 ~2ms，跳过部分 SIFT 调用）
        # ======================================================
        lk_result = self._lk_track(minimap_gray)
        if lk_result is not None and not run_sift:
            dx_map, dy_map, lk_conf = lk_result
            if lk_conf >= self._lk_min_conf:
                tx = int(round(self.last_x + dx_map))
                ty = int(round(self.last_y + dy_map))
                if (0 <= tx < self.map_width and 0 <= ty < self.map_height
                        and abs(dx_map) + abs(dy_map) < self.JUMP_THRESHOLD):
                    found, center_x, center_y = True, tx, ty
                    match_quality = lk_conf * 0.7

        # ======================================================
        # 第二层：SIFT 精确匹配（周期性运行 + LK 失败时触发）
        # ======================================================
        if not found:
            # 提取特征
            kp_mini, des_mini = self._extract_minimap_features(minimap_gray)

            # 超低纹理 + 特征过少 → 更激进的预处理重试
            if texture_std < 15 and (des_mini is None or len(kp_mini) < 15):
                enhanced = self._enhance_for_texture(minimap_gray_raw, texture_std=5)
                kp_retry, des_retry = self._extract_minimap_features(enhanced)
                if des_retry is not None and len(kp_retry) > (len(kp_mini) if kp_mini is not None else 0):
                    kp_mini, des_mini = kp_retry, des_retry

            # 动态匹配参数：按 texture_std 连续插值
            t = max(0.0, min(1.0, (texture_std - 15) / 15))
            eff_match_ratio = 0.88 - t * 0.06  # 0.88(极低纹理) → 0.82(标准)
            eff_min_match = 3 if texture_std < 30 else config.SIFT_MIN_MATCH_COUNT

            if des_mini is not None:
                # ---- SIFT 第一轮：局部 → 全局回退 ----
                for search_round in range(2):
                    if search_round == 0 and self.using_local:
                        current_kp, current_flann = self.kp_local, self.flann_local
                    elif search_round == 0:
                        current_kp, current_flann = list(self.kp_big_all), self.flann_global
                    else:
                        if not self.using_local:
                            break
                        current_kp, current_flann = list(self.kp_big_all), self.flann_global

                    result = self._sift_match_region(
                        kp_mini, des_mini, minimap_gray.shape,
                        current_kp, current_flann, eff_match_ratio, eff_min_match)

                    if result is not None:
                        tx, ty, inlier_count, quality = result
                        if self.last_x is not None:
                            max_jump = self.JUMP_THRESHOLD if search_round == 0 else self.JUMP_THRESHOLD * 2
                            if abs(tx - self.last_x) + abs(ty - self.last_y) >= max_jump:
                                continue
                        found, center_x, center_y, match_quality = True, tx, ty, quality
                        # SIFT 成功 → 更新 LK 比例和跟踪点
                        if self._last_sift_scale is not None:
                            self._lk_map_scale = self._last_sift_scale
                        break

                # ---- SIFT 第二轮：附近搜索（使用 FLANN 缓存）----
                if not found and self.last_x is not None:
                    nr = self.NEARBY_SEARCH_RADIUS
                    bucket = (self.last_x // self._nearby_flann_bucket,
                              self.last_y // self._nearby_flann_bucket)
                    if bucket in self._nearby_flann_cache:
                        nearby_kp, nearby_des, nearby_flann = self._nearby_flann_cache[bucket]
                    else:
                        dx_arr = np.abs(self.kp_coords[:, 0] - self.last_x)
                        dy_arr = np.abs(self.kp_coords[:, 1] - self.last_y)
                        indices = np.where((dx_arr < nr) & (dy_arr < nr))[0]
                        if len(indices) >= 10:
                            nearby_kp = [self.kp_big_all[i] for i in indices]
                            nearby_des = self.des_big_all[indices]
                            nearby_flann = self._create_flann(nearby_des)
                            # 缓存（最多保留 30 个桶）
                            if len(self._nearby_flann_cache) >= 30:
                                self._nearby_flann_cache.pop(next(iter(self._nearby_flann_cache)))
                            self._nearby_flann_cache[bucket] = (nearby_kp, nearby_des, nearby_flann)
                        else:
                            nearby_kp = nearby_des = nearby_flann = None

                    if nearby_flann is not None:
                        result = self._sift_match_region(
                            kp_mini, des_mini, minimap_gray.shape,
                            nearby_kp, nearby_flann, 0.90, 3)
                        if result is not None:
                            tx, ty, inlier_count, quality = result
                            if abs(tx - self.last_x) + abs(ty - self.last_y) < self.JUMP_THRESHOLD * 1.5:
                                found, center_x, center_y = True, tx, ty
                                match_quality = quality * 0.8

                    # ---- ORB 兜底 ----
                    if not found:
                        orb_result = self._orb_nearby_match(
                            minimap_gray, self.last_x, self.last_y, nr)
                        if orb_result is not None:
                            found, center_x, center_y = True, *orb_result
                            match_quality = 0.4

                    # ---- ECC 像素级兜底（低/中纹理均可，海洋必备）----
                    if not found:
                        ecc_result = self._ecc_nearby_match(
                            minimap_gray, self.last_x, self.last_y)
                        if ecc_result is not None:
                            found, center_x, center_y = True, *ecc_result
                            match_quality = 0.3

        # 更新 LK 上一帧（不论用哪层跟踪，都更新灰度图）
        self._lk_prev_gray = minimap_gray.copy()
        if found and not is_inertial:
            # 重新提取当前帧关键点给下一帧 LK 使用
            h_mm, w_mm = minimap_gray.shape[:2]
            circ_mask = self._get_circular_mask(h_mm, w_mm)
            pts = cv2.goodFeaturesToTrack(
                minimap_gray, maxCorners=60, qualityLevel=0.01,
                minDistance=7, mask=circ_mask)
            self._lk_prev_pts = pts  # shape (N,1,2) or None

        # ---- 状态更新 ----
        if found and not is_inertial:
            self.last_x, self.last_y = center_x, center_y
            self.lost_frames = 0
            self.local_fail_count = 0
            self._switch_to_local(center_x, center_y)
        else:
            self.lost_frames += 1
            if self.using_local:
                self.local_fail_count += 1
                if self.local_fail_count >= self.LOCAL_FAIL_LIMIT:
                    self._switch_to_global()
                    self._nearby_flann_cache.clear()  # 切全局时清缓存
            if self.last_x is not None and self.lost_frames <= config.MAX_LOST_FRAMES:
                found, center_x, center_y = True, self.last_x, self.last_y
                is_inertial = True
            elif self.lost_frames > config.MAX_LOST_FRAMES:
                self.last_x = self.last_y = None
                self._lk_prev_gray = None
                self._lk_prev_pts = None
                if self.using_local:
                    self._switch_to_global()

        # 箭头方向
        arrow_stopped = True
        if found and not is_inertial:
            arrow_angle, arrow_stopped = self._arrow_dir.update(center_x, center_y)
        elif found and is_inertial:
            arrow_angle, arrow_stopped = self._arrow_dir.update(None, None)
        else:
            arrow_angle = self._arrow_dir._last_angle
            arrow_stopped = self._arrow_dir._is_stopped
        self._last_arrow_angle = arrow_angle
        self._last_arrow_stopped = arrow_stopped

        return self._make_result(
            found, center_x, center_y, t_start=t_start,
            arrow_angle=arrow_angle, arrow_stopped=arrow_stopped,
            is_inertial=is_inertial, match_count=0,
            match_quality=match_quality)


class LoFTRMapTracker:
    """LoFTR AI 深度学习匹配引擎（仅在需要时才 import torch/kornia）"""

    def __init__(self):
        import torch as _torch
        import kornia as _K
        from kornia.feature import LoFTR as _LoFTR

        print("正在加载 LoFTR AI 模型...")
        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        print(f"当前计算设备: {self.device}")

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
        # FP16 半精度推理（GPU 时性能提升约 40%，减少显存占用）
        self._use_fp16 = (self.device.type == 'cuda')
        if self._use_fp16:
            self.matcher.half()
            print("已启用 FP16 半精度推理")
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

        self._lock = Lock()

        import time as _time
        self._perf_time = _time

    def preprocess_image(self, img_bgr, target_size=None):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if target_size is not None:
            img_gray = cv2.resize(img_gray, (target_size, target_size))
        h, w = img_gray.shape
        new_h = h - (h % 8)
        new_w = w - (w % 8)
        img_gray = cv2.resize(img_gray, (new_w, new_h))
        tensor = self.kornia.image_to_tensor(img_gray, False).float() / 255.0
        if self._use_fp16:
            tensor = tensor.half()
        return tensor.to(self.device)

    # ------------------------------------------
    # 轻量级重定位（供混合模式后台线程调用）
    # 两阶段金字塔: 粗扫(低分辨率) → 精定位(原尺寸 crop)
    # ------------------------------------------
    def relocate(self, minimap_bgr):
        """
        后台重定位：在全图上找到 minimap 对应位置。
        返回 (found, center_x, center_y) 或 (False, None, None)。
        不修改自身状态（state/last_x/last_y），线程安全。
        """
        t0 = self._perf_time.perf_counter()
        coarse_tile = getattr(config, 'HYBRID_COARSE_TILE', 400)
        coarse_step = getattr(config, 'HYBRID_COARSE_STEP', 350)
        fine_radius = getattr(config, 'HYBRID_FINE_RADIUS', 300)
        mini_sz = getattr(config, 'HYBRID_MINI_SIZE', 128)

        # === 阶段1: 粗扫 — 缩小分辨率快速扫全图 ===
        best_score = 0
        best_cx, best_cy = None, None

        tensor_mini = self.preprocess_image(minimap_bgr, target_size=mini_sz)

        y = 0
        while y < self.map_height:
            x = 0
            while x < self.map_width:
                x2 = min(self.map_width, x + coarse_tile * 4)
                y2 = min(self.map_height, y + coarse_tile * 4)
                crop = self.logic_map_bgr[y:y2, x:x2]
                if crop.shape[0] < 16 or crop.shape[1] < 16:
                    x += coarse_step * 4
                    continue

                tensor_crop = self.preprocess_image(crop, target_size=coarse_tile)
                input_dict = {"image0": tensor_mini, "image1": tensor_crop}
                with self.torch.no_grad():
                    corr = self.matcher(input_dict)

                # GPU 端置信度过滤
                conf = corr['confidence']
                valid = conf > config.AI_CONFIDENCE_THRESHOLD
                n_valid = int(valid.sum().item())

                if n_valid > best_score and n_valid >= 3:
                    mkpts1 = corr['keypoints1'][valid].cpu().numpy()
                    # 反算回原图坐标
                    scale_x = (x2 - x) / coarse_tile
                    scale_y = (y2 - y) / coarse_tile
                    cx_local = float(mkpts1[:, 0].mean()) * scale_x + x
                    cy_local = float(mkpts1[:, 1].mean()) * scale_y + y
                    if 0 <= cx_local < self.map_width and 0 <= cy_local < self.map_height:
                        best_score = n_valid
                        best_cx, best_cy = cx_local, cy_local

                x += coarse_step * 4
            y += coarse_step * 4

        if best_cx is None:
            t1 = self._perf_time.perf_counter()
            print(f"[混合-LoFTR] 粗扫失败 ({(t1-t0)*1000:.0f}ms)")
            return False, None, None

        # === 阶段2: 精定位 — 在粗扫命中区域取原尺寸 crop ===
        fx1 = max(0, int(best_cx) - fine_radius)
        fy1 = max(0, int(best_cy) - fine_radius)
        fx2 = min(self.map_width, int(best_cx) + fine_radius)
        fy2 = min(self.map_height, int(best_cy) + fine_radius)
        fine_crop = self.logic_map_bgr[fy1:fy2, fx1:fx2]

        if fine_crop.shape[0] < 16 or fine_crop.shape[1] < 16:
            return False, None, None

        # 缓存全分辨率 minimap tensor（避免重复预处理）
        tensor_mini_fine = self.preprocess_image(minimap_bgr)
        tensor_fine = self.preprocess_image(fine_crop)
        with self.torch.no_grad():
            corr2 = self.matcher({"image0": tensor_mini_fine, "image1": tensor_fine})

        # GPU 端置信度过滤（减少 GPU→CPU 传输量）
        conf2 = corr2['confidence']
        valid_mask = conf2 > config.AI_CONFIDENCE_THRESHOLD
        mkpts0 = corr2['keypoints0'][valid_mask].cpu().numpy()
        mkpts1 = corr2['keypoints1'][valid_mask].cpu().numpy()

        if len(mkpts0) < config.AI_MIN_MATCH_COUNT:
            t1 = self._perf_time.perf_counter()
            print(f"[混合-LoFTR] 精定位失败 (匹配{len(mkpts0)}点, {(t1-t0)*1000:.0f}ms)")
            return False, None, None

        M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, config.AI_RANSAC_THRESHOLD)
        if M is None:
            return False, None, None

        h, w = minimap_bgr.shape[:2]
        center_pt = np.float32([[[w / 2, h / 2]]])
        dst = cv2.perspectiveTransform(center_pt, M)
        final_x = int(dst[0][0][0]) + fx1
        final_y = int(dst[0][0][1]) + fy1

        if 0 <= final_x < self.map_width and 0 <= final_y < self.map_height:
            t1 = self._perf_time.perf_counter()
            print(f"[混合-LoFTR] ✅ 重定位成功 ({final_x}, {final_y}) "
                  f"匹配{len(mkpts0)}点 粗扫最佳{best_score}点 耗时{(t1-t0)*1000:.0f}ms")
            return True, final_x, final_y

        return False, None, None

    def match(self, minimap_bgr):
        with self._lock:
            return self._match_impl(minimap_bgr)

    def _match_impl(self, minimap_bgr):
        t_start = self._perf_time.perf_counter()
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
            display_crop = cv2.resize(display_crop, (config.VIEW_SIZE, int(config.VIEW_SIZE * max(1, y2 - x1) / max(1, x2 - x1))))
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

            with self.torch.no_grad():
                correspondences = self.matcher(input_dict)

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
