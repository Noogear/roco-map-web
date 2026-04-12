"""
tracker_engines.py - 纯识别引擎模块（无 Web 框架依赖）

包含:
  SIFTMapTracker    SIFT 传统特征匹配引擎

算法层已拆分到 backend/core/：
  enhance.py  图像增强（CLAHE / 色温补偿）
  features.py SIFT/ORB 特征提取与匹配
  flow.py     LK 光流追踪器
  ecc.py      ECC 像素级对齐
  cold_start.py 低纹理冷启动
"""

import math
import os
import cv2
import numpy as np
from collections import deque
from threading import Lock
from backend import config

from backend.core.enhance import (make_clahe_pair, adaptive_clahe, adaptive_clahe_map,
                                  enhance_for_texture, correct_color_temperature)
from backend.core.features import (CircularMaskCache, create_flann,
                                   extract_minimap_features, sift_match_region)
from backend.core.flow import LKTracker
from backend.core.ecc import ecc_align
from backend.core.cold_start import ColdStarter
from backend.tracking.direction import ArrowDirectionSystem


class SIFTMapTracker:
    """SIFT 传统特征匹配引擎"""

    def __init__(self):
        print("正在初始化 SIFT 引擎...")
        # 自适应 CLAHE（双档），及掩码缓存
        self._clahe_normal, self._clahe_low = make_clahe_pair()
        self._low_texture_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
        self._mask_cache = CircularMaskCache()

        # SIFT 检测器
        _sift_contrast = getattr(config, 'SIFT_CONTRAST_THRESHOLD', 0.02)
        self.sift = cv2.SIFT_create(contrastThreshold=_sift_contrast)

        # 上一帧箭头角度
        self._last_arrow_angle = 0.0
        self._last_arrow_stopped = True
        # 箭头方向系统（纯坐标驱动）
        self._arrow_dir = ArrowDirectionSystem()

        # 线程锁（防止并发请求导致 kp_local / flann 竞态）
        self._lock = Lock()

        logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = logic_map_bgr.shape[:2]

        logic_map_gray = cv2.cvtColor(logic_map_bgr, cv2.COLOR_BGR2GRAY)
        self._logic_map_gray = logic_map_gray
        logic_map_enhanced = adaptive_clahe_map(logic_map_gray, self._clahe_normal, self._clahe_low,
                                                self._low_texture_thresh)
        print("正在提取大地图 SIFT 特征点...")
        self.kp_big_all, self.des_big_all = self.sift.detectAndCompute(logic_map_enhanced, None)
        print(f"✅ 全局特征点: {len(self.kp_big_all)} 个")

        self.kp_coords = np.array([kp.pt for kp in self.kp_big_all], dtype=np.float32)

        print("正在构建全局 FLANN 索引...")
        self.flann_global = create_flann(self.des_big_all)

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

        # LK 光流追踪器
        self._lk = LKTracker(
            enabled=getattr(config, 'LK_ENABLED', True),
            sift_every=getattr(config, 'LK_SIFT_INTERVAL', 4),
            min_conf=getattr(config, 'LK_MIN_CONFIDENCE', 0.5),
            mask_cache=self._mask_cache,
        )
        self._lk.map_scale = 4.0   # 初始比例，SIFT 匹配成功后更新

        # ECC 低纹理兜底
        self._ecc_enabled = getattr(config, 'ECC_ENABLED', True)
        self._ecc_min_cc = getattr(config, 'ECC_MIN_CORRELATION', 0.25)
        self._last_sift_scale = self._lk.map_scale  # 默认与 LK scale 一致

        # 海洋/低纹理区域冷启动
        self._cold_starter = ColdStarter(
            map_width=self.map_width,
            map_height=self.map_height,
            logic_map_gray=logic_map_gray,
            mask_cache=self._mask_cache,
        )
        self._cold_starter.build_candidates()
        print(f"🌊 低纹理区域候选: {len(self._cold_starter._candidates)} 个")
        self._cold_start_cd = 0   # 冷启动冷却帧数

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
        self.flann_local = create_flann(self.des_local)
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

        minimap_bgr = correct_color_temperature(minimap_bgr)
        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = adaptive_clahe(minimap_gray, self._clahe_normal, self._clahe_low, self._low_texture_thresh)
        kp_mini, des_mini = extract_minimap_features(minimap_gray, self.sift, self._mask_cache)
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
        locked_flann = create_flann(locked_des)

        # 逐步放宽阈值重试
        for ratio, min_m in [(0.85, 5), (0.78, 4), (0.70, 3)]:
            result = sift_match_region(
                kp_mini, des_mini, minimap_gray.shape,
                locked_kp, locked_flann, ratio, min_m,
                self.map_width, self.map_height)
            if result is not None:
                tx, ty, inlier_count, quality, _scale = result
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
                     match_count=0, match_quality=0.0, _locked_state='',
                     _tp_far_candidate=None):
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
            '_tp_far_candidate': _tp_far_candidate,
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
    def _orb_nearby_match(self, minimap_gray, cx_hint, cy_hint, radius):
        """ORB 附近搜索：在 (cx_hint, cy_hint) 周围 radius 内做 ORB 匹配"""
        if self.orb is None:
            return None
        h_mm, w_mm = minimap_gray.shape[:2]
        circ_mask = self._mask_cache.get(h_mm, w_mm)
        kp_orb_mini, des_orb_mini = self.orb.detectAndCompute(minimap_gray, circ_mask)
        if des_orb_mini is None or len(kp_orb_mini) < 3:
            return None

        x1 = max(0, cx_hint - radius)
        y1 = max(0, cy_hint - radius)
        x2 = min(self.map_width, cx_hint + radius)
        y2 = min(self.map_height, cy_hint + radius)

        local_gray = self._logic_map_gray[y1:y2, x1:x2]
        local_gray = adaptive_clahe(local_gray, self._clahe_normal, self._clahe_low, self._low_texture_thresh)
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

        # 护眼/防蓝光软件叠加层色温补偿（硬件 Night Light 不影响截图，无需处理）
        minimap_bgr = correct_color_temperature(minimap_bgr)
        minimap_gray_raw = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        texture_std = float(np.std(minimap_gray_raw))

        # 自适应 CLAHE + 纹理增强
        minimap_gray = adaptive_clahe(minimap_gray_raw, self._clahe_normal, self._clahe_low, self._low_texture_thresh)
        minimap_gray = enhance_for_texture(minimap_gray, texture_std, self._clahe_low)

        self._lk.frame_num += 1
        run_sift = self._lk.should_run_sift()

        _tp_far_candidate = None  # 传送候选：全局 SIFT 匹配成功但超跳变阈值

        # ======================================================
        # 第一层：LK 光流快速跟踪（每帧 ~2ms，跳过部分 SIFT 调用）
        # ======================================================
        lk_result = self._lk.track(minimap_gray, self.last_x)
        if lk_result is not None and not run_sift:
            dx_map, dy_map, lk_conf = lk_result
            if lk_conf >= self._lk.min_conf:
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
            kp_mini, des_mini = extract_minimap_features(minimap_gray, self.sift, self._mask_cache)

            # 超低纹理 + 特征过少 → 更激进的预处理重试
            if texture_std < 15 and (des_mini is None or len(kp_mini) < 15):
                enhanced = enhance_for_texture(minimap_gray_raw, texture_std=5, clahe_low=self._clahe_low)
                kp_retry, des_retry = extract_minimap_features(enhanced, self.sift, self._mask_cache)
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

                    result = sift_match_region(
                        kp_mini, des_mini, minimap_gray.shape,
                        current_kp, current_flann, eff_match_ratio, eff_min_match,
                        self.map_width, self.map_height)

                    if result is not None:
                        tx, ty, inlier_count, quality, avg_scale = result
                        if self.last_x is not None:
                            max_jump = self.JUMP_THRESHOLD if search_round == 0 else self.JUMP_THRESHOLD * 2
                            if abs(tx - self.last_x) + abs(ty - self.last_y) >= max_jump:
                                # 超跳变阈值：保留为传送候选（取质量最高的）
                                if quality >= 0.3 and (_tp_far_candidate is None or quality > _tp_far_candidate[2]):
                                    _tp_far_candidate = (tx, ty, quality)
                                continue
                        found, center_x, center_y, match_quality = True, tx, ty, quality
                        # SIFT 成功 → 同步 scale
                        self._last_sift_scale = avg_scale
                        self._lk.map_scale = avg_scale
                        self._cold_starter.update_scale(avg_scale)
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
                            nearby_flann = create_flann(nearby_des)
                            # 缓存（最多保留 30 个桶）
                            if len(self._nearby_flann_cache) >= 30:
                                self._nearby_flann_cache.pop(next(iter(self._nearby_flann_cache)))
                            self._nearby_flann_cache[bucket] = (nearby_kp, nearby_des, nearby_flann)
                        else:
                            nearby_kp = nearby_des = nearby_flann = None

                    if nearby_flann is not None:
                        result = sift_match_region(
                            kp_mini, des_mini, minimap_gray.shape,
                            nearby_kp, nearby_flann, 0.90, 3,
                            self.map_width, self.map_height)
                        if result is not None:
                            tx, ty, inlier_count, quality, avg_scale = result
                            if abs(tx - self.last_x) + abs(ty - self.last_y) < self.JUMP_THRESHOLD * 1.5:
                                found, center_x, center_y = True, tx, ty
                                match_quality = quality * 0.8
                                self._last_sift_scale = avg_scale
                                self._lk.map_scale = avg_scale
                                self._cold_starter.update_scale(avg_scale)

                    # ---- ORB 兜底 ----
                    if not found:
                        orb_result = self._orb_nearby_match(
                            minimap_gray, self.last_x, self.last_y, nr)
                        if orb_result is not None:
                            found, center_x, center_y = True, *orb_result
                            match_quality = 0.4

                    # ---- ECC 像素级兜底（低/中纹理均可，海洋必备）----
                    if not found:
                        ecc_result = ecc_align(
                            minimap_gray, self._logic_map_gray,
                            self.last_x, self.last_y,
                            self._last_sift_scale, self.map_width, self.map_height,
                            self.JUMP_THRESHOLD, self._ecc_min_cc)
                        if ecc_result is not None:
                            found, center_x, center_y = True, *ecc_result
                            match_quality = 0.3

        # ---- 冷启动低纹理场景兜底（海洋/大片裸地/海岸线）----
        # 触发条件1: 完全丢失跟踪（last_x=None）+ 小地图低纹理 + 冷却结束
        # 触发条件2: 半丢失(连续丢帧>8但<MAX)+ 低纹理 — 海洋近岸/横渡场景专用
        # 冷却机制：
        #   成功：冻结 60 帧（~2秒），下帧由 LK+SIFT 接管，不会再触发
        #   失败：冻结 20 帧（~0.7秒），尽快重试，适应横渡过程中持续找不到的情况
        # ★ 成功后恢复：cold_result 进入普通状态更新，last_x/last_y 赋值，
        #   下一帧 LK + SIFT 自动接管，不再触发（last_x 已非 None）
        if self._cold_start_cd > 0:
            self._cold_start_cd -= 1
        _ocean_texture_thresh = 55      # fully_lost 阈值：含海岸线混合纹理（比纯海洋高）
        # semi_lost 用更严格的纹理阈值，与 _build_ocean_candidates 保持一致，
        # 避免陆地低纹理区（草地/森林）短暂丢帧后误触发冷启动跳到海洋
        _semi_lost_texture_thresh = getattr(config, 'OCEAN_STD_THRESHOLD', 35)
        _semi_lost = (self.last_x is not None and self.lost_frames >= 12
                      and texture_std < _semi_lost_texture_thresh)
        _fully_lost = (self.last_x is None and texture_std < _ocean_texture_thresh)
        if not found and (_fully_lost or _semi_lost) and self._cold_start_cd == 0:
            cold_result = self._cold_starter.locate(minimap_gray_raw, self.last_x, self.last_y, self.lost_frames)
            if cold_result is not None:
                # semi_lost 模式下拒绝距上次坐标超过 JUMP_THRESHOLD 的冷启动结果：
                # 防止陆地误匹配到远处海洋候选区后 ECC/LK 在错误位置自我强化锁死
                if (self.last_x is not None
                        and abs(cold_result[0] - self.last_x)
                        + abs(cold_result[1] - self.last_y) >= self.JUMP_THRESHOLD):
                    cold_result = None
            if cold_result is not None:
                found, center_x, center_y = True, cold_result[0], cold_result[1]
                match_quality = 0.25
                self._last_sift_scale = self._cold_starter._last_scale
                self._lk.map_scale = self._cold_starter._last_scale
                self._cold_start_cd = 60   # 成功：长冷却，让 LK/SIFT 接管
            else:
                self._cold_start_cd = 20   # 失败：短冷却，横渡时尽快重试

        # 更新 LK 上一帧（不论用哪层跟踪，都更新灰度图）
        self._lk.prev_gray = minimap_gray.copy()
        if found and not is_inertial:
            # 重新提取当前帧关键点给下一帧 LK 使用
            h_mm, w_mm = minimap_gray.shape[:2]
            circ_mask = self._mask_cache.get(h_mm, w_mm)
            pts = cv2.goodFeaturesToTrack(
                minimap_gray, maxCorners=60, qualityLevel=0.01,
                minDistance=7, mask=circ_mask)
            self._lk.prev_pts = pts  # shape (N,1,2) or None

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
                self._lk.prev_gray = None
                self._lk.prev_pts = None
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
            match_quality=match_quality,
            _tp_far_candidate=_tp_far_candidate)

