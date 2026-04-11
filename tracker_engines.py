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


class _ArrowCache:
    """
    箭头检测运行时缓存：模板快速路径 + HSV 自适应学习 + 角度 EMA 平滑。
    箭头形状/颜色/大小在游戏中固定，利用这一先验逐步提升检测成功率。
    """

    def __init__(self):
        # 模板缓存
        self.template_gray = None       # 上次成功的灰度箭头 patch
        self.template_mask = None       # 对应的二值 mask (前景区域)
        self.last_bbox = None           # (x1, y1, x2, y2) in minimap coords
        self.hit_count = 0              # 连续模板命中次数
        self.miss_streak = 0            # 连续检测失败次数

        # HSV 自适应范围
        self._hsv_lower = np.array([15, 120, 180], dtype=np.uint8)
        self._hsv_upper = np.array([35, 255, 255], dtype=np.uint8)
        self._learned_lower = None
        self._learned_upper = None
        self._hsv_samples = 0
        self._hsv_learn_alpha = getattr(config, 'ARROW_HSV_LEARN_ALPHA', 0.1)

        # 角度 EMA
        self._ema_angle = None
        self._angle_alpha = getattr(config, 'ARROW_ANGLE_SMOOTH_ALPHA', 0.4)

        # wake 检测状态（本帧是否成功从白色区域推算了方向）
        self.last_wake_success = False

    # ---------- 模板快速路径 ----------
    def try_template_match(self, minimap_bgr):
        """在上次位置附近用模板匹配快速检测箭头，返回 (cx, cy) 或 None"""
        if self.template_gray is None or self.last_bbox is None:
            return None

        h, w = minimap_bgr.shape[:2]
        pad = getattr(config, 'ARROW_ROI_PADDING', 15)
        x1, y1, x2, y2 = self.last_bbox
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        if rx2 <= rx1 or ry2 <= ry1:
            return None
        roi = cv2.cvtColor(minimap_bgr[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        th, tw = self.template_gray.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return None

        threshold = getattr(config, 'ARROW_TEMPLATE_MATCH_THRESHOLD', 0.65)
        res = cv2.matchTemplate(roi, self.template_gray, cv2.TM_CCOEFF_NORMED,
                                mask=self.template_mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            # 命中: 模板左上角 → 中心坐标(minimap 坐标系)
            cx = rx1 + max_loc[0] + tw / 2.0
            cy = ry1 + max_loc[1] + th / 2.0
            self.hit_count += 1
            self.miss_streak = 0
            return cx, cy
        return None

    def update_template(self, minimap_bgr, mask, bbox):
        """成功检测后更新模板缓存"""
        x1, y1, x2, y2 = bbox
        h, w = minimap_bgr.shape[:2]
        pad = 3
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)

        patch = cv2.cvtColor(minimap_bgr[y1p:y2p, x1p:x2p], cv2.COLOR_BGR2GRAY)
        patch_mask = mask[y1p:y2p, x1p:x2p]
        if patch.size > 0 and patch_mask.size > 0:
            self.template_gray = patch.copy()
            self.template_mask = patch_mask.copy()
            self.last_bbox = (x1p, y1p, x2p, y2p)

    # ---------- HSV 自适应学习 ----------
    def get_hsv_bounds(self):
        """返回当前使用的 HSV 阈值 (lower, upper)"""
        if self._learned_lower is not None and self._hsv_samples >= 20:
            return self._learned_lower.copy(), self._learned_upper.copy()
        return self._hsv_lower.copy(), self._hsv_upper.copy()

    def learn_hsv(self, hsv_img, mask):
        """从成功检测的像素中学习 HSV 范围"""
        pixels = hsv_img[mask > 0]
        if len(pixels) < 10:
            return
        h_min, s_min, v_min = pixels.min(axis=0)
        h_max, s_max, v_max = pixels.max(axis=0)
        # 留 margin 防止过拟合
        margin = np.array([3, 15, 15], dtype=np.float32)
        obs_lower = np.clip(np.array([h_min, s_min, v_min], dtype=np.float32) - margin, 0, 255)
        obs_upper = np.clip(np.array([h_max, s_max, v_max], dtype=np.float32) + margin, 0, 255)

        alpha = self._hsv_learn_alpha
        if self._learned_lower is None:
            self._learned_lower = obs_lower
            self._learned_upper = obs_upper
        else:
            self._learned_lower = (1 - alpha) * self._learned_lower + alpha * obs_lower
            self._learned_upper = (1 - alpha) * self._learned_upper + alpha * obs_upper
        self._hsv_samples += 1

    # ---------- 角度 EMA ----------
    def smooth_angle(self, raw_angle):
        """圆形 EMA 平滑角度"""
        if self._ema_angle is None:
            self._ema_angle = raw_angle
            return raw_angle
        diff = _angular_diff(raw_angle, self._ema_angle)
        self._ema_angle = (self._ema_angle + self._angle_alpha * diff) % 360
        return self._ema_angle

    def record_miss(self):
        """检测失败时更新状态"""
        self.miss_streak += 1
        self.hit_count = 0
        if self.miss_streak > 10:
            self._ema_angle = None  # 过久未检测到 → 重置角度平滑


class SIFTMapTracker:
    """SIFT 传统特征匹配引擎"""

    def __init__(self):
        print("正在初始化 SIFT 引擎...")
        # 自适应 CLAHE：3 档预创建（低/中/高纹理场景）
        _cl_low = getattr(config, 'CLAHE_LIMIT_LOW_TEXTURE', 5.0)
        _cl_mid = config.SIFT_CLAHE_LIMIT
        _cl_high = getattr(config, 'CLAHE_LIMIT_HIGH_TEXTURE', 2.0)
        self._clahe_low  = cv2.createCLAHE(clipLimit=_cl_low,  tileGridSize=(8, 8))
        self._clahe_mid  = cv2.createCLAHE(clipLimit=_cl_mid,  tileGridSize=(8, 8))
        self._clahe_high = cv2.createCLAHE(clipLimit=_cl_high, tileGridSize=(8, 8))
        self.clahe = self._clahe_mid  # 默认（大地图特征提取用中档）
        self._clahe_low_thresh  = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
        self._clahe_high_thresh = getattr(config, 'CLAHE_HIGH_TEXTURE_THRESHOLD', 60)
        # SIFT 检测器：降低 contrastThreshold 提取更多弱纹理特征
        _sift_contrast = getattr(config, 'SIFT_CONTRAST_THRESHOLD', 0.02)
        self.sift = cv2.SIFT_create(contrastThreshold=_sift_contrast)

        # 上一帧箭头角度（用于箭头检测失败时降级）
        self._last_arrow_angle = 0.0
        # 箭头运行时缓存（模板快速路径 + HSV 自适应 + 角度平滑）
        self._arrow_cache = _ArrowCache()

        # 线程锁（防止并发请求导致 kp_local / flann 竞态）
        self._lock = Lock()

        logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = logic_map_bgr.shape[:2]

        logic_map_gray = cv2.cvtColor(logic_map_bgr, cv2.COLOR_BGR2GRAY)
        self._logic_map_gray = logic_map_gray  # 留存灰度大地图（ORB 局部匹配用）
        logic_map_enhanced = self._adaptive_clahe_map(logic_map_gray)
        print("正在提取大地图 SIFT 特征点...")
        self.kp_big_all, self.des_big_all = self.sift.detectAndCompute(logic_map_enhanced, None)
        print(f"✅ 全局特征点: {len(self.kp_big_all)} 个")

        # 预存全局特征点坐标（用于快速局部筛选）
        self.kp_coords = np.array([kp.pt for kp in self.kp_big_all], dtype=np.float32)

        # === 全局 FLANN 索引（常驻不销毁）===
        print("正在构建全局 FLANN 索引...")
        self.flann_global = self._create_flann(self.des_big_all)

        # === 局部 FLANN（动态重建，初始=全局）===
        self.kp_local = list(self.kp_big_all)
        self.des_local = self.des_big_all.copy()
        self.flann_local = self.flann_global
        self.using_local = False
        self.local_fail_count = 0

        # === 局部搜索参数 ===
        self.SEARCH_RADIUS = getattr(config, 'SEARCH_RADIUS', 400)
        self.LOCAL_FAIL_LIMIT = getattr(config, 'LOCAL_FAIL_LIMIT', 5)
        self.JUMP_THRESHOLD = getattr(config, 'SIFT_JUMP_THRESHOLD', 500)

        # ORB 快速备份引擎（SIFT 失败时的轻量候补）
        _orb_enabled = getattr(config, 'ORB_BACKUP_ENABLED', True)
        if _orb_enabled:
            _orb_n = getattr(config, 'ORB_NFEATURES', 500)
            self.orb = cv2.ORB_create(nfeatures=_orb_n)
            self.orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.orb = None
            self.orb_bf = None
        # ORB 局部描述子（在 _switch_to_local 时一并构建）
        self.orb_local_kp = None
        self.orb_local_des = None

        # 惯性导航状态
        self.last_x = None
        self.last_y = None
        self.lost_frames = 0
        self._sift_confused = False  # scale 异常时标记，供混合引擎快速升级
        self._arrow_offset = (0.0, 0.0)  # 箭头中心相对 minimap 中心的偏移（降级复用）

        # === 状态冻结（BLOCK 期间保持局部索引 + 坐标，恢复时直接复用）===
        self._frozen = False
        self._frozen_last_x = None
        self._frozen_last_y = None
        self._frozen_using_local = False
        self._frozen_kp_local = None
        self._frozen_des_local = None
        self._frozen_flann_local = None
        self._frozen_orb_local_kp = None
        self._frozen_orb_local_des = None
        import time as _frozen_time
        self._frozen_time_mod = _frozen_time
        self._frozen_at = 0.0

        # 性能监控
        import time as _time
        self._perf_time = _time
        self._frame_times = []

        # === 坐标锁定模式 ===
        self.coord_lock_enabled = False
        self._lock_history_size = getattr(config, 'COORD_LOCK_HISTORY_SIZE', 10)
        self._lock_search_radius = getattr(config, 'COORD_LOCK_SEARCH_RADIUS', 400)
        self._lock_max_retries = getattr(config, 'COORD_LOCK_MAX_RETRIES', 5)
        self._lock_min_to_activate = getattr(config, 'COORD_LOCK_MIN_HISTORY_TO_ACTIVATE', 15)

        # 移动预测与静止检测（用位移历史推算朝向，防止静止时 wake 随镜头漂移）
        self._prev_frame_x = None   # 上一帧定位坐标（用于计算位移）
        self._prev_frame_y = None
        self._last_moving_angle = None   # 最近一次移动状态下的可信朝向角
        self._stopped_frames = 0         # 连续静止帧计数

    # ------------------------------------------
    # 状态冻结 / 恢复（战斗/背包期间保持局部索引+坐标）
    # ------------------------------------------
    def _freeze_state(self):
        """进入 BLOCK 时冻结当前状态，避免局部索引/坐标在丢帧中被破坏"""
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
            self._frozen_orb_local_kp = self.orb_local_kp
            self._frozen_orb_local_des = self.orb_local_des
        self._frozen_at = self._frozen_time_mod.time()
        if self._frozen_last_x is not None:
            print(f"❄️ 状态冻结 ({self._frozen_last_x}, {self._frozen_last_y}) "
                  f"局部={'✓' if self.using_local else '✗'}")

    def _thaw_state(self):
        """小地图重现时恢复冻结的状态，跳过全局搜索冷启动"""
        if not self._frozen:
            return
        self._frozen = False
        elapsed = self._frozen_time_mod.time() - self._frozen_at
        timeout = getattr(config, 'FREEZE_TIMEOUT', 30.0)

        if elapsed > timeout:
            print(f"🔥 冻结已超时 ({elapsed:.0f}s > {timeout}s)，不恢复旧状态")
            return

        # 恢复坐标
        if self._frozen_last_x is not None:
            self.last_x = self._frozen_last_x
            self.last_y = self._frozen_last_y
        # 恢复局部索引
        if self._frozen_using_local and self._frozen_flann_local is not None:
            self.kp_local = self._frozen_kp_local
            self.des_local = self._frozen_des_local
            self.flann_local = self._frozen_flann_local
            self.orb_local_kp = self._frozen_orb_local_kp
            self.orb_local_des = self._frozen_orb_local_des
            self.using_local = True
        # 重置失败计数
        self.lost_frames = 0
        self.local_fail_count = 0
        print(f"♻️ 状态恢复 ({self.last_x}, {self.last_y}) "
              f"局部={'✓' if self.using_local else '✗'} 冻结时长={elapsed:.1f}s")

    @property
    def frozen(self):
        return self._frozen

    @property
    def frozen_position(self):
        """冻结期间的静态坐标（供 Kalman 暂停时使用）"""
        if self._frozen and self._frozen_last_x is not None:
            return self._frozen_last_x, self._frozen_last_y
        return None

    # ------------------------------------------
    # 自适应 CLAHE 选择
    # ------------------------------------------
    def _adaptive_clahe(self, gray):
        """根据纹理标准差选择合适的 CLAHE 档位"""
        std = np.std(gray)
        if std < self._clahe_low_thresh:
            return self._clahe_low.apply(gray)
        elif std > self._clahe_high_thresh:
            return self._clahe_high.apply(gray)
        return self._clahe_mid.apply(gray)

    def _adaptive_clahe_map(self, gray):
        """对大地图进行区域自适应 CLAHE 增强（与小地图一致，避免描述符不匹配）"""
        h, w = gray.shape[:2]
        tile = 256
        result = np.empty_like(gray)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                patch = gray[y:y+tile, x:x+tile]
                std = np.std(patch)
                if std < self._clahe_low_thresh:
                    result[y:y+tile, x:x+tile] = self._clahe_low.apply(patch)
                elif std > self._clahe_high_thresh:
                    result[y:y+tile, x:x+tile] = self._clahe_high.apply(patch)
                else:
                    result[y:y+tile, x:x+tile] = self._clahe_mid.apply(patch)
        return result

    # ------------------------------------------
    # 构建/重建 FLANN 索引
    # ------------------------------------------
    @staticmethod
    def _create_flann(descriptors):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flann.add([descriptors])
        flann.train()
        return flann

    # ------------------------------------------
    # 切换到局部搜索模式
    # ------------------------------------------
    def _switch_to_local(self, cx, cy):
        """以 (cx, cy) 为中心，提取半径内的特征点，重建局部 FLANN"""
        # 静止优化：中心未显著移动时跳过重建（节省 ~5-15ms/帧）
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

        if len(indices) < 20:   # 特征点太少不值得切局部
            return

        self.kp_local = [self.kp_big_all[i] for i in indices]
        self.des_local = self.des_big_all[indices]
        self.flann_local = self._create_flann(self.des_local)
        self.using_local = True
        self.local_fail_count = 0
        self._local_center = (cx, cy)

        # 一并构建 ORB 局部描述子
        if self.orb is not None:
            try:
                local_gray_kps = [cv2.KeyPoint(x=self.kp_coords[i][0], y=self.kp_coords[i][1], size=31)
                                  for i in indices]
                # ORB 在已知位置上 compute 描述子（用大地图灰度图的局部区域）
                # 但大地图灰度图未持久化，改用 ORB detectAndCompute 在局部区域
                # 简化方案：用 SIFT 局部关键点位置直接构建 ORB 的搜索空间标记
                self.orb_local_kp = local_gray_kps
                self.orb_local_des = None  # 延迟到实际 ORB 匹配时再计算
            except Exception:
                self.orb_local_kp = None
                self.orb_local_des = None

    # ------------------------------------------
    # 回退到全局搜索模式
    # ------------------------------------------
    def _switch_to_global(self):
        """回退全局（全局 FLANN 常驻，无需重建）"""
        self.using_local = False
        self.local_fail_count = 0
        self._local_center = None
        print(f"🌍 已切换到 全局 搜索模式 | 特征点: {len(self.kp_big_all)} | lost: {self.lost_frames}")

    # ------------------------------------------
    # 坐标锁定模式
    # ------------------------------------------
    def set_coord_lock(self, enabled):
        """开启/关闭坐标锁定模式"""
        if enabled:
            if len(self.kp_coords) == 0:
                return False
            print(f"🔒 坐标锁定已启用 (历史{self._lock_history_size}均值±{self._lock_search_radius})")
        else:
            print(f"🔓 坐标锁定已关闭，恢复正常搜索")
        self.coord_lock_enabled = enabled
        return True

    @staticmethod
    def _compute_lock_anchor(pos_deque, n=10):
        """从坐标历史 deque 计算最近 N 个坐标的平均值"""
        if pos_deque is None or len(pos_deque) < n:
            return None
        recent = list(pos_deque)[-n:]
        ax = sum(p[0] for p in recent) / len(recent)
        ay = sum(p[1] for p in recent) / len(recent)
        return int(ax), int(ay)

    def _match_locked(self, minimap_bgr, anchor_x, anchor_y):
        """
        锁定模式下的匹配：以 anchor 为中心，限定搜索范围 ±RADIUS。
        失败则逐步放宽阈值重试，最多 MAX_RETRIES 次。
        返回与 _match_impl 相同格式的 dict
        """
        t_start = self._perf_time.perf_counter()
        r = self._lock_search_radius

        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = self._adaptive_clahe(minimap_gray)
        h_mm, w_mm = minimap_gray.shape[:2]
        circ_mask = self._get_circular_mask(h_mm, w_mm)
        kp_mini, des_mini = self.sift.detectAndCompute(minimap_gray, circ_mask)

        if des_mini is None or len(kp_mini) < 2:
            return self._make_locked_result(False, None, None, t_start, "NO_FEATURES")

        # 在全局特征点中筛选锚点范围内的点
        dx = np.abs(self.kp_coords[:, 0] - anchor_x)
        dy = np.abs(self.kp_coords[:, 1] - anchor_y)
        mask = (dx < r) & (dy < r)
        indices = np.where(mask)[0]

        if len(indices) < 5:
            return self._make_locked_result(False, None, None, t_start, "FEW_KPTS")

        locked_kp = [self.kp_big_all[i] for i in indices]
        locked_des = self.des_big_all[indices]
        locked_flann = self._create_flann(locked_des)

        # 逐步放宽阈值的重试策略
        retry_ratios = [0.9, 0.8, 0.75, 0.7, 0.65]
        retry_min_matches = [5, 4, 4, 3, 3]

        for attempt in range(min(self._lock_max_retries, len(retry_ratios))):
            ratio = retry_ratios[attempt] if attempt < len(retry_ratios) else 0.65
            min_match = retry_min_matches[attempt] if attempt < len(retry_min_matches) else 3

            try:
                matches = locked_flann.knnMatch(des_mini, k=2)
            except cv2.error:
                continue

            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio * n.distance:
                        good.append(m)

            if len(good) >= min_match:
                src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([locked_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)
                if M is not None:
                    inlier_count = int(mask.sum()) if mask is not None else 0
                    if inlier_count >= min_match:
                        # Homography 缩放合理性检查
                        sx = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
                        sy = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
                        avg_scale = (sx + sy) / 2
                        max_scale = getattr(config, 'SIFT_MAX_HOMOGRAPHY_SCALE', 8.0)
                        if avg_scale > max_scale or avg_scale < 1.0 / max_scale:
                            continue
                        h, w = minimap_gray.shape
                        arrow_center = self._detect_arrow_center(minimap_bgr)
                        if arrow_center is not None:
                            ref_x, ref_y, ref_angle = arrow_center
                            self._last_arrow_angle = ref_angle
                        else:
                            # 降级：使用 minimap 中心点
                            ref_x, ref_y = w / 2.0, h / 2.0
                            ref_angle = self._last_arrow_angle

                        center_pt = np.float32([[[ref_x, ref_y]]])
                        dst_center = cv2.perspectiveTransform(center_pt, M)
                        tx = int(dst_center[0][0][0])
                        ty = int(dst_center[0][0][1])

                        if abs(tx - anchor_x) <= r * 1.5 and abs(ty - anchor_y) <= r * 1.5:
                            if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
                                return self._make_locked_result(True, tx, ty, t_start, "LOCKED", arrow_angle=ref_angle,
                                                           retry=attempt + 1)

        # 所有重试都失败
        return self._make_locked_result(False, None, None, t_start, "LOCK_FAIL")

    def _make_locked_result(self, found, cx, cy, t_start, state, arrow_angle=None, retry=0):
        """构造锁定模式的返回字典"""
        t_elapsed = (self._perf_time.perf_counter() - t_start) * 1000
        self._frame_times.append(t_elapsed)
        if len(self._frame_times) >= 60:
            avg = sum(self._frame_times) / len(self._frame_times)
            print(f"[🔒锁定] 平均耗时: {avg:.1f}ms | state: {state} | retries: {retry}")
            self._frame_times.clear()
        return {
            'found': found,
            'center_x': cx,
            'center_y': cy,
            'arrow_angle': arrow_angle,
            'is_inertial': not found,
            'match_count': 0,
            'map_width': self.map_width,
            'map_height': self.map_height,
            '_locked_state': state,
        }

    # ------------------------------------------
    # 箭头检测：从圆形小地图中提取玩家箭头中心坐标
    # ------------------------------------------
    def _detect_arrow_center(self, minimap_bgr):
        """
        在小地图图像中检测黄色/橙色箭头，返回 (cx, cy, angle) 或 None。
        两阶段检测：
          1. 快速路径: 模板匹配（cached patch + 小 ROI）→ ~0.3ms
          2. 完整路径: HSV + 形态学 → ~1.5ms
        角度由白色泛光(wake)方向计算，不再依赖凸包尖端。
        """
        h, w = minimap_bgr.shape[:2]
        if h < 10 or w < 10:
            return None

        cache = self._arrow_cache

        # === 快速路径：模板匹配 ===
        tmpl_hit = cache.try_template_match(minimap_bgr)
        if tmpl_hit is not None:
            cx, cy = tmpl_hit
            raw_wake = self._compute_wake_angle(minimap_bgr)
            cache.last_wake_success = raw_wake is not None
            angle = raw_wake if raw_wake is not None else (
                cache._ema_angle if cache._ema_angle is not None else 0.0)
            angle = cache.smooth_angle(angle)
            return (float(cx), float(cy), float(angle))

        # === 完整路径：HSV 颜色检测 ===
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        lower, upper = cache.get_hsv_bounds()
        mask = cv2.inRange(hsv, lower, upper)

        # 圆形掩码：排除小地图四角以外区域（防止误检建筑物橙色）
        circ_mask = self._get_circular_mask(h, w)
        mask = mask & circ_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cache.record_miss()
            return None

        # 选取最靠近小地图中心的轮廓（箭头总在中心，最大轮廓可能是边缘建筑）
        mcx, mcy = w / 2.0, h / 2.0
        best_cnt = None
        best_dist = float('inf')
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                continue
            M_c = cv2.moments(cnt)
            if M_c['m00'] < 1e-6:
                continue
            ccx = M_c['m10'] / M_c['m00']
            ccy = M_c['m01'] / M_c['m00']
            dist = (ccx - mcx) ** 2 + (ccy - mcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cnt = cnt

        if best_cnt is None:
            cache.record_miss()
            return None

        M_moments = cv2.moments(best_cnt)
        cx = M_moments['m10'] / M_moments['m00']
        cy = M_moments['m01'] / M_moments['m00']

        # 角度计算：白色泛光(wake)方向
        raw_wake = self._compute_wake_angle(minimap_bgr)
        cache.last_wake_success = raw_wake is not None
        angle = raw_wake if raw_wake is not None else (
            cache._ema_angle if cache._ema_angle is not None else 0.0)
        angle = cache.smooth_angle(angle)

        # 更新缓存
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        cache.update_template(minimap_bgr, mask, (bx, by, bx + bw, by + bh))
        cache.learn_hsv(hsv, mask)
        cache.miss_streak = 0

        return (float(cx), float(cy), float(angle))

    @staticmethod
    def _compute_wake_angle(minimap_bgr):
        """
        通过白色泛光(wake)方向计算箭头朝向。
        箭头在朝向方向会发出类似手电筒的白色/亮色光晕，
        这些低饱和度高亮度像素的质心偏移方向即为箭头朝向。
        在亮色地形（雪地/沙漠等）中像素计数过高时返回 None，交由 EMA 平滑。
        返回角度(0=北/上, 顺时针增加)或 None。
        """
        h, w = minimap_bgr.shape[:2]
        mcx, mcy = w // 2, h // 2

        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 距离中心的距离平方映射
        yy, xx = np.ogrid[:h, :w]
        dist_sq = (xx - mcx) ** 2 + (yy - mcy) ** 2

        # 内环（跳过箭头图标本身）和外环上限（圆形小地图边界内）
        inner_r = max(5, min(h, w) // 12)
        outer_r = min(h, w) // 2 - 3

        # 泛光像素：低饱和 + 高亮度 + 在环形区域内
        wake = ((sat < 60) & (val > 180)
                & (dist_sq > inner_r * inner_r)
                & (dist_sq < outer_r * outer_r))

        wy, wx = np.where(wake)
        count = len(wx)

        # 太少说明无泛光信号；太多说明地形整体偏亮（雪地等），信噪比太低
        if count < 15 or count > 3000:
            return None

        # 亮度加权质心
        weights = val[wake].astype(np.float64)
        w_dx = np.average(wx.astype(np.float64), weights=weights) - mcx  # East
        w_dy = np.average(wy.astype(np.float64), weights=weights) - mcy  # South

        # 偏移量太小说明泛光均匀分布，无法判断方向
        if math.sqrt(w_dx * w_dx + w_dy * w_dy) < 3.0:
            return None

        # 图像坐标 → 导航角度 (0=北/上, 90=东/右, 180=南/下, 270=西/左)
        angle = math.degrees(math.atan2(w_dx, -w_dy)) % 360
        return angle

    @staticmethod
    def _draw_arrow_marker(img_bgr, cx, cy, size=None, angle=0):
        """
        在地图上绘制玩家方向箭头标记。
        angle: 旋转角度（度），0=朝上，顺时针增加
        """
        if size is None:
            size = 12

        pts = np.array([
            [cx, cy - size],
            [cx - size * 0.6, cy + size * 0.7],
            [cx + size * 0.6, cy + size * 0.7],
        ], dtype=np.float64)

        if angle != 0:
            rad = math.radians(angle)   # 正角度 = 从北顺时针旋转（修正左右镜像）
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            centered = pts - np.array([cx, cy])
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = (rot_mat @ centered.T).T + np.array([cx, cy])
            pts = rotated.astype(np.int32)
        else:
            pts = pts.astype(np.int32)

        overlay = img_bgr.copy()
        cv2.fillPoly(overlay, [pts], (48, 182, 254))
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)

    @staticmethod
    def _paste_arrow_patch(img_bgr, cx, cy, arrow_patch, offset_x, offset_y, scale=1.0):
        """将从小地图抠出的箭头图像块(BGRA)贴到目标图像上"""
        if arrow_patch is None or arrow_patch.size == 0:
            return

        ph, pw = arrow_patch.shape[:2]

        if scale != 1.0:
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            arrow_patch = cv2.resize(arrow_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            offset_x = int(offset_x * scale)
            offset_y = int(offset_y * scale)
            ph, pw = new_w, new_h

        paste_x = cx - offset_x
        paste_y = cy - offset_y

        dh, dw = img_bgr.shape[:2]

        sx1 = max(0, -paste_x)
        sy1 = max(0, -paste_y)
        dx1 = max(0, paste_x)
        dy1 = max(0, paste_y)
        dx2 = min(dw, paste_x + pw)
        dy2 = min(dh, paste_y + ph)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        if dx2 <= dx1 or dy2 <= dy1 or sx2 <= sx1 or sy2 <= sy1:
            return

        roi = img_bgr[dy1:dy2, dx1:dx2]
        patch_region = arrow_patch[sy1:sy2, sx1:sx2]

        bgr_part = patch_region[:, :, :3]
        alpha_part = patch_region[:, :, 3:4].astype(np.float32) / 255.0

        blended = (bgr_part.astype(np.float32) * alpha_part +
                   roi.astype(np.float32) * (1.0 - alpha_part))
        img_bgr[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)

    # ------------------------------------------
    # ORB 快速备份匹配（局部模式下 SIFT 失败时的候补）
    # ------------------------------------------
    def _orb_local_match(self, minimap_bgr, minimap_gray):
        """
        ORB 备份：在 SIFT 局部匹配失败后尝试 ORB 匹配。
        比 SIFT 快 ~3x，精度略低但足以维持跟踪连续性。
        返回 (tx, ty) 或 None。
        """
        if not self.using_local or self.orb is None:
            return None

        try:
            h_mm, w_mm = minimap_gray.shape[:2]
            circ_mask = self._get_circular_mask(h_mm, w_mm)
            kp_orb_mini, des_orb_mini = self.orb.detectAndCompute(minimap_gray, circ_mask)
            if des_orb_mini is None or len(kp_orb_mini) < 3:
                return None

            # 在局部区域的大地图上做 ORB 检测
            if self.last_x is None or self.last_y is None:
                return None
            r = self.SEARCH_RADIUS
            lx, ly = self.last_x, self.last_y
            x1 = max(0, lx - r)
            y1 = max(0, ly - r)
            x2 = min(self.map_width, lx + r)
            y2 = min(self.map_height, ly + r)

            # 大地图局部灰度（从预存灰度大地图裁剪）
            local_gray = self._logic_map_gray[y1:y2, x1:x2]
            local_gray = self._adaptive_clahe(local_gray)

            kp_orb_map, des_orb_map = self.orb.detectAndCompute(local_gray, None)
            if des_orb_map is None or len(kp_orb_map) < 5:
                return None

            matches = self.orb_bf.knnMatch(des_orb_mini, des_orb_map, k=2)
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < 5:
                return None

            src_pts = np.float32([kp_orb_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_orb_map[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            if M is None:
                return None
            inlier_count = int(mask.sum()) if mask is not None else 0
            if inlier_count < 4:
                return None

            # 缩放合理性
            sx_h = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
            sy_h = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
            avg_scale = (sx_h + sy_h) / 2
            if avg_scale > 5.0 or avg_scale < 0.2:
                return None

            center_pt = np.float32([[[w_mm / 2.0, h_mm / 2.0]]])
            dst_center = cv2.perspectiveTransform(center_pt, M)
            tx = int(dst_center[0][0][0]) + x1  # 局部→全局坐标
            ty = int(dst_center[0][0][1]) + y1

            if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
                # 跳变检查（使用与主匹配相同的阈值）
                jump = abs(tx - lx) + abs(ty - ly)
                if jump < self.JUMP_THRESHOLD:
                    print(f"[ORB备份] ✅ 匹配成功 ({tx},{ty}) inliers={inlier_count}")
                    return tx, ty
        except Exception as e:
            pass
        return None

    # ------------------------------------------
    # 核心匹配（两轮搜索 + 跳变过滤）
    # ------------------------------------------
    def match(self, minimap_bgr):
        with self._lock:
            return self._match_impl(minimap_bgr)

    @staticmethod
    def _make_circular_mask(h, w):
        """为圆形小地图生成圆形掩码，裁掉四角噪声区域"""
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        r = min(cx, cy) - 2  # 留2px安全边距
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask

    def _get_circular_mask(self, h, w):
        """带缓存的圆形掩码（小地图尺寸校准后固定，避免每帧重建）"""
        key = (h, w)
        if not hasattr(self, '_circ_mask_cache'):
            self._circ_mask_cache = {}
        cached = self._circ_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = self._make_circular_mask(h, w)
        self._circ_mask_cache[key] = mask
        return mask

    def _match_impl(self, minimap_bgr):
        t_start = self._perf_time.perf_counter()
        found = False
        center_x, center_y = None, None
        arrow_angle = None
        is_inertial = False
        match_quality = 0.0  # 0-1 匹配质量
        self._sift_confused = False  # 每帧重置

        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = self._adaptive_clahe(minimap_gray)

        # 圆形 mask：过滤圆形小地图四角噪声特征点
        h_mm, w_mm = minimap_gray.shape[:2]
        circ_mask = self._get_circular_mask(h_mm, w_mm)
        kp_mini, des_mini = self.sift.detectAndCompute(minimap_gray, circ_mask)

        if des_mini is not None and len(kp_mini) >= 2:

            for search_round in range(2):

                if search_round == 0 and self.using_local:
                    current_kp = self.kp_local
                    current_flann = self.flann_local
                elif search_round == 0 and not self.using_local:
                    current_kp = list(self.kp_big_all)
                    current_flann = self.flann_global
                else:
                    if not self.using_local:
                        break
                    current_kp = list(self.kp_big_all)
                    current_flann = self.flann_global

                try:
                    matches = current_flann.knnMatch(des_mini, k=2)
                except cv2.error:
                    matches = []

                good = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                            good.append(m)

                if len(good) >= config.SIFT_MIN_MATCH_COUNT:
                    src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)

                    if M is not None:
                        inlier_count = int(mask.sum()) if mask is not None else 0
                        if inlier_count < config.SIFT_MIN_MATCH_COUNT:
                            continue

                        # Homography 缩放合理性检查（防止UI误匹配）
                        sx = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
                        sy = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
                        avg_scale = (sx + sy) / 2
                        max_scale = getattr(config, 'SIFT_MAX_HOMOGRAPHY_SCALE', 8.0)
                        if avg_scale > max_scale or avg_scale < 1.0 / max_scale:
                            self._sift_confused = True  # 几何异常 → 标记混乱
                            continue

                        h, w = minimap_gray.shape

                        # 箭头检测：获取精确中心+泛光方向
                        arrow_center = self._detect_arrow_center(minimap_bgr)
                        if arrow_center is not None:
                            ref_x, ref_y, ref_angle = arrow_center
                            arrow_angle = ref_angle
                            self._last_arrow_angle = ref_angle
                            # 缓存箭头中心偏移量（相对 minimap 中心），降级时复用
                            self._arrow_offset = (ref_x - w / 2.0, ref_y - h / 2.0)
                            quality_bonus = 1.0
                        else:
                            # 降级：使用上次箭头偏移量估算（保持参考点稳定，避免震荡）
                            ox, oy = getattr(self, '_arrow_offset', (0.0, 0.0))
                            ref_x, ref_y = w / 2.0 + ox, h / 2.0 + oy
                            arrow_angle = self._last_arrow_angle
                            quality_bonus = 0.6

                        center_pt = np.float32([[[ref_x, ref_y]]])
                        dst_center = cv2.perspectiveTransform(center_pt, M)
                        tx = int(dst_center[0][0][0])
                        ty = int(dst_center[0][0][1])

                        # 置信度评分
                        inlier_ratio = inlier_count / max(len(good), 1)
                        match_quality = min(1.0, inlier_ratio * quality_bonus)

                        if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
                            if self.last_x is not None:
                                jump = abs(tx - self.last_x) + abs(ty - self.last_y)
                                # 局部搜索用更紧的阈值，全局回退放宽但仍然有限
                                max_jump = self.JUMP_THRESHOLD if search_round == 0 else self.JUMP_THRESHOLD * 2
                                if jump < max_jump:
                                    found = True
                                    center_x, center_y = tx, ty
                                    break
                                else:
                                    continue
                            else:
                                found = True
                                center_x, center_y = tx, ty
                                break

        # --- 状态更新 ---
        if found:
            self.last_x = center_x
            self.last_y = center_y
            self.lost_frames = 0
            self.local_fail_count = 0
            self._switch_to_local(center_x, center_y)

        else:
            # === ORB 快速备份：SIFT 失败 + 局部模式 → ORB 尝试挽救 ===
            if (not found and self.using_local and self.orb is not None
                    and des_mini is not None and len(kp_mini) >= 2):
                orb_result = self._orb_local_match(minimap_bgr, minimap_gray)
                if orb_result is not None:
                    found = True
                    center_x, center_y = orb_result
                    match_quality = 0.5  # ORB 匹配质量标记为中等
                    arrow_angle = self._last_arrow_angle
                    self.last_x = center_x
                    self.last_y = center_y
                    self.lost_frames = 0
                    self.local_fail_count = 0
                    self._switch_to_local(center_x, center_y)

            if not found:
                self.lost_frames += 1

                if self.using_local:
                    self.local_fail_count += 1
                    if self.local_fail_count >= self.LOCAL_FAIL_LIMIT:
                        self._switch_to_global()

                if self.last_x is not None and self.lost_frames <= config.MAX_LOST_FRAMES:
                    found = True
                    center_x = self.last_x
                    center_y = self.last_y
                    is_inertial = True
                elif self.lost_frames > config.MAX_LOST_FRAMES:
                    self.last_x = None
                    self.last_y = None
                    if self.using_local:
                        self._switch_to_global()

        # === 性能统计 ===
        t_elapsed = (self._perf_time.perf_counter() - t_start) * 1000
        self._frame_times.append(t_elapsed)
        if len(self._frame_times) >= 60:
            avg = sum(self._frame_times) / len(self._frame_times)
            mode = "局部" if self.using_local else "全局"
            feat = len(self.kp_local) if self.using_local else len(self.kp_big_all)
            print(f"[{mode}模式] 平均耗时: {avg:.1f}ms | "
                  f"特征点: {feat} | lost: {self.lost_frames} | "
                  f"fps: {1000/avg:.1f}")
            self._frame_times.clear()

        # ===== 移动方向预测 + 静止检测 =====
        # 目的1: 静止时 wake（白色区域）跟随镜头旋转而非玩家朝向，需冻结角度
        # 目的2: 移动中 wake 失效时，用位移方向角（地图坐标系推算）作为备选
        if found and not is_inertial and self._prev_frame_x is not None:
            dx = center_x - self._prev_frame_x
            dy = center_y - self._prev_frame_y
            dist = math.sqrt(dx * dx + dy * dy)
            _thr = getattr(config, 'ARROW_STOPPED_THRESHOLD', 6)
            _min_fr = getattr(config, 'ARROW_STOPPED_FRAMES_MIN', 4)
            if dist > _thr:
                # 玩家正在移动
                self._stopped_frames = 0
                if self._arrow_cache.last_wake_success:
                    # wake 本帧成功：记录为可信移动角
                    self._last_moving_angle = arrow_angle
                else:
                    # wake 本帧失败：用地图位移方向角补偿
                    # 地图坐标系: x 右=东, y 下=南 → atan2(dx, -dy) = 导航角(北=0, 顺时针)
                    move_angle = math.degrees(math.atan2(dx, -dy)) % 360
                    arrow_angle = move_angle
                    self._last_moving_angle = move_angle
            else:
                # 玩家静止：white area 随镜头旋转，角度不可信
                self._stopped_frames += 1
                if (self._stopped_frames >= _min_fr
                        and self._last_moving_angle is not None):
                    # 冻结：使用最后一次移动时确认的角度
                    arrow_angle = self._last_moving_angle
        if found and not is_inertial:
            self._prev_frame_x = center_x
            self._prev_frame_y = center_y

        return {
            'found': found,
            'center_x': center_x,
            'center_y': center_y,
            'arrow_angle': arrow_angle,
            'is_inertial': is_inertial,
            'match_count': 0,
            'match_quality': match_quality,
            'map_width': self.map_width,
            'map_height': self.map_height,
        }


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
