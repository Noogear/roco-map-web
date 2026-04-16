"""
tracker_engines.py - 识别引擎主流程（无 Web 框架依赖）

算法层拆分到 backend/core/：
    enhance.py    图像增强（CLAHE / 色温补偿）
    features.py   SIFT / RootSIFT 特征提取与匹配
    flow.py       LK 光流追踪器
    ecc.py        ECC 像素级对齐
    hash_index.py 感知哈希粗定位索引
"""

import cv2
import numpy as np
import math
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from backend import config

from backend.core.enhance import (make_clahe_pair, adaptive_clahe_map,
                                  enhance_minimap, correct_color_temperature,
                                  classify_scene_by_color, make_scene_boosted_gray)
from backend.core.features import (CircularMaskCache, create_flann,
                                   extract_map_features_tiled, extract_minimap_features,
                                   sift_match_region)
from backend.core.flow import LKTracker
from backend.core.ecc import ecc_align, phase_correlate
from backend.core.hash_index import MapHashIndex
from backend.core.icon_mask import detect_ui_icon_exclusion_mask
from backend.core.temporal_bridge import LowTextureTemporalBridge, BridgeObservation
from backend.core.data_standards import DataScope, bind_scope
from backend.tracking.direction import ArrowDirectionSystem


# ==================== 共享只读 SIFT 资源（全局单例）====================

class SharedSIFTResources:
    """
    全局共享的只读 SIFT 资源：特征点、描述子、FLANN 索引、哈希索引等。

    这些数据在服务器启动时提取一次，所有 SIFTMapTracker 实例共享只读引用。
    knnMatch() 对 FLANN KD-Tree 只做只读查询，线程安全。
    """

    def __init__(self) -> None:
        bind_scope(self, DataScope.GLOBAL_SHARED)
        print("=" * 50)
        print("正在加载共享 SIFT 资源...")

        _sift_contrast = getattr(config, 'SIFT_CONTRAST_THRESHOLD', 0.02)
        self.sift = cv2.SIFT_create(contrastThreshold=_sift_contrast)

        logic_map_gray = cv2.imread(config.LOGIC_MAP_PATH, cv2.IMREAD_GRAYSCALE)
        if logic_map_gray is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = logic_map_gray.shape[:2]
        self._logic_map_gray = logic_map_gray

        # CLAHE 增强后提取全局特征（临时 CLAHE 对象，仅用于地图预处理）
        _clahe_normal, _clahe_low = make_clahe_pair()
        _low_texture_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
        logic_map_enhanced = adaptive_clahe_map(logic_map_gray, _clahe_normal, _clahe_low,
                                                _low_texture_thresh)

        _global_tile_size = getattr(config, 'SIFT_GLOBAL_TILE_SIZE', 1536)
        _global_tile_overlap = getattr(config, 'SIFT_GLOBAL_TILE_OVERLAP', 96)
        _global_tile_feature_cap = getattr(config, 'SIFT_GLOBAL_MAX_FEATURES_PER_TILE', 0)
        print(f"正在分块提取大地图 SIFT 特征点... (tile={_global_tile_size}, overlap={_global_tile_overlap})")
        self.kp_big_all, self.des_big_all = extract_map_features_tiled(
            logic_map_enhanced,
            self.sift,
            tile_size=_global_tile_size,
            overlap=_global_tile_overlap,
            max_features_per_tile=_global_tile_feature_cap,
        )
        if self.des_big_all is None or not self.kp_big_all:
            raise RuntimeError('全局 SIFT 特征提取失败：未找到可用特征点')
        print(f"✅ 全局特征点: {len(self.kp_big_all)} 个")

        # ECC 专用：与运行时小地图同域的 CLAHE 增强大图（避免 ECC 两侧亮度域不匹配）
        self._logic_map_gray_clahe = logic_map_enhanced
        del logic_map_enhanced   # 让变量名失效，实际数据由 _logic_map_gray_clahe 持有

        self.kp_coords = np.array([kp.pt for kp in self.kp_big_all], dtype=np.float32)

        print("正在构建全局 FLANN 索引...")
        self.flann_global = create_flann(self.des_big_all)

        # 哈希索引：用于全局丢失、冻结恢复后的快速粗定位
        self._hash_index = MapHashIndex(
            logic_map_gray,
            map_width=self.map_width,
            map_height=self.map_height,
        )
        print(f"🔑 哈希索引条目: {len(self._hash_index._xs)} 个")

        # ---- S 通道辅助索引（低纹理/海洋场景补充特征）----
        # 内存策略: BGR → HSV 转换后立即释放 BGR/HSV，仅保留 S 通道和其增强副本
        # 峰值: gray(1x) + BGR(3x) + HSV(3x) ≈ 7x gray，随后降至 gray(1x) + S_clahe(1x)
        self.kp_big_sat = None
        self.des_big_sat = None
        self.kp_coords_sat = None
        self.flann_global_sat = None
        self._logic_map_sat_clahe = None
        _sat_enabled = getattr(config, 'SIFT_SAT_ENABLED', True)
        if _sat_enabled:
            print("正在提取饱和度通道辅助特征（低纹理/海洋场景）...")
            _map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
            if _map_bgr is not None:
                _map_hsv = cv2.cvtColor(_map_bgr, cv2.COLOR_BGR2HSV)
                del _map_bgr                               # 立即释放 BGR
                _map_sat_raw = _map_hsv[:, :, 1].copy()   # 提取 S 通道
                del _map_hsv                               # 立即释放 HSV
                # 对 S 通道做逐块 CLAHE（不走 enhance_minimap 以避免灰度专用的双边滤波）
                _clahe_sat_init = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
                _sat_tile_patch = 512
                _h_sat, _w_sat = _map_sat_raw.shape[:2]
                _map_sat_clahe = np.empty_like(_map_sat_raw)
                for _sy in range(0, _h_sat, _sat_tile_patch):
                    for _sx in range(0, _w_sat, _sat_tile_patch):
                        _patch = _map_sat_raw[_sy:_sy + _sat_tile_patch, _sx:_sx + _sat_tile_patch]
                        _map_sat_clahe[_sy:_sy + _sat_tile_patch, _sx:_sx + _sat_tile_patch] = \
                            _clahe_sat_init.apply(_patch)
                del _map_sat_raw                           # raw S 通道不再需要
                self._logic_map_sat_clahe = _map_sat_clahe
                # 分块提取 SIFT 特征，缩小每块 tile 特征数限制内存峰值
                _sat_tile_size = getattr(config, 'SIFT_SAT_TILE_SIZE', 1536)
                _sat_cap = getattr(config, 'SIFT_SAT_MAX_FEATURES_PER_TILE', 600)
                print(f"  S通道分块提取... (tile={_sat_tile_size}, cap/tile={_sat_cap})")
                _kp_sat, _des_sat = extract_map_features_tiled(
                    _map_sat_clahe, self.sift,
                    tile_size=_sat_tile_size,
                    overlap=_global_tile_overlap,
                    max_features_per_tile=_sat_cap,
                )
                if _des_sat is not None and _kp_sat:
                    self.kp_big_sat = _kp_sat
                    self.des_big_sat = _des_sat
                    self.kp_coords_sat = np.array([kp.pt for kp in _kp_sat], dtype=np.float32)
                    self.flann_global_sat = create_flann(_des_sat)
                    print(f"✅ S通道特征点: {len(_kp_sat)} 个")
                else:
                    print("⚠️ S通道特征提取失败，将跳过 SAT 辅助匹配")
            else:
                print("⚠️ 无法加载彩色地图，跳过 S 通道索引")

        print("=" * 50)


_shared_sift: SharedSIFTResources | None = None
_shared_sift_lock = Lock()


def get_shared_sift() -> SharedSIFTResources:
    """双检锁懒加载全局共享 SIFT 资源。"""
    global _shared_sift
    if _shared_sift is not None:
        return _shared_sift
    with _shared_sift_lock:
        if _shared_sift is not None:
            return _shared_sift
        _shared_sift = SharedSIFTResources()
        return _shared_sift


@dataclass(frozen=True)
class EngineTrackingSnapshot:
    """引擎跟踪态快照，供编排层做一次性回滚。"""

    last_x: int | None
    last_y: int | None
    lost_frames: int
    using_local: bool
    local_fail_count: int
    kp_local: list
    des_local: np.ndarray | None
    flann_local: cv2.FlannBasedMatcher
    local_center: tuple[int, int] | None
    last_arrow_angle: float
    last_arrow_stopped: bool
    arrow_history: tuple[tuple[int, int], ...]
    arrow_dir_last_angle: float
    arrow_dir_is_stopped: bool
    arrow_dir_stop_streak: int
    lk_prev_gray: np.ndarray | None
    lk_prev_pts: np.ndarray | None
    lk_map_scale: float
    lk_frame_num: int
    last_sift_scale: float
    relocalize_cd: int
    local_success_streak: int
    force_global_revalidate: bool
    force_global_revalidate_frame: int
    watchdog_accum_dx: float
    watchdog_accum_dy: float
    watchdog_suspect_streak: int
    watchdog_last_sift_x: int | None
    watchdog_last_sift_y: int | None
    watchdog_consecutive_ok: int
    watchdog_static_streak: int
    watchdog_hash_mismatch_streak: int
    watchdog_cooldown: int
    pending_freeze_resume_hint: dict | None


def classify_scene(texture_std: float) -> str:
    """按纹理强弱做粗分类，用于选择匹配路径。"""
    if texture_std < 15:
        return 'ocean'       # 纯海洋/纯色
    if texture_std < 35:
        return 'low_texture'  # 海岸线/低纹理裸地
    if texture_std < 55:
        return 'mixed'        # 混合纹理
    return 'urban'            # 高纹理城镇/森林


class SIFTMapTracker:
    """SIFT 传统特征匹配引擎（per-session 实例，共享只读资源）"""

    def __init__(self, shared: SharedSIFTResources):
        bind_scope(self, DataScope.SESSION_SCOPED)
        # 引用共享只读资源
        self.sift = shared.sift
        self._logic_map_gray = shared._logic_map_gray
        self._logic_map_gray_clahe = shared._logic_map_gray_clahe
        self.map_height = shared.map_height
        self.map_width = shared.map_width
        self.kp_big_all = shared.kp_big_all
        self.des_big_all = shared.des_big_all
        self.kp_coords = shared.kp_coords
        self.flann_global = shared.flann_global
        self._hash_index = shared._hash_index

        # S 通道辅助资源（只读共享，低纹理/海洋场景）
        self.kp_big_sat = shared.kp_big_sat
        self.des_big_sat = shared.des_big_sat
        self.kp_coords_sat = shared.kp_coords_sat
        self.flann_global_sat = shared.flann_global_sat
        self._logic_map_sat_clahe = shared._logic_map_sat_clahe
        # per-session：CLAHE 有内部状态不能共享，仅在 SAT 可用时创建
        self._clahe_sat = (cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
                           if self.flann_global_sat is not None else None)

        # per-session: CLAHE 有内部状态，不能共享
        self._clahe_normal, self._clahe_low = make_clahe_pair()
        self._low_texture_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
        self._mask_cache = CircularMaskCache()

        # 上一帧箭头角度
        self._last_arrow_angle = 0.0
        self._last_arrow_stopped = True
        # 箭头方向系统（纯坐标驱动，支持小变化防抖 + 大方向敏感）
        self._arrow_dir = ArrowDirectionSystem(
            history_size=getattr(config, 'ARROW_POS_HISTORY_LEN', 4),
            ema_alpha=getattr(config, 'ARROW_ANGLE_SMOOTH_ALPHA', 0.35),
            stop_speed_px=getattr(config, 'ARROW_MOVE_MIN_DISPLACEMENT', 6),
            stop_debounce=getattr(config, 'ARROW_STOPPED_DEBOUNCE', 20),
            small_change_threshold=getattr(config, 'ARROW_SMALL_CHANGE_THRESHOLD', 12.0),
            big_change_threshold=getattr(config, 'ARROW_BIG_CHANGE_THRESHOLD', 45.0),
        )

        # 线程锁（防止并发请求导致 kp_local / flann 竞态）
        self._lock = Lock()

        # 局部 FLANN（动态重建）
        self.kp_local = self.kp_big_all
        self.des_local = self.des_big_all
        self.flann_local = self.flann_global
        self.using_local = False
        self.local_fail_count = 0

        self.SEARCH_RADIUS = getattr(config, 'SEARCH_RADIUS', 400)
        self.LOCAL_FAIL_LIMIT = getattr(config, 'LOCAL_FAIL_LIMIT', 5)
        self.JUMP_THRESHOLD = getattr(config, 'SIFT_JUMP_THRESHOLD', 500)
        self.NEARBY_SEARCH_RADIUS = getattr(config, 'NEARBY_SEARCH_RADIUS', 600)
        self._local_center = None
        self._local_revalidate_interval = getattr(config, 'LOCAL_REVALIDATE_INTERVAL', 6)
        self._local_revalidate_min_quality = getattr(config, 'LOCAL_REVALIDATE_MIN_QUALITY', 0.45)
        self._local_revalidate_margin = getattr(config, 'LOCAL_REVALIDATE_MARGIN', 0.08)
        self._local_revalidate_diff = getattr(config, 'LOCAL_REVALIDATE_DIFF', 220)
        self._local_success_streak = 0
        self._force_global_revalidate = False
        self._force_global_revalidate_frame = 0

        # LK 光流追踪器
        self._lk = LKTracker(
            enabled=getattr(config, 'LK_ENABLED', True),
            sift_every=getattr(config, 'LK_SIFT_INTERVAL', 4),
            min_conf=getattr(config, 'LK_MIN_CONFIDENCE', 0.5),
            mask_cache=self._mask_cache,
        )
        self._lk.map_scale = 4.0   # 初始比例，SIFT 匹配成功后更新

        # ECC 局部像素对齐兜底
        self._ecc_enabled = getattr(config, 'ECC_ENABLED', True)
        self._ecc_min_cc = getattr(config, 'ECC_MIN_CORRELATION', 0.25)
        self._last_sift_scale = self._lk.map_scale  # 默认与 LK scale 一致

        self._relocalize_cd = 0   # 粗定位冷却帧数

        # 看门狗：LK 累积位移 vs SIFT 结果一致性，连续不一致时强制解锁死锁状态
        self._watchdog_accum_dx: float = 0.0
        self._watchdog_accum_dy: float = 0.0
        self._watchdog_suspect_streak: int = 0
        self._watchdog_last_sift_x: int | None = None
        self._watchdog_last_sift_y: int | None = None
        self._watchdog_consecutive_ok: int = 0
        self._watchdog_static_streak: int = 0
        self._watchdog_hash_mismatch_streak: int = 0
        self._watchdog_cooldown: int = 0

        # 附近搜索 FLANN 缓存（OrderedDict LRU，避免每帧重建）
        self._nearby_flann_cache: OrderedDict = OrderedDict()
        self._nearby_flann_bucket = 200  # 网格粒度（地图像素）

        # 低纹理时序桥接（多假设轨迹）
        self._bridge = LowTextureTemporalBridge(
            enabled=getattr(config, 'LOW_TEXTURE_BRIDGE_ENABLED', True),
            top_k=getattr(config, 'LOW_TEXTURE_BRIDGE_TOP_K', 8),
            decay=getattr(config, 'LOW_TEXTURE_BRIDGE_DECAY', 0.92),
            transition_penalty=getattr(config, 'LOW_TEXTURE_BRIDGE_TRANSITION_PENALTY', 0.002),
            min_obs_quality=getattr(config, 'LOW_TEXTURE_BRIDGE_MIN_OBS_QUALITY', 0.08),
            strong_source_bonus=getattr(config, 'LOW_TEXTURE_BRIDGE_STRONG_SOURCE_BONUS', 0.12),
            exit_min_quality=getattr(config, 'LOW_TEXTURE_BRIDGE_EXIT_MIN_QUALITY', 0.55),
        )

        # 惯性导航状态
        self.last_x = None
        self.last_y = None
        self.lost_frames = 0
        self._watchdog_triggered = False

        # INERTIAL 坐标锁定检测：SIFT 连续失败时若画面仍在变化，提前退出惯性保持以免坐标卡死
        self._inertial_entry_gray: np.ndarray | None = None
        self._inertial_failed_sift_count: int = 0

        # 状态冻结
        self._frozen = False
        self._frozen_last_x = None
        self._frozen_last_y = None
        self._frozen_local_kp = None
        self._frozen_local_flann = None
        self._pending_freeze_resume_hint = None


    # ---- 状态冻结 / 恢复 ----
    def _freeze_state(self):
        if self._frozen:
            return
        self._frozen = True
        self._pending_freeze_resume_hint = None
        self._frozen_last_x = self.last_x
        self._frozen_last_y = self.last_y
        self._frozen_local_kp = None
        self._frozen_local_flann = None
        if self.using_local:
            self._frozen_local_kp = self.kp_local
            self._frozen_local_flann = self.flann_local

        # 进入冻结后，活体状态清空，只保留一次性恢复所需快照，避免跨场景脏状态续算。
        self._lk.reset()
        _bridge = getattr(self, '_bridge', None)
        if _bridge is not None:
            _bridge.reset()
        self.last_x = None
        self.last_y = None
        self._switch_to_global()
        self._nearby_flann_cache.clear()

    def _thaw_state(self):
        if not self._frozen:
            return
        self._frozen = False
        self._pending_freeze_resume_hint = None
        # 冻结结束后统一生成一次性恢复提示；首帧先做轻量恢复匹配，再决定是否回退到重定位路径。
        if self._frozen_last_x is not None and self._frozen_last_y is not None:
            self._pending_freeze_resume_hint = {
                'x': self._frozen_last_x,
                'y': self._frozen_last_y,
                'kp_local': self._frozen_local_kp,
                'flann_local': self._frozen_local_flann,
            }

        self._frozen_last_x = None
        self._frozen_last_y = None
        self._frozen_local_kp = None
        self._frozen_local_flann = None
        self.lost_frames = 0
        self.local_fail_count = 0
        self._watchdog_consecutive_ok = 0
        self._watchdog_static_streak = 0

        # thaw 后首帧不要复用冻结前的 LK 帧缓存，也不要保留旧 nearby cache。
        self._lk.reset()
        self._nearby_flann_cache.clear()

    def freeze_for_scene_change(self) -> None:
        """公开冻结入口：暂停跟踪并保留一次性恢复快照。"""
        self._freeze_state()

    def resume_after_scene_change(self) -> None:
        """公开恢复入口：从冻结态回到常规跟踪流程。"""
        self._thaw_state()

    def _get_or_build_nearby_flann(self, cx, cy, radius):
        """按中心点和半径获取附近区域 FLANN，带 LRU 缓存。"""
        bucket = (cx // self._nearby_flann_bucket,
                  cy // self._nearby_flann_bucket,
                  int(radius))
        if bucket in self._nearby_flann_cache:
            self._nearby_flann_cache.move_to_end(bucket)
            return self._nearby_flann_cache[bucket]

        dx_arr = np.abs(self.kp_coords[:, 0] - cx)
        dy_arr = np.abs(self.kp_coords[:, 1] - cy)
        indices = np.where((dx_arr < radius) & (dy_arr < radius))[0]
        if len(indices) < 10:
            return None, None, None

        nearby_kp = [self.kp_big_all[i] for i in indices]
        nearby_des = self.des_big_all[indices]
        nearby_flann = create_flann(nearby_des)
        if len(self._nearby_flann_cache) >= 30:
            self._nearby_flann_cache.popitem(last=False)  # LRU: 淘汰最久未使用
        self._nearby_flann_cache[bucket] = (nearby_kp, nearby_des, nearby_flann)
        return nearby_kp, nearby_des, nearby_flann

    def _try_nearby_match_stack(self, kp_mini, des_mini, minimap_gray, hint_x, hint_y,
                                radius, allow_ecc=True, source_prefix=''):
        """复用的附近搜索链：SIFT nearby → 可选 ECC → Template Matching兜底。"""
        nearby_kp, nearby_des, nearby_flann = self._get_or_build_nearby_flann(hint_x, hint_y, radius)
        if nearby_flann is not None:
            result = sift_match_region(
                kp_mini, des_mini, minimap_gray.shape,
                nearby_kp, nearby_flann, 0.90, 3,
                self.map_width, self.map_height)
            if result is not None:
                tx, ty, inlier_count, quality, avg_scale = result
                if abs(tx - hint_x) + abs(ty - hint_y) < self.JUMP_THRESHOLD * 1.5:
                    src = (source_prefix + '_SIFT_NEARBY') if source_prefix else 'SIFT_NEARBY'
                    return tx, ty, quality * 0.8, avg_scale, src, inlier_count

        if allow_ecc and self._ecc_enabled:
            ecc_result = ecc_align(
                minimap_gray, self._logic_map_gray_clahe,
                hint_x, hint_y,
                self._last_sift_scale, self.map_width, self.map_height,
                self.JUMP_THRESHOLD, self._ecc_min_cc)
            if ecc_result is not None:
                src = (source_prefix + '_ECC') if source_prefix else 'ECC'
                return ecc_result[0], ecc_result[1], 0.3, None, src, 0

        return None

    def _consume_freeze_resume_match(self, kp_mini, des_mini, minimap_gray,
                                     match_ratio, min_match):
        """消费一次冻结恢复提示：local → nearby → hash 粗定位。"""
        hint = self._pending_freeze_resume_hint
        self._pending_freeze_resume_hint = None
        if hint is None:
            return None

        hint_x = hint.get('x')
        hint_y = hint.get('y')
        if hint_x is None or hint_y is None:
            return None

        kp_local = hint.get('kp_local')
        flann_local = hint.get('flann_local')
        if kp_local is not None and flann_local is not None:
            result = sift_match_region(
                kp_mini, des_mini, minimap_gray.shape,
                kp_local, flann_local, match_ratio, min_match,
                self.map_width, self.map_height)
            if result is not None:
                tx, ty, inlier_count, quality, avg_scale = result
                if abs(tx - hint_x) + abs(ty - hint_y) < self.JUMP_THRESHOLD * 1.5:
                    return tx, ty, quality, avg_scale, 'FREEZE_RESUME_LOCAL', inlier_count

        radius = max(self.NEARBY_SEARCH_RADIUS,
                     getattr(config, 'FREEZE_RESUME_SEARCH_RADIUS', self.SEARCH_RADIUS * 2))
        nearby = self._try_nearby_match_stack(
            kp_mini, des_mini, minimap_gray,
            hint_x, hint_y, radius,
            allow_ecc=False,
            source_prefix='FREEZE_RESUME')
        if nearby is not None:
            return nearby

        # nearby 也失败时再退回哈希粗定位，避免恢复路径直接掉回全局搜索
        hash_candidates = self._hash_index.locate(
            minimap_gray, last_x=hint_x, last_y=hint_y,
            radius=radius, max_results=1)
        if hash_candidates:
            hx, hy, hdist = hash_candidates[0]
            if abs(hx - hint_x) + abs(hy - hint_y) < self.JUMP_THRESHOLD * 2:
                return hx, hy, max(0.15, 0.35 - hdist * 0.02), None, 'FREEZE_RESUME_HASH', 0
        return None

    @property
    def frozen(self):
        return self._frozen

    @property
    def last_position(self) -> tuple[int, int] | None:
        if self.last_x is None or self.last_y is None:
            return None
        return self.last_x, self.last_y

    @property
    def last_arrow_state(self) -> tuple[float, bool]:
        return self._last_arrow_angle, self._last_arrow_stopped

    @property
    def frozen_position(self):
        """冻结期间的静态坐标（供 Kalman 暂停时使用）"""
        if self._frozen and self._frozen_last_x is not None:
            return self._frozen_last_x, self._frozen_last_y
        return None

    def snapshot_tracking_state(self) -> EngineTrackingSnapshot:
        """导出当前跟踪状态，供上层在拒绝测量时回滚。"""
        arrow_dir = self._arrow_dir
        return EngineTrackingSnapshot(
            last_x=self.last_x,
            last_y=self.last_y,
            lost_frames=self.lost_frames,
            using_local=self.using_local,
            local_fail_count=self.local_fail_count,
            kp_local=self.kp_local,
            des_local=self.des_local,
            flann_local=self.flann_local,
            local_center=self._local_center,
            last_arrow_angle=self._last_arrow_angle,
            last_arrow_stopped=self._last_arrow_stopped,
            arrow_history=tuple(arrow_dir._history),
            arrow_dir_last_angle=arrow_dir._last_angle,
            arrow_dir_is_stopped=arrow_dir._is_stopped,
            arrow_dir_stop_streak=arrow_dir._low_move_streak,
            lk_prev_gray=(None if self._lk.prev_gray is None else self._lk.prev_gray.copy()),
            lk_prev_pts=(None if self._lk.prev_pts is None else self._lk.prev_pts.copy()),
            lk_map_scale=self._lk.map_scale,
            lk_frame_num=self._lk.frame_num,
            last_sift_scale=self._last_sift_scale,
            relocalize_cd=self._relocalize_cd,
            local_success_streak=self._local_success_streak,
            force_global_revalidate=self._force_global_revalidate,
            force_global_revalidate_frame=self._force_global_revalidate_frame,
            watchdog_accum_dx=self._watchdog_accum_dx,
            watchdog_accum_dy=self._watchdog_accum_dy,
            watchdog_suspect_streak=self._watchdog_suspect_streak,
            watchdog_last_sift_x=self._watchdog_last_sift_x,
            watchdog_last_sift_y=self._watchdog_last_sift_y,
            watchdog_consecutive_ok=self._watchdog_consecutive_ok,
            watchdog_static_streak=self._watchdog_static_streak,
            watchdog_hash_mismatch_streak=self._watchdog_hash_mismatch_streak,
            watchdog_cooldown=self._watchdog_cooldown,
            pending_freeze_resume_hint=(None if self._pending_freeze_resume_hint is None
                                        else dict(self._pending_freeze_resume_hint)),
        )

    def restore_tracking_state(self, snapshot: EngineTrackingSnapshot) -> None:
        """恢复到历史快照，阻断错误识别结果继续自我强化。"""
        arrow_dir = self._arrow_dir
        with self._lock:
            self.last_x = snapshot.last_x
            self.last_y = snapshot.last_y
            self.lost_frames = snapshot.lost_frames
            self.using_local = snapshot.using_local
            self.local_fail_count = snapshot.local_fail_count
            self.kp_local = snapshot.kp_local
            self.des_local = snapshot.des_local
            self.flann_local = snapshot.flann_local
            self._local_center = snapshot.local_center
            self._last_arrow_angle = snapshot.last_arrow_angle
            self._last_arrow_stopped = snapshot.last_arrow_stopped
            arrow_dir._history.clear()
            arrow_dir._history.extend(snapshot.arrow_history)
            arrow_dir._last_angle = snapshot.arrow_dir_last_angle
            arrow_dir._is_stopped = snapshot.arrow_dir_is_stopped
            arrow_dir._low_move_streak = snapshot.arrow_dir_stop_streak
            self._lk.prev_gray = (None if snapshot.lk_prev_gray is None
                                  else snapshot.lk_prev_gray.copy())
            self._lk.prev_pts = (None if snapshot.lk_prev_pts is None
                                 else snapshot.lk_prev_pts.copy())
            self._lk.map_scale = snapshot.lk_map_scale
            self._lk.frame_num = snapshot.lk_frame_num
            self._last_sift_scale = snapshot.last_sift_scale
            self._relocalize_cd = snapshot.relocalize_cd
            self._local_success_streak = snapshot.local_success_streak
            self._force_global_revalidate = snapshot.force_global_revalidate
            self._force_global_revalidate_frame = snapshot.force_global_revalidate_frame
            self._watchdog_accum_dx = snapshot.watchdog_accum_dx
            self._watchdog_accum_dy = snapshot.watchdog_accum_dy
            self._watchdog_suspect_streak = snapshot.watchdog_suspect_streak
            self._watchdog_last_sift_x = snapshot.watchdog_last_sift_x
            self._watchdog_last_sift_y = snapshot.watchdog_last_sift_y
            self._watchdog_consecutive_ok = snapshot.watchdog_consecutive_ok
            self._watchdog_static_streak = snapshot.watchdog_static_streak
            self._watchdog_hash_mismatch_streak = snapshot.watchdog_hash_mismatch_streak
            self._watchdog_cooldown = snapshot.watchdog_cooldown
            self._pending_freeze_resume_hint = (None if snapshot.pending_freeze_resume_hint is None
                                                else dict(snapshot.pending_freeze_resume_hint))
            self._watchdog_triggered = False

    def mark_measurement_rejected(self) -> None:
        """登记一次被上层拒绝的测量，避免无限沿用旧基点并强制进入更保守的重定位路径。"""
        with self._lock:
            self._local_success_streak = 0
            self._force_global_revalidate = True
            self._force_global_revalidate_frame = self._lk.frame_num
            self._watchdog_accum_dx = 0.0
            self._watchdog_accum_dy = 0.0
            self._watchdog_last_sift_x = None
            self._watchdog_last_sift_y = None
            self._watchdog_consecutive_ok = 0
            self._watchdog_static_streak = 0
            self._watchdog_hash_mismatch_streak = 0
            self._watchdog_cooldown = 0
            self._watchdog_triggered = False
            self._bridge.reset()

            self.lost_frames += 1
            if self.using_local:
                self.local_fail_count += 1
                if self.local_fail_count >= self.LOCAL_FAIL_LIMIT:
                    self._switch_to_global()
                    self._nearby_flann_cache.clear()

            if self.lost_frames > config.MAX_LOST_FRAMES:
                self.last_x = None
                self.last_y = None
                self._switch_to_global()
                self._nearby_flann_cache.clear()
                self._lk.reset()

    def sync_external_position(self, x: int, y: int) -> None:
        """传送后立即重置引擎状态，确保不被旧基点干扰。
        
        调用场景：传送确认 → 坐标已由外部验证，立即应用并清理所有旧状态
        特别清理冻结状态和待恢复提示，避免下一帧被冻结前的信息误导。
        """
        with self._lock:
            # === 重置后续追踪基点 ===
            self.last_x = x
            self.last_y = y
            self.lost_frames = 0
            self.local_fail_count = 0
            
            # === 清空冻结相关状态 ===
            # 传送极有可能跨场景，必须彻底断绝冻结前的任何恢复参考
            self._frozen = False
            self._frozen_last_x = None
            self._frozen_last_y = None
            self._frozen_local_kp = None
            self._frozen_local_flann = None
            self._pending_freeze_resume_hint = None
            
            # === 重置活体追踪状态（避免与传送前混混） ===
            # LK 光流上一帧状态（下一帧前将被重算）
            self._lk.reset()
            self.using_local = False  # 从全局模式开始，逐帧建立局部信心
            
            # 清空 local 特征快照，后续会根据新坐标重建
            self.kp_local = None
            self.des_local = None
            self.flann_local = None
            self._local_center = None
            
            # === 重置看门狗和重定位状态 ===
            # Watchdog / 重定位冷却等都应该刷新，给下一帧充分的探索空间
            self._relocalize_cd = 0
            self._watchdog_accum_dx = 0.0
            self._watchdog_accum_dy = 0.0
            self._watchdog_suspect_streak = 0
            self._watchdog_last_sift_x = None
            self._watchdog_last_sift_y = None
            self._watchdog_consecutive_ok = 0
            self._watchdog_static_streak = 0
            self._watchdog_hash_mismatch_streak = 0
            self._watchdog_cooldown = 0
            self._watchdog_triggered = False
            self._force_global_revalidate = False
            self._bridge.reset()
            
            # === 清空缓存 ===
            self._nearby_flann_cache.clear()
            
            # === 最后：建立新的局部追踪基点（从全局特征坐标开始） ===
            self._switch_to_local(x, y)

    # ---- 局部/全局搜索切换 ----
    def _switch_to_local(self, cx, cy):
        """以 (cx, cy) 为中心，提取半径内的特征点，重建局部 FLANN"""
        if self.using_local and self._local_center is not None:
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
        # 切全局时一并清除看门狗累积，避免历史位移数据干扰新轮次判断
        self._watchdog_accum_dx = 0.0
        self._watchdog_accum_dy = 0.0
        self._watchdog_last_sift_x = None
        self._watchdog_last_sift_y = None
        self._watchdog_consecutive_ok = 0
        self._watchdog_static_streak = 0
        self._watchdog_hash_mismatch_streak = 0

    def _trigger_watchdog_unlock(self, reason: str) -> None:
        """统一执行看门狗解锁，并进入短冷却避免刚解锁又被同一批脏状态连环触发。

        设计约束：看门狗是“死锁解锁器”，不是“坐标重置器”。
        触发后只重置局部跟踪链路（local/LK/cache/revalidate），保留坐标锚点，
        由后续常规状态机走 inertial/全局复核，避免坐标因看门狗本身乱飘。
        """
        self._switch_to_global()
        self._nearby_flann_cache.clear()
        self._lk.reset()
        self._force_global_revalidate = True
        self._force_global_revalidate_frame = self._lk.frame_num
        self._watchdog_suspect_streak = 0
        self._watchdog_hash_mismatch_streak = 0
        self._watchdog_triggered = True
        self._watchdog_cooldown = max(
            self._watchdog_cooldown,
            int(getattr(config, 'WATCHDOG_TRIGGER_COOLDOWN', 24)),
        )
        print(f"[看门狗] {reason} → 进入 {self._watchdog_cooldown} 帧冷却")

    def _resolve_local_revalidation(self, local_match, global_match, tp_far_candidate):
        """
        比较局部命中与全局复核结果。

        若全局结果更可信且与局部结果显著冲突，则优先使用全局结果；
        若全局结果对应的是远距离跳点，则只作为传送候选上报，当前帧回退为未命中，
        防止错误局部坐标继续喂给 LK / ECC / 本地 FLANN 形成自我强化锁死。
        """
        tx, ty, inlier_count, quality, avg_scale, _source = local_match
        gx, gy, ginliers, gquality, gscale = global_match
        diff = abs(gx - tx) + abs(gy - ty)
        if diff < self._local_revalidate_diff:
            return local_match, tp_far_candidate, False
        if gquality + self._local_revalidate_margin < quality:
            return local_match, tp_far_candidate, False

        if (self.last_x is not None and self.last_y is not None
                and abs(gx - self.last_x) + abs(gy - self.last_y) >= self.JUMP_THRESHOLD):
            if gquality >= 0.3 and (tp_far_candidate is None or gquality > tp_far_candidate[2]):
                tp_far_candidate = (gx, gy, gquality)
            return None, tp_far_candidate, True

        return (gx, gy, ginliers, gquality, gscale, 'SIFT_GLOBAL_REVALIDATED'), tp_far_candidate, True

    def _maybe_revalidate_local_match(self, kp_mini, des_mini, mm_shape,
                                      ratio, min_match, local_match, tp_far_candidate):
        tx, ty, inlier_count, quality, avg_scale, source = local_match
        self._local_success_streak += 1
        need_validate = (
            self._force_global_revalidate
            or quality < self._local_revalidate_min_quality
            or self._local_success_streak >= self._local_revalidate_interval
        )
        # _force_global_revalidate 超过 3 个 SIFT 周期未消费则自动过期
        if self._force_global_revalidate:
            _max_stale = self._lk.sift_every * 3
            if self._lk.frame_num - self._force_global_revalidate_frame > _max_stale:
                self._force_global_revalidate = False
                need_validate = (
                    quality < self._local_revalidate_min_quality
                    or self._local_success_streak >= self._local_revalidate_interval
                )
        if not need_validate:
            return local_match, tp_far_candidate, False

        self._force_global_revalidate = False
        self._local_success_streak = 0
        global_result = sift_match_region(
            kp_mini, des_mini, mm_shape,
            self.kp_big_all, self.flann_global, ratio, min_match,
            self.map_width, self.map_height)
        if global_result is None:
            return local_match, tp_far_candidate, False
        return self._resolve_local_revalidation(local_match, global_result, tp_far_candidate)

    def _make_result(self, found, cx=None, cy=None,
                     arrow_angle=None, arrow_stopped=True, is_inertial=False,
                     match_count=0, match_quality=0.0, _locked_state='',
                     _tp_far_candidate=None, source='',
                     _watchdog_triggered=False):
        """统一构造返回字典"""
        return {
            'found': found, 'center_x': cx, 'center_y': cy,
            'arrow_angle': arrow_angle, 'arrow_stopped': arrow_stopped,
            'is_inertial': is_inertial, 'match_count': match_count,
            'match_quality': match_quality,
            'map_width': self.map_width, 'map_height': self.map_height,
            '_locked_state': _locked_state,
            '_tp_far_candidate': _tp_far_candidate,
            'source': source,
            '_watchdog_triggered': _watchdog_triggered,
        }

    def _collect_bridge_observations(
        self,
        scene: str,
        minimap_gray_raw: np.ndarray,
        found: bool,
        center_x: int | None,
        center_y: int | None,
        match_quality: float,
        source: str,
        frame_cfg: dict | None = None,
    ) -> list[BridgeObservation]:
        observations: list[BridgeObservation] = []

        if found and center_x is not None and center_y is not None:
            observations.append(BridgeObservation(
                x=int(center_x),
                y=int(center_y),
                quality=float(max(0.0, match_quality)),
                source=str(source or 'OBS'),
            ))

        if scene not in ('ocean', 'low_texture'):
            return observations

        if frame_cfg is None:
            max_hash_obs = int(getattr(config, 'LOW_TEXTURE_BRIDGE_HASH_CANDIDATES', 4))
            radius = int(getattr(config, 'LOW_TEXTURE_BRIDGE_HASH_RADIUS', 900))
        else:
            max_hash_obs = int(frame_cfg.get('bridge_hash_candidates', 4))
            radius = int(frame_cfg.get('bridge_hash_radius', 900))
        if max_hash_obs <= 0:
            return observations

        hash_candidates = self._hash_index.locate(
            minimap_gray_raw,
            last_x=self.last_x,
            last_y=self.last_y,
            radius=radius if self.last_x is not None and self.last_y is not None else 0,
            max_results=max_hash_obs,
        )
        for hx, hy, hdist in hash_candidates:
            h_quality = max(0.08, 0.35 - hdist * 0.02)
            observations.append(BridgeObservation(
                x=int(hx),
                y=int(hy),
                quality=float(h_quality),
                source='HASH_INDEX',
            ))

        return observations

    def _apply_low_texture_bridge(
        self,
        scene: str,
        minimap_gray_raw: np.ndarray,
        found: bool,
        center_x: int | None,
        center_y: int | None,
        match_quality: float,
        source: str,
        lk_result,
        frame_cfg: dict | None = None,
    ) -> tuple[bool, int | None, int | None, float, str]:
        observations = self._collect_bridge_observations(
            scene,
            minimap_gray_raw,
            found,
            center_x,
            center_y,
            match_quality,
            source,
            frame_cfg=frame_cfg,
        )

        motion_hint = (0.0, 0.0)
        use_arrow_fallback = False
        
        if lk_result is not None:
            dx_map, dy_map, lk_conf = lk_result
            if lk_conf >= self._lk.min_conf:
                motion_hint = (float(dx_map), float(dy_map))
            else:
                use_arrow_fallback = True
        else:
            use_arrow_fallback = True
            
        # 巧妙复用已有的低纹理桥接器 (TemporalBridge)：
        # 如果 LK 光流在纯色区宣告失效（通常因为没提取到角点），我们把方向系统的推算结果
        # 直接作为先验（motion_hint）喂给桥接器，这样桥接器内部的假设轨迹就会顺着箭头滑行！
        if use_arrow_fallback and scene in ('ocean', 'low_texture') and not self._arrow_dir._is_stopped:
            rad = math.radians(self._arrow_dir._last_angle)
            speed = 4.0  # 游戏默认推算是常量步幅
            motion_hint = (float(speed * math.sin(rad)), float(-speed * math.cos(rad)))

        bridged = self._bridge.step(
            scene=scene,
            observations=observations,
            motion_hint=motion_hint,
            map_width=self.map_width,
            map_height=self.map_height,
        )

        if bridged is None:
            return found, center_x, center_y, match_quality, source

        should_override = (not found) or (bridged.quality >= match_quality)
        if not should_override:
            return found, center_x, center_y, match_quality, source

        return True, bridged.x, bridged.y, float(bridged.quality), bridged.source

    def _build_frame_runtime_config(self) -> dict:
        """构建当前帧运行时配置快照，减少热路径重复 getattr(config, ...)。"""
        wd_limit = int(getattr(config, 'WATCHDOG_SUSPECT_LIMIT', 3))
        return {
            'low_texture_like_std_threshold': float(getattr(config, 'LOW_TEXTURE_LIKE_STD_THRESHOLD', 40.0)),
            'phase_correlate_enabled': bool(getattr(config, 'PHASE_CORRELATE_ENABLED', True)),
            'phase_correlate_min_response': float(getattr(config, 'PHASE_CORRELATE_MIN_RESPONSE', 0.05)),
            'phase_correlate_crop_ratio': float(getattr(config, 'PHASE_CORRELATE_CROP_RATIO', 2.0)),
            'sat_match_ratio': float(getattr(config, 'SIFT_SAT_MATCH_RATIO', 0.90)),
            'sat_min_match': int(getattr(config, 'SIFT_SAT_MIN_MATCH', 3)),
            'sat_ecc_min_cc': float(getattr(config, 'SIFT_SAT_ECC_MIN_CC', 0.28)),
            'watchdog_suspect_limit': wd_limit,
            'watchdog_hash_mismatch_limit': int(getattr(config, 'WATCHDOG_HASH_MISMATCH_LIMIT', wd_limit * 2)),
            'watchdog_mad_threshold': float(getattr(config, 'WATCHDOG_MAD_THRESHOLD', 24.0)),
            'watchdog_static_limit': int(getattr(config, 'WATCHDOG_STATIC_LIMIT', 8)),
            'watchdog_hash_min_quality': float(getattr(config, 'WATCHDOG_HASH_MIN_MATCH_QUALITY', 0.55)),
            'watchdog_hash_min_matches': int(getattr(config, 'WATCHDOG_HASH_MIN_MATCH_COUNT', 6)),
            'watchdog_hash_check_radius': int(getattr(config, 'WATCHDOG_HASH_CHECK_RADIUS', 320)),
            'watchdog_hash_hamming_margin': int(getattr(config, 'WATCHDOG_HASH_HAMMING_MARGIN', 6)),
            'watchdog_inertial_min_sift_fails': int(getattr(config, 'WATCHDOG_INERTIAL_MIN_SIFT_FAILS', 3)),
            'watchdog_inertial_mad_threshold': float(getattr(config, 'WATCHDOG_INERTIAL_MAD_THRESHOLD', 22.0)),
            'max_lost_frames': int(getattr(config, 'MAX_LOST_FRAMES', 50)),
            'bridge_hash_candidates': int(getattr(config, 'LOW_TEXTURE_BRIDGE_HASH_CANDIDATES', 4)),
            'bridge_hash_radius': int(getattr(config, 'LOW_TEXTURE_BRIDGE_HASH_RADIUS', 900)),
            # 色彩智能增强开关
            'icon_mask_enabled': bool(getattr(config, 'ICON_MASK_ENABLED', True)),
            'scene_color_enabled': bool(getattr(config, 'SCENE_COLOR_ENABLED', True)),
            'scene_boosted_gray_enabled': bool(getattr(config, 'SCENE_BOOSTED_GRAY_ENABLED', True)),
        }

    # ---- 核心匹配 ----
    def match(self, minimap_bgr):
        with self._lock:
            return self._match_impl(minimap_bgr)

    def _match_impl(self, minimap_bgr):
        frame_cfg = self._build_frame_runtime_config()
        found = False
        center_x, center_y = None, None
        arrow_angle = None
        is_inertial = False
        match_count = 0
        match_quality = 0.0
        source = ''

        # 护眼/防蓝光软件叠加层色温补偿（硬件 Night Light 不影响截图，无需处理）
        minimap_bgr = correct_color_temperature(minimap_bgr)
        minimap_gray_raw = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        texture_std = float(np.std(minimap_gray_raw))
        scene = classify_scene(texture_std)

        # ── 色彩场景细化（仅非城市场景，~0.3ms）──────────────────────────────
        # 纯纹理分类对草原/雪地/海岸辨别力弱；颜色分析可区分绿草 vs 雪地 vs 蓝海，
        # 并将结果回映到现有 scene 标签，使后续低纹理兜底链路正确激活。
        scene_detail = scene
        if frame_cfg['scene_color_enabled'] and scene != 'urban':
            scene_detail = classify_scene_by_color(minimap_bgr, scene)
            # 草原和雪地按低纹理路由，以启用 phase_correlate / SAT / bridge 等兜底
            if scene_detail in ('grassland', 'snow') and scene in ('mixed', 'low_texture'):
                scene = 'low_texture'

        low_texture_like = (
            scene in ('ocean', 'low_texture')
            or texture_std < frame_cfg['low_texture_like_std_threshold']
        )

        # ── 场景优化灰度图（仅 ocean/grassland/snow，~0.2ms）─────────────────
        # 按主色调调整通道权重，使小地图灰度图在该场景下对比度/梯度更丰富，
        # SIFT 可提取更多高质量特征。minimap_gray_raw 保持标准灰度（供哈希用）。
        _sift_input_gray = minimap_gray_raw
        if frame_cfg['scene_boosted_gray_enabled'] and scene_detail in ('ocean', 'grassland', 'snow'):
            _boosted = make_scene_boosted_gray(minimap_bgr, scene_detail)
            if _boosted is not None:
                _sift_input_gray = _boosted

        # 自适应 CLAHE + 纹理增强（合并为单次调用，消除双 CLAHE）
        minimap_gray = enhance_minimap(_sift_input_gray, texture_std,
                                       self._clahe_normal, self._clahe_low,
                                       self._low_texture_thresh)

        # ── 任务图标排除遮罩（~1ms，高饱和色块检测）────────────────────────
        # 游戏任务图标（！/？/其他标记）为高饱和度彩色圆点，在低纹理区会产生
        # 大量"假特征"污染全局 SIFT 匹配池；检测并在特征提取时排除。
        _icon_excl: np.ndarray | None = None
        if frame_cfg['icon_mask_enabled']:
            _icon_excl = detect_ui_icon_exclusion_mask(minimap_bgr)

        self._lk.frame_num += 1
        run_sift = self._lk.should_run_sift()

        if self._watchdog_cooldown > 0:
            self._watchdog_cooldown -= 1

        if self._relocalize_cd > 0:
            self._relocalize_cd -= 1

        _tp_far_candidate = None  # 传送候选：全局 SIFT 匹配成功但超跳变阈值

        # ======================================================
        # 第一层：LK 光流快速跟踪（每帧 ~2ms，跳过部分 SIFT 调用）
        # ======================================================
        lk_result = self._lk.track(minimap_gray, self.last_x)

        # 看门狗：LK 每帧位移累积（不论当帧是否采用 LK 结果，都参与累积）
        if lk_result is not None:
            _wdx, _wdy, _wconf = lk_result
            if _wconf >= self._lk.min_conf:
                self._watchdog_accum_dx += _wdx
                self._watchdog_accum_dy += _wdy

        if lk_result is not None and not run_sift:
            dx_map, dy_map, lk_conf = lk_result
            if lk_conf >= self._lk.min_conf:
                tx = int(round(self.last_x + dx_map))
                ty = int(round(self.last_y + dy_map))
                if (0 <= tx < self.map_width and 0 <= ty < self.map_height
                        and abs(dx_map) + abs(dy_map) < self.JUMP_THRESHOLD):
                    found, center_x, center_y = True, tx, ty
                    match_quality = lk_conf * 0.7
                    source = 'LK'

        # ======================================================
        # 第二层：SIFT 精确匹配（周期性运行 + LK 失败时触发）
        # ======================================================
        if not found:
            # 提取特征
            kp_mini, des_mini = extract_minimap_features(minimap_gray, self.sift, self._mask_cache,
                                                           texture_std=texture_std,
                                                           exclude_mask=_icon_excl)

            # 超低/低纹理 + 特征过少 → 更激进的预处理重试
            # 方案A: 扩展到 low_texture（海岸线），不只限于 ocean
            if low_texture_like and (des_mini is None or len(kp_mini) < 15):
                enhanced = enhance_minimap(_sift_input_gray, 5,
                                           self._clahe_normal, self._clahe_low,
                                           self._low_texture_thresh)
                kp_retry, des_retry = extract_minimap_features(enhanced, self.sift, self._mask_cache,
                                                                texture_std=texture_std,
                                                                exclude_mask=_icon_excl)
                if des_retry is not None and len(kp_retry) > (len(kp_mini) if kp_mini is not None else 0):
                    kp_mini, des_mini = kp_retry, des_retry

            # 动态匹配参数：按场景分类路由
            if scene == 'ocean':
                eff_match_ratio = 0.86
                eff_min_match = 4  # 稍微提高一点以防御UI干扰，但保持灵活
            elif scene == 'low_texture' or (scene == 'mixed' and low_texture_like):
                eff_match_ratio = 0.84
                eff_min_match = 4
            else:
                eff_match_ratio = config.SIFT_MATCH_RATIO  # 0.82
                eff_min_match = config.SIFT_MIN_MATCH_COUNT

            if des_mini is not None:
                revalidation_blocked = False

                # ---- 冻结恢复首帧：先尝试旧坐标/局部索引的轻量恢复，再决定是否进入常规重定位 ----
                resume_match = self._consume_freeze_resume_match(
                    kp_mini, des_mini, minimap_gray,
                    eff_match_ratio, eff_min_match)
                if resume_match is not None:
                    tx, ty, quality, avg_scale, source, match_count = resume_match
                    found, center_x, center_y, match_quality = True, tx, ty, quality
                    self._local_success_streak = 0
                    self._force_global_revalidate = True
                    self._force_global_revalidate_frame = self._lk.frame_num
                    if avg_scale is not None:
                        self._last_sift_scale = avg_scale
                        self._lk.map_scale = avg_scale

                # ---- SIFT 第一轮：局部 → 全局回退 ----
                for search_round in range(2):
                    if found:
                        break
                    if search_round == 0 and self.using_local:
                        current_kp, current_flann = self.kp_local, self.flann_local
                    elif search_round == 0:
                        current_kp, current_flann = self.kp_big_all, self.flann_global
                    else:
                        if not self.using_local:
                            break
                        current_kp, current_flann = self.kp_big_all, self.flann_global

                    result = sift_match_region(
                        kp_mini, des_mini, minimap_gray.shape,
                        current_kp, current_flann, eff_match_ratio, eff_min_match,
                        self.map_width, self.map_height)

                    if result is not None:
                        tx, ty, inlier_count, quality, avg_scale = result
                        candidate_source = 'SIFT_LOCAL' if (search_round == 0 and self.using_local) else 'SIFT_GLOBAL'
                        if self.last_x is not None:
                            max_jump = self.JUMP_THRESHOLD if search_round == 0 else self.JUMP_THRESHOLD * 2
                            if abs(tx - self.last_x) + abs(ty - self.last_y) >= max_jump:
                                # 超跳变阈值：保留为传送候选（取质量最高的）
                                if quality >= 0.3 and (_tp_far_candidate is None or quality > _tp_far_candidate[2]):
                                    _tp_far_candidate = (tx, ty, quality)
                                continue
                        if candidate_source == 'SIFT_LOCAL':
                            local_match = (tx, ty, inlier_count, quality, avg_scale, candidate_source)
                            local_match, _tp_far_candidate, revalidation_blocked = self._maybe_revalidate_local_match(
                                kp_mini, des_mini, minimap_gray.shape,
                                eff_match_ratio, eff_min_match, local_match, _tp_far_candidate)
                            if local_match is None:
                                self._local_success_streak = 0
                                break
                            tx, ty, inlier_count, quality, avg_scale, candidate_source = local_match
                        else:
                            self._local_success_streak = 0
                            self._force_global_revalidate = False
                        match_count = inlier_count
                        found, center_x, center_y, match_quality = True, tx, ty, quality
                        source = candidate_source
                        # SIFT 成功 → 同步 scale
                        self._last_sift_scale = avg_scale
                        self._lk.map_scale = avg_scale
                        break

                # ---- SIFT 第二轮：附近搜索（使用 FLANN 缓存）----
                if not found and self.last_x is not None and not revalidation_blocked:
                    nr = self.NEARBY_SEARCH_RADIUS
                    nearby_match = self._try_nearby_match_stack(
                        kp_mini, des_mini, minimap_gray,
                        self.last_x, self.last_y, nr,
                        allow_ecc=True,
                        source_prefix='')
                    if nearby_match is not None:
                        tx, ty, quality, avg_scale, source, match_count = nearby_match
                        found, center_x, center_y = True, tx, ty
                        match_quality = quality
                        if avg_scale is not None:
                            self._last_sift_scale = avg_scale
                            self._lk.map_scale = avg_scale
            # ---- phaseCorrelate：频域互相关（有位置提示时的低纹理/海洋兜底）----
            # 在 SIFT+ECC 全失败后触发；FFT 全局最优，不受重复纹理/描述子歧义影响
            # 约 1-2ms；需要 last_x/last_y 作为搜索中心
            if (not found and low_texture_like
                    and self.last_x is not None
                    and frame_cfg['phase_correlate_enabled']):
                _pc_min = frame_cfg['phase_correlate_min_response']
                _pc_ratio = frame_cfg['phase_correlate_crop_ratio']
                # 1. 灰度通道
                _pc = phase_correlate(
                    minimap_gray, self._logic_map_gray_clahe,
                    self.last_x, self.last_y,
                    self._last_sift_scale, self.map_width, self.map_height,
                    self.JUMP_THRESHOLD * 2, _pc_min, _pc_ratio)
                if _pc is not None:
                    found, center_x, center_y = True, _pc[0], _pc[1]
                    match_quality = 0.35
                    source = 'PHASE_CORRELATE'
                # 2. S 通道（海洋场景辨别力更强）
                if not found and self._logic_map_sat_clahe is not None \
                        and self._clahe_sat is not None:
                    _hsv_pc = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
                    _sat_mini_pc = self._clahe_sat.apply(_hsv_pc[:, :, 1])
                    _pc_s = phase_correlate(
                        _sat_mini_pc, self._logic_map_sat_clahe,
                        self.last_x, self.last_y,
                        self._last_sift_scale, self.map_width, self.map_height,
                        self.JUMP_THRESHOLD * 2, _pc_min, _pc_ratio)
                    if _pc_s is not None:
                        found, center_x, center_y = True, _pc_s[0], _pc_s[1]
                        match_quality = 0.30
                        source = 'PHASE_CORRELATE_SAT'

            # ---- 方案B: S通道辅助匹配（低纹理/海洋场景最终兜底）----
            # 灰度 SIFT 全链路失败后才触发，运行时开销 ~3-5ms（含 cvtColor + SIFT）
            if not found and low_texture_like and self._clahe_sat is not None:
                _hsv_mini = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
                _sat_mini = self._clahe_sat.apply(_hsv_mini[:, :, 1])
                # inner_ratio 固定用 hard 值，排除玩家箭头对 S 通道的干扰
                kp_sat_mini, des_sat_mini = extract_minimap_features(
                    _sat_mini, self.sift, self._mask_cache,
                    texture_std=None, inner_ratio=0.18, exclude_mask=_icon_excl)
                if des_sat_mini is not None:
                    _sat_ratio = frame_cfg['sat_match_ratio']
                    _sat_min = frame_cfg['sat_min_match']
                    _res_sat = sift_match_region(
                        kp_sat_mini, des_sat_mini, _sat_mini.shape,
                        self.kp_big_sat, self.flann_global_sat,
                        _sat_ratio, _sat_min,
                        self.map_width, self.map_height)
                    if _res_sat is not None:
                        tx, ty, inlier_count, quality, avg_scale = _res_sat
                        _jmp_ok = (self.last_x is None
                                   or abs(tx - self.last_x) + abs(ty - self.last_y) < self.JUMP_THRESHOLD)
                        if _jmp_ok:
                            found, center_x, center_y = True, tx, ty
                            match_quality = quality * 0.85  # 轻微折扣：S 通道描述子空间与灰度不同
                            match_count = inlier_count
                            source = 'SIFT_SAT'
                            if avg_scale is not None:
                                self._last_sift_scale = avg_scale
                                self._lk.map_scale = avg_scale

                # ---- S通道 ECC 兜底（有位置提示、SAT SIFT 依然失败时）----
                # _sat_mini 已在上方计算，直接复用，无额外 cvtColor 开销
                if not found and self.last_x is not None \
                        and self._logic_map_sat_clahe is not None and self._ecc_enabled:
                    _sat_ecc_min = frame_cfg['sat_ecc_min_cc']
                    _ecc_sat = ecc_align(
                        _sat_mini, self._logic_map_sat_clahe,
                        self.last_x, self.last_y,
                        self._last_sift_scale, self.map_width, self.map_height,
                        self.JUMP_THRESHOLD, _sat_ecc_min)
                    if _ecc_sat is not None:
                        found, center_x, center_y = True, _ecc_sat[0], _ecc_sat[1]
        # =========================================================
        # 🐶 看门狗异常检测系统 (Watchdog) - 帧差法重用重设版本
        # 职责：检测“游戏画面在滚动，但识别坐标死死卡住”的明确死锁状态
        # 特性：抛弃脆弱且易漏的 LK 位移累加，采用绝对强硬的中心画面 MAD (像素差) 确认法
        # =========================================================
        _wd_is_sift_source = source in ('SIFT_LOCAL', 'SIFT_GLOBAL', 'SIFT_GLOBAL_REVALIDATED')
        _wd_limit = frame_cfg['watchdog_suspect_limit']
        _hash_limit = frame_cfg['watchdog_hash_mismatch_limit']

        if run_sift:
            if self._watchdog_cooldown > 0:
                # 冷却中，仅保持同步基准
                if found and not is_inertial and _wd_is_sift_source:
                    self._watchdog_last_sift_x = center_x
                    self._watchdog_last_sift_y = center_y
                    self._watchdog_anchor_gray = minimap_gray_raw.copy()
                self._watchdog_suspect_streak = 0
                self._watchdog_hash_mismatch_streak = 0
                self._watchdog_static_streak = 0
            
            elif found and not is_inertial and _wd_is_sift_source:
                if self._watchdog_last_sift_x is None or not hasattr(self, '_watchdog_anchor_gray'):
                    # 舒适初始化第一帧基准
                    self._watchdog_last_sift_x = center_x
                    self._watchdog_last_sift_y = center_y
                    self._watchdog_anchor_gray = minimap_gray_raw.copy()
                else:
                    _sift_moved = abs(center_x - self._watchdog_last_sift_x) + abs(center_y - self._watchdog_last_sift_y)
                    
                    if _sift_moved < 15:
                        self._watchdog_suspect_streak += 1
                        
                        # SIFT 连续几帧卡在同一个极小的坐标内
                        if self._watchdog_suspect_streak >= _wd_limit:
                            # 终极拷问：既然坐标没变，那游戏小地图内的画面到底动没动？
                            # 提取中心 160x160 区域计算像素平差 (MAD)
                            h_mm, w_mm = minimap_gray_raw.shape[:2]
                            ch, cw = h_mm // 2, w_mm // 2
                            p = 80
                            
                            # 防御性保护：确保不会越界；若前后帧圆提取尺寸略有抖动，则退化为比较公共中心区域
                            if ch - p >= 0 and cw - p >= 0 and ch + p <= h_mm and cw + p <= w_mm:
                                roi_now = minimap_gray_raw[ch-p:ch+p, cw-p:cw+p]
                                old_gray = self._watchdog_anchor_gray
                                oh, ow = old_gray.shape[:2]
                                och, ocw = oh // 2, ow // 2
                                if och - p >= 0 and ocw - p >= 0 and och + p <= oh and ocw + p <= ow:
                                    roi_old = old_gray[och-p:och+p, ocw-p:ocw+p]
                                else:
                                    roi_old = old_gray

                                common_h = min(roi_now.shape[0], roi_old.shape[0])
                                common_w = min(roi_now.shape[1], roi_old.shape[1])
                                if common_h >= 40 and common_w >= 40:
                                    y1_now = (roi_now.shape[0] - common_h) // 2
                                    x1_now = (roi_now.shape[1] - common_w) // 2
                                    y1_old = (roi_old.shape[0] - common_h) // 2
                                    x1_old = (roi_old.shape[1] - common_w) // 2
                                    roi_now_cmp = roi_now[y1_now:y1_now + common_h, x1_now:x1_now + common_w]
                                    roi_old_cmp = roi_old[y1_old:y1_old + common_h, x1_old:x1_old + common_w]
                                    mad = float(np.mean(cv2.absdiff(roi_now_cmp, roi_old_cmp)))
                                else:
                                    mad = 0.0
                            else:
                                mad = 0.0 # 异常小地图时不触发死锁

                            _wd_mad_thresh = frame_cfg['watchdog_mad_threshold']
                            if mad > _wd_mad_thresh:
                                # 实锤死锁：画面变化极大（玩家在动）但 SIFT 坐标卡死
                                print(f"[看门狗-死锁确诊] 画面剧烈波动(MAD={mad:.1f}) 但 SIFT 坐标卡死在({center_x},{center_y})!")
                                self._trigger_watchdog_unlock('基于全图真差的死锁确诊强制解绑')
                                found = False
                                self._watchdog_suspect_streak = 0
                            else:
                                # 玩家真正挂机静止：MAD 极低，证明此时不需要移动，SIFT 卡着是对的
                                self._watchdog_static_streak += 1
                                _wd_static_limit = frame_cfg['watchdog_static_limit']
                                if self._watchdog_static_streak >= _wd_static_limit:
                                    self._force_global_revalidate = True
                                    self._force_global_revalidate_frame = self._lk.frame_num
                                    self._watchdog_static_streak = 0
                                    # 定期更新基准，防止极缓慢的环境变化积累成假性变化
                                    self._watchdog_anchor_gray = minimap_gray_raw.copy()
                                    self._watchdog_last_sift_x = center_x
                                    self._watchdog_last_sift_y = center_y
                    else:
                        # 识别结果产生了实际位移，打破死锁循环，重置锚点
                        self._watchdog_suspect_streak = 0
                        self._watchdog_static_streak = 0
                        self._watchdog_last_sift_x = center_x
                        self._watchdog_last_sift_y = center_y
                        self._watchdog_anchor_gray = minimap_gray_raw.copy()

                    # =========================================================
                    # 当处于移动时再检查哈希。防止静止且弹出全屏 UI 导致哈希崩坏触发误解。
                    # =========================================================
                    _hash_min_quality = frame_cfg['watchdog_hash_min_quality']
                    _hash_min_matches = frame_cfg['watchdog_hash_min_matches']
                    
                    if match_quality >= _hash_min_quality and match_count >= _hash_min_matches:
                        if _sift_moved >= 15 or self._watchdog_suspect_streak == 0:
                            _hash_radius = frame_cfg['watchdog_hash_check_radius']
                            _hash_margin = frame_cfg['watchdog_hash_hamming_margin']
                            consistent, best_dist, _ = self._hash_index.check_consistency_details(
                                minimap_gray_raw, center_x, center_y, check_radius=_hash_radius)
                            
                            if consistent:
                                self._watchdog_hash_mismatch_streak = 0
                            else:
                                _thresh = int(getattr(self._hash_index, '_hamming_thresh', 12))
                                _effective_dist = best_dist if best_dist is not None else (_thresh + _hash_margin + 1)
                                if _effective_dist > _thresh + _hash_margin:
                                    self._watchdog_hash_mismatch_streak += 1
                                    if self._watchdog_hash_mismatch_streak >= _hash_limit:
                                        print(f'[看门狗-哈希幻觉] 视觉不一致确诊! dist={_effective_dist}, streak={self._watchdog_hash_mismatch_streak}')
                                        self._trigger_watchdog_unlock('移动中检测到哈希极度偏离产生幻觉')
                                        found = False
                                else:
                                    self._watchdog_hash_mismatch_streak = 0
                        else:
                            self._watchdog_hash_mismatch_streak = 0

            # 每个周期清理无用的 LK 历史，防止副作用累积
            self._watchdog_accum_dx = 0.0
            self._watchdog_accum_dy = 0.0

        # =========================================================
        # 粗定位回退...
        _semi_lost = (self.last_x is not None and self.lost_frames >= 8
                  and low_texture_like)
        _fully_lost = (self.last_x is None)
        if not found and (_fully_lost or _semi_lost) and self._relocalize_cd == 0:
            _hash_radius = 0 if _fully_lost else min(2400, 1200 + max(0, self.lost_frames - 8) * 60)
            hash_candidates = self._hash_index.locate(
                minimap_gray_raw,
                last_x=self.last_x, last_y=self.last_y,
                radius=_hash_radius, max_results=3)
            hash_hit = None
            for hx, hy, hdist in hash_candidates:
                # semi_lost 模式下拒绝超跳变阈值的结果
                if self.last_x is not None and abs(hx - self.last_x) + abs(hy - self.last_y) >= self.JUMP_THRESHOLD:
                    continue
                hash_hit = (hx, hy, hdist)
                break
            if hash_hit is not None:
                found, center_x, center_y = True, hash_hit[0], hash_hit[1]
                match_quality = max(0.15, 0.35 - hash_hit[2] * 0.02)
                source = 'HASH_INDEX'
                self._relocalize_cd = 40
            else:
                self._relocalize_cd = 15

        # 低纹理时序桥接：融合弱观测 + 运动先验，避免逐帧贪心导致错误吸附。
        found, center_x, center_y, match_quality, source = self._apply_low_texture_bridge(
            'low_texture' if low_texture_like else scene, minimap_gray_raw,
            found, center_x, center_y,
            match_quality, source,
            lk_result,
            frame_cfg=frame_cfg,
        )

        # 更新 LK 上一帧（不论用哪层跟踪，都更新灰度图）
        self._lk.prev_gray = minimap_gray.copy()
        if found and not is_inertial:
            self._lk.refresh_tracking_points(minimap_gray)
        elif self._lk.prev_pts is not None and len(self._lk.prev_pts) < 15:
            # LK 点快耗尽时主动补点，避免下一帧直接失明。
            self._lk.refresh_tracking_points(minimap_gray)

        # ---- 状态更新 ----
        if found and not is_inertial:
            self.last_x, self.last_y = center_x, center_y
            self.lost_frames = 0
            self.local_fail_count = 0
            self._inertial_entry_gray = None      # 已恢复 FOUND，清除惯性基准
            self._inertial_failed_sift_count = 0
            self._switch_to_local(center_x, center_y)
            if source != 'SIFT_LOCAL':
                self._local_success_streak = 0
            if source in ('ECC', 'HASH_INDEX', 'SIFT_NEARBY'):
                self._force_global_revalidate = True
                self._force_global_revalidate_frame = self._lk.frame_num
            elif source in ('SIFT_GLOBAL', 'SIFT_GLOBAL_REVALIDATED'):
                self._force_global_revalidate = False
        else:
            self._local_success_streak = 0
            self.lost_frames += 1
            if self.using_local:
                self.local_fail_count += 1
                if self.local_fail_count >= self.LOCAL_FAIL_LIMIT:
                    self._switch_to_global()
                    self._nearby_flann_cache.clear()  # 切全局时清缓存
            if self.last_x is not None and self.lost_frames <= frame_cfg['max_lost_frames']:
                # ── INERTIAL 坐标锁定检测 ──────────────────────────────────────────────
                # 非低纹理场景：SIFT 若连续失败且画面已明显变化，则提前退出 INERTIAL
                # 防止"坐标卡死但系统以为还在原地"的死坐标锁定问题
                if self.lost_frames == 1:
                    # 首次进入 INERTIAL：记录入场基准帧，用于后续帧差检测
                    self._inertial_entry_gray = minimap_gray_raw.copy()
                    self._inertial_failed_sift_count = 0
                elif run_sift and not low_texture_like:
                    # 非低纹理场景每次 SIFT 运行时累计失败次数，并检查画面是否变化
                    self._inertial_failed_sift_count += 1
                    _iner_min = frame_cfg['watchdog_inertial_min_sift_fails']
                    _iner_thresh = frame_cfg['watchdog_inertial_mad_threshold']
                    if self._inertial_failed_sift_count >= _iner_min and self._inertial_entry_gray is not None:
                        _h_i, _w_i = minimap_gray_raw.shape[:2]
                        _ch_i, _cw_i = _h_i // 2, _w_i // 2
                        _p_i = 80
                        if _ch_i - _p_i >= 0 and _cw_i - _p_i >= 0 and _ch_i + _p_i <= _h_i and _cw_i + _p_i <= _w_i:
                            _roi_now_i = minimap_gray_raw[_ch_i-_p_i:_ch_i+_p_i, _cw_i-_p_i:_cw_i+_p_i]
                            _eh_i, _ew_i = self._inertial_entry_gray.shape[:2]
                            _ech_i, _ecw_i = _eh_i // 2, _ew_i // 2
                            if _ech_i - _p_i >= 0 and _ecw_i - _p_i >= 0 and _ech_i + _p_i <= _eh_i and _ecw_i + _p_i <= _ew_i:
                                _roi_old_i = self._inertial_entry_gray[_ech_i-_p_i:_ech_i+_p_i, _ecw_i-_p_i:_ecw_i+_p_i]
                                _ch2_i = min(_roi_now_i.shape[0], _roi_old_i.shape[0])
                                _cw2_i = min(_roi_now_i.shape[1], _roi_old_i.shape[1])
                                if _ch2_i >= 40 and _cw2_i >= 40:
                                    _y1n_i = (_roi_now_i.shape[0] - _ch2_i) // 2
                                    _x1n_i = (_roi_now_i.shape[1] - _cw2_i) // 2
                                    _y1o_i = (_roi_old_i.shape[0] - _ch2_i) // 2
                                    _x1o_i = (_roi_old_i.shape[1] - _cw2_i) // 2
                                    _iner_mad = float(np.mean(cv2.absdiff(
                                        _roi_now_i[_y1n_i:_y1n_i+_ch2_i, _x1n_i:_x1n_i+_cw2_i],
                                        _roi_old_i[_y1o_i:_y1o_i+_ch2_i, _x1o_i:_x1o_i+_cw2_i]
                                    )))
                                    if _iner_mad > _iner_thresh:
                                        print(f"[锁定-INERTIAL漂移] SIFT连续失败{self._inertial_failed_sift_count}次, "
                                              f"画面MAD={_iner_mad:.1f}>{_iner_thresh}, 提前退出INERTIAL")
                                        self.last_x = self.last_y = None
                                        self._lk.prev_gray = None
                                        self._lk.prev_pts = None
                                        self._bridge.reset()
                                        if self.using_local:
                                            self._switch_to_global()
                                            self._nearby_flann_cache.clear()
                                        self._inertial_entry_gray = None
                                        self._inertial_failed_sift_count = 0
                # ────────────────────────────────────────────────────────────────────

                if self.last_x is not None:  # 可能被坐标锁定检测提前重置
                    found = True
                    is_inertial = True

                    # 终极冷启动推演网：如果玩家上线就在海面，Bridge 因为没有任何初始信标而无法启动，
                    # 此时全链路崩溃。我们直接在此进行终极接管，更新 last_x，确保系统能在盲态下起步。
                    if low_texture_like and not self._arrow_dir._is_stopped:
                        rad = math.radians(self._arrow_dir._last_angle)
                        speed = 4.0
                        center_x = int(round(self.last_x + speed * math.sin(rad)))
                        center_y = int(round(self.last_y - speed * math.cos(rad)))
                        self.last_x = center_x
                        self.last_y = center_y
                        self.lost_frames = max(1, self.lost_frames - 1)
                    else:
                        center_x, center_y = self.last_x, self.last_y
            elif self.lost_frames > frame_cfg['max_lost_frames']:
                self.last_x = self.last_y = None
                self._lk.prev_gray = None
                self._lk.prev_pts = None
                self._bridge.reset()
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

        _wd_triggered = self._watchdog_triggered
        self._watchdog_triggered = False

        return self._make_result(
            found, center_x, center_y,
            arrow_angle=arrow_angle, arrow_stopped=arrow_stopped,
            is_inertial=is_inertial, match_count=match_count,
            match_quality=match_quality,
            _tp_far_candidate=_tp_far_candidate,
            source=source,
            _watchdog_triggered=_wd_triggered)

