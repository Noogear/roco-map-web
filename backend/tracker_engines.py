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
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from backend import config

from backend.core.enhance import (make_clahe_pair, adaptive_clahe_map,
                                  enhance_minimap, correct_color_temperature)
from backend.core.features import (CircularMaskCache, create_flann,
                                   extract_map_features_tiled, extract_minimap_features,
                                   sift_match_region)
from backend.core.flow import LKTracker
from backend.core.ecc import ecc_align
from backend.core.hash_index import MapHashIndex
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

        # per-session: CLAHE 有内部状态，不能共享
        self._clahe_normal, self._clahe_low = make_clahe_pair()
        self._low_texture_thresh = getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30)
        self._mask_cache = CircularMaskCache()

        # 上一帧箭头角度
        self._last_arrow_angle = 0.0
        self._last_arrow_stopped = True
        # 箭头方向系统（纯坐标驱动）
        self._arrow_dir = ArrowDirectionSystem()

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

        # 附近搜索 FLANN 缓存（OrderedDict LRU，避免每帧重建）
        self._nearby_flann_cache: OrderedDict = OrderedDict()
        self._nearby_flann_bucket = 200  # 网格粒度（地图像素）

        # 惯性导航状态
        self.last_x = None
        self.last_y = None
        self.lost_frames = 0
        self._watchdog_triggered = False

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
        """复用的附近搜索链：SIFT nearby → 可选 ECC。"""
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
            arrow_dir_stop_streak=arrow_dir._stop_streak,
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
            arrow_dir._stop_streak = snapshot.arrow_dir_stop_streak

    def sync_external_position(self, x: int, y: int) -> None:
        """让外部确认后的坐标立即成为引擎新基点。"""
        with self._lock:
            self.last_x = x
            self.last_y = y
            self.lost_frames = 0
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

    # ---- 核心匹配 ----
    def match(self, minimap_bgr):
        with self._lock:
            return self._match_impl(minimap_bgr)

    def _match_impl(self, minimap_bgr):
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

        # 自适应 CLAHE + 纹理增强（合并为单次调用，消除双 CLAHE）
        minimap_gray = enhance_minimap(minimap_gray_raw, texture_std,
                                       self._clahe_normal, self._clahe_low,
                                       self._low_texture_thresh)

        self._lk.frame_num += 1
        run_sift = self._lk.should_run_sift()

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
                                                           texture_std=texture_std)

            # 超低纹理 + 特征过少 → 更激进的预处理重试
            if scene == 'ocean' and (des_mini is None or len(kp_mini) < 15):
                enhanced = enhance_minimap(minimap_gray_raw, 5,
                                           self._clahe_normal, self._clahe_low,
                                           self._low_texture_thresh)
                kp_retry, des_retry = extract_minimap_features(enhanced, self.sift, self._mask_cache,
                                                                texture_std=texture_std)
                if des_retry is not None and len(kp_retry) > (len(kp_mini) if kp_mini is not None else 0):
                    kp_mini, des_mini = kp_retry, des_retry

            # 动态匹配参数：按场景分类路由
            if scene == 'ocean':
                eff_match_ratio = 0.88
                eff_min_match = 3
            elif scene == 'low_texture':
                eff_match_ratio = 0.86
                eff_min_match = 3
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

        # 看门狗：用 LK 累积位移检测“SIFT 卡在错位置但看起来还活着”的局部死锁。
        _wd_min_move = getattr(config, 'WATCHDOG_LK_MIN_MOVE', 60)
        _wd_limit = getattr(config, 'WATCHDOG_SUSPECT_LIMIT', 3)
        if run_sift:
            _accum_dist = abs(self._watchdog_accum_dx) + abs(self._watchdog_accum_dy)
            _wd_sift_src = source in ('SIFT_LOCAL', 'SIFT_GLOBAL', 'SIFT_GLOBAL_REVALIDATED')
            if found and not is_inertial and _wd_sift_src and self._watchdog_last_sift_x is not None:
                if _accum_dist > _wd_min_move:
                    _sift_moved = (abs(center_x - self._watchdog_last_sift_x)
                                   + abs(center_y - self._watchdog_last_sift_y))
                    if _sift_moved < _wd_min_move * 0.35:
                        # 玩家移动但 SIFT 原地不动 → 可疑
                        self._watchdog_suspect_streak += 1
                        self._watchdog_consecutive_ok = 0
                        self._watchdog_static_streak = 0
                        print(f"[看门狗] 可疑帧 #{self._watchdog_suspect_streak}: "
                              f"LK累积={_accum_dist:.0f}px, SIFT位移={_sift_moved:.0f}px")
                        if self._watchdog_suspect_streak >= _wd_limit:
                            print(f"[看门狗] 死锁解除! 连续{self._watchdog_suspect_streak}次不一致 "
                                  f"→ 清空基点+切全局重定位")
                            self._switch_to_global()       # 内部同时清除看门狗累积/基准点
                            self._nearby_flann_cache.clear()
                            self._lk.reset()
                            self._force_global_revalidate = True
                            self._force_global_revalidate_frame = self._lk.frame_num
                            self._watchdog_suspect_streak = 0
                            self._watchdog_triggered = True  # 通知上层清空 pos_history
                            self.last_x = None             # 摆脱错误基点，下帧全局无阈值接受
                            self.last_y = None
                            self.lost_frames = 0
                            found = False
                    else:
                        # SIFT 位移正常；需连续多帧正常才清除可疑计数（防 ABAB 交替绕过）
                        self._watchdog_consecutive_ok += 1
                        self._watchdog_static_streak = 0
                        if self._watchdog_consecutive_ok >= 3:
                            self._watchdog_suspect_streak = 0
                        self._watchdog_last_sift_x = center_x
                        self._watchdog_last_sift_y = center_y
                else:
                    # LK 累积不足（玩家静止/慢走），更新基准点并计数静止帧
                    self._watchdog_last_sift_x = center_x
                    self._watchdog_last_sift_y = center_y
                    self._watchdog_static_streak += 1
                    _wd_static_limit = getattr(config, 'WATCHDOG_STATIC_LIMIT', 8)
                    if self._watchdog_static_streak >= _wd_static_limit:
                        self._force_global_revalidate = True
                        self._force_global_revalidate_frame = self._lk.frame_num
                        self._watchdog_static_streak = 0
            elif found and not is_inertial and _wd_sift_src:
                # 首次 SIFT 命中：初始化看门狗基准点
                self._watchdog_last_sift_x = center_x
                self._watchdog_last_sift_y = center_y
            # 每个 SIFT 帧结束后重置 LK 累积（不论本帧是否命中）
            self._watchdog_accum_dx = 0.0
            self._watchdog_accum_dy = 0.0

            # 哈希校验补充“视觉内容与当前位置是否匹配”这条证据链。
            if found and not is_inertial and _wd_sift_src and center_x is not None:
                if not self._hash_index.check_consistency(minimap_gray_raw, center_x, center_y):
                    self._watchdog_suspect_streak += 1
                    if self._watchdog_suspect_streak >= _wd_limit:
                        print(f'[看门狗-哈希] 视觉不一致触发解锁! SIFT报告({center_x},{center_y})与哈希索引不匹配')
                        self._switch_to_global()
                        self._nearby_flann_cache.clear()
                        self._lk.reset()
                        self._force_global_revalidate = True
                        self._force_global_revalidate_frame = self._lk.frame_num
                        self._watchdog_suspect_streak = 0
                        self._watchdog_triggered = True
                        self.last_x = None
                        self.last_y = None
                        self.lost_frames = 0
                        found = False

        # 粗定位回退：低纹理或完全丢失时，先用哈希索引给出廉价候选。
        _semi_lost = (self.last_x is not None and self.lost_frames >= 8
                      and scene in ('ocean', 'low_texture'))
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

