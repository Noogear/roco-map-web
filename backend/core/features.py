"""
core/features.py - ORB + BEBLID 特征提取与匹配

说明：
- 运行时强依赖 opencv-contrib 的 xfeatures2d.BEBLID。
"""

from __future__ import annotations

import numpy as np
import cv2
import hashlib
import os
from backend import config


class _StaticBFMatcher:
    """静态描述子匹配器：封装 BFMatcher 接口，兼容旧 knnMatch 调用。"""

    def __init__(self, train_des: np.ndarray) -> None:
        self._train_des = np.asarray(train_des, dtype=np.uint8)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def knnMatch(self, query_des: np.ndarray, k: int = 2):
        query = np.asarray(query_des, dtype=np.uint8)
        return self._bf.knnMatch(query, self._train_des, k=k)


class _StaticFlannLSHMatcher:
    """静态描述子匹配器：FLANN-LSH（用于二进制描述子大规模候选加速）。"""

    def __init__(self, train_des: np.ndarray, *, checks: int = 50) -> None:
        self._train_des = np.ascontiguousarray(train_des, dtype=np.uint8)
        index_params = dict(
            algorithm=6,          # FLANN_INDEX_LSH
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=max(1, int(checks)))
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)
        self._flann.add([self._train_des])
        self._flann.train()

    def knnMatch(self, query_des: np.ndarray, k: int = 2):
        query = np.ascontiguousarray(query_des, dtype=np.uint8)
        return self._flann.knnMatch(query, k=k)


class ORBBeblidFeature2D:
    """组合特征算子：ORB 提取关键点 + BEBLID 计算描述子。"""

    def __init__(self) -> None:
        ensure_beblid_runtime()
        nfeatures = int(getattr(config, 'ORB_BEBLID_NFEATURES', 7000))
        scale_factor = float(getattr(config, 'ORB_BEBLID_SCALE_FACTOR', 1.0))
        n_bits = int(getattr(config, 'ORB_BEBLID_BITS', 512))
        fast_threshold = int(getattr(config, 'ORB_BEBLID_FAST_THRESHOLD', 6))
        edge_threshold = int(getattr(config, 'ORB_BEBLID_EDGE_THRESHOLD', 15))

        self._orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=edge_threshold,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=fast_threshold,
        )
        if n_bits == 256:
            beblid_bits = cv2.xfeatures2d.BEBLID_SIZE_256_BITS
        else:
            beblid_bits = cv2.xfeatures2d.BEBLID_SIZE_512_BITS
        self._beblid = cv2.xfeatures2d.BEBLID_create(scale_factor, beblid_bits)

    def detectAndCompute(self, image_gray: np.ndarray, mask: np.ndarray | None):
        kp = self._orb.detect(image_gray, mask)
        if not kp:
            return [], None
        kp, des = self._beblid.compute(image_gray, kp)
        if des is None or len(kp) < 2:
            return [], None
        return kp, np.asarray(des, dtype=np.uint8)


def ensure_beblid_runtime() -> None:
    """启动期强校验：缺少 BEBLID 能力时直接失败。"""
    has_xf = hasattr(cv2, 'xfeatures2d')
    has_beblid = has_xf and hasattr(cv2.xfeatures2d, 'BEBLID_create')
    if not has_beblid:
        raise RuntimeError(
            'BEBLID 不可用：请安装 opencv-contrib-python，并确认 cv2.xfeatures2d.BEBLID_create 可用。'
        )


def create_orb_beblid_feature2d() -> ORBBeblidFeature2D:
    return ORBBeblidFeature2D()


# ---------------------------------------------------------------------------
# 圆形掩码（带 LRU 缓存）
# ---------------------------------------------------------------------------

def _resolve_circular_mask_geometry(
    h: int,
    w: int,
    *,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
) -> tuple[int, int, int]:
    if center is None:
        cx = w / 2.0
        cy = h / 2.0
    else:
        cx = float(center[0])
        cy = float(center[1])

    if radius is None:
        base_radius = min(cx, cy, max(0.0, w - cx), max(0.0, h - cy))
    else:
        base_radius = float(radius)

    margin_ratio = float(getattr(config, 'MINIMAP_MASK_EDGE_MARGIN_RATIO', 0.08))
    min_margin = int(getattr(config, 'MINIMAP_MASK_EDGE_MARGIN_MIN_PIXELS', 6))
    margin = max(2, min_margin, int(round(base_radius * margin_ratio)))
    outer_radius = max(1, int(round(base_radius - margin)))
    return int(round(cx)), int(round(cy)), outer_radius


def make_circular_mask(
    h: int,
    w: int,
    inner_ratio: float = 0.0,
    *,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
) -> np.ndarray:
    """生成大小为 (h, w) 的圆形掩码；可选中心挖空（用于排除玩家箭头）。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy, r_outer = _resolve_circular_mask_geometry(h, w, center=center, radius=radius)
    cv2.circle(mask, (cx, cy), r_outer, 255, -1)
    if inner_ratio > 0:
        r_inner = max(1, int(round(r_outer * float(inner_ratio))))
        if r_inner < r_outer:
            cv2.circle(mask, (cx, cy), r_inner, 0, -1)
    return mask


class CircularMaskCache:
    """按 (h, w, inner_px) 缓存圆形掩码，避免重复生成。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def get(
        self,
        h: int,
        w: int,
        inner_ratio: float = 0.0,
        *,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
    ) -> np.ndarray:
        if center is not None or radius is not None:
            return make_circular_mask(h, w, inner_ratio, center=center, radius=radius)

        _, _, r_outer = _resolve_circular_mask_geometry(h, w)
        ratio = max(0.0, min(0.95, float(inner_ratio)))
        inner_px = int(round(r_outer * ratio)) if ratio > 0 else 0
        key = (h, w, inner_px, r_outer)
        if key not in self._cache:
            self._cache[key] = make_circular_mask(h, w, inner_px / float(r_outer))
        return self._cache[key]


# ---------------------------------------------------------------------------
# 匹配器工厂（兼容旧接口名 create_flann）
# ---------------------------------------------------------------------------

def create_flann(descriptors: np.ndarray):
    """兼容旧函数名：二进制描述子走 BF-Hamming，浮点描述子走 FLANN(KDTree)。"""
    des = np.asarray(descriptors)
    if np.issubdtype(des.dtype, np.floating):
        des = np.asarray(des, dtype=np.float32)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flann.add([des])
        flann.train()
        return flann
    des = np.asarray(des, dtype=np.uint8)
    return _StaticBFMatcher(des)


def create_flann_lsh(descriptors: np.ndarray, *, checks: int = 50):
    """二进制描述子 FLANN-LSH 匹配器（大候选场景加速）。"""
    des = np.asarray(descriptors, dtype=np.uint8)
    return _StaticFlannLSHMatcher(des, checks=checks)


# ---------------------------------------------------------------------------
# 小地图特征提取
# ---------------------------------------------------------------------------

def extract_minimap_features(
    minimap_gray: np.ndarray,
    sift,
    mask_cache: CircularMaskCache,
    *,
    texture_std: float | None = None,
    inner_ratio: float | None = None,
    mask_center: tuple[float, float] | None = None,
    mask_radius: float | None = None,
) -> tuple[list | None, np.ndarray | None]:
    """提取小地图特征（带圆形掩码）。返回 (kp, des) 或 (None, None)。

    texture_std:  按场景分类（ocean/low_texture/mixed/urban）自动选择挖空比例。
    inner_ratio:  直接指定挖空比例，覆盖 config 及 texture_std 逻辑（多用于测试）。
    """
    h, w = minimap_gray.shape[:2]
    if inner_ratio is None:
        r_urban = float(getattr(config, 'MINIMAP_CENTER_EXCLUDE_RATIO', 0.16))
        r_mixed = float(getattr(config, 'MINIMAP_CENTER_EXCLUDE_RATIO_MIXED', r_urban))
        r_hard  = float(getattr(config, 'MINIMAP_CENTER_EXCLUDE_RATIO_HARD', r_mixed))
        if texture_std is not None:
            if texture_std < 35:    # low_texture + ocean
                inner_ratio = r_hard
            elif texture_std < 55:  # mixed
                inner_ratio = r_mixed
            else:                   # urban
                inner_ratio = r_urban
        else:
            inner_ratio = r_urban

    mask = mask_cache.get(
        h, w,
        inner_ratio=inner_ratio,
        center=mask_center,
        radius=mask_radius,
    )

    kp, des = sift.detectAndCompute(minimap_gray, mask)
    if des is None or len(kp) < 2:
        return None, None
    return kp, np.asarray(des)


def extract_map_features_tiled(
    map_gray: np.ndarray,
    sift,
    *,
    tile_size: int,
    overlap: int = 0,
    max_features_per_tile: int = 0,
) -> tuple[list, np.ndarray | None]:
    """
    以分块方式提取整张大地图的 SIFT 特征，降低 detectAndCompute 的峰值内存。

    overlap 用于覆盖 tile 边界附近的特征；通过“归属区域”裁剪避免重复保留。
    """
    h, w = map_gray.shape[:2]
    tile_size = max(256, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    step = max(1, tile_size - overlap)
    half_overlap = overlap // 2

    all_keypoints: list = []
    descriptor_chunks: list[np.ndarray] = []

    for y0 in range(0, h, step):
        y1 = min(h, y0 + tile_size)
        own_top = y0 if y0 == 0 else y0 + half_overlap
        own_bottom = y1 if y1 == h else y1 - half_overlap

        for x0 in range(0, w, step):
            x1 = min(w, x0 + tile_size)
            own_left = x0 if x0 == 0 else x0 + half_overlap
            own_right = x1 if x1 == w else x1 - half_overlap

            tile = map_gray[y0:y1, x0:x1]
            kp_tile, des_tile = sift.detectAndCompute(tile, None)
            if des_tile is None or not kp_tile:
                continue

            kept: list[tuple[int, float, float]] = []
            for idx, kp in enumerate(kp_tile):
                gx = kp.pt[0] + x0
                gy = kp.pt[1] + y0
                if own_left <= gx < own_right and own_top <= gy < own_bottom:
                    kept.append((idx, gx, gy))

            if not kept:
                continue

            if max_features_per_tile > 0 and len(kept) > max_features_per_tile:
                responses = np.array([kp_tile[idx].response for idx, _, _ in kept], dtype=np.float32)
                top = np.argpartition(responses, -max_features_per_tile)[-max_features_per_tile:]
                kept = [kept[i] for i in sorted(top)]

            descriptor_chunks.append(np.asarray(des_tile[[idx for idx, _, _ in kept]]))
            for idx, gx, gy in kept:
                kp = kp_tile[idx]
                all_keypoints.append(cv2.KeyPoint(
                    x=float(gx),
                    y=float(gy),
                    size=kp.size,
                    angle=kp.angle,
                    response=kp.response,
                    octave=kp.octave,
                    class_id=kp.class_id,
                ))

    if not descriptor_chunks:
        return [], None

    des_all = np.vstack(descriptor_chunks)
    return all_keypoints, des_all


# ---------------------------------------------------------------------------
# 区域匹配
# ---------------------------------------------------------------------------

def match_region(
    kp_mini: list,
    des_mini: np.ndarray,
    mm_shape: tuple[int, int],
    region_kp: list,
    region_flann,
    ratio: float,
    min_match: int,
    map_width: int,
    map_height: int,
    *,
    minimap_center: tuple[float, float] | None = None,
    use_gms: bool = False,
    gms_train_shape: tuple[int, int] | None = None,
    gms_min_matches: int = 20,
    gms_with_rotation: bool = True,
    gms_with_scale: bool = False,
) -> tuple[int, int, int, float, float] | None:
    """
    通用 ORB+BEBLID 区域匹配（保留旧函数名）。
    返回 (tx, ty, inlier_count, quality, avg_scale) 或 None。
    """
    if des_mini is None or len(kp_mini) < 2:
        return None

    query = np.asarray(des_mini)
    matches = None
    for _query in (query, np.asarray(query, dtype=np.uint8), np.asarray(query, dtype=np.float32)):
        try:
            matches = region_flann.knnMatch(_query, k=2)
            break
        except cv2.error:
            continue
    if matches is None:
        return None

    ratio_good = [m for m_n in matches if len(m_n) == 2
                  for m, n in [m_n] if m.distance < ratio * n.distance]

    good = ratio_good

    # 可选 GMS：仅在样本足够时启用；失败自动回退 ratio_good。
    if use_gms and len(kp_mini) >= max(min_match, 6):
        has_xf = hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'matchGMS')
        if has_xf:
            try:
                tentative = [m_n[0] for m_n in matches if len(m_n) >= 1]
                if len(tentative) >= max(gms_min_matches, min_match):
                    hq, wq = mm_shape[:2]
                    if gms_train_shape is not None:
                        ht, wt = gms_train_shape[:2]
                    else:
                        _xs = [kp.pt[0] for kp in region_kp]
                        _ys = [kp.pt[1] for kp in region_kp]
                        wt = max(16, int(max(_xs) - min(_xs) + 1)) if _xs else wq
                        ht = max(16, int(max(_ys) - min(_ys) + 1)) if _ys else hq
                    gms_matches = cv2.xfeatures2d.matchGMS(
                        (int(wq), int(hq)), (int(wt), int(ht)),
                        kp_mini, region_kp, tentative,
                        withRotation=bool(gms_with_rotation),
                        withScale=bool(gms_with_scale),
                    )
                    if gms_matches is not None and len(gms_matches) >= max(min_match, gms_min_matches):
                        good = list(gms_matches)
            except Exception:
                # GMS 不可用/异常，回退 ratio 结果
                good = ratio_good

    if len(good) < min_match:
        return None

    src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([region_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC,
        ransacReprojThreshold=float(getattr(config, 'ORB_BEBLID_RANSAC_THRESHOLD', 6.0)))
    if M is None:
        return None

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if inlier_count < min_match:
        return None

    # AffinePartial2D 返回 2x3 矩阵: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
    avg_scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    max_scale = float(getattr(config, 'ORB_BEBLID_MAX_AFFINE_SCALE', 2.0))
    if avg_scale > max_scale or avg_scale < 1.0 / max_scale:
        return None

    h, w = mm_shape[:2]
    if minimap_center is None:
        center_x, center_y = w / 2.0, h / 2.0
    else:
        center_x, center_y = float(minimap_center[0]), float(minimap_center[1])
    center_src = np.array([center_x, center_y, 1.0], dtype=np.float64)
    dst_center = M @ center_src
    tx, ty = int(dst_center[0]), int(dst_center[1])

    if not (0 <= tx < map_width and 0 <= ty < map_height):
        return None

    inlier_ratio = inlier_count / max(len(good), 1)
    count_conf = min(1.0, inlier_count / max(1.0, float(getattr(config, 'ORB_BEBLID_QUALITY_NORM_COUNT', 18.0))))
    quality = min(1.0, inlier_ratio * count_conf)
    return tx, ty, inlier_count, quality, avg_scale


# ---------------------------------------------------------------------------
# 特征磁盘缓存
# ---------------------------------------------------------------------------

def _compute_feature_fingerprint(
    map_path: str,
    scales: list[float],
    tile_size: int,
    overlap: int,
    nfeatures: int,
    n_bits: int,
) -> str:
    """计算特征缓存指纹（地图文件属性 + 提取参数）。任一变化则缓存自动失效。"""
    try:
        stat = os.stat(map_path)
        file_info = f"{stat.st_size}:{int(stat.st_mtime * 1000)}"
    except OSError:
        file_info = "?"
    params = (
        f"scales={sorted(scales)}|tile={tile_size}|overlap={overlap}"
        f"|nf={nfeatures}|bits={n_bits}"
    )
    return hashlib.md5(f"{file_info}|{params}".encode()).hexdigest()[:16]


def save_features_npz(
    path: str,
    kp_list: list,
    des: np.ndarray,
    fingerprint: str = '',
) -> None:
    """将关键点列表和描述子压缩保存到 .npz。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kp_arr = np.array(
        [
            (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in kp_list
        ],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('size', 'f4'), ('angle', 'f4'),
            ('response', 'f4'), ('octave', 'i4'), ('class_id', 'i4'),
        ],
    )
    np.savez_compressed(path, keypoints=kp_arr, descriptors=des,
                        fingerprint=np.array([fingerprint]))


def load_features_npz(
    path: str,
    fingerprint: str = '',
) -> tuple[list, np.ndarray] | None:
    """从 .npz 恢复关键点和描述子；fingerprint 不匹配或文件损坏时返回 None。"""
    npz_path = path if path.endswith('.npz') else path + '.npz'
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path, allow_pickle=False)
        if fingerprint:
            saved_fp = str(data['fingerprint'][0])
            if saved_fp != fingerprint:
                return None
        kp_arr = data['keypoints']
        des = np.asarray(data['descriptors'])
        kp_list = [
            cv2.KeyPoint(
                x=float(r['x']), y=float(r['y']), size=float(r['size']),
                angle=float(r['angle']), response=float(r['response']),
                octave=int(r['octave']), class_id=int(r['class_id']),
            )
            for r in kp_arr
        ]
        return kp_list, des
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 多尺度特征提取
# ---------------------------------------------------------------------------

def extract_map_features_multiscale(
    map_gray: np.ndarray,
    sift,
    scales: list[float],
    *,
    tile_size: int,
    overlap: int = 0,
    max_features_per_tile: int = 0,
) -> tuple[list, np.ndarray | None]:
    """
    多尺度分块特征提取：在多个缩放倍率下提取，坐标归一化回原图坐标系后合并。

    相比单尺度，能有效应对游戏视角缩放（小地图 FOV 变化）时的匹配率下降问题。
    scales 为空时等价于单尺度 extract_map_features_tiled。
    """
    if not scales:
        return extract_map_features_tiled(
            map_gray, sift, tile_size=tile_size, overlap=overlap,
            max_features_per_tile=max_features_per_tile)

    h0, w0 = map_gray.shape[:2]
    all_kp: list = []
    all_des_chunks: list[np.ndarray] = []

    for s in scales:
        if abs(s - 1.0) < 1e-6:
            layer = map_gray
        else:
            nw = max(16, int(round(w0 * s)))
            nh = max(16, int(round(h0 * s)))
            layer = cv2.resize(map_gray, (nw, nh), interpolation=cv2.INTER_LINEAR)

        kp_s, des_s = extract_map_features_tiled(
            layer, sift,
            tile_size=tile_size,
            overlap=overlap,
            max_features_per_tile=max_features_per_tile,
        )
        if des_s is None or not kp_s:
            continue

        # 坐标归一化：缩放坐标 → 原图坐标系
        if abs(s - 1.0) >= 1e-6:
            for kp in kp_s:
                kp.pt = (kp.pt[0] / s, kp.pt[1] / s)

        all_kp.extend(kp_s)
        all_des_chunks.append(des_s)

    if not all_des_chunks:
        return [], None

    return all_kp, np.vstack(all_des_chunks)

