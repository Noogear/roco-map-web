"""
core/hash_index.py - 感知哈希粗定位索引（MapHashIndex）

用途：
    1. 低纹理 / 全局丢失时给出廉价粗定位候选
    2. 冻结恢复后提供快速回退路径
    3. 为看门狗补充视觉一致性校验
"""

from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np

from backend import config


def _phash64(gray: np.ndarray, hash_size: int = 8) -> int:
    """计算 64-bit 感知哈希（DCT 低频）。"""
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def _prepare_query_patch(minimap_gray: np.ndarray, patch_size: int) -> tuple[np.ndarray, float]:
    """把圆形小地图转成用于哈希查询的方形 patch，并返回 patch 与其均值。"""
    h_mm, w_mm = minimap_gray.shape[:2]
    circ_mask = np.zeros((h_mm, w_mm), dtype=np.uint8)
    cx, cy = w_mm // 2, h_mm // 2
    cv2.circle(circ_mask, (cx, cy), min(cx, cy) - 2, 255, -1)
    mini_mean = float(np.mean(minimap_gray))
    filled = minimap_gray.copy()
    filled[circ_mask == 0] = int(mini_mean)
    patch = cv2.resize(filled, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return patch, float(np.mean(patch))


def _compute_hamming_distances(candidate_hashes: np.ndarray, query_hash: int) -> np.ndarray:
    """批量计算候选哈希到 query 的汉明距离。"""
    xored = candidate_hashes ^ np.uint64(query_hash)
    dists = np.zeros(len(xored), dtype=np.int32)
    for shift in range(0, 64, 8):
        byte_val = (xored >> np.uint64(shift)) & np.uint64(0xFF)
        dists += _POPCOUNT_TABLE[byte_val.astype(np.int32)]
    return dists


class MapHashIndex:
    """
    感知哈希地图索引。

    初始化时以 step 为步长遍历大地图，对每个位置取 patch
    计算 pHash，存入反向索引。运行时查找汉明距离 ≤ threshold
    的候选位置。

    磁盘缓存：按地图文件 MD5 + 参数指纹自动缓存 .npz
    """

    def __init__(
        self,
        logic_map_gray: np.ndarray,
        map_width: int,
        map_height: int,
        *,
        step: int = 0,
        patch_scale: float = 0.0,
        patch_size: int = 0,
        hamming_threshold: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        self._map_width = map_width
        self._map_height = map_height
        self._step = step or getattr(config, 'HASH_INDEX_STEP', 60)
        self._patch_scale = patch_scale or getattr(config, 'HASH_INDEX_PATCH_SCALE', 4.0)
        self._patch_size = patch_size or getattr(config, 'HASH_INDEX_PATCH_SIZE', 128)
        self._hamming_thresh = hamming_threshold or getattr(config, 'HASH_INDEX_HAMMING_THRESHOLD', 12)

        self._xs: np.ndarray = np.empty(0, dtype=np.int32)
        self._ys: np.ndarray = np.empty(0, dtype=np.int32)
        self._hashes: np.ndarray = np.empty(0, dtype=np.uint64)
        self._means: np.ndarray = np.empty(0, dtype=np.float32)
        self._build(logic_map_gray, cache_dir)

    # ------------------------------------------------------------------
    def _cache_path(self, cache_dir: str | None) -> str | None:
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(config.LOGIC_MAP_PATH), '.cache')
        fp = f'{self._map_width}x{self._map_height}_s{self._step}_ps{self._patch_size}_sc{self._patch_scale}'
        return os.path.join(cache_dir, f'hash_index_{hashlib.md5(fp.encode()).hexdigest()[:12]}.npz')

    def _build(self, logic_map_gray: np.ndarray, cache_dir: str | None) -> None:
        cache = self._cache_path(cache_dir)
        if cache and os.path.isfile(cache):
            try:
                data = np.load(cache)
                self._xs = data['xs']
                self._ys = data['ys']
                self._hashes = data['hashes']
                self._means = data['means']
                print(f'✅ 哈希索引从缓存加载: {len(self._xs)} 条目')
                return
            except Exception:
                pass

        print(f'正在构建哈希索引... (step={self._step}, patch={self._patch_size})')
        h, w = logic_map_gray.shape[:2]
        step = self._step
        ps = self._patch_size
        half_crop = int(ps * self._patch_scale / 2)

        xs_list, ys_list, hash_list, mean_list = [], [], [], []

        for cy in range(half_crop, h - half_crop, step):
            for cx in range(half_crop, w - half_crop, step):
                y1, y2 = cy - half_crop, cy + half_crop
                x1, x2 = cx - half_crop, cx + half_crop
                patch = logic_map_gray[y1:y2, x1:x2]
                # 缩放到 patch_size 模拟小地图视野
                patch_resized = cv2.resize(patch, (ps, ps), interpolation=cv2.INTER_AREA)
                h_val = _phash64(patch_resized)
                xs_list.append(cx)
                ys_list.append(cy)
                hash_list.append(h_val)
                mean_list.append(float(np.mean(patch_resized)))

        self._xs = np.array(xs_list, dtype=np.int32)
        self._ys = np.array(ys_list, dtype=np.int32)
        self._hashes = np.array(hash_list, dtype=np.uint64)
        self._means = np.array(mean_list, dtype=np.float32)

        print(f'[OK] 哈希索引构建完成: {len(self._xs)} 条目')

        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            try:
                np.savez_compressed(cache,
                                    xs=self._xs, ys=self._ys, hashes=self._hashes,
                                    means=self._means)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def locate(
        self,
        minimap_gray: np.ndarray,
        last_x: int | None = None,
        last_y: int | None = None,
        radius: int = 0,
        max_results: int = 5,
        hamming_threshold: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """
        查找与小地图最匹配的候选位置。

        Args:
            minimap_gray: 灰度小地图（未增强的原始图像）。
            last_x/y:     有位置提示时，优先返回距此处较近的候选。
            radius:       >0 时只搜索 last_x±radius 范围内的候选。
            max_results:  最多返回几个候选。
            hamming_threshold: 可选汉明阈值；None 表示使用索引默认阈值。

        Returns:
            [(x, y, hamming_distance), ...] 按汉明距离升序排列。
        """
        if len(self._hashes) == 0:
            return []

        query_patch, query_mean = _prepare_query_patch(minimap_gray, self._patch_size)
        query_hash = _phash64(query_patch)

        # 预过滤：均值灰度差异过大的排除
        color_thresh = getattr(config, 'HASH_INDEX_COLOR_THRESH', 50)
        mask = np.abs(self._means - query_mean) < color_thresh

        # 空间限制
        if radius > 0 and last_x is not None and last_y is not None:
            mask &= (np.abs(self._xs - last_x) < radius) & (np.abs(self._ys - last_y) < radius)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        candidate_hashes = self._hashes[indices]
        dists = _compute_hamming_distances(candidate_hashes, query_hash)

        # 阈值过滤
        threshold = self._hamming_thresh if hamming_threshold is None else int(hamming_threshold)
        ok = dists <= threshold
        ok_indices = indices[ok]
        ok_dists = dists[ok]
        if len(ok_indices) == 0:
            return []

        if last_x is not None and last_y is not None:
            spatial = np.abs(self._xs[ok_indices] - last_x) + np.abs(self._ys[ok_indices] - last_y)
            order = np.lexsort((spatial, ok_dists))
        else:
            order = np.argsort(ok_dists)
        order = order[:max_results]

        return [(int(self._xs[ok_indices[i]]),
                 int(self._ys[ok_indices[i]]),
                 int(ok_dists[i]))
                for i in order]

    # ------------------------------------------------------------------
    def check_consistency(
        self,
        minimap_gray: np.ndarray,
        expected_x: int,
        expected_y: int,
        check_radius: int = 0,
    ) -> bool:
        """
        检查当前小地图的 pHash 是否与 expected 位置附近的索引条目一致。

        用于辅助看门狗：若返回 False，说明视觉内容与当前认为的位置不匹配。
        """
        consistent, _, _ = self.check_consistency_details(
            minimap_gray=minimap_gray,
            expected_x=expected_x,
            expected_y=expected_y,
            check_radius=check_radius,
        )
        return consistent

    # ------------------------------------------------------------------
    def check_consistency_details(
        self,
        minimap_gray: np.ndarray,
        expected_x: int,
        expected_y: int,
        check_radius: int = 0,
    ) -> tuple[bool, int | None, int]:
        """一致性细节版本。

        Returns:
            (consistent, best_distance, evaluated_count)
        """
        if len(self._hashes) == 0:
            return True, None, 0  # 无索引时不阻断

        query_patch, query_mean = _prepare_query_patch(minimap_gray, self._patch_size)
        query_hash = _phash64(query_patch)

        r = check_radius or self._step * 3
        mask = ((np.abs(self._xs - expected_x) < r) &
                (np.abs(self._ys - expected_y) < r))
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return True, None, 0  # 附近无索引条目

        # 与 locate 同步：先按亮度均值做粗过滤，避免明显不可能候选干扰距离判断。
        color_thresh = getattr(config, 'HASH_INDEX_COLOR_THRESH', 50)
        local_means = self._means[indices]
        mean_mask = np.abs(local_means - query_mean) < color_thresh
        if np.any(mean_mask):
            indices = indices[mean_mask]

        if len(indices) == 0:
            return True, None, 0

        candidate_hashes = self._hashes[indices]
        dists = _compute_hamming_distances(candidate_hashes, query_hash)
        best = int(np.min(dists)) if len(dists) > 0 else None
        if best is None:
            return True, None, 0

        # 最近邻的距离 ≤ 阈值则一致
        return best <= self._hamming_thresh, best, int(len(dists))


# popcount 查找表（256 字节）
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
