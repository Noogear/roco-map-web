"""recognize_single 的 hash + ECC 兜底策略。"""

from __future__ import annotations

from typing import Any

from backend import config
from backend.core.ecc import ecc_align


def locate_hash_candidates_adaptive(hash_index, gray_raw, max_results: int = 3):
    """无状态场景下自适应放宽 hash 阈值，提升低纹理召回率。"""
    if hash_index is None:
        return []

    thresholds = [None, 16, 20]
    # 注意：不要临时改写共享索引的 _hamming_thresh，避免跨会话并发污染。
    for th in thresholds:
        candidates = hash_index.locate(
            gray_raw,
            last_x=None,
            last_y=None,
            radius=0,
            max_results=max_results,
            hamming_threshold=th,
        )
        if candidates:
            return candidates
    return []


def build_hash_ecc_or_hash_fallback_status(
    *,
    shared: Any,
    hash_index,
    gray,
    gray_raw,
    source_suffix: str,
):
    """构建 hash+ECC 兜底状态。返回 status dict 或 None。"""
    if hash_index is None:
        return None

    candidates = locate_hash_candidates_adaptive(
        hash_index=hash_index,
        gray_raw=gray_raw,
        max_results=3,
    )
    if not candidates:
        return None

    logic_map_for_ecc = getattr(shared, '_logic_map_gray_clahe', shared._logic_map_gray)
    ecc_min_cc = float(getattr(config, 'SINGLE_ECC_MIN_CORRELATION', 0.30))
    ecc_jump = int(getattr(config, 'SINGLE_ECC_JUMP_THRESHOLD', 360))
    ecc_scale_hint = float(getattr(config, 'HASH_INDEX_PATCH_SCALE', 4.0))

    best_hit = None
    best_key = None

    for hx, hy, hdist in candidates:
        refined = ecc_align(
            gray,
            logic_map_for_ecc,
            int(hx),
            int(hy),
            ecc_scale_hint,
            shared.map_width,
            shared.map_height,
            ecc_jump,
            min_cc=ecc_min_cc,
            return_cc=True,
        )
        if refined is not None:
            ex, ey, ecc_cc = refined
            consistent = True
            consistency_error = False
            try:
                consistent = bool(hash_index.check_consistency(gray_raw, int(ex), int(ey)))
            except Exception:
                # fail-closed: 一致性检查异常时不再“放行”，而是降权。
                consistent = False
                consistency_error = True

            base_q = max(0.18, 0.42 - float(hdist) * 0.015)
            if consistent:
                base_q = max(base_q, 0.35)
            else:
                base_q = min(base_q, 0.30)
                if consistency_error:
                    base_q = min(base_q, 0.24)

            hit_key = (float(ecc_cc), 1 if consistent else 0, float(base_q), -int(hdist))
            if best_key is None or hit_key > best_key:
                best_key = hit_key
                best_hit = {
                    'x': int(ex),
                    'y': int(ey),
                    'quality': float(min(base_q, 0.65)),
                }

    if best_hit is not None:
        return {
            'state': 'FOUND',
            'found': True,
            'source': f'HASH_ECC_{source_suffix}',
            'position': {'x': best_hit['x'], 'y': best_hit['y']},
            'matches': 0,
            'match_quality': best_hit['quality'],
            'mode': 'sift',
        }

    hx, hy, hdist = candidates[0]
    return {
        'state': 'HASH_FOUND',
        'found': True,
        'source': f'HASH_INDEX_{source_suffix}',
        'position': {'x': int(hx), 'y': int(hy)},
        'matches': 0,
        'match_quality': max(0.15, 0.35 - float(hdist) * 0.02),
        'mode': 'sift',
    }
