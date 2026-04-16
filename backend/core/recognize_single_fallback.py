"""recognize_single 的 hash + ECC 兜底策略。"""

from __future__ import annotations

from typing import Any

from backend import config
from backend.core.ecc import ecc_align


def locate_hash_candidates_adaptive(hash_index, gray_raw, hash_query_lock, max_results: int = 3):
    """无状态场景下自适应放宽 hash 阈值，提升低纹理召回率。"""
    if hash_index is None:
        return []

    thresholds = [None, 16, 20]
    with hash_query_lock:
        original = int(getattr(hash_index, '_hamming_thresh', 12))
        try:
            for th in thresholds:
                if th is not None:
                    hash_index._hamming_thresh = int(th)
                candidates = hash_index.locate(gray_raw, last_x=None, last_y=None, radius=0, max_results=max_results)
                if candidates:
                    return candidates
            return []
        finally:
            hash_index._hamming_thresh = original


def build_hash_ecc_or_hash_fallback_status(
    *,
    shared: Any,
    hash_index,
    hash_query_lock,
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
        hash_query_lock=hash_query_lock,
        max_results=3,
    )
    if not candidates:
        return None

    logic_map_for_ecc = getattr(shared, '_logic_map_gray_clahe', shared._logic_map_gray)
    ecc_min_cc = float(getattr(config, 'SINGLE_ECC_MIN_CORRELATION', 0.30))
    ecc_jump = int(getattr(config, 'SINGLE_ECC_JUMP_THRESHOLD', 360))
    ecc_scale_hint = float(getattr(config, 'HASH_INDEX_PATCH_SCALE', 4.0))

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
        )
        if refined is not None:
            ex, ey = refined
            consistent = True
            try:
                consistent = bool(hash_index.check_consistency(gray_raw, int(ex), int(ey)))
            except Exception:
                consistent = True
            base_q = max(0.18, 0.42 - float(hdist) * 0.015)
            if consistent:
                base_q = max(base_q, 0.35)
            return {
                'state': 'FOUND',
                'found': True,
                'source': f'HASH_ECC_{source_suffix}',
                'position': {'x': int(ex), 'y': int(ey)},
                'matches': 0,
                'match_quality': float(min(base_q, 0.65)),
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
