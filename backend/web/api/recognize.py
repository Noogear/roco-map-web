from __future__ import annotations

import numpy as np

from backend import config
from backend.vision.engine.feature_map_tracker import FeatureMapTracker
from backend.vision.engine.shared_feature_factory import get_shared_feature
from backend.map.autodetect import detect_minimap_circle
from backend.map.minimap import CircleCalibrator, detect_and_extract_with_meta

_shared = None
_matcher = None


def ensure_initialized():
    global _shared, _matcher
    if _shared is None or _matcher is None:
        _shared = get_shared_feature()
        _matcher = FeatureMapTracker(_shared)
    return _shared, _matcher


def extract_minimap_with_autodetect(img: np.ndarray, calibrator=None):
    """返回 (minimap_img, mm_center, mm_radius, source_tag, raw_crop)
    calibrator: CircleCalibrator | None，传入可复用历史（稳定检测）；None 时使用临时实例。
    raw_crop: MinimapCrop | None，含 orig_cx/orig_cy，供调用方取原帧圆坐标。
    """
    _cal = calibrator if calibrator is not None else CircleCalibrator()
    source_tag = 'RAW'
    extracted = detect_and_extract_with_meta(img, _cal, engine_frozen=False)
    if extracted is not None:
        return extracted.image, extracted.center_xy, extracted.radius, 'LOCAL_CIRCLE', extracted
    input_h, input_w = img.shape[:2]
    if min(input_h, input_w) >= 320:
        det = detect_minimap_circle(img)
        if det is not None:
            h, w = img.shape[:2]
            bs = min(w, h)
            cx = float(det['cx']) * w
            cy = float(det['cy']) * h
            r = float(det['r']) * bs
            margin = max(1.0, float(getattr(config, 'MINIMAP_CAPTURE_MARGIN', 1.4)))
            sz = max(10, int(round(r * 2 * margin)))
            rx = max(0, min(int(round(cx - sz / 2)), w - sz))
            ry = max(0, min(int(round(cy - sz / 2)), h - sz))
            crop = img[ry:ry + sz, rx:rx + sz].copy()
            extracted2 = detect_and_extract_with_meta(crop, CircleCalibrator(), engine_frozen=False)
            if extracted2 is not None:
                stag = f"AUTO_DETECT_{det.get('layout', 'full').upper()}"
                return extracted2.image, extracted2.center_xy, extracted2.radius, stag, extracted2
            source_tag = f"AUTO_DETECT_{det.get('layout', 'full').upper()}"
            return crop, None, None, source_tag, None
    return img, None, None, source_tag, None


def analyze_image(img):
    _, matcher = ensure_initialized()
    minimap_img, minimap_center, minimap_radius, source_tag, _ = extract_minimap_with_autodetect(img)
    result = matcher.match(minimap_img, minimap_center=minimap_center, minimap_radius=minimap_radius)
    if not result.get('found', False):
        return {'success': False, 'error': 'No match found', 'source': f"{result.get('source', '')}_{source_tag}"}
    return {'success': True, 'position': {'x': int(result['center_x']), 'y': int(result['center_y'])}, 'match_quality': float(result.get('match_quality', 0.0)), 'source': f"{result.get('source', '')}_{source_tag}"}
