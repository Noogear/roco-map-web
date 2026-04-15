"""
tracking/autodetect.py - 全帧小地图圆自动识别（Web 屏幕捕获）

目标：
  - 支持不同分辨率下的整帧自动定位
  - 支持多帧投票，降低单帧动画/特效/遮挡造成的误检
  - 用 circle / edge / template 组合打分，提高候选区分度

纯函数模块，无 Flask / Web 依赖。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from backend import config


_TEMPLATE_CACHE: dict[int, np.ndarray] = {}


def _get_ring_template(size: int) -> np.ndarray:
    """生成合成圆环模板（0..1 float32），用于模板相关性评分。"""
    size = max(16, int(size))
    if size in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[size]

    tmpl = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    r = max(3, int(round(size * 0.38)))
    thickness = max(2, int(round(size * 0.08)))
    cv2.circle(tmpl, (c, c), r, 1.0, thickness=thickness)
    tmpl = cv2.GaussianBlur(tmpl, (0, 0), sigmaX=max(0.8, size / 30.0))
    _TEMPLATE_CACHE[size] = tmpl
    return tmpl


def _resize_for_detection(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """将整帧缩放到合理上限，返回 (缩放后图像, scale)。"""
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    limit = int(getattr(config, 'AUTO_DETECT_MAX_SIDE', 1280))
    if max_side <= limit:
        return img_bgr, 1.0

    scale = limit / float(max_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale


def _layout_regions(w: int, h: int) -> list[tuple[str, tuple[int, int, int, int], int]]:
    """返回候选布局 ROI：(layout_name, (x1,y1,x2,y2), param2)。"""
    p2_roi = int(getattr(config, 'AUTO_DETECT_HOUGH_PARAM2_ROI', 24))
    p2_full = int(getattr(config, 'AUTO_DETECT_HOUGH_PARAM2_FULL', 20))
    return [
        ('top-right', (int(w * 0.50), 0, w, int(h * 0.52)), p2_roi),
        ('top-left', (0, 0, int(w * 0.50), int(h * 0.52)), p2_roi),
        ('full', (0, 0, w, h), p2_full),
    ]


def _collect_hough_candidates(
    blur_gray: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    layout: str,
    param2: int,
) -> list[dict]:
    """在指定 ROI 内收集 Hough 圆候选。"""
    roi = blur_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    h, w = blur_gray.shape[:2]
    base = min(h, w)
    min_r = max(10, int(base * float(getattr(config, 'AUTO_DETECT_MIN_RADIUS_RATIO', 0.035))))
    max_r = max(min_r + 6, int(base * float(getattr(config, 'AUTO_DETECT_MAX_RADIUS_RATIO', 0.16))))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(roi.shape[:2]) // 4),
        param1=int(getattr(config, 'AUTO_DETECT_HOUGH_PARAM1', 90)),
        param2=max(8, int(param2)),
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    candidates = []
    for c in np.round(circles[0, :]).astype(int):
        cx, cy, r = int(c[0]) + x1, int(c[1]) + y1, int(c[2])
        if r <= 0:
            continue
        candidates.append({'cx': cx, 'cy': cy, 'r': r, 'layout': layout})
    return candidates


def _edge_score(edges: np.ndarray, cx: int, cy: int, r: int) -> float:
    """圆环边缘强度评分。"""
    h, w = edges.shape[:2]
    pad = max(4, int(r * 1.25))
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(w, cx + pad)
    y2 = min(h, cy + pad)
    roi = edges[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    band = max(2.0, r * 0.12)
    ring_mask = (dist >= (r - band)) & (dist <= (r + band))
    if not np.any(ring_mask):
        return 0.0
    return float(np.mean(roi[ring_mask]) / 255.0)


def _template_score(edges: np.ndarray, cx: int, cy: int, r: int) -> float:
    """圆环模板相关性评分。"""
    tmpl_size = int(getattr(config, 'AUTO_DETECT_TEMPLATE_SIZE', 64))
    side = max(24, int(round(r * 2.6)))
    patch = cv2.getRectSubPix(edges, (side, side), (float(cx), float(cy)))
    if patch is None or patch.size == 0:
        return 0.0

    patch = cv2.resize(patch, (tmpl_size, tmpl_size), interpolation=cv2.INTER_AREA)
    patch = patch.astype(np.float32) / 255.0
    tmpl = _get_ring_template(tmpl_size)
    score = float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
    return max(0.0, min(1.0, (score + 1.0) * 0.5))


def _texture_score(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """圆内纹理方差评分，避免空白圆形 UI 误判。"""
    h, w = gray.shape[:2]
    pad = max(4, int(r * 1.1))
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(w, cx + pad)
    y2 = min(h, cy + pad)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    inner_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (max(2, int(r * 0.82)) ** 2)
    if not np.any(inner_mask):
        return 0.0
    std_val = float(np.std(roi[inner_mask]))
    return max(0.0, min(1.0, std_val / 48.0))


def _circle_score(w: int, h: int, cx: int, cy: int, r: int, layout: str) -> float:
    """circle 项评分：位置先验 + 半径合理性。"""
    base = float(min(h, w))
    if base <= 0:
        return 0.0

    right = cx / max(w, 1.0)
    left = 1.0 - right
    top = 1.0 - (cy / max(h, 1.0))

    if layout == 'top-right':
        pos_score = 0.65 * right + 0.35 * top
    elif layout == 'top-left':
        pos_score = 0.65 * left + 0.35 * top
    else:
        # 全图回退：保留轻微“靠上”先验，但不强制右上角
        pos_score = 0.55 * top + 0.45 * max(left, right)

    min_ratio = float(getattr(config, 'AUTO_DETECT_MIN_RADIUS_RATIO', 0.035))
    max_ratio = float(getattr(config, 'AUTO_DETECT_MAX_RADIUS_RATIO', 0.16))
    target_ratio = (min_ratio + max_ratio) * 0.5
    r_ratio = r / base
    half_span = max((max_ratio - min_ratio) * 0.6, 1e-6)
    size_score = max(0.0, 1.0 - abs(r_ratio - target_ratio) / half_span)

    return 0.6 * pos_score + 0.4 * size_score


def _prefilter_candidates(gray: np.ndarray, edges: np.ndarray, candidates: list[dict]) -> list[dict]:
    """
    候选预筛：先用便宜的 circle + edge 分数筛掉大部分候选，
    再对少量候选做模板/纹理等高成本评分。
    """
    h, w = gray.shape[:2]
    prefiltered = []
    for candidate in candidates:
        cx, cy, r = candidate['cx'], candidate['cy'], candidate['r']
        layout = candidate['layout']
        circle = _circle_score(w, h, cx, cy, r, layout)
        edge = _edge_score(edges, cx, cy, r)
        quick = 0.62 * circle + 0.38 * edge
        pre = dict(candidate)
        pre.update({
            'circle_score': circle,
            'edge_score': edge,
            'quick_score': quick,
        })
        prefiltered.append(pre)

    prefiltered = _dedupe_candidates(prefiltered, score_key='quick_score')
    limit = max(1, int(getattr(config, 'AUTO_DETECT_MAX_REFINED_CANDIDATES', 6)))
    return prefiltered[:limit]


def _score_candidate(gray: np.ndarray, edges: np.ndarray, candidate: dict) -> dict:
    """对单个候选综合打分。"""
    h, w = gray.shape[:2]
    cx, cy, r = candidate['cx'], candidate['cy'], candidate['r']
    layout = candidate['layout']

    circle = float(candidate.get('circle_score', _circle_score(w, h, cx, cy, r, layout)))
    edge = float(candidate.get('edge_score', _edge_score(edges, cx, cy, r)))
    template = _template_score(edges, cx, cy, r)
    texture = _texture_score(gray, cx, cy, r)

    final = 0.30 * circle + 0.28 * edge + 0.27 * template + 0.15 * texture
    scored = dict(candidate)
    scored.update({
        'circle_score': circle,
        'edge_score': edge,
        'template_score': template,
        'texture_score': texture,
        'score': final,
    })
    return scored


def _dedupe_candidates(scored: list[dict], score_key: str = 'score') -> list[dict]:
    """按空间位置与半径去重，保留高分候选。"""
    deduped: list[dict] = []
    for cand in sorted(scored, key=lambda d: d.get(score_key, 0.0), reverse=True):
        duplicate = False
        for kept in deduped:
            dist = math.hypot(cand['cx'] - kept['cx'], cand['cy'] - kept['cy'])
            if dist <= 14 and abs(cand['r'] - kept['r']) <= 10:
                duplicate = True
                break
        if not duplicate:
            deduped.append(cand)
    return deduped


def detect_minimap_circle(img_bgr: np.ndarray) -> dict | None:
    """单帧检测：返回归一化小地图圆信息或 None。"""
    if img_bgr is None or img_bgr.size == 0:
        return None

    h0, w0 = img_bgr.shape[:2]
    if h0 < 32 or w0 < 32:
        return None

    work, scale = _resize_for_detection(img_bgr)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blur, 60, 160)

    raw_candidates: list[dict] = []
    for layout, (x1, y1, x2, y2), param2 in _layout_regions(w, h):
        raw_candidates.extend(_collect_hough_candidates(blur, x1, y1, x2, y2, layout, param2))

    if not raw_candidates:
        return None

    # 先做便宜的预筛，再做昂贵的模板/纹理评分，减少整帧自动识别的 CPU 开销
    refined_candidates = _prefilter_candidates(gray, edges, raw_candidates)
    scored = [_score_candidate(gray, edges, cand) for cand in refined_candidates]
    deduped = _dedupe_candidates(scored, score_key='score')
    if not deduped:
        return None

    best = deduped[0]
    min_score = float(getattr(config, 'AUTO_DETECT_MIN_SCORE', 0.26))
    if best['score'] < min_score:
        return None

    px = int(round(best['cx'] / scale))
    py = int(round(best['cy'] / scale))
    pr = int(round(best['r'] / scale))
    frame_h, frame_w = img_bgr.shape[:2]

    return {
        'cx': max(0.0, min(1.0, px / max(frame_w, 1))),
        'cy': max(0.0, min(1.0, py / max(frame_h, 1))),
        'r': max(0.02, min(0.49, pr / max(min(frame_w, frame_h), 1))),
        'px': px,
        'py': py,
        'pr': pr,
        'confidence': round(float(best['score']), 3),
        'layout': best['layout'],
        'frame_width': frame_w,
        'frame_height': frame_h,
        'components': {
            'circle': round(float(best['circle_score']), 3),
            'edge': round(float(best['edge_score']), 3),
            'template': round(float(best['template_score']), 3),
            'texture': round(float(best['texture_score']), 3),
        },
    }


def detect_minimap_circle_batch(frames: list[np.ndarray]) -> dict | None:
    """多帧投票检测：用于降低单帧抖动/动画/特效造成的误检。"""
    valid_frames = [f for f in frames if f is not None and getattr(f, 'size', 0) > 0]
    if not valid_frames:
        return None

    detections = [det for det in (detect_minimap_circle(f) for f in valid_frames) if det is not None]
    if not detections:
        return None

    if len(valid_frames) == 1 or len(detections) == 1:
        result = dict(max(detections, key=lambda d: d['confidence']))
        result['votes'] = 1
        result['frames_received'] = len(valid_frames)
        result['detections_found'] = len(detections)
        result['scatter_px'] = 0.0
        result['components'] = dict(result.get('components', {}))
        result['components']['stability'] = 1.0
        return result

    pos_tol = int(getattr(config, 'AUTO_DETECT_VOTE_POSITION_TOLERANCE', 24))
    radius_tol = int(getattr(config, 'AUTO_DETECT_VOTE_RADIUS_TOLERANCE', 18))
    required_votes = max(2, min(int(getattr(config, 'AUTO_DETECT_MIN_VOTES', 3)), len(detections)))

    best_cluster: list[dict] = []
    best_cluster_score = -1.0
    for ref in detections:
        cluster = [
            d for d in detections
            if math.hypot(d['px'] - ref['px'], d['py'] - ref['py']) <= pos_tol
            and abs(d['pr'] - ref['pr']) <= radius_tol
        ]
        mean_conf = sum(d['confidence'] for d in cluster) / max(len(cluster), 1)
        cluster_score = len(cluster) * 10.0 + mean_conf
        if cluster_score > best_cluster_score:
            best_cluster = cluster
            best_cluster_score = cluster_score

    if len(best_cluster) < required_votes:
        return None

    med_px = int(round(float(np.median([d['px'] for d in best_cluster]))))
    med_py = int(round(float(np.median([d['py'] for d in best_cluster]))))
    med_pr = int(round(float(np.median([d['pr'] for d in best_cluster]))))
    scatter = max(math.hypot(d['px'] - med_px, d['py'] - med_py) for d in best_cluster)
    stability = max(0.0, min(1.0, 1.0 - scatter / max(pos_tol * 1.5, 1.0)))

    representative = min(
        best_cluster,
        key=lambda d: math.hypot(d['px'] - med_px, d['py'] - med_py) + abs(d['pr'] - med_pr),
    )
    frame_w = representative['frame_width']
    frame_h = representative['frame_height']
    mean_conf = sum(d['confidence'] for d in best_cluster) / len(best_cluster)
    final_conf = max(0.0, min(1.0, mean_conf * (0.65 + 0.35 * stability)))

    return {
        'cx': max(0.0, min(1.0, med_px / max(frame_w, 1))),
        'cy': max(0.0, min(1.0, med_py / max(frame_h, 1))),
        'r': max(0.02, min(0.49, med_pr / max(min(frame_w, frame_h), 1))),
        'px': med_px,
        'py': med_py,
        'pr': med_pr,
        'confidence': round(float(final_conf), 3),
        'layout': representative.get('layout', 'full'),
        'votes': len(best_cluster),
        'frames_received': len(valid_frames),
        'detections_found': len(detections),
        'scatter_px': round(float(scatter), 1),
        'components': {
            'circle': round(float(np.mean([d['components']['circle'] for d in best_cluster])), 3),
            'edge': round(float(np.mean([d['components']['edge'] for d in best_cluster])), 3),
            'template': round(float(np.mean([d['components']['template'] for d in best_cluster])), 3),
            'texture': round(float(np.mean([d['components']['texture'] for d in best_cluster])), 3),
            'stability': round(float(stability), 3),
        },
    }
