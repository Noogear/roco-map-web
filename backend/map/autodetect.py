from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from backend import config

_TEMPLATE_CACHE: dict[int, np.ndarray] = {}
ScopeName = Literal['top_right', 'top_band', 'bottom_left', 'fullscreen', 'custom']


@dataclass
class ScoreContext:
	"""统一的评分上下文，避免重复传递参数。"""
	hsv: np.ndarray
	gray: np.ndarray
	edges: np.ndarray
	w: int
	h: int
	cx: int
	cy: int
	r: int
	
	def roi_rect(self, scale: float = 1.2) -> tuple[int, int, int, int]:
		"""生成以 (cx, cy) 为中心、半径 scale*r 的矩形 ROI."""
		pad = max(4, int(self.r * scale))
		x1 = max(0, self.cx - pad)
		y1 = max(0, self.cy - pad)
		x2 = min(self.w, self.cx + pad)
		y2 = min(self.h, self.cy + pad)
		return x1, y1, x2, y2


def _get_ring_template(size: int) -> np.ndarray:
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


def _resolve_scope(search_scope: str | None) -> ScopeName:
	valid: set[str] = {'top_right', 'top_band', 'bottom_left', 'fullscreen', 'custom'}
	default_scope = str(getattr(config, 'AUTO_DETECT_DEFAULT_SCOPE', 'top_right')).strip().lower()
	if default_scope not in valid:
		default_scope = 'top_right'
	candidate = (search_scope or default_scope).strip().lower()
	return candidate if candidate in valid else default_scope


def _normalized_roi_to_pixels(w: int, h: int, roi_norm: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
	x1n, y1n, x2n, y2n = roi_norm
	x1n = max(0.0, min(1.0, float(x1n)))
	y1n = max(0.0, min(1.0, float(y1n)))
	x2n = max(0.0, min(1.0, float(x2n)))
	y2n = max(0.0, min(1.0, float(y2n)))
	if x2n <= x1n or y2n <= y1n:
		return None
	x1 = int(round(x1n * w))
	y1 = int(round(y1n * h))
	x2 = int(round(x2n * w))
	y2 = int(round(y2n * h))
	if x2 - x1 < 8 or y2 - y1 < 8:
		return None
	return x1, y1, x2, y2


def _layout_regions_with_scope(
	w: int,
	h: int,
	search_scope: ScopeName,
	custom_scope_norm: tuple[float, float, float, float] | None,
	enable_full_fallback: bool,
) -> list[tuple[str, tuple[int, int, int, int], int]]:
	p2_roi = int(getattr(config, 'AUTO_DETECT_HOUGH_PARAM2_ROI', 24))
	p2_full = int(getattr(config, 'AUTO_DETECT_HOUGH_PARAM2_FULL', 20))
	top_right_region = ('top-right', (int(w * 0.50), 0, w, int(h * 0.52)), p2_roi)
	top_left_region = ('top-left', (0, 0, int(w * 0.50), int(h * 0.52)), p2_roi)
	bottom_left_region = ('bottom-left', (0, int(h * 0.48), int(w * 0.50), h), p2_roi)
	full_region = ('full', (0, 0, w, h), p2_full)
	regions: list[tuple[str, tuple[int, int, int, int], int]] = []
	if search_scope == 'top_right':
		regions.append(top_right_region)
	elif search_scope == 'top_band':
		regions.append(top_right_region)
		regions.append(top_left_region)
	elif search_scope == 'bottom_left':
		regions.append(bottom_left_region)
	elif search_scope == 'custom' and custom_scope_norm is not None:
		custom_px = _normalized_roi_to_pixels(w, h, custom_scope_norm)
		if custom_px is not None:
			regions.append(('custom', custom_px, p2_roi))
	elif search_scope == 'fullscreen':
		regions.append(top_right_region)
		regions.append(bottom_left_region)

	if enable_full_fallback or search_scope == 'fullscreen':
		regions.append(full_region)
	return regions


def _collect_hough_candidates(blur_gray: np.ndarray, x1: int, y1: int, x2: int, y2: int, layout: str, param2: int) -> list[dict]:
	roi = blur_gray[y1:y2, x1:x2]
	if roi.size == 0:
		return []
	h, w = blur_gray.shape[:2]
	base = min(h, w)
	min_r = max(10, int(base * float(getattr(config, 'AUTO_DETECT_MIN_RADIUS_RATIO', 0.035))))
	max_r = max(min_r + 6, int(base * float(getattr(config, 'AUTO_DETECT_MAX_RADIUS_RATIO', 0.16))))
	circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(20, min(roi.shape[:2]) // 4), param1=int(getattr(config, 'AUTO_DETECT_HOUGH_PARAM1', 90)), param2=max(8, int(param2)), minRadius=min_r, maxRadius=max_r)
	if circles is None:
		return []
	candidates = []
	for c in np.round(circles[0, :]).astype(int):
		cx, cy, r = int(c[0]) + x1, int(c[1]) + y1, int(c[2])
		if r <= 0:
			continue
		candidates.append({'cx': cx, 'cy': cy, 'r': r, 'layout': layout})
	return candidates


def _edge_score(ctx: ScoreContext) -> float:
	"""圆环边缘响应度评分（Canny 边缘）。"""
	x1, y1, x2, y2 = ctx.roi_rect(1.25)
	roi = ctx.edges[y1:y2, x1:x2]
	if roi.size == 0:
		return 0.0
	yy, xx = np.ogrid[y1:y2, x1:x2]
	dist = np.sqrt((xx - ctx.cx) ** 2 + (yy - ctx.cy) ** 2)
	band = max(2.0, ctx.r * 0.12)
	ring_mask = (dist >= (ctx.r - band)) & (dist <= (ctx.r + band))
	if not np.any(ring_mask):
		return 0.0
	return float(np.mean(roi[ring_mask]) / 255.0)


def _template_score(ctx: ScoreContext) -> float:
	"""模板匹配得分（圆环模板与边缘图匹配度）。"""
	tmpl_size = int(getattr(config, 'AUTO_DETECT_TEMPLATE_SIZE', 64))
	side = max(24, int(round(ctx.r * 2.6)))
	patch = cv2.getRectSubPix(ctx.edges, (side, side), (float(ctx.cx), float(ctx.cy)))
	if patch is None or patch.size == 0:
		return 0.0
	patch = cv2.resize(patch, (tmpl_size, tmpl_size), interpolation=cv2.INTER_AREA)
	patch = patch.astype(np.float32) / 255.0
	tmpl = _get_ring_template(tmpl_size)
	score = float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
	return max(0.0, min(1.0, (score + 1.0) * 0.5))


def _texture_score(ctx: ScoreContext) -> float:
	"""内部纹理复杂度评分（圆内标准差）。"""
	x1, y1, x2, y2 = ctx.roi_rect(1.1)
	roi = ctx.gray[y1:y2, x1:x2]
	if roi.size == 0:
		return 0.0
	yy, xx = np.ogrid[y1:y2, x1:x2]
	inner_mask = ((xx - ctx.cx) ** 2 + (yy - ctx.cy) ** 2) <= (max(2, int(ctx.r * 0.82)) ** 2)
	if not np.any(inner_mask):
		return 0.0
	std_val = float(np.std(roi[inner_mask]))
	return max(0.0, min(1.0, std_val / 48.0))


def _ring_color_score(ctx: ScoreContext) -> float:
	"""圆环颜色匹配得分（偏黄/偏绿 UI 边框）。"""
	x1, y1, x2, y2 = ctx.roi_rect(1.2)
	if x2 <= x1 or y2 <= y1:
		return 0.0
	hsv = ctx.hsv[y1:y2, x1:x2]
	if hsv.size == 0:
		return 0.0
	H = hsv[:, :, 0]
	S = hsv[:, :, 1]
	V = hsv[:, :, 2]
	yy, xx = np.ogrid[y1:y2, x1:x2]
	dist = np.sqrt((xx - ctx.cx) ** 2 + (yy - ctx.cy) ** 2)
	ring = (dist >= (ctx.r * 0.88)) & (dist <= (ctx.r * 1.12))
	if not np.any(ring):
		return 0.0
	yellow = (H >= 12) & (H <= 50) & (S >= 55) & (V >= 85)
	green = (H >= 50) & (H <= 95) & (S >= 45) & (V >= 70)
	target = yellow | green
	den = float(np.sum(ring))
	num = float(np.sum(ring & target))
	if den <= 1e-6:
		return 0.0
	return max(0.0, min(1.0, num / den))


def _ring_continuity_score(ctx: ScoreContext, bins: int = 36) -> float:
	"""圆环边缘连续性得分（角度覆盖率）。"""
	x1, y1, x2, y2 = ctx.roi_rect(1.2)
	if x2 <= x1 or y2 <= y1:
		return 0.0
	roi = ctx.edges[y1:y2, x1:x2]
	if roi.size == 0:
		return 0.0
	gy, gx = np.where(roi > 0)
	if len(gx) == 0:
		return 0.0
	x = gx + x1
	y = gy + y1
	dx = x - float(ctx.cx)
	dy = y - float(ctx.cy)
	d = np.sqrt(dx * dx + dy * dy)
	mask = (d >= (ctx.r * 0.86)) & (d <= (ctx.r * 1.14))
	if not np.any(mask):
		return 0.0
	ang = (np.arctan2(dy[mask], dx[mask]) + 2.0 * math.pi) % (2.0 * math.pi)
	idx = np.floor(ang / (2.0 * math.pi) * bins).astype(np.int32)
	occ = np.zeros((bins,), dtype=np.uint8)
	occ[np.clip(idx, 0, bins - 1)] = 1
	return float(np.mean(occ))


def _apply_post_validation(best: dict, use_layout_prior: bool) -> bool:
	"""全屏模式下的后验校验（统一框架）。"""
	if use_layout_prior:
		return True  # 布局先验模式不需要额外校验
	
	post_val = getattr(config, 'AUTO_DETECT_POST_VALIDATION', {})
	if not post_val:
		return True  # 无配置时默认通过
	
	# 条件 1: corner_any 得分下限
	min_corner = post_val.get('min_corner_any_score', 0.24)
	if float(best.get('corner_any_score', 0.0)) < min_corner:
		return False
	
	# 条件 2: ring_continuity 得分下限
	min_cont = post_val.get('min_ring_continuity', 0.42)
	if float(best.get('ring_continuity_score', 0.0)) < min_cont:
		return False
	
	# 条件 3: 复合认证得分
	auth_w = post_val.get('auth_score_weights', {})
	auth = (auth_w.get('ring_continuity', 0.45) * float(best.get('ring_continuity_score', 0.0)) +
			auth_w.get('template', 0.35) * float(best.get('template_score', 0.0)) +
			auth_w.get('ui_radius', 0.20) * float(best.get('ui_radius_score', 0.0)))
	min_auth = post_val.get('min_auth_score', 0.50)
	if auth < min_auth:
		return False
	
	# 条件 4: 场景伪圆拒绝
	for rule in post_val.get('scene_artifact_rules', []):
		texture_min = rule.get('texture_min', 0.90)
		edge_max = rule.get('edge_max', 0.08)
		if (float(best.get('texture_score', 0.0)) > texture_min and
			float(best.get('edge_score', 0.0)) < edge_max):
			return False
	
	# 条件 5: 布局特定规则
	layout = str(best.get('layout', ''))
	layout_rules = post_val.get('layout_specific_rules', {})
	if layout in layout_rules:
		rule = layout_rules[layout]
		ring_color = float(best.get('ring_color_score', 0.0))
		ring_cont = float(best.get('ring_continuity_score', 0.0))
		if 'ring_color_max' in rule and 'ring_continuity_min' in rule:
			if ring_color < float(rule['ring_color_max']) and ring_cont < float(rule['ring_continuity_min']):
				return False
	
	return True


def _circle_score(w: int, h: int, cx: int, cy: int, r: int, layout: str) -> float:
	base = float(min(h, w))
	if base <= 0:
		return 0.0
	right = cx / max(w, 1.0)
	left = 1.0 - right
	top = 1.0 - (cy / max(h, 1.0))
	bottom = cy / max(h, 1.0)
	if layout == 'top-right':
		pos_score = 0.65 * right + 0.35 * top
	elif layout == 'top-left':
		pos_score = 0.65 * left + 0.35 * top
	elif layout == 'bottom-right':
		pos_score = 0.65 * right + 0.35 * bottom
	elif layout == 'bottom-left':
		pos_score = 0.65 * left + 0.35 * bottom
	else:
		pos_score = 0.45 * max(top, bottom) + 0.55 * max(left, right)
	min_ratio = float(getattr(config, 'AUTO_DETECT_MIN_RADIUS_RATIO', 0.035))
	max_ratio = float(getattr(config, 'AUTO_DETECT_MAX_RADIUS_RATIO', 0.16))
	target_ratio = (min_ratio + max_ratio) * 0.5
	r_ratio = r / base
	half_span = max((max_ratio - min_ratio) * 0.6, 1e-6)
	size_score = max(0.0, 1.0 - abs(r_ratio - target_ratio) / half_span)
	edge_dist = min(cx, cy, w - cx, h - cy)
	if edge_dist < r * 0.5:
		edge_penalty = 0.3
	elif edge_dist < r:
		edge_penalty = 0.7 + 0.3 * ((edge_dist - r * 0.5) / max(r * 0.5, 1.0))
	else:
		edge_penalty = 1.0
	return (0.6 * pos_score + 0.4 * size_score) * edge_penalty


def _anchor_score(w: int, h: int, cx: int, cy: int, r: int) -> float:
	"""基于 HUD 结构的右上角锚点评分。
	
	经验上小地图圆心满足：
	- 距离右边界约 1.0~1.9 个半径
	- 距离顶部约 1.2~2.6 个半径
	
	该评分为软约束，不直接一票否决，避免不同 UI 缩放导致漏检。
	"""
	r = max(1, int(r))
	right_gap = max(0.0, float(w - cx))
	top_gap = max(0.0, float(cy))
	right_ratio = right_gap / float(r)
	top_ratio = top_gap / float(r)

	# 目标值与容忍半径采用较宽范围，保证跨分辨率稳定
	right_term = max(0.0, 1.0 - abs(right_ratio - 1.45) / 1.25)
	top_term = max(0.0, 1.0 - abs(top_ratio - 1.85) / 1.35)
	return 0.65 * right_term + 0.35 * top_term


def _ui_radius_score(w: int, h: int, r: int) -> float:
	"""小地图在屏幕中的典型半径比例评分（分辨率无关）。"""
	base = float(min(h, w))
	if base <= 1:
		return 0.0
	r_ratio = float(r) / base
	# 经验目标区间约 0.055~0.105，中心值 0.078
	target = 0.078
	half_span = 0.035
	return max(0.0, 1.0 - abs(r_ratio - target) / half_span)


def _corner_any_score(w: int, h: int, cx: int, cy: int, r: int) -> float:
	"""与具体象限无关的角落贴近度评分（支持右上/左下等任意角落）。"""
	r = max(1.0, float(r))
	corners = [(0.0, 0.0), (float(w), 0.0), (0.0, float(h)), (float(w), float(h))]
	best = 0.0
	for x0, y0 in corners:
		d = math.hypot(float(cx) - x0, float(cy) - y0)
		ratio = d / r
		# UI 小地图通常离角落约 1.8~3.3 个半径
		s = max(0.0, 1.0 - abs(ratio - 2.45) / 1.35)
		if s > best:
			best = s
	return best


def _dedupe_candidates(scored: list[dict], score_key: str = 'score') -> list[dict]:
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


def _compute_final_score(components: dict, use_layout_prior: bool) -> float:
	"""从 9 分量及权重表计算最终得分（配置驱动）。"""
	weights_key = 'layout_prior' if use_layout_prior else 'fullscreen'
	weights_dict = getattr(config, 'AUTO_DETECT_WEIGHTS', {})[weights_key]
	return (weights_dict.get('circle', 0.11) * components.get('circle', 0.0) +
			weights_dict.get('edge', 0.10) * components.get('edge', 0.0) +
			weights_dict.get('template', 0.16) * components.get('template', 0.0) +
			weights_dict.get('texture', 0.11) * components.get('texture', 0.0) +
			weights_dict.get('anchor', 0.0) * components.get('anchor', 0.0) +
			weights_dict.get('ring_color', 0.14) * components.get('ring_color', 0.0) +
			weights_dict.get('ring_continuity', 0.14) * components.get('ring_continuity', 0.0) +
			weights_dict.get('ui_radius', 0.14) * components.get('ui_radius', 0.0) +
			weights_dict.get('corner_any', 0.10) * components.get('corner_any', 0.0))


def _prefilter_candidates(gray: np.ndarray, edges: np.ndarray, candidates: list[dict], w: int, h: int, hsv: np.ndarray, use_layout_prior: bool) -> list[dict]:

	# 布局先验启用时，优先贴近右上锚点的候选；全屏模式则不过滤位置。
	if use_layout_prior:
		anchored = []
		for c in candidates:
			r = max(1, int(c['r']))
			right_gap_ratio = (w - int(c['cx'])) / float(r)
			top_gap_ratio = int(c['cy']) / float(r)
			if right_gap_ratio <= 2.0 and top_gap_ratio <= 2.8:
				anchored.append(c)
		if anchored:
			candidates = anchored

	prefiltered = []
	quick_weights = getattr(config, 'AUTO_DETECT_QUICK_WEIGHTS', {})
	quick_layout = quick_weights.get('layout_prior', {})
	quick_full = quick_weights.get('fullscreen', {})
	for candidate in candidates:
		cx, cy, r = candidate['cx'], candidate['cy'], candidate['r']
		layout = candidate['layout']
		ctx = ScoreContext(hsv=hsv, gray=gray, edges=edges, w=w, h=h, cx=cx, cy=cy, r=r)
		circle = _circle_score(w, h, cx, cy, r, layout)
		edge = _edge_score(ctx)
		anchor = _anchor_score(w, h, cx, cy, r)
		if use_layout_prior:
			quick = (
				float(quick_layout.get('circle', 0.50)) * circle
				+ float(quick_layout.get('edge', 0.28)) * edge
				+ float(quick_layout.get('anchor', 0.22)) * anchor
			)
		else:
			quick = (
				float(quick_full.get('circle', 0.62)) * circle
				+ float(quick_full.get('edge', 0.38)) * edge
			)
		pre = dict(candidate)
		pre.update({'circle_score': circle, 'edge_score': edge, 'anchor_score': anchor, 'quick_score': quick})
		prefiltered.append(pre)
	prefiltered = _dedupe_candidates(prefiltered, score_key='quick_score')
	limit = max(1, int(getattr(config, 'AUTO_DETECT_MAX_REFINED_CANDIDATES', 5)))
	return prefiltered[:limit]


def _score_candidate(candidate: dict, use_layout_prior: bool, w: int, h: int, hsv_img: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> dict:
	cx, cy, r = candidate['cx'], candidate['cy'], candidate['r']
	layout = candidate['layout']
	ctx = ScoreContext(hsv=hsv_img, gray=gray, edges=edges, w=w, h=h, cx=cx, cy=cy, r=r)
	circle = float(candidate.get('circle_score', _circle_score(w, h, cx, cy, r, layout)))
	edge = float(candidate.get('edge_score', _edge_score(ctx)))
	anchor = float(candidate.get('anchor_score', _anchor_score(w, h, cx, cy, r)))
	template = _template_score(ctx)
	texture = _texture_score(ctx)
	ring_color = _ring_color_score(ctx)
	ring_cont = _ring_continuity_score(ctx)
	ui_radius = _ui_radius_score(w, h, r)
	corner_any = _corner_any_score(w, h, cx, cy, r)

	components = {'circle': circle, 'edge': edge, 'template': template, 'texture': texture, 'anchor': anchor, 'ring_color': ring_color, 'ring_continuity': ring_cont, 'ui_radius': ui_radius, 'corner_any': corner_any}
	final = _compute_final_score(components, use_layout_prior)

	scored = dict(candidate)
	scored.update({'circle_score': circle, 'edge_score': edge, 'anchor_score': anchor, 'template_score': template, 'texture_score': texture, 'ring_color_score': ring_color, 'ring_continuity_score': ring_cont, 'ui_radius_score': ui_radius, 'corner_any_score': corner_any, 'score': final})
	return scored


def detect_minimap_circle(
	img_bgr: np.ndarray,
	debug: bool = False,
	search_scope: str | None = None,
	custom_scope_norm: tuple[float, float, float, float] | None = None,
	enable_full_fallback: bool | None = None,
) -> dict | None:
	"""检测小地图圆形区域。

	Args:
		img_bgr: 输入 BGR 图像。
		debug: 是否输出候选统计。
		search_scope: 匹配范围，可选
			- 'top_right' (默认，低延迟)
			- 'top_band'
			- 'bottom_left'
			- 'fullscreen'
			- 'custom'（需配合 custom_scope_norm）
		custom_scope_norm: 自定义归一化 ROI=(x1,y1,x2,y2)，范围 [0,1]。
		enable_full_fallback: 指定范围失败后是否回退全图检测。

	Returns:
		检测结果字典；未检测到返回 None。
	"""
	if img_bgr is None or img_bgr.size == 0:
		return None
	h0, w0 = img_bgr.shape[:2]
	if h0 < 32 or w0 < 32:
		return None
	work, scale = _resize_for_detection(img_bgr)
	h, w = work.shape[:2]
	gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
	blur = cv2.GaussianBlur(gray, (9, 9), 2)
	edges = cv2.Canny(blur, 60, 160)
	raw_candidates: list[dict] = []
	scope = _resolve_scope(search_scope)
	if enable_full_fallback is None:
		enable_full_fallback = bool(getattr(config, 'AUTO_DETECT_ENABLE_FULL_FALLBACK', False))
	use_layout_prior = scope in {'top_right', 'top_band'}
	for layout, (x1, y1, x2, y2), param2 in _layout_regions_with_scope(w, h, scope, custom_scope_norm, bool(enable_full_fallback)):
		found = _collect_hough_candidates(blur, x1, y1, x2, y2, layout, param2)
		if debug:
			print(f'  [autodetect] scope={scope} {layout}: {len(found)} candidates')
		raw_candidates.extend(found)
	if not raw_candidates:
		return None
	refined_candidates = _prefilter_candidates(gray, edges, raw_candidates, w, h, hsv, use_layout_prior)
	scored = [_score_candidate(cand, use_layout_prior, w, h, hsv, gray, edges) for cand in refined_candidates]
	deduped = _dedupe_candidates(scored, score_key='score')
	if not deduped:
		return None
	if use_layout_prior:
		_min_cx_ratio = 0.75
		_max_cy_ratio = 0.35
		filtered = []
		for c in deduped:
			norm_cx = (c['cx'] / scale) / max(w0, 1)
			norm_cy = (c['cy'] / scale) / max(h0, 1)
			if norm_cx >= _min_cx_ratio and norm_cy <= _max_cy_ratio:
				corner_dist = ((1.0 - norm_cx) ** 2 + norm_cy ** 2) ** 0.5
				corner_bonus = max(0, 0.15 * (1.0 - corner_dist))
				c['_corner_bonus'] = corner_bonus
				c['_adjusted_score'] = c['score'] + corner_bonus
				filtered.append(c)
		if not filtered:
			return None
	else:
		filtered = list(deduped)
		for c in filtered:
			c['_adjusted_score'] = c['score']

	filtered.sort(key=lambda c: c.get('_adjusted_score', c['score']), reverse=True)
	best = filtered[0]
	min_score = float(getattr(config, 'AUTO_DETECT_MIN_SCORE', 0.26))
	if best['score'] < min_score:
		return None

	if not _apply_post_validation(best, use_layout_prior):
		return None

	px = int(round(best['cx'] / scale))
	py = int(round(best['cy'] / scale))
	pr = int(round(best['r'] / scale))
	frame_h, frame_w = img_bgr.shape[:2]
	return {'cx': max(0.0, min(1.0, px / max(frame_w, 1))), 'cy': max(0.0, min(1.0, py / max(frame_h, 1))), 'r': max(0.02, min(0.49, pr / max(min(frame_w, frame_h), 1))), 'px': px, 'py': py, 'pr': pr, 'confidence': round(float(best['score']), 3), 'layout': best['layout'], 'frame_width': frame_w, 'frame_height': frame_h, 'components': {'circle': round(float(best['circle_score']), 3), 'edge': round(float(best['edge_score']), 3), 'anchor': round(float(best.get('anchor_score', 0.0)), 3), 'template': round(float(best['template_score']), 3), 'texture': round(float(best['texture_score']), 3), 'ring_color': round(float(best.get('ring_color_score', 0.0)), 3), 'ring_continuity': round(float(best.get('ring_continuity_score', 0.0)), 3), 'ui_radius': round(float(best.get('ui_radius_score', 0.0)), 3), 'corner_any': round(float(best.get('corner_any_score', 0.0)), 3)}}


def detect_minimap_circle_batch(
	frames: list[np.ndarray],
	search_scope: str | None = None,
	custom_scope_norm: tuple[float, float, float, float] | None = None,
	enable_full_fallback: bool | None = None,
) -> dict | None:
	valid_frames = [f for f in frames if f is not None and getattr(f, 'size', 0) > 0]
	if not valid_frames:
		return None
	detections = [
		det
		for det in (
			detect_minimap_circle(
				f,
				search_scope=search_scope,
				custom_scope_norm=custom_scope_norm,
				enable_full_fallback=enable_full_fallback,
			)
			for f in valid_frames
		)
		if det is not None
	]
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
		cluster = [d for d in detections if math.hypot(d['px'] - ref['px'], d['py'] - ref['py']) <= pos_tol and abs(d['pr'] - ref['pr']) <= radius_tol]
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
	representative = min(best_cluster, key=lambda d: math.hypot(d['px'] - med_px, d['py'] - med_py) + abs(d['pr'] - med_pr))
	frame_w = representative['frame_width']
	frame_h = representative['frame_height']
	mean_conf = sum(d['confidence'] for d in best_cluster) / len(best_cluster)
	final_conf = max(0.0, min(1.0, mean_conf * (0.65 + 0.35 * stability)))
	return {'cx': max(0.0, min(1.0, med_px / max(frame_w, 1))), 'cy': max(0.0, min(1.0, med_py / max(frame_h, 1))), 'r': max(0.02, min(0.49, med_pr / max(min(frame_w, frame_h), 1))), 'px': med_px, 'py': med_py, 'pr': med_pr, 'confidence': round(float(final_conf), 3), 'layout': representative.get('layout', 'full'), 'votes': len(best_cluster), 'frames_received': len(valid_frames), 'detections_found': len(detections), 'scatter_px': round(float(scatter), 1), 'components': {'circle': round(float(np.mean([d['components']['circle'] for d in best_cluster])), 3), 'edge': round(float(np.mean([d['components']['edge'] for d in best_cluster])), 3), 'anchor': round(float(np.mean([d['components'].get('anchor', 0.0) for d in best_cluster])), 3), 'template': round(float(np.mean([d['components']['template'] for d in best_cluster])), 3), 'texture': round(float(np.mean([d['components']['texture'] for d in best_cluster])), 3), 'ring_color': round(float(np.mean([d['components'].get('ring_color', 0.0) for d in best_cluster])), 3), 'ring_continuity': round(float(np.mean([d['components'].get('ring_continuity', 0.0) for d in best_cluster])), 3), 'ui_radius': round(float(np.mean([d['components'].get('ui_radius', 0.0) for d in best_cluster])), 3), 'corner_any': round(float(np.mean([d['components'].get('corner_any', 0.0) for d in best_cluster])), 3), 'stability': round(float(stability), 3)}}
