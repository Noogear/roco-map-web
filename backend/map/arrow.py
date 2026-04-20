"""
backend/map/arrow.py - 玩家方向箭头检测核心算法

从小地图裁剪区（crop_bgr）提取玩家箭头朝向（罗盘角，上=0° 顺时针）。

公开接口
--------
detect_arrow(crop_bgr, prev_stable_angle=None) -> dict | None
    从裁剪区直接检测。

detect_arrow_from_frame(frame_bgr, mx, my, mr, prev_stable_angle=None) -> dict | None
    从完整帧 + 小地图圆心坐标检测（内部自动截取裁剪区）。

返回字典字段
------------
angle_deg   : float  罗盘角（0-360，上=0° 顺时针）
aspect      : float  minAreaRect 长宽比（兼容旧接口）
skewness    : float  PCA 主轴偏度（正/负决定尖端方向）
mask_pixels : int    有效黄色像素数
warning     : str|None  "asp_low"/"shield_occluded"/"[amb_wb]" 等
head_xy     : tuple  箭尖在裁剪坐标系内的坐标（可视化用）
tail_xy     : tuple  箭尾在裁剪坐标系内的坐标（可视化用）
centroid    : tuple  黄色质心坐标（裁剪坐标系）
rect        : cv2.RotatedRect  原始 minAreaRect（可视化用）
"""
from __future__ import annotations

import math
import sys
import cv2
import numpy as np
from pathlib import Path

# 导入项目根目录模块
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from path_config import PLAYER_ARROW_TEMPLATE_NPY_PATH


# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------

ARROW_CROP_RADIUS = 20   # 裁剪半径（像素）；实际取 max(ARROW_CROP_RADIUS, mr*0.25)
APERTURE_RADIUS   = 15   # 黄色检测光圈半径

YELLOW_HSV_LOWER  = np.array([10, 110, 140], dtype=np.uint8)
YELLOW_HSV_UPPER  = np.array([30, 255, 255], dtype=np.uint8)
YELLOW_HSV_LOWER2 = np.array([ 8,  60, 150], dtype=np.uint8)
YELLOW_HSV_UPPER2 = np.array([35, 255, 255], dtype=np.uint8)

WHITE_HSV_LOWER = np.array([  0,  0, 185], dtype=np.uint8)
WHITE_HSV_UPPER = np.array([180, 45, 255], dtype=np.uint8)
WHITE_APERTURE_EXTRA = 8

# 黄白双色共现强约束
ENABLE_STRONG_YW_CONSTRAINT = True
WHITE_MIN_PIXELS = 5
WHITE_YELLOW_MIN_RATIO = 0.05

# 形态学梯度锐边约束
ENABLE_GRADIENT_CONSTRAINT = True
GRADIENT_THRESHOLD = 30
GRADIENT_DILATE_ITERS = 2

GAUSSIAN_CENTER_SIGMA = 5.0

MIN_ARROW_PIXELS    = 10
MAX_ARROW_PIXELS    = 600
MIN_ASPECT_RATIO    = 1.15   # effective_aspect（PCA 伸长率与 minAreaRect 取最大）低于此值时告警
MIN_ASPECT_RATIO_HARD = 1.08  # minAreaRect aspect 低于此值时直接拒绝（PCA 轴不可信）
SHIELD_PIXEL_THRESH  = 450
SHIELD_ASPECT_THRESH = 1.20

# 歧义帧时序融合
ENABLE_TEMPORAL_BLEND = True
AMB_SKEW_ABS   = 0.10
AMB_ASPECT_MIN = 1.00
AMB_ASPECT_MAX = 1.55
AMB_CURR_WEIGHT = 0.50

# 弱偏度反转保护（时序管线用）
ANTIFLIP_SKEW_THRESH  = 0.25   # |skewness| < 此值时视为方向不可信
ANTIFLIP_ANGLE_THRESH = 155.0  # 与上一稳定角相差超过此值才触发翻转

# ---------------------------------------------------------------------------
# 模板消歧配置（利用图标形状区分箭头头/尾方向）
# ---------------------------------------------------------------------------

TEMPLATE_SIZE = 40                    # 内部使用的模板边长（像素）
TEMPLATE_MIN_DIFF_RATIO = 0.04        # 正/反模板得分差 < 4% 时放弃（差异不可信）
TEMPLATE_NPY = PLAYER_ARROW_TEMPLATE_NPY_PATH

_tmpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # size → (tmpl, tmpl_flip)
_rot_tmpl_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # (angle_5deg, size) → (rot_up, rot_down)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _blend_angles_deg(prev_deg: float, curr_deg: float, curr_weight: float) -> float:
    """圆周空间加权融合角度，避免 0/360 跨界抖动。"""
    wp = max(0.0, 1.0 - curr_weight)
    wc = max(0.0, curr_weight)
    rp, rc = math.radians(prev_deg), math.radians(curr_deg)
    x = wp * math.cos(rp) + wc * math.cos(rc)
    y = wp * math.sin(rp) + wc * math.sin(rc)
    if abs(x) < 1e-8 and abs(y) < 1e-8:
        return curr_deg
    return math.degrees(math.atan2(y, x)) % 360.0


def _make_arrow_template(size: int = TEMPLATE_SIZE) -> np.ndarray:
    """运行时兜底模板（头朝上），仅用于 npy 缺失或损坏时。"""
    img = np.zeros((size, size), dtype=np.float32)
    c = size // 2

    head_bot = int(size * 0.57)
    head_hw = int(size * 0.46)
    cv2.fillPoly(
        img,
        [
            np.array(
                [
                    [c, max(1, int(size * 0.04))],
                    [c - head_hw, head_bot],
                    [c + head_hw, head_bot],
                ],
                dtype=np.int32,
            )
        ],
        1.0,
    )

    tail_hw = int(size * 0.15)
    cv2.fillPoly(
        img,
        [
            np.array(
                [
                    [c, int(size * 0.60)],
                    [c - tail_hw, int(size * 0.78)],
                    [c, int(size * 0.96)],
                    [c + tail_hw, int(size * 0.78)],
                ],
                dtype=np.int32,
            )
        ],
        0.75,
    )
    return img


def _get_templates(size: int = TEMPLATE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (tmpl_head_up, tmpl_head_down)，带模块级缓存。
    运行时仅依赖 .npy（由构建流程生成）；缺失时回退合成模板。
    """
    if size in _tmpl_cache:
        return _tmpl_cache[size]

    tmpl: np.ndarray | None = None
    if TEMPLATE_NPY.exists():
        try:
            raw_npy = np.load(str(TEMPLATE_NPY)).astype(np.float32)
            if raw_npy.ndim == 2 and raw_npy.max() > 0:
                tmpl = cv2.resize(raw_npy, (size, size), interpolation=cv2.INTER_AREA)
                tmpl /= tmpl.max()
        except Exception:
            tmpl = None

    if tmpl is None:
        tmpl = _make_arrow_template(size)

    tmpl_flip = np.flipud(tmpl)
    _tmpl_cache[size] = (tmpl, tmpl_flip)
    return tmpl, tmpl_flip


def _template_flip_score(mask: np.ndarray, pca_dir: np.ndarray) -> float | None:
    """
    旋转 mask 使 pca_dir 指向图像正下方，然后与头朝上/头朝下模板做内积比较，
    返回 shield_sign（+1 或 -1），差异不足 TEMPLATE_MIN_DIFF_RATIO 时返回 None。

    shield_sign +1 → 头部在 +pca_dir 方向
    shield_sign -1 → 头部在 -pca_dir 方向
    """
    h, w = mask.shape[:2]
    side = min(h, w, TEMPLATE_SIZE)
    alpha = math.degrees(math.atan2(float(pca_dir[0]), float(pca_dir[1])))

    # 等价变换：旋转模板而非 mask，便于按角度缓存（全程无 warpAffine on mask）
    # 把 pca_dir 转到 [0,1] 等于把模板转 −alpha，缓存步进 5°
    alpha_r = int(round(alpha / 5.0)) * 5
    cache_key = (alpha_r, side)
    if cache_key not in _rot_tmpl_cache:
        tmpl, tmpl_flip = _get_templates(side)
        M = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), -alpha_r, 1.0)
        t_r  = cv2.warpAffine(tmpl,      M, (side, side), flags=cv2.INTER_NEAREST, borderValue=0.0)
        t_fr = cv2.warpAffine(tmpl_flip, M, (side, side), flags=cv2.INTER_NEAREST, borderValue=0.0)
        _rot_tmpl_cache[cache_key] = (t_r, t_fr)
    tmpl_r, tmpl_flip_r = _rot_tmpl_cache[cache_key]

    mask_rs = cv2.resize(mask.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)
    mx = float(mask_rs.max())
    if mx < 1e-6:
        return None
    mask_rs /= mx

    s_up   = float(np.dot(mask_rs.ravel(), tmpl_r.ravel()))
    s_down = float(np.dot(mask_rs.ravel(), tmpl_flip_r.ravel()))

    total = s_up + s_down
    if total < 1e-6 or abs(s_up - s_down) / total < TEMPLATE_MIN_DIFF_RATIO:
        return None

    # 模板方向：tmpl_r 的头部对应 -pca_dir（即图像上方），s_up > s_down → shield_sign = -1
    return -1.0 if s_up > s_down else 1.0


def _compute_strong_edge(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    _, edge = cv2.threshold(grad, GRADIENT_THRESHOLD, 255, cv2.THRESH_BINARY)
    return cv2.dilate(edge, k, iterations=GRADIENT_DILATE_ITERS)


def _extract_yellow_mask(crop_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    h, w = crop_bgr.shape[:2]
    aperture = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(aperture, (w // 2, h // 2), APERTURE_RADIUS, 255, -1)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    strong_edge = _compute_strong_edge(crop_bgr) if ENABLE_GRADIENT_CONSTRAINT else None

    def _build(lower, upper):
        m = cv2.inRange(hsv, lower, upper)
        if strong_edge is not None:
            m = cv2.bitwise_and(m, strong_edge)
        m = cv2.bitwise_and(m, aperture)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = _build(YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    if int(np.count_nonzero(mask)) >= MIN_ARROW_PIXELS:
        return mask, "normal"
    return _build(YELLOW_HSV_LOWER2, YELLOW_HSV_UPPER2), "wide"


def _extract_white_border(crop_bgr: np.ndarray, yellow_mask: np.ndarray,
                          aperture_radius: int) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nearby   = cv2.dilate(yellow_mask, k5, iterations=2)
    excluded = cv2.dilate(yellow_mask, k2, iterations=1)
    ring = cv2.bitwise_and(nearby, cv2.bitwise_not(excluded))
    return cv2.bitwise_and(white, ring)


def _keep_center_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    clabel = int(labels[h // 2, w // 2])
    keep = clabel if clabel > 0 else -1
    if keep <= 0:
        best_id, best_d = -1, 1e18
        for cid in range(1, n):
            if stats[cid, cv2.CC_STAT_AREA] < MIN_ARROW_PIXELS:
                continue
            d2 = (cents[cid, 0] - w * 0.5) ** 2 + (cents[cid, 1] - h * 0.5) ** 2
            if d2 < best_d:
                best_d, best_id = d2, cid
        keep = best_id
    if keep <= 0:
        return mask
    out = np.zeros_like(mask)
    out[labels == keep] = 255
    return out


def _best_arrow_contour(mask: np.ndarray) -> np.ndarray:
    if int(np.count_nonzero(mask)) == 0:
        return mask
    h, w = mask.shape[:2]
    cx0, cy0 = w * 0.5, h * 0.5
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    if int(np.count_nonzero(work)) < MIN_ARROW_PIXELS:
        work = mask.copy()
    n, labels, stats, cents = cv2.connectedComponentsWithStats(work, connectivity=8)
    if n <= 1:
        return mask
    best_id, best_score = -1, -1e9
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_ARROW_PIXELS:
            continue
        ww, hh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        asp = max(ww, hh) / max(min(ww, hh), 1e-6)
        dist = math.hypot(cents[i, 0] - cx0, cents[i, 1] - cy0)
        g = math.exp(-(dist ** 2) / (2.0 * GAUSSIAN_CENTER_SIGMA ** 2))
        ch = 1.0 if abs(cents[i, 0] - cx0) <= w * 0.20 and abs(cents[i, 1] - cy0) <= h * 0.20 else 0.0
        a_s = max(0.0, 1.0 - abs(area - 260.0) / 450.0)
        r_s = min(1.0, (asp - 1.0) / 2.0)
        score = (a_s * 0.4 + r_s * 0.6) * 0.30 + g * 0.60 + ch * 0.10
        if score > best_score:
            best_score, best_id = score, i
    if best_id < 0:
        return mask
    out = np.zeros_like(mask)
    out[labels == best_id] = 255
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)


CURVATURE_K = 12  # 曲率计算的轮廓邻域步长（大K平滑V形凹口等局部凹角）
CURVATURE_MIN_RATIO = 0.70  # tip曲率 / 中值曲率 < 此值才信任曲率结果


def _curvature_arrow_angle(mask: np.ndarray, pts: np.ndarray,
                           cx: float, cy: float,
                           rect, pixels: int) -> dict | None:
    """
    当 minAreaRect aspect 过低时（mask 近圆形），
    用轮廓曲率找最尖锐的 **凸** 角作为箭头头部，从质心→尖端确定方向。
    仅在尖端曲率显著小于中值（有明确尖角）时才返回结果，否则返回 None 走正常 PCA。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < CURVATURE_K * 2 + 5:
        return None

    cnt_pts = cnt.reshape(-1, 2).astype(np.float64)
    n = len(cnt_pts)

    # 计算每个轮廓点的曲率（邻域向量夹角，越小越尖锐）
    curvatures = np.full(n, math.pi)
    for i in range(n):
        v1 = cnt_pts[(i - CURVATURE_K) % n] - cnt_pts[i]
        v2 = cnt_pts[(i + CURVATURE_K) % n] - cnt_pts[i]
        len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
        curvatures[i] = math.acos(cos_a)

    best_idx = int(np.argmin(curvatures))
    best_curv = curvatures[best_idx]
    median_curv = float(np.median(curvatures))

    # 质量检查：如果尖端曲率不够突出（近圆形 mask），放弃曲率结果
    if median_curv < 1e-6 or best_curv / median_curv > CURVATURE_MIN_RATIO:
        return None

    tip = cnt_pts[best_idx]
    dx = float(tip[0] - cx)
    dy = float(tip[1] - cy)
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None

    angle_deg = math.degrees(math.atan2(dx, -dy)) % 360.0
    ndx, ndy = dx / length, dy / length

    return {
        "angle_deg":   round(angle_deg, 1),
        "head_xy":     (float(tip[0]), float(tip[1])),
        "tail_xy":     (cx - ndx * length, cy - ndy * length),
        "centroid":    (cx, cy),
        "mask_pixels": pixels,
        "aspect":      round(max(rect[1]) / max(min(rect[1]), 1e-6), 2),
        "skewness":    0.0,
        "rect":        rect,
        "warning":     "curvature_fallback",
        "use_white":   False,
    }


def _compute_arrow_angle(mask: np.ndarray, white_border: np.ndarray) -> dict | None:
    """
    PCA 主轴 + 偏度法 + 白边 tie-breaker 计算箭头角度。

    effective_aspect = max(minAreaRect 伸长率, PCA 方差伸长率)，
    对"圆形头部+喙"型图标更准确，减少误报 asp_low。
    """
    pixels = int(np.count_nonzero(mask))
    if pixels < MIN_ARROW_PIXELS:
        return None
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) < 5:
        return None

    # minAreaRect 伸长率
    rect = cv2.minAreaRect(pts)
    rw, rh = rect[1]
    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)

    # minAreaRect 近圆形时 PCA 轴退化 → 尝试轮廓曲率找箭头尖端
    if aspect < MIN_ASPECT_RATIO_HARD:
        curv_result = _curvature_arrow_angle(mask, pts, cx, cy, rect, pixels)
        if curv_result is not None:
            return curv_result
        # 曲率不可靠（轮廓近圆形），继续走正常 PCA 路径

    # PCA 主轴
    _, ev = cv2.PCACompute(pts, mean=None)
    pca_dir  = np.array([float(ev[0, 0]), float(ev[0, 1])])
    pca_perp = np.array([-float(ev[0, 1]), float(ev[0, 0])])

    center = np.array([cx, cy])
    proj      = (pts - center) @ pca_dir
    proj_perp = (pts - center) @ pca_perp

    # PCA 分布伸长率（对"圆形+喙"型图标比 minAreaRect 更准确）
    var_along = float(np.var(proj))
    var_perp  = float(np.var(proj_perp))
    pca_aspect = float(np.sqrt(var_along / max(var_perp, 1e-6)))
    effective_aspect = max(aspect, pca_aspect)

    # 警告分类
    warning: str | None = None
    if pixels > MAX_ARROW_PIXELS:
        warning = f"pix>{MAX_ARROW_PIXELS}({pixels})"
    elif pixels > SHIELD_PIXEL_THRESH and effective_aspect < SHIELD_ASPECT_THRESH:
        warning = f"shield_occluded(px={pixels},asp={effective_aspect:.2f})"
    elif effective_aspect < MIN_ASPECT_RATIO:
        warning = f"asp_low({effective_aspect:.2f})"

    # 偏度法确定尖端方向
    pstd = float(np.std(proj))
    if pstd < 1e-6:
        return None
    skewness = float(np.mean(((proj - float(np.mean(proj))) / pstd) ** 3))
    shield_sign = 1.0 if skewness > 0 else -1.0

    # ---- 消歧：模板优先 → 白边兜底 ----
    if abs(skewness) < AMB_SKEW_ABS:
        tmpl_sign = _template_flip_score(mask, pca_dir)
        if tmpl_sign is not None:
            shield_sign = tmpl_sign
            warning = (warning or "") + "[amb_tmpl]"
        else:
            wy, wx = np.where(white_border > 0)
            if len(wx) >= WHITE_MIN_PIXELS:
                wpts = np.column_stack([wx, wy]).astype(np.float32)
                mean_proj = float(np.mean((wpts - center) @ pca_dir))
                shield_sign = 1.0 if mean_proj >= 0.0 else -1.0
                warning = (warning or "") + "[amb_wb]"

    dx = pca_dir[0] * shield_sign
    dy = pca_dir[1] * shield_sign
    angle_deg = math.degrees(math.atan2(dx, -dy)) % 360.0

    head_xy = (cx + dx * 12, cy + dy * 12)
    tail_xy = (cx - dx * 12, cy - dy * 12)

    return {
        "angle_deg":   round(angle_deg, 1),
        "head_xy":     (float(head_xy[0]), float(head_xy[1])),
        "tail_xy":     (float(tail_xy[0]), float(tail_xy[1])),
        "centroid":    (cx, cy),
        "mask_pixels": pixels,
        "aspect":      round(aspect, 2),
        "skewness":    round(skewness, 3),
        "rect":        rect,
        "warning":     warning or None,
        "use_white":   False,
    }


# ---------------------------------------------------------------------------
# 内部：完整检测流水线（crop_bgr → result）
# ---------------------------------------------------------------------------

def _run_pipeline(crop_bgr: np.ndarray,
                  prev_stable_angle: float | None = None) -> dict | None:
    mask, _ = _extract_yellow_mask(crop_bgr)
    mask = _best_arrow_contour(mask)
    mask_before = mask.copy()

    # 中心核心圈裁剪
    h_m, w_m = mask.shape[:2]
    core = np.zeros_like(mask)
    core_r = max(10, APERTURE_RADIUS - 2)
    cv2.circle(core, (w_m // 2, h_m // 2), core_r, 255, -1)
    mask_cropped = cv2.bitwise_and(mask, core)

    # 若裁剪导致细长目标丢角严重则回退
    before_px = int(np.count_nonzero(mask_before))
    after_px  = int(np.count_nonzero(mask_cropped))
    if before_px > 0:
        bx, by, bw, bh = cv2.boundingRect(mask_before)
        pre_aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
        # 像素损失严重，或 aspect 退化严重（尖端被截断）
        if pre_aspect >= 1.25:
            px_loss = after_px / before_px < 0.93
            pts_after = np.column_stack(np.where(mask_cropped > 0)[::-1]).astype(np.float32)
            if len(pts_after) >= 5:
                rect_after = cv2.minAreaRect(pts_after)
                rw_a, rh_a = rect_after[1]
                post_aspect = max(rw_a, rh_a) / max(min(rw_a, rh_a), 1e-6)
                aspect_drop = post_aspect / pre_aspect < 0.80
            else:
                aspect_drop = True
            if px_loss or aspect_drop:
                mask_cropped = mask_before
    mask = _keep_center_component(mask_cropped)

    white_border = _extract_white_border(crop_bgr, mask, APERTURE_RADIUS)

    # 黄白双色共现硬约束
    if ENABLE_STRONG_YW_CONSTRAINT:
        wp = int(np.count_nonzero(white_border))
        yp = int(np.count_nonzero(mask))
        if wp < WHITE_MIN_PIXELS or wp / max(yp, 1) < WHITE_YELLOW_MIN_RATIO:
            return None

    result = _compute_arrow_angle(mask, white_border)
    if result is None:
        return None

    # 歧义帧时序融合（曲率回退成功时视为可信，不做融合）
    if ENABLE_TEMPORAL_BLEND and prev_stable_angle is not None:
        is_curvature = "curvature_fallback" in (result.get("warning") or "")
        amb = (not is_curvature and
               abs(float(result.get("skewness", 0.0))) < AMB_SKEW_ABS and
               AMB_ASPECT_MIN <= float(result.get("aspect", 0.0)) <= AMB_ASPECT_MAX)
        if amb:
            curr_angle = float(result["angle_deg"])
            # 歧义帧反转保护：若当前方向与稳定参考差 ~180°，先翻转再融合
            delta = abs(curr_angle - prev_stable_angle) % 360.0
            delta = min(delta, 360.0 - delta)
            if delta > ANTIFLIP_ANGLE_THRESH:
                curr_angle = (curr_angle + 180.0) % 360.0
                result["head_xy"], result["tail_xy"] = result["tail_xy"], result["head_xy"]
                result["warning"] = (result.get("warning") or "") + "[antiflip]"

            new_angle = _blend_angles_deg(float(prev_stable_angle),
                                          curr_angle,
                                          AMB_CURR_WEIGHT)
            result["angle_deg"] = round(new_angle, 1)
            ccx, ccy = result["centroid"]
            hx, hy = result["head_xy"]
            length = max(6.0, math.hypot(hx - ccx, hy - ccy))
            rad = math.radians(new_angle)
            ddx, ddy = math.sin(rad), -math.cos(rad)
            result["head_xy"] = (ccx + ddx * length, ccy + ddy * length)
            result["tail_xy"] = (ccx - ddx * length, ccy - ddy * length)
            result["warning"] = (result.get("warning") or "") + "[temporal_blend]"

    return result


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def detect_arrow(crop_bgr: np.ndarray,
                 prev_stable_angle: float | None = None) -> dict | None:
    """
    从小地图中心裁剪区检测玩家箭头朝向。

    Parameters
    ----------
    crop_bgr : np.ndarray
        BGR 格式的裁剪图像（通常 40×40 左右，由调用方从完整帧截取）。
    prev_stable_angle : float | None
        上一稳定帧的角度，用于歧义帧时序融合。None 表示不融合。

    Returns
    -------
    dict | None
        成功时返回包含 angle_deg、aspect、skewness、warning 等字段的字典；
        检测失败（无法找到图标）时返回 None。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    return _run_pipeline(crop_bgr, prev_stable_angle)


def detect_arrow_from_frame(frame_bgr: np.ndarray,
                            mx: int, my: int, mr: int,
                            prev_stable_angle: float | None = None,
                            crop_buf: list | None = None) -> dict | None:
    """
    从完整游戏帧 + 小地图圆心坐标检测玩家箭头朝向。

    Parameters
    ----------
    frame_bgr : np.ndarray
        完整 BGR 帧。
    mx, my : int
        小地图圆心坐标（像素）。
    mr : int
        小地图圆形半径（像素）。
    prev_stable_angle : float | None
        上一稳定帧的角度，用于歧义帧时序融合。
    crop_buf : list | None
        时序中值缓冲（list of recent crop ndarray）。
        若提供且长度>=2，对历史帧+当前帧取像素中值，降低视频压缩噪声。

    Returns
    -------
    dict | None
        同 detect_arrow() 的返回值；额外包含 crop_bgr 字段（裁剪区图像）。
    """
    crop_r = max(ARROW_CROP_RADIUS, int(mr * 0.25))
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, mx - crop_r)
    y1 = max(0, my - crop_r)
    x2 = min(w_img, mx + crop_r)
    y2 = min(h_img, my + crop_r)
    raw_crop = frame_bgr[y1:y2, x1:x2]
    if raw_crop.size == 0:
        return None

    # 时序中值融合
    if crop_buf is not None and len(crop_buf) >= 2:
        same_shape = all(c.shape == raw_crop.shape for c in crop_buf)
        if same_shape:
            stack = np.stack(crop_buf + [raw_crop], axis=0)
            crop_bgr = np.median(stack, axis=0).astype(np.uint8)
        else:
            crop_bgr = raw_crop
    else:
        crop_bgr = raw_crop

    result = _run_pipeline(crop_bgr, prev_stable_angle)
    if result is not None:
        result["crop_bgr"] = crop_bgr
    return result
