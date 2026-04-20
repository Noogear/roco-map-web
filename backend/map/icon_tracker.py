"""
backend/map/icon_tracker.py - 基于小地图图标位移的方向与距离估算

核心组件：
  - OpticalFlowTracker: LK 光流追踪器，多点中值共识
  - load_icon_templates: 加载图标模板（多尺度，按对比度排序）
  - detect_initial_icons: 模板匹配检测初始图标位置

架构（两阶段）：
  1. 初始化：模板匹配检测 3~5 个高置信度图标位置
  2. 追踪：LK 光流追踪这些点（每帧 <2ms），丢失后重新初始化

过滤策略：
  1. 排除边缘图标（距小地图圆心太远）
  2. 排除中心区域（玩家箭头覆盖区）
  3. 排除静止不动的图标（不参与位移共识）
  4. 排除与主流方向不一致的异常值
  5. 外观校验（NCC）防止追踪点身份漂移
  6. 单点共识方向一致性检查
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class IconTemplate:
    """预处理后的图标模板（多尺度）。"""
    name: str
    gray_scales: list[tuple[np.ndarray, np.ndarray, int, int]]  # [(gray, mask, w, h), ...]


@dataclass
class TrackedPoint:
    """一个被追踪的点。"""
    pt: np.ndarray           # [x, y] float32
    name: str                # 图标名 or 特征点 ID
    init_frame: int
    age: int = 0             # 存活帧数
    total_disp: float = 0.0  # 累计位移
    init_patch: np.ndarray | None = None  # 初始外观 patch（身份校验用）


@dataclass
class FrameResult:
    """单帧的追踪结果。"""
    frame_idx: int
    n_tracked: int = 0
    n_moving: int = 0
    consensus_dx: float = 0.0
    consensus_dy: float = 0.0
    consensus_dist: float = 0.0
    latency_ms: float = 0.0
    tracked_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 图标模板加载
# ---------------------------------------------------------------------------

def load_icon_templates(
    icon_dir: Path,
    icon_names: set[str] | None = None,
    target_radius: float = 70.0,
    max_templates: int = 15,
) -> list[IconTemplate]:
    """加载图标模板，只保留最具辨识度的 *max_templates* 个。"""
    all_templates: list[tuple[float, IconTemplate]] = []

    for icon_path in sorted(icon_dir.glob("*.png")):
        name = icon_path.stem
        if icon_names and name not in icon_names:
            continue

        icon_rgba = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
        if icon_rgba is None:
            continue

        # Alpha 切割
        if icon_rgba.ndim == 2:
            bgr = cv2.cvtColor(icon_rgba, cv2.COLOR_GRAY2BGR)
            alpha = np.full(icon_rgba.shape[:2], 255, dtype=np.uint8)
        elif icon_rgba.shape[2] == 4:
            bgr = icon_rgba[:, :, :3]
            alpha = icon_rgba[:, :, 3]
        else:
            bgr = icon_rgba
            alpha = np.full(icon_rgba.shape[:2], 255, dtype=np.uint8)

        ys, xs = np.where(alpha > 12)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bgr = bgr[y1:y2, x1:x2]

        oh, ow = bgr.shape[:2]
        if ow < 8 or oh < 8 or ow * oh < 40:
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        if contrast < 3.0:
            continue

        # 前景掩膜
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        fg = (((s >= 32) & (v >= 48)) | (v >= 180)).astype(np.uint8) * 255
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3, iterations=1)

        # 缩放尺度
        target_w = max(14.0, min(target_radius * 0.35, 42.0))
        s0 = target_w / max(float(ow), 1.0)
        scale_factors = sorted({
            max(0.15, min(1.2, round(v_, 3)))
            for v_ in [0.85 * s0, 1.0 * s0, 1.2 * s0]
        })

        variants: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for sc in scale_factors:
            tw = max(10, int(round(ow * sc)))
            th = max(10, int(round(oh * sc)))
            t_gray = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
            t_mask = cv2.resize(fg, (tw, th), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(t_mask > 12) >= 15:
                variants.append((t_gray, t_mask, tw, th))

        if variants:
            all_templates.append((contrast, IconTemplate(name=name, gray_scales=variants)))

    all_templates.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in all_templates[:max_templates]]


# ---------------------------------------------------------------------------
# 初始化检测
# ---------------------------------------------------------------------------

# 空间排除比例常量
CENTER_RATIO = 0.25   # 距圆心 < 25% 半径的区域排除（箭头覆盖区）
EDGE_RATIO = 0.82     # 距圆心 > 82% 半径的区域排除（小地图边缘）
TRACKING_CENTER_RATIO = 0.18  # 追踪中的中心排除（略小于初始化）


def detect_initial_icons(
    minimap_gray: np.ndarray,
    center_xy: tuple[float, float],
    radius: float,
    templates: list[IconTemplate],
    score_threshold: float = 0.50,
    edge_ratio: float = EDGE_RATIO,
    center_ratio: float = CENTER_RATIO,
    max_icons: int = 5,
) -> list[tuple[str, float, float, float]]:
    """检测初始图标位置。

    Returns:
        [(name, cx, cy, score), ...]
    """
    h, w = minimap_gray.shape[:2]
    hits: list[tuple[str, float, float, float, int, int]] = []

    for tmpl in templates:
        best_score = -1.0
        best_loc = None
        best_tw, best_th = 0, 0

        for t_gray, t_mask, tw, th in tmpl.gray_scales:
            if tw >= w or th >= h:
                continue
            try:
                corr = cv2.matchTemplate(
                    minimap_gray, t_gray, cv2.TM_CCOEFF_NORMED, mask=t_mask)
            except Exception:
                continue
            if corr.size == 0:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(corr)
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_tw, best_th = tw, th

        if best_score < score_threshold or best_loc is None:
            continue

        cx = best_loc[0] + best_tw / 2.0
        cy = best_loc[1] + best_th / 2.0
        dist = math.hypot(cx - center_xy[0], cy - center_xy[1])
        dist_ratio = dist / max(radius, 1.0)

        if dist_ratio > edge_ratio or dist_ratio < center_ratio:
            continue

        hits.append((tmpl.name, cx, cy, best_score, best_tw, best_th))

    # NMS
    hits.sort(key=lambda x: x[3], reverse=True)
    final: list[tuple[str, float, float, float, int, int]] = []
    for name, cx, cy, score, tw, th in hits:
        overlap = any(
            math.hypot(cx - fx, cy - fy) < max(tw, th) * 0.7
            for _, fx, fy, _, _, _ in final
        )
        if not overlap:
            final.append((name, cx, cy, score, tw, th))
        if len(final) >= max_icons:
            break

    return [(name, cx, cy, score) for name, cx, cy, score, _, _ in final]


# ---------------------------------------------------------------------------
# 外观 patch 工具
# ---------------------------------------------------------------------------

_PATCH_HALF = 10


def _extract_patch(gray: np.ndarray, x: float, y: float) -> np.ndarray | None:
    h, w = gray.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    r = _PATCH_HALF
    if ix - r < 0 or iy - r < 0 or ix + r + 1 > w or iy + r + 1 > h:
        return None
    return gray[iy - r:iy + r + 1, ix - r:ix + r + 1].copy()


def _patch_ncc(p1: np.ndarray, p2: np.ndarray) -> float:
    a = p1.astype(np.float32).ravel()
    b = p2.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# LK 光流追踪器
# ---------------------------------------------------------------------------

# 性能 / 质量可调参数
NCC_CHECK_INTERVAL = 10   # 每隔多少帧做一次外观校验
NCC_MIN_THRESHOLD = 0.25  # NCC 低于此值判定为身份漂移
FB_ERR_MAX = 2.0          # 前后验证最大允许误差 (px)
SINGLE_POINT_DISP_MULT = 2.0  # 单点共识所需位移倍数（× stationary_threshold）
SINGLE_POINT_ANGLE_TOL = math.pi / 2  # 单点方向与多点参考最大偏差 (90°)


class OpticalFlowTracker:
    """基于 LK 金字塔光流的多点追踪器，中值共识输出位移。"""

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(
        self,
        center_xy: tuple[float, float],
        radius: float,
        stationary_threshold: float = 0.8,
        max_jump: float = 25.0,
        consensus_angle_tol: float = 60.0,
    ):
        self.center_xy = center_xy
        self.radius = radius
        self.stationary_threshold = stationary_threshold
        self.max_jump = max_jump
        self.consensus_angle_tol = consensus_angle_tol

        self._prev_gray: np.ndarray | None = None
        self._tracked_points: list[TrackedPoint] = []
        self._frame_results: list[FrameResult] = []
        self._cumulative_dx: float = 0.0
        self._cumulative_dy: float = 0.0
        self._total_path: float = 0.0
        self._last_drop_count: int = 0
        self._last_multi_angle: float | None = None

    # -- 只读属性 --

    @property
    def cumulative_displacement(self) -> float:
        return math.hypot(self._cumulative_dx, self._cumulative_dy)

    @property
    def cumulative_distance(self) -> float:
        return self._total_path

    @property
    def frame_results(self) -> list[FrameResult]:
        return self._frame_results

    @property
    def n_tracked(self) -> int:
        return len(self._tracked_points)

    @property
    def last_drop_count(self) -> int:
        return self._last_drop_count

    # -- 初始化 / 补充 --

    def init_points(
        self, gray: np.ndarray, icons: list[tuple[str, float, float, float]],
    ) -> None:
        """用检测到的图标位置初始化追踪点（会重置 _prev_gray）。"""
        self._prev_gray = gray.copy()
        frame_idx = self._frame_results[-1].frame_idx if self._frame_results else 0
        for name, cx, cy, _score in icons:
            if any(math.hypot(tp.pt[0] - cx, tp.pt[1] - cy) < 10
                   for tp in self._tracked_points):
                continue
            patch = _extract_patch(gray, cx, cy)
            self._tracked_points.append(TrackedPoint(
                pt=np.array([cx, cy], dtype=np.float32),
                name=name, init_frame=frame_idx, init_patch=patch,
            ))

    def supplement_points(
        self, gray: np.ndarray, icons: list[tuple[str, float, float, float]],
    ) -> None:
        """在 update() 之后安全补充追踪点（不覆盖 _prev_gray）。"""
        frame_idx = self._frame_results[-1].frame_idx if self._frame_results else 0
        for name, cx, cy, _score in icons:
            if any(math.hypot(tp.pt[0] - cx, tp.pt[1] - cy) < 10
                   for tp in self._tracked_points):
                continue
            patch = _extract_patch(gray, cx, cy)
            self._tracked_points.append(TrackedPoint(
                pt=np.array([cx, cy], dtype=np.float32),
                name=name, init_frame=frame_idx, init_patch=patch,
            ))

    def add_feature_points(
        self, gray: np.ndarray, frame_idx: int, max_extra: int = 3,
    ) -> None:
        """用 Shi-Tomasi 角点补充追踪点。"""
        if len(self._tracked_points) >= 4:
            return

        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(self.center_xy[0]), int(self.center_xy[1])),
                   int(self.radius * EDGE_RATIO), 255, -1)
        cv2.circle(mask, (int(self.center_xy[0]), int(self.center_xy[1])),
                   int(self.radius * CENTER_RATIO), 0, -1)
        for tp in self._tracked_points:
            cv2.circle(mask, (int(tp.pt[0]), int(tp.pt[1])), 15, 0, -1)

        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=max_extra, qualityLevel=0.15,
            minDistance=15, mask=mask, blockSize=7,
        )
        if corners is not None:
            for c in corners:
                x, y = float(c[0, 0]), float(c[0, 1])
                patch = _extract_patch(gray, x, y)
                self._tracked_points.append(TrackedPoint(
                    pt=np.array([x, y], dtype=np.float32),
                    name=f"feat_{frame_idx}_{int(x)}_{int(y)}",
                    init_frame=frame_idx, init_patch=patch,
                ))

        self._prev_gray = gray.copy()

    # -- 核心更新 --

    def update(self, gray: np.ndarray, frame_idx: int) -> FrameResult:
        """用光流追踪更新所有点，返回帧位移共识。"""
        result = FrameResult(frame_idx=frame_idx)

        if self._prev_gray is None or not self._tracked_points:
            self._prev_gray = gray.copy()
            self._frame_results.append(result)
            return result

        prev_pts = np.array(
            [tp.pt for tp in self._tracked_points], dtype=np.float32,
        ).reshape(-1, 1, 2)

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **self.LK_PARAMS)
        if next_pts is None or status is None:
            self._prev_gray = gray.copy()
            self._frame_results.append(result)
            return result

        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, next_pts, None, **self.LK_PARAMS)

        n_before = len(self._tracked_points)
        surviving: list[TrackedPoint] = []
        displacements: list[tuple[float, float]] = []

        for i, tp in enumerate(self._tracked_points):
            if status[i, 0] == 0:
                continue
            if back_status is not None and back_status[i, 0] == 0:
                continue

            nx, ny = float(next_pts[i, 0, 0]), float(next_pts[i, 0, 1])

            # 前后验证
            if back_pts is not None:
                fb_err = math.hypot(
                    float(back_pts[i, 0, 0]) - float(prev_pts[i, 0, 0]),
                    float(back_pts[i, 0, 1]) - float(prev_pts[i, 0, 1]),
                )
                if fb_err > FB_ERR_MAX:
                    continue

            # 空间排除
            dist = math.hypot(nx - self.center_xy[0], ny - self.center_xy[1])
            if dist > self.radius * 0.95 or dist < self.radius * TRACKING_CENTER_RATIO:
                continue

            dx = nx - tp.pt[0]
            dy = ny - tp.pt[1]
            d = math.hypot(dx, dy)
            if d > self.max_jump:
                continue

            # 外观校验
            if (tp.init_patch is not None
                    and tp.age > 0
                    and tp.age % NCC_CHECK_INTERVAL == 0):
                cur_patch = _extract_patch(gray, nx, ny)
                if cur_patch is not None and _patch_ncc(tp.init_patch, cur_patch) < NCC_MIN_THRESHOLD:
                    continue

            tp.pt = np.array([nx, ny], dtype=np.float32)
            tp.age += 1
            tp.total_disp += d
            surviving.append(tp)

            if d >= self.stationary_threshold:
                displacements.append((dx, dy))

        self._tracked_points = surviving
        self._last_drop_count = n_before - len(surviving)
        self._prev_gray = gray.copy()

        result.n_tracked = len(surviving)
        result.tracked_names = [tp.name for tp in surviving]

        # ---- 共识计算 ----
        if len(displacements) >= 2:
            dxs = np.array([d[0] for d in displacements])
            dys = np.array([d[1] for d in displacements])
            angles = np.degrees(np.arctan2(dys, dxs))
            median_angle = float(np.median(angles))
            angle_diffs = np.abs(angles - median_angle)
            angle_diffs = np.minimum(angle_diffs, 360.0 - angle_diffs)
            inlier = angle_diffs <= self.consensus_angle_tol

            if np.any(inlier):
                result.consensus_dx = float(np.median(dxs[inlier]))
                result.consensus_dy = float(np.median(dys[inlier]))
                result.consensus_dist = math.hypot(
                    result.consensus_dx, result.consensus_dy)
                result.n_moving = int(np.sum(inlier))
                self._last_multi_angle = math.atan2(
                    result.consensus_dy, result.consensus_dx)

        elif len(displacements) == 1:
            d1 = math.hypot(displacements[0][0], displacements[0][1])
            if d1 >= self.stationary_threshold * SINGLE_POINT_DISP_MULT:
                accept = True
                if self._last_multi_angle is not None:
                    sp_angle = math.atan2(displacements[0][1], displacements[0][0])
                    delta = abs(sp_angle - self._last_multi_angle)
                    if delta > math.pi:
                        delta = 2 * math.pi - delta
                    if delta > SINGLE_POINT_ANGLE_TOL:
                        accept = False
                if accept:
                    result.consensus_dx, result.consensus_dy = displacements[0]
                    result.consensus_dist = d1
                    result.n_moving = 1

        self._cumulative_dx += result.consensus_dx
        self._cumulative_dy += result.consensus_dy
        self._total_path += result.consensus_dist

        self._frame_results.append(result)
        return result
