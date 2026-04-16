"""
core/enhance.py - 图像增强算法

提供无状态函数和可复用的 CLAHE 工厂。
无 Web 框架依赖，仅依赖 cv2 / numpy / config。
"""

import numpy as np
import cv2
from backend import config


def make_clahe_pair():
    """创建 (clahe_normal, clahe_low) 双档 CLAHE 对象。"""
    clahe_normal = cv2.createCLAHE(
        clipLimit=getattr(config, 'CLAHE_LIMIT_NORMAL', 3.0), tileGridSize=(8, 8))
    clahe_low = cv2.createCLAHE(
        clipLimit=getattr(config, 'CLAHE_LIMIT_LOW_TEXTURE', 6.0), tileGridSize=(8, 8))
    return clahe_normal, clahe_low


def adaptive_clahe_map(gray: np.ndarray, clahe_normal, clahe_low,
                       low_thresh: float, tile: int = 256) -> np.ndarray:
    """大地图分块自适应 CLAHE + 锐化，与小地图 enhance_minimap 保持一致增强路径。"""
    h, w = gray.shape[:2]
    result = np.empty_like(gray)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            patch = gray[y:y + tile, x:x + tile]
            std = float(np.std(patch))
            result[y:y + tile, x:x + tile] = enhance_minimap(
                patch, std, clahe_normal, clahe_low, low_thresh)
    return result


def enhance_minimap(gray_raw: np.ndarray, texture_std: float,
                    clahe_normal, clahe_low, low_thresh: float) -> np.ndarray:
    """
    统一小地图增强：一次 CLAHE + 按纹理强度做锐化（合并原 adaptive_clahe + enhance_for_texture，消除双 CLAHE）。

    texture_std < 15 : 超低纹理（海面），双边保边 + clahe_low + 强 unsharp mask
    texture_std < 30 : 低纹理（雪地），clahe_low + 温和锐化
    texture_std < 55 : 中低纹理（草地），clahe_normal + 极温和 unsharp
    else             : 标准纹理，clahe_normal
    """
    if texture_std < 15:
        filtered = cv2.bilateralFilter(gray_raw, 9, 75, 75)
        enhanced = clahe_low.apply(filtered)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
        return cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)
    if texture_std < 30:
        enhanced = clahe_low.apply(gray_raw)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    if texture_std < 55:
        enhanced = clahe_normal.apply(gray_raw)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
        return cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)
    return clahe_normal.apply(gray_raw)


def correct_color_temperature(bgr: np.ndarray) -> np.ndarray:
    """
    检测并补偿护眼/防蓝光软件叠加层引起的色温漂移（Gray-World 假设）。
    仅当 R_mean - B_mean 超过阈值时触发，否则原图直接返回（零开销）。
    """
    bias_thresh = getattr(config, 'COLOR_TEMP_BIAS_THRESH', 15)
    b_mean = float(np.mean(bgr[:, :, 0]))
    r_mean = float(np.mean(bgr[:, :, 2]))
    if r_mean - b_mean < bias_thresh:
        return bgr
    g_mean = float(np.mean(bgr[:, :, 1]))
    overall = (b_mean + g_mean + r_mean) / 3.0
    gains = [
        float(np.clip(overall / ch, 0.5, 2.0)) if ch > 1e-3 else 1.0
        for ch in (b_mean, g_mean, r_mean)
    ]
    corrected = bgr.astype(np.float32)
    for i, g in enumerate(gains):
        if abs(g - 1.0) > 0.01:
            corrected[:, :, i] = np.clip(corrected[:, :, i] * g, 0, 255)
    return corrected.astype(np.uint8)


def classify_scene_by_color(bgr: np.ndarray, prior_scene: str) -> str:
    """
    利用颜色信息对纹理分类结果做细化，弥补 texture_std 的盲区。

    默认会细化所有场景；对 prior_scene='urban' 可通过配置选择是否细化。

    细化场景
    --------
    ocean     : 蓝色主导（H≈90-130, S>50, V>50），覆盖率≥ocean_thresh
    grassland : 绿色主导（H≈35-80, S>40），覆盖率≥grass_thresh
    snow      : 低饱和+高亮度（S<45, V>185），覆盖率≥snow_thresh

    若任何细化条件不满足则返回 prior_scene（保持原有分类）。

    Config keys
    -----------
    SCENE_COLOR_OCEAN_THRESH  float  0.28
    SCENE_COLOR_GRASS_THRESH  float  0.28
    SCENE_COLOR_SNOW_THRESH   float  0.38
    """
    refine_urban = bool(getattr(config, 'SCENE_COLOR_REFINE_URBAN', True))
    if prior_scene == 'urban' and not refine_urban:
        return prior_scene

    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(8, min(cx, cy) - 4)
    # 只分析中心圆形区域，避免扫描整张图
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    roi = bgr[y0:y1, x0:x1]
    if roi.size < 100:
        return prior_scene

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0].ravel().astype(np.float32)  # OpenCV H: 0-179
    s_ch = hsv[:, :, 1].ravel().astype(np.float32)
    v_ch = hsv[:, :, 2].ravel().astype(np.float32)
    n = float(max(1, h_ch.size))

    ocean_h_min = float(getattr(config, 'SCENE_COLOR_OCEAN_H_MIN', 90))
    ocean_h_max = float(getattr(config, 'SCENE_COLOR_OCEAN_H_MAX', 130))
    ocean_s_min = float(getattr(config, 'SCENE_COLOR_OCEAN_S_MIN', 50))
    ocean_v_min = float(getattr(config, 'SCENE_COLOR_OCEAN_V_MIN', 50))
    grass_h_min = float(getattr(config, 'SCENE_COLOR_GRASS_H_MIN', 30))
    grass_h_max = float(getattr(config, 'SCENE_COLOR_GRASS_H_MAX', 92))
    grass_s_min = float(getattr(config, 'SCENE_COLOR_GRASS_S_MIN', 28))
    snow_s_max = float(getattr(config, 'SCENE_COLOR_SNOW_S_MAX', 45))
    snow_v_min = float(getattr(config, 'SCENE_COLOR_SNOW_V_MIN', 185))

    # 海洋：蓝色主导
    ocean_pct = float(np.sum(
        (h_ch >= ocean_h_min) & (h_ch <= ocean_h_max) & (s_ch > ocean_s_min) & (v_ch > ocean_v_min)
    )) / n
    # 草原：绿色主导（范围放宽以覆盖偏黄草地）
    grass_pct = float(np.sum(
        (h_ch >= grass_h_min) & (h_ch <= grass_h_max) & (s_ch > grass_s_min)
    )) / n
    # 雪地：低饱和 + 高亮
    snow_pct = float(np.sum(
        (s_ch < snow_s_max) & (v_ch > snow_v_min)
    )) / n

    ocean_thresh = float(getattr(config, 'SCENE_COLOR_OCEAN_THRESH', 0.28))
    grass_thresh = float(getattr(config, 'SCENE_COLOR_GRASS_THRESH', 0.22))
    grass_thresh_urban = float(getattr(config, 'SCENE_COLOR_GRASS_THRESH_URBAN', 0.30))
    snow_thresh  = float(getattr(config, 'SCENE_COLOR_SNOW_THRESH',  0.38))

    if ocean_pct >= ocean_thresh and ocean_pct > grass_pct:
        return 'ocean'
    if snow_pct >= snow_thresh:
        return 'snow'
    eff_grass_thresh = grass_thresh_urban if prior_scene == 'urban' else grass_thresh
    if grass_pct >= eff_grass_thresh:
        return 'grassland'
    return prior_scene


# BGR channel weights for scene-optimized grayscale conversion
# Key: scene_detail → (B_weight, G_weight, R_weight)
# Standard cv2.BGR2GRAY uses (0.114, 0.587, 0.299) — optimal for urban
_SCENE_GRAY_WEIGHTS: dict[str, tuple[float, float, float]] = {
    # 海洋：提高 B 权重，凸显蓝色区域内的梯度（岸线、水流）
    'ocean':     (0.50, 0.40, 0.10),
    # 草原：提高 G 权重，放大草地与道路/岩石之间的对比
    'grassland': (0.06, 0.78, 0.16),
    # 雪地：均等权重（雪地无主色），转换后做百分位拉伸
    'snow':      (0.114, 0.587, 0.299),
}


def make_scene_boosted_gray(bgr: np.ndarray, scene_detail: str) -> np.ndarray | None:
    """
    按场景类型生成色彩优化灰度图，最大化该场景下 SIFT 特征数量。

    仅对 ocean / grassland / snow 三种场景做优化；其余返回 None，
    调用方退回到标准 cv2.BGR2GRAY 路径。

    海洋
        B 通道权重提至 0.50，岸线轮廓在灰度图中对比更强，SIFT 可提取
        到更多描述岸线走向的特征点。
    草原
        G 通道权重提至 0.70，草地 vs 细小道路/石块的亮度差被放大，
        低纹理草原中的稀疏路径能产生更多可重复匹配的特征。
    雪地
        标准灰度转换后做 5%~95% 百分位拉伸，将雪地中的微弱纹理（
        坡度阴影、融雪水迹）拉伸至全动态范围，供 SIFT 抓取。

    Returns
    -------
    uint8 grayscale ndarray，或 None（非目标场景）。
    """
    weights = _SCENE_GRAY_WEIGHTS.get(scene_detail)
    if weights is None:
        return None

    bw, gw, rw = weights
    bgr_f = bgr.astype(np.float32)
    gray = bgr_f[:, :, 0] * bw + bgr_f[:, :, 1] * gw + bgr_f[:, :, 2] * rw

    if scene_detail == 'snow':
        lo = float(np.percentile(gray, 5))
        hi = float(np.percentile(gray, 95))
        if hi - lo > 8:
            gray = (gray - lo) / (hi - lo) * 255.0

    return np.clip(gray, 0, 255).astype(np.uint8)

