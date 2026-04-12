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


def adaptive_clahe(gray: np.ndarray, clahe_normal, clahe_low, low_thresh: float) -> np.ndarray:
    """
    低纹理用 clahe_low，否则 clahe_normal。

    Args:
        gray:          输入灰度图
        clahe_normal:  标准纹理 CLAHE
        clahe_low:     低纹理 CLAHE
        low_thresh:    std 低于此值视为低纹理
    """
    if np.std(gray) < low_thresh:
        return clahe_low.apply(gray)
    return clahe_normal.apply(gray)


def adaptive_clahe_map(gray: np.ndarray, clahe_normal, clahe_low,
                       low_thresh: float, tile: int = 256) -> np.ndarray:
    """大地图分块自适应 CLAHE，启动时对逻辑地图整体处理一次。"""
    h, w = gray.shape[:2]
    result = np.empty_like(gray)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            patch = gray[y:y + tile, x:x + tile]
            clahe = clahe_low if np.std(patch) < low_thresh else clahe_normal
            result[y:y + tile, x:x + tile] = clahe.apply(patch)
    return result


def enhance_for_texture(gray: np.ndarray, texture_std: float, clahe_low) -> np.ndarray:
    """
    按纹理 std 连续决定增强强度。

    texture_std < 15  : 超低纹理（海面），双边保边 + 高CLAHE + 强锐化
    texture_std < 30  : 低纹理（雪地），温和锐化
    else              : 标准纹理，原图返回
    """
    if texture_std < 15:
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = clahe_low.apply(filtered)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
        return cv2.addWeighted(enhanced, 2.0, blurred, -1.0, 0)
    if texture_std < 30:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return gray


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
