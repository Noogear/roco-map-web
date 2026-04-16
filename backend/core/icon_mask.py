"""
core/icon_mask.py — 游戏 UI 图标检测与 SIFT 排除遮罩

小地图上的任务图标、兴趣点标记通常为高饱和度、高亮度的彩色圆点，
与地形纹理（海洋/草原/雪地）差异明显。

本模块通过 HSV 阈值 + 连通域分析检测这些区域，生成排除遮罩，
在 SIFT/ORB 特征提取时排除 UI 干扰，减少"假特征"混入匹配池。

设计约束
---------
- 纯 CPU，依赖 OpenCV + NumPy
- 每帧调用开销 ≤ 2ms（小地图通常 150×150 ~ 300×300）
- 保守默认值：宁可漏掉一两个图标，也不误伤地形特征
"""

from __future__ import annotations

import cv2
import numpy as np

from backend import config


def detect_ui_icon_exclusion_mask(bgr: np.ndarray) -> np.ndarray | None:
    """
    检测小地图中的游戏 UI 图标（任务标记、兴趣点等）并生成像素级排除遮罩。

    算法
    ----
    1. BGR → HSV；高饱和度 + 高亮度区域即为图标候选（游戏 UI 用色高度饱和）
    2. 形态学开运算去除单像素噪声
    3. 连通域过滤：按面积、形状宽高比、与圆心距离筛选
    4. 通过膨胀稍微扩大遮罩以覆盖图标光晕边缘

    Returns
    -------
    uint8 ndarray (h, w)：255 = UI 图标区域（排除），0 = 地形（保留）
    若未检测到任何图标则返回 None，避免不必要的 bitwise 合并操作。

    Config keys（均有默认值，可在 config.py 中覆盖）
    --------------------------------------------------
    ICON_MASK_ENABLED          bool   True   - 总开关
    ICON_MASK_SAT_THRESH       int    160    - 饱和度下限 (0-255)
    ICON_MASK_VAL_THRESH       int    170    - 亮度下限 (0-255)
    ICON_MASK_CENTER_EXCL_RATIO float 0.15   - 中心玩家标记排除半径比（占小图半径）
    ICON_MASK_MIN_AREA         int    18     - 候选连通域最小面积（像素²）
    ICON_MASK_MAX_AREA         int    900    - 候选连通域最大面积（过大 = 非图标）
    ICON_MASK_MAX_ASPECT       float  3.5    - 最大宽高比（过扁 = 文字/线段，非图标）
    ICON_MASK_DILATE_RADIUS    int    5      - 膨胀半径（像素）
    """
    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    sat_thresh = int(getattr(config, 'ICON_MASK_SAT_THRESH', 160))
    val_thresh = int(getattr(config, 'ICON_MASK_VAL_THRESH', 170))

    # 高饱和度 + 高亮度：游戏 UI 图标的视觉特征
    # 自然地形（海洋/草原/雪地）的饱和度通常远低于 160
    raw_mask = ((s_ch >= sat_thresh) & (v_ch >= val_thresh)).astype(np.uint8) * 255

    # 形态学开运算去噪
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)

    if not np.any(cleaned):
        return None

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return None

    cx, cy = w // 2, h // 2
    r_minimap = max(1, min(cx, cy) - 2)
    center_excl_r2 = (r_minimap * float(getattr(config, 'ICON_MASK_CENTER_EXCL_RATIO', 0.15))) ** 2

    min_area = int(getattr(config, 'ICON_MASK_MIN_AREA', 18))
    max_area = int(getattr(config, 'ICON_MASK_MAX_AREA', 900))
    max_aspect = float(getattr(config, 'ICON_MASK_MAX_ASPECT', 3.5))

    result = np.zeros((h, w), dtype=np.uint8)
    found_any = False

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 1 or bh < 1:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > max_aspect:
            continue  # 过于细长 → 不是圆形图标

        # 质心（近似）
        bx = int(stats[i, cv2.CC_STAT_LEFT]) + bw // 2
        by_ = int(stats[i, cv2.CC_STAT_TOP]) + bh // 2

        # 排除中心区域（玩家自身标记）
        dist2 = (bx - cx) ** 2 + (by_ - cy) ** 2
        if dist2 < center_excl_r2:
            continue

        result[labels == i] = 255
        found_any = True

    if not found_any:
        return None

    # 膨胀：覆盖图标光晕/边缘渐变区域
    dilate_r = int(getattr(config, 'ICON_MASK_DILATE_RADIUS', 5))
    if dilate_r > 0:
        d = dilate_r * 2 + 1
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        result = cv2.dilate(result, k_dil)

    return result
