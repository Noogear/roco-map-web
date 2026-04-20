"""
icon_matcher_v2 配置和性能调优参数

可按以下策略调整：
1. 识别率优先：降低 score_threshold (0.45-0.50), 增加 bright_v_threshold, 降低 edge_threshold
2. 精确率优先：提高 score_threshold (0.65-0.75), 降低 bright_v_threshold, 提高 edge_threshold  
3. 性能优先：增加 center_hole_radius, 增加 edge_margin (快速缩小搜索区域)
4. 抗干扰优先：提高 vivid_saturation, 提高 masked_ncc_weight (依赖纹理匹配)
"""

# ============================================================================
# 图像预处理参数
# ============================================================================

# 中心黑洞配置（遮蔽玩家箭头）
CENTER_HOLE_RADIUS_RATIO = 0.15  # 相对于小地图半径

# 边缘死亡环配置（去掉卡在边缘的远方指示符）
EDGE_MARGIN_PIXELS = 6  # 像素值

# ============================================================================
# 候选提取参数 (第三阶段粗筛)
# ============================================================================

# 高光亮度阈值（排除纯白背景如雪地）
# 较低 (175) → 更宽松，容易误匹配雪地
# 较高 (210) → 更严格，可能漏掉高亮真实图标
BRIGHT_V_THRESHOLD = 200

# 高饱和度检测
VIVID_SATURATION_THRESHOLD = 50  # 更高 → 更依赖彩色图标
VIVID_VALUE_THRESHOLD = 65

# 梯度边缘阈值
# 较低 (32) → 包含更多平缓边界
# 较高 (40) → 只包含锐利边缘（更抗干扰）
GRADIENT_THRESHOLD = 36

# 候选框的尺寸约束
MIN_CANDIDATE_AREA = 24
MAX_CANDIDATE_AREA = 4000
MIN_CANDIDATE_SIZE = 6  # 最小宽/高
MAX_CANDIDATE_SIZE = 110
MAX_ASPECT_RATIO = 3.0
MIN_FILL_RATIO = 0.04

# Padding 扩展（为滑动窗口留出空间）
CANDIDATE_PADDING = 8

# ============================================================================
# 精排参数 (第四阶段)
# ============================================================================

# 滑动窗口匹配
# masked_ncc < 这个值 → 直接跳过
MASKED_NCC_REJECT_THRESHOLD = -0.8

# 颜色相似度阈值
MIN_COLOR_SIM = 0.0  # 基本无下限（由最终得分控制）

# 多尺度权重调整
# 推荐配置: masked_ncc:0.70, color_sim:0.18, hsv_gate:0.12
MASKED_NCC_WEIGHT = 0.70  # 越高越依赖纹理匹配（更精准，但更严格）
COLOR_SIM_WEIGHT = 0.18
HSV_GATE_WEIGHT = 0.12

# ============================================================================
# 后处理参数 (第五阶段NMS)
# ============================================================================

# NMS IoU 阈值 (去重重叠框)
NMS_IOU_THRESHOLD = 0.4

# 判定为"边缘图标"的条件
# is_edge = (distance + 0.35*max(w,h)) >= (radius * edge_ratio)
EDGE_DISTANCE_RATIO = 0.90  # 相对于小地图半径

# ============================================================================
# 特征路由参数 (颜色分组匹配)
# ============================================================================

# 色彩分组的邻近关系（用于增加鲁棒性）
# 当补丁颜色无法确定时，会查询邻近色组的图标
COLOR_NEIGHBORS = {
    "red": ["yellow", "white"],
    "yellow": ["red", "green"],
    "green": ["yellow", "cyan"],
    "blue": ["cyan", "white"],
    "cyan": ["green", "blue"],
    "white": ["red", "blue", "gray"],
    "gray": ["white"],
}

# ============================================================================
# 性能目标与推荐配置
# ============================================================================

PRESET_CONFIGS = {
    "balanced": {
        "description": "平衡识别率与精确率",
        "score_threshold": 0.60,
        "bright_v_threshold": 200,
        "vivid_saturation_threshold": 50,
        "gradient_threshold": 36,
        "masked_ncc_weight": 0.70,
    },
    "high_recall": {
        "description": "优先高识别率（容忍更多误匹配）",
        "score_threshold": 0.48,
        "bright_v_threshold": 175,
        "vivid_saturation_threshold": 40,
        "gradient_threshold": 30,
        "masked_ncc_weight": 0.65,
    },
    "high_precision": {
        "description": "优先高精确率（严格筛选）",
        "score_threshold": 0.70,
        "bright_v_threshold": 220,
        "vivid_saturation_threshold": 55,
        "gradient_threshold": 42,
        "masked_ncc_weight": 0.75,
    },
    "fast": {
        "description": "优先性能（快速识别）",
        "score_threshold": 0.55,
        "center_hole_radius_ratio": 0.20,  # 加大中心黑洞以减少搜索
        "edge_margin_pixels": 8,  # 加大边缘缓冲
        "max_candidate_area": 3000,  # 减少候选框尺寸上限
    },
}
