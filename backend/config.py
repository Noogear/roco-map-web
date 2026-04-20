import os as _os
import sys
from pathlib import Path

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# 导入根目录的 path_config
_PROJECT_ROOT = Path(_ROOT).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from path_config import LOGIC_MAP_PATH, DISPLAY_MAP_PATH

PORT = 8686

MINIMAP = {'top': 292, 'left': 1853, 'width': 150, 'height': 150}
WINDOW_GEOMETRY = '400x400+1500+100'
VIEW_SIZE = 400
JPEG_PAD = 40

MINIMAP_CAPTURE_MARGIN = 1.4
MINIMAP_CIRCLE_CALIBRATION_FRAMES = 8
MINIMAP_CIRCLE_R_TOLERANCE = 8
MINIMAP_CIRCLE_CENTER_TOLERANCE = 15
MINIMAP_CIRCLE_RECALIBRATE_MISS = 30
MINIMAP_LOCAL_MIN_RADIUS_RATIO = 0.22
MINIMAP_LOCAL_MAX_RADIUS_RATIO = 0.48
MINIMAP_LOCAL_MIN_SCORE = 0.22
# 候选圆半径必须 >= 最大候选圆半径 × 此比例，才会参与评分（过滤任务图标等小圆）
# 游戏内任务图标半径约为小地图的 1/10，此值设 0.5 可有效过滤
MINIMAP_CANDIDATE_MIN_R_RATIO = 0.50

AUTO_DETECT_MAX_SIDE = 1280
AUTO_DETECT_MIN_RADIUS_RATIO = 0.035
AUTO_DETECT_MAX_RADIUS_RATIO = 0.16
AUTO_DETECT_HOUGH_PARAM1 = 90
AUTO_DETECT_HOUGH_PARAM2_ROI = 34
AUTO_DETECT_HOUGH_PARAM2_FULL = 20
AUTO_DETECT_MIN_SCORE = 0.26
AUTO_DETECT_FRAME_COUNT = 5
AUTO_DETECT_MIN_VOTES = 3
AUTO_DETECT_VOTE_POSITION_TOLERANCE = 24
AUTO_DETECT_VOTE_RADIUS_TOLERANCE = 18
AUTO_DETECT_TEMPLATE_SIZE = 64
AUTO_DETECT_MAX_REFINED_CANDIDATES = 5

HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30

RENDER_STILL_THRESHOLD = 5
RENDER_STOPPED_STILL_THRESHOLD = 15
RENDER_EMA_ALPHA = 0.35
RENDER_EMA_ALPHA_MAX = 0.92
RENDER_EMA_SLOW_DIST = 6
RENDER_EMA_FAST_DIST = 45

TP_JUMP_THRESHOLD = 300
TP_CONFIRM_FRAMES = 3
TP_CLUSTER_RADIUS = 150

# ========== 阶段 2-3 优化：权重配置化 + 后验校验整合 ==========

# 得分权重表（按不同模式）
AUTO_DETECT_WEIGHTS = {
	'layout_prior': {  # 布局先验模式（右上角优先）
		'circle': 0.13,
		'edge': 0.11,
		'template': 0.13,
		'texture': 0.10,
		'anchor': 0.12,
		'ring_color': 0.11,
		'ring_continuity': 0.10,
		'ui_radius': 0.10,
		'corner_any': 0.10,
	},
	'fullscreen': {  # 全屏模式（任意位置）
		'circle': 0.11,
		'edge': 0.10,
		'template': 0.16,
		'texture': 0.11,
		'anchor': 0.0,  # 不使用锚点约束
		'ring_color': 0.14,
		'ring_continuity': 0.14,
		'ui_radius': 0.14,
		'corner_any': 0.10,
	},
}

# 快速预过滤权重（仅用 circle + edge + anchor）
AUTO_DETECT_QUICK_WEIGHTS = {
	'layout_prior': {'circle': 0.50, 'edge': 0.28, 'anchor': 0.22},
	'fullscreen': {'circle': 0.62, 'edge': 0.38},
}

# 后验校验规则（全屏模式）
AUTO_DETECT_POST_VALIDATION = {
	'min_corner_any_score': 0.24,
	'min_ring_continuity': 0.42,
	'auth_score_weights': {'ring_continuity': 0.45, 'template': 0.35, 'ui_radius': 0.20},
	'min_auth_score': 0.50,
	'scene_artifact_rules': [
		{'texture_min': 0.90, 'edge_max': 0.08},  # 高纹理 + 低边缘 = 场景噪声
	],
	'layout_specific_rules': {
		'top-left': {'ring_color_max': 0.18, 'ring_continuity_min': 0.75},  # 左上角软抑制
	},
}

# 检测范围控制（默认压缩到右上角以获得更低延迟）
# 可选: top_right / top_band / bottom_left / fullscreen / custom
AUTO_DETECT_DEFAULT_SCOPE = 'top_right'
# 当指定范围未命中时，是否兜底跑 full 范围（默认关闭以保证低延迟）
AUTO_DETECT_ENABLE_FULL_FALLBACK = False

