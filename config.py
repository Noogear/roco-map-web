# ==========================================
# 游戏地图跟点助手 - 全局配置文件
# ==========================================
PORT = 8686

# --- 屏幕截图区域 ---
MINIMAP = {"top": 292, "left": 1853, "width": 150, "height": 150}

# --- UI ---
WINDOW_GEOMETRY = "400x400+1500+100"
VIEW_SIZE = 400

# --- 地图文件 ---
LOGIC_MAP_PATH = "web/big_map.png"
DISPLAY_MAP_PATH = "web/big_map-1.png"

# --- 惯性导航 ---
MAX_LOST_FRAMES = 50

# ==========================================
# SIFT 引擎核心参数
# ==========================================
SIFT_REFRESH_RATE = 50
SIFT_MATCH_RATIO = 0.82               # Lowe's Ratio (标准纹理)
SIFT_MIN_MATCH_COUNT = 5              # 最低匹配点数 (标准纹理)
SIFT_RANSAC_THRESHOLD = 8.0
SIFT_CONTRAST_THRESHOLD = 0.02        # SIFT 特征对比度 (越低越多弱纹理特征)
SIFT_MAX_HOMOGRAPHY_SCALE = 8.0

# --- CLAHE 自适应 ---
# 双档: 纹理 std < LOW_TEXTURE_THRESHOLD → 低纹理(海洋/草/雪), clip 自动插值
CLAHE_LOW_TEXTURE_THRESHOLD = 30
CLAHE_LIMIT_NORMAL = 3.0              # 标准纹理 clip
CLAHE_LIMIT_LOW_TEXTURE = 6.0         # 低纹理 clip (连续插值到此)

# --- 搜索策略 ---
SEARCH_RADIUS = 400
LOCAL_FAIL_LIMIT = 3
SIFT_JUMP_THRESHOLD = 500
NEARBY_SEARCH_RADIUS = 600            # 匹配失败时邻近搜索范围

# --- 状态冻结 ---
FREEZE_TIMEOUT = 30.0

# --- 小地图圆形检测 ---
MINIMAP_CAPTURE_MARGIN = 1.4
MINIMAP_CIRCLE_CALIBRATION_FRAMES = 8
MINIMAP_CIRCLE_R_TOLERANCE = 8
MINIMAP_CIRCLE_CENTER_TOLERANCE = 15
MINIMAP_CIRCLE_RECALIBRATE_MISS = 30

# --- 箭头方向 ---
ARROW_ANGLE_SMOOTH_ALPHA = 0.35
ARROW_MOVE_MIN_DISPLACEMENT = 6
ARROW_POS_HISTORY_LEN = 4
ARROW_STOPPED_DEBOUNCE = 20
ARROW_SNAP_THRESHOLD = 90

# --- 坐标锁定 ---
COORD_LOCK_ENABLED = False
COORD_LOCK_HISTORY_SIZE = 10
COORD_LOCK_SEARCH_RADIUS = 400
COORD_LOCK_MAX_RETRIES = 5
COORD_LOCK_MIN_HISTORY_TO_ACTIVATE = 10

# --- 线性过滤 ---
LINEAR_FILTER_ENABLED = True
LINEAR_FILTER_WINDOW = 10
LINEAR_FILTER_MAX_DEVIATION = 120
LINEAR_FILTER_MAX_CONSECUTIVE = 10
RENDER_OFFSET_X = 0
RENDER_OFFSET_Y = 0

# --- 渲染平滑 ---
RENDER_STILL_THRESHOLD = 5
RENDER_EMA_ALPHA = 0.35

# --- 传送检测 ---
TP_JUMP_THRESHOLD = 300
TP_CONFIRM_FRAMES = 3
TP_CLUSTER_RADIUS = 150

# ==========================================
# LoFTR AI 引擎
# ==========================================
AI_REFRESH_RATE = 200
AI_CONFIDENCE_THRESHOLD = 0.25
AI_MIN_MATCH_COUNT = 6
AI_RANSAC_THRESHOLD = 8.0
AI_SCAN_SIZE = 1600
AI_SCAN_STEP = 1400
AI_TRACK_RADIUS = 500

# --- 混合引擎 ---
HYBRID_ENABLED = True
HYBRID_TRIGGER_LOST_FRAMES = 5
HYBRID_CONFUSED_IMMEDIATE = True       # SIFT 几何混乱时立即触发 LoFTR（不等 N 帧）
HYBRID_COOLDOWN = 3.0                  # LoFTR 两次触发最小间隔（秒）
HYBRID_COARSE_TILE = 400               # 粗扫阶段的 tile 目标尺寸（像素，缩放后）
HYBRID_COARSE_STEP = 350               # 粗扫步长（缩放前 = step*4）
HYBRID_FINE_RADIUS = 300               # 精定位阶段的 crop 半径（像素）
HYBRID_MINI_SIZE = 128                 # LoFTR 输入的 minimap 统一尺寸