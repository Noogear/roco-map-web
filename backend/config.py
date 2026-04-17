# ==========================================
# 游戏地图跟点助手 - 全局配置文件
# ==========================================
import os as _os
# 项目根目录（backend/ 的上一级）
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

PORT = 8686

# --- 屏幕截图区域 ---
MINIMAP = {"top": 292, "left": 1853, "width": 150, "height": 150}

# --- UI ---
WINDOW_GEOMETRY = "400x400+1500+100"
VIEW_SIZE = 400
JPEG_PAD = 40   # JPEG 模式扩边像素（单边），前端用此余量做亚帧微平移

# --- 地图文件 ---
_MAP_DIR = _os.path.join(_ROOT, "frontend", "img", "map")
LOGIC_MAP_PATH = _os.path.join(_MAP_DIR, "map_z7.webp")
DISPLAY_MAP_PATH = LOGIC_MAP_PATH

# --- 惯性导航 ---
MAX_LOST_FRAMES = 50
INERTIAL_MAX_HOLD_FRAMES = 12         # 识别连续失败时，惯性坐标最多保持 N 帧（10fps 下约 1.2s）
INERTIAL_MAX_HOLD_FRAMES_LOW_TEXTURE = 50  # 低纹理/海洋场景惯性保持上限，避免频繁清锚导致跳变

# ==========================================
# 特征匹配引擎核心参数（ORB + BEBLID）
# ==========================================
FEATURE_REFRESH_RATE = 50
FEATURE_MATCH_RATIO = 0.82               # Lowe's Ratio (标准纹理)
FEATURE_MIN_MATCH_COUNT = 5              # 最低匹配点数 (标准纹理)
FEATURE_RANSAC_THRESHOLD = 8.0
FEATURE_CONTRAST_THRESHOLD = 0.02        # 特征检测对比度阈值（ORB FAST threshold 参考值）
FEATURE_MAX_HOMOGRAPHY_SCALE = 8.0
FEATURE_GLOBAL_TILE_SIZE = 1536          # 全局特征分块边长，避免全图 detectAndCompute 峰值内存过高
FEATURE_GLOBAL_TILE_OVERLAP = 96         # 分块重叠，覆盖 tile 边缘特征
FEATURE_GLOBAL_MAX_FEATURES_PER_TILE = 0 # 0 表示不额外裁剪单块特征数量

# --- 匹配策略路由（BF / FLANN-LSH / GMS）---
MATCHER_POLICY_ENABLE = True
# 全局轮在高纹理场景下可优先尝试 FLANN-LSH（失败自动回退 BF）
MATCHER_GLOBAL_USE_LSH = True
MATCHER_LSH_CHECKS = 50
MATCHER_LSH_MIN_KP = 120
# GMS 仅在高纹理且特征点足够时启用（默认保守，只用于全局轮）
MATCHER_GMS_ENABLE = True
MATCHER_GMS_GLOBAL_ONLY = True
MATCHER_GMS_MIN_KP = 140
MATCHER_GMS_MIN_MATCHES = 20
MATCHER_GMS_WITH_ROTATION = False
MATCHER_GMS_WITH_SCALE = False

# --- 特征磁盘缓存（.npz 压缩，命中时跳过全图特征提取，大幅加速冷启动）---
FEATURE_CACHE_ENABLED = True             # 启用磁盘缓存；地图文件或提取参数变化时自动失效
FEATURE_CACHE_PATH = _os.path.join(_ROOT, 'assets', 'feature_cache_gray.npz')
FEATURE_CACHE_SAT_PATH = _os.path.join(_ROOT, 'assets', 'feature_cache_sat.npz')

# --- 多尺度特征池（应对游戏视角缩放，坐标归一化到原图坐标系后合并）---
FEATURE_MULTISCALE_ENABLED = True        # 启用多尺度提取；False 时退化为单尺度分块提取
FEATURE_MULTISCALE_SCALES = [0.75, 1.0, 1.25]  # 提取倍率列表

# --- CLAHE 自适应 ---
# 双档: 纹理 std < LOW_TEXTURE_THRESHOLD → 低纹理(海洋/草/雪), clip 自动插值
CLAHE_LOW_TEXTURE_THRESHOLD = 40      # 扩大至 40，使草地等混合低纹理区也走低纹理 CLAHE 路径
LOW_TEXTURE_LIKE_STD_THRESHOLD = 40   # mixed 帧中低于此阈值时，允许启用低纹理兜底链路（草坪边界场景）
CLAHE_LIMIT_NORMAL = 3.0              # 标准纹理 clip
CLAHE_LIMIT_LOW_TEXTURE = 6.0         # 低纹理 clip (连续插值到此)

# --- 搜索策略 ---
SEARCH_RADIUS = 400
LOCAL_FAIL_LIMIT = 3
FEATURE_JUMP_THRESHOLD = 500
NEARBY_SEARCH_RADIUS = 600            # 匹配失败时邻近搜索范围
LOCAL_REVALIDATE_INTERVAL = 15        # 连续局部命中 N 次后强制做一次全局复核（仅 quality<0.8 时计数）
LOCAL_REVALIDATE_MIN_QUALITY = 0.45   # 局部质量偏低时提前触发全局复核
LOCAL_REVALIDATE_MARGIN = 0.08        # 全局质量至少比局部高出该冗余才覆盖局部结果
LOCAL_REVALIDATE_DIFF = 220           # 全局/局部差异超过此值视为冲突，防局部错误自我强化

# --- LK 光流加速（每帧 ~2ms，降低特征匹配调用频率）---
LK_ENABLED = True
LK_FEATURE_INTERVAL = 4               # 每 N 帧强制跡一次特征匹配做漂移校正
LK_MIN_CONFIDENCE = 0.5               # 光流跟踪点中至少有此比例有效才采信

# --- ECC 低纹理兜底 ---
ECC_ENABLED = True
ECC_MIN_CORRELATION = 0.40            # findTransformECC 结果低于此阈值则放弃（提高：防草地/雪地重复纹理假成功）

# --- 哈希粗定位（低纹理 / 全局丢失回退） ---
HASH_INDEX_STEP = 60                  # 地图索引采样步长；越小越准，初始化越慢
HASH_INDEX_PATCH_SCALE = 4.0          # 逻辑地图 patch 相对小地图的缩放倍数
HASH_INDEX_PATCH_SIZE = 128           # 哈希前统一缩放到的 patch 尺寸
HASH_INDEX_HAMMING_THRESHOLD = 12     # 汉明距离上限；越小越严格
HASH_INDEX_COLOR_THRESH = 50          # 候选区块均值灰度与当前小地图均值的最大差异

# --- 运动看门狗（防止错误坐标自我强化死锁）---
WATCHDOG_LK_MIN_MOVE = 60            # LK 帧间累积位移超过此值（像素）才触发一致性检查
WATCHDOG_SUSPECT_LIMIT = 3           # 连续 N 次特征匹配帧检测到位移不一致则强制全局重定位
WATCHDOG_TRIGGER_COOLDOWN = 24       # 看门狗解锁后冷却帧数；冷却期内不再次触发硬解锁，避免短时间连环炸
WATCHDOG_HASH_MISMATCH_LIMIT = 6     # 哈希视觉不一致连续 N 次才触发解锁（独立于 LK 可疑计数）
WATCHDOG_HASH_HAMMING_MARGIN = 6     # 哈希“边缘不一致”容忍边际；仅明显超阈值才累计不一致
WATCHDOG_HASH_CHECK_RADIUS = 320     # 哈希一致性校验搜索半径；0 表示按索引默认半径
WATCHDOG_HASH_MIN_MATCH_QUALITY = 0.55  # 仅当特征匹配质量达到该阈值时才让哈希不一致参与看门狗计数
WATCHDOG_HASH_MIN_MATCH_COUNT = 6       # 仅当特征匹配内点数达到该阈值时才让哈希不一致参与看门狗计数
WATCHDOG_MAD_THRESHOLD = 27.0           # 看门狗主MAD阈值：高于此值认定画面明显变化（死锁确诊）；18→24→27减少低纹理/特效区边缘误触
WATCHDOG_INERTIAL_MIN_FEATURE_FAILS = 3   # INERTIAL 坐标锁定检测：特征匹配连续失败 N 次后开始检测画面变化
WATCHDOG_INERTIAL_MAD_THRESHOLD = 22.0 # INERTIAL 状态下画面MAD超过该阈值时提前退出（防止坐标卡死）

# --- 状态冻结 ---
FREEZE_RESUME_SEARCH_RADIUS = 900      # 冻结恢复首帧的轻量提示搜索半径（先尝试旧坐标附近）
FREEZE_RESUME_TELEPORT_MAD = 42.0      # 解冻时冻结前后小地图 MAD 超过此阈值则判定为传送，跳过旧坐标恢复提示

# --- 低纹理弱重定位防飘（抑制 phase/hash/ecc 单帧误跳） ---
WEAK_RELOCATE_GUARD_ENABLED = True
WEAK_RELOCATE_GUARD_JUMP = 120         # 弱来源跳变超过该阈值进入候选确认，不立即采信
WEAK_RELOCATE_GUARD_RADIUS = 90        # 连续候选聚合半径
WEAK_RELOCATE_GUARD_FRAMES = 2         # 连续多少帧候选聚合后才采信
WEAK_RELOCATE_GUARD_ADAPTIVE = True    # 按 LK 置信度自适应调整门槛
WEAK_RELOCATE_GUARD_CONF_FLOOR = 0.35  # LK 置信度低于该值视为低可信，采用更严格门槛
WEAK_RELOCATE_GUARD_JUMP_SCALE_MIN = 0.70   # 低可信时 jump 阈值缩放（更严格）
WEAK_RELOCATE_GUARD_JUMP_SCALE_MAX = 1.25   # 高可信时 jump 阈值缩放（更灵活）
WEAK_RELOCATE_GUARD_RADIUS_SCALE_MIN = 0.70 # 低可信时候选半径缩放（更严格）
WEAK_RELOCATE_GUARD_RADIUS_SCALE_MAX = 1.20 # 高可信时候选半径缩放（更灵活）
WEAK_RELOCATE_GUARD_FRAMES_MAX = 3          # 自适应下最大确认帧数

# --- 小地图圆形检测 ---
MINIMAP_CAPTURE_MARGIN = 1.4
MINIMAP_CIRCLE_CALIBRATION_FRAMES = 8
MINIMAP_CIRCLE_R_TOLERANCE = 8
MINIMAP_CIRCLE_CENTER_TOLERANCE = 15
MINIMAP_CIRCLE_RECALIBRATE_MISS = 30
MINIMAP_LOCAL_MIN_RADIUS_RATIO = 0.22   # 局部方形截取中，小地图半径占短边的最小比例
MINIMAP_LOCAL_MAX_RADIUS_RATIO = 0.48   # 局部方形截取中，小地图半径占短边的最大比例
MINIMAP_LOCAL_MIN_SCORE = 0.22          # 局部圆检测最低接受分（冻结态重建时防误判）
MINIMAP_MASK_EDGE_MARGIN_RATIO = 0.08   # 小地图圆形 mask 外圈内缩比例，避免 UI 边框/阴影泄漏
MINIMAP_MASK_EDGE_MARGIN_MIN_PIXELS = 6 # 小地图圆形 mask 最小内缩像素
MINIMAP_CENTER_EXCLUDE_RATIO = 0.16        # 高纹理场景（urban）挖空比例
MINIMAP_CENTER_EXCLUDE_RATIO_MIXED = 0.18  # 混合纹理场景（mixed）挖空比例
MINIMAP_CENTER_EXCLUDE_RATIO_HARD = 0.20  # 低纹理/海洋场景（low_texture/ocean）挖空比例

# --- 全帧自动识别小地图圆（Web 屏幕捕获用） ---
AUTO_DETECT_MAX_SIDE = 1280             # 自动定位前整帧缩放的最大边，平衡速度与精度
AUTO_DETECT_MIN_RADIUS_RATIO = 0.035    # 小地图半径占整帧短边的最小比例
AUTO_DETECT_MAX_RADIUS_RATIO = 0.16     # 小地图半径占整帧短边的最大比例
AUTO_DETECT_HOUGH_PARAM1 = 90           # 全帧 HoughCircles 高阈值
AUTO_DETECT_HOUGH_PARAM2_ROI = 24       # 右上/左上 ROI 的 Hough 判定阈值（更严格）
AUTO_DETECT_HOUGH_PARAM2_FULL = 20      # 全图回退扫描的 Hough 判定阈值（更宽松）
AUTO_DETECT_MIN_SCORE = 0.26            # 单帧候选最低接受分
AUTO_DETECT_FRAME_COUNT = 5             # 自动定位默认采样帧数（多帧投票）
AUTO_DETECT_MIN_VOTES = 3               # 多帧投票至少命中的帧数
AUTO_DETECT_VOTE_POSITION_TOLERANCE = 24  # 多帧投票中圆心聚类容差（像素）
AUTO_DETECT_VOTE_RADIUS_TOLERANCE = 18    # 多帧投票中半径聚类容差（像素）
AUTO_DETECT_TEMPLATE_SIZE = 64          # 合成圆环模板尺寸
AUTO_DETECT_MAX_REFINED_CANDIDATES = 6  # 昂贵模板/纹理评分前最多保留的候选数

# ==========================================
# 饱和度(S)通道辅助 ORB（低纹理/海洋场景）
# ==========================================
# 是否启用 S 通道辅助特征索引（启动时额外耗时 ~10-20s，内存 +30-50MB）
SAT_ORB_ENABLED = True
# S 通道分块提取边长（同灰度通道）
SAT_ORB_TILE_SIZE = 1536
# 每块最大特征数：控制总特征量和内存，0 表示不限（建议 500-800）
SAT_ORB_MAX_FEATURES_PER_TILE = 600
# S 通道匹配 Lowe ratio（宽松，海洋特征重复性高）
SAT_ORB_MATCH_RATIO = 0.90
# S 通道最低有效匹配点数
SAT_ORB_MIN_MATCH = 3
# S 通道 ECC 相关系数阈值（比灰度 ECC 稍低，因 S 通道对比度更弱）
SAT_ORB_ECC_MIN_CC = 0.28

# ==========================================
# 频域归一化互相关（phaseCorrelate）—— 低纹理/海洋兜底
# ==========================================
# 是否启用（依赖 last_x/last_y 位置提示）
PHASE_CORRELATE_ENABLED = True
# 响应值阈值：海洋场景实测约 0.85-0.95（+5px 偏移），保守阈值设 0.05
PHASE_CORRELATE_MIN_RESPONSE = 0.05
# 搜索窗口放大比：crop = minimap × scale × ratio；2.0 → ±150px 搜索范围（scale=4.0 时）
PHASE_CORRELATE_CROP_RATIO = 2.0

# ==========================================
# 低纹理时序桥接（多假设轨迹）
# ==========================================
LOW_TEXTURE_BRIDGE_ENABLED = True
LOW_TEXTURE_BRIDGE_TOP_K = 8
LOW_TEXTURE_BRIDGE_DECAY = 0.92
LOW_TEXTURE_BRIDGE_TRANSITION_PENALTY = 0.002
LOW_TEXTURE_BRIDGE_MIN_OBS_QUALITY = 0.08
LOW_TEXTURE_BRIDGE_STRONG_SOURCE_BONUS = 0.12
LOW_TEXTURE_BRIDGE_EXIT_MIN_QUALITY = 0.55
LOW_TEXTURE_BRIDGE_HASH_CANDIDATES = 4
LOW_TEXTURE_BRIDGE_HASH_RADIUS = 900

# ==========================================
# ORB + BEBLID 核心参数（硬依赖 opencv-contrib-python）
# ==========================================
ORB_BEBLID_NFEATURES = 7000
ORB_BEBLID_FAST_THRESHOLD = 6
ORB_BEBLID_EDGE_THRESHOLD = 15
ORB_BEBLID_SCALE_FACTOR = 1.0
ORB_BEBLID_BITS = 512
ORB_BEBLID_RANSAC_THRESHOLD = 6.0
ORB_BEBLID_MAX_AFFINE_SCALE = 2.0
ORB_BEBLID_QUALITY_NORM_COUNT = 18.0

# 匹配质量过滤：按描述子距离排序取前 K 个最佳匹配，去除噪声匹配（0=禁用）
FEATURE_MATCH_TOP_K = 0

# --- 网格均匀特征提取（地图侧，借鉴 LKMapTools 关键技术）---
# 混合模式：标准 tiled 提取为主，低特征密度 tile 自动补充网格提取
# 确保草坪/海洋等低纹理区也有特征覆盖，同时不降低高纹理区描述子质量
FEATURE_GRID_ENABLED = True
FEATURE_GRID_CELL_SIZE = 96              # 子格边长（px），需 >= 64 以容纳 BEBLID patch
FEATURE_GRID_FEATURES_PER_CELL = 30      # 每子格最多提取特征数
FEATURE_GRID_FAST_THRESHOLD = 2          # 子格 ORB FAST 阈值（极低，检测草坪微弱角点）
FEATURE_GRID_EDGE_THRESHOLD = 1          # 子格 ORB 边缘阈值（极低，允许边缘特征）
FEATURE_GRID_MIN_TILE_FEATURES = 2000    # tile 特征少于此值时启用网格补充

# ==========================================
# 场景颜色细化（grass/ocean/snow）
# ==========================================
SCENE_COLOR_ENABLED = True
SCENE_BOOSTED_GRAY_ENABLED = True

# 是否在 prior_scene='urban' 时也做颜色细化（提升草坪识别召回）
SCENE_COLOR_REFINE_URBAN = True

# 覆盖率阈值（比例）
SCENE_COLOR_OCEAN_THRESH = 0.28
SCENE_COLOR_GRASS_THRESH = 0.22
SCENE_COLOR_GRASS_THRESH_URBAN = 0.30
SCENE_COLOR_SNOW_THRESH = 0.38

# H/S/V 规则阈值（OpenCV H: 0-179）
SCENE_COLOR_OCEAN_H_MIN = 90
SCENE_COLOR_OCEAN_H_MAX = 130
SCENE_COLOR_OCEAN_S_MIN = 50
SCENE_COLOR_OCEAN_V_MIN = 50

SCENE_COLOR_GRASS_H_MIN = 30
SCENE_COLOR_GRASS_H_MAX = 92
SCENE_COLOR_GRASS_S_MIN = 28

SCENE_COLOR_SNOW_S_MAX = 45
SCENE_COLOR_SNOW_V_MIN = 185

# 草坪场景强制进入低纹理稳态链路（phase/sat/hash/bridge）
SCENE_COLOR_FORCE_LOW_TEXTURE_GRASSLAND = True
SCENE_COLOR_FORCE_LOW_TEXTURE_STD_MAX = 36.0

# --- 箭头方向 ---
ARROW_ANGLE_SMOOTH_ALPHA = 0.35
ARROW_MOVE_MIN_DISPLACEMENT = 6
ARROW_POS_HISTORY_LEN = 4
ARROW_STOPPED_DEBOUNCE = 20
ARROW_SMALL_CHANGE_THRESHOLD = 12.0    # 小于此角度变化时抑制输出（防抖）
ARROW_BIG_CHANGE_THRESHOLD = 45.0      # 大于此角度变化时立即切换（保持敏感）

# --- 线性过滤 ---
LINEAR_FILTER_ENABLED = True
LINEAR_FILTER_WINDOW = 10
LINEAR_FILTER_MAX_DEVIATION = 120
LINEAR_FILTER_MAX_CONSECUTIVE = 10
RENDER_OFFSET_X = 0
RENDER_OFFSET_Y = 0

# --- 渲染平滑 ---
RENDER_STILL_THRESHOLD = 2
RENDER_STOPPED_STILL_THRESHOLD = 2  # 静止判定时的渲染死区；与 STILL 一致以避免低速长时间不刷新
RENDER_EMA_ALPHA = 0.35          # 慢速时的最低 alpha（最平滑）
RENDER_EMA_ALPHA_MAX = 0.92      # 快速时的最高 alpha（立即跟随，无视觉滞后）
RENDER_EMA_SLOW_DIST = 6         # 低于此 px 差距使用 ALPHA 最小值
RENDER_EMA_FAST_DIST = 45        # 高于此 px 差距使用 ALPHA 最大值

# --- 传送检测 ---
TP_JUMP_THRESHOLD = 300
TP_CONFIRM_FRAMES = 3
TP_CLUSTER_RADIUS = 150

