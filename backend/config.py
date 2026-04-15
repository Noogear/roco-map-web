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

# ==========================================
# SIFT 引擎核心参数
# ==========================================
SIFT_REFRESH_RATE = 50
SIFT_MATCH_RATIO = 0.82               # Lowe's Ratio (标准纹理)
SIFT_MIN_MATCH_COUNT = 5              # 最低匹配点数 (标准纹理)
SIFT_RANSAC_THRESHOLD = 8.0
SIFT_CONTRAST_THRESHOLD = 0.02        # SIFT 特征对比度 (越低越多弱纹理特征)
SIFT_MAX_HOMOGRAPHY_SCALE = 8.0
SIFT_GLOBAL_TILE_SIZE = 1536          # 全局特征分块边长，避免全图 detectAndCompute 峰值内存过高
SIFT_GLOBAL_TILE_OVERLAP = 96         # 分块重叠，覆盖 tile 边缘特征
SIFT_GLOBAL_MAX_FEATURES_PER_TILE = 0 # 0 表示不额外裁剪单块特征数量

# --- CLAHE 自适应 ---
# 双档: 纹理 std < LOW_TEXTURE_THRESHOLD → 低纹理(海洋/草/雪), clip 自动插值
CLAHE_LOW_TEXTURE_THRESHOLD = 40      # 扩大至 40，使草地等混合低纹理区也走低纹理 CLAHE 路径
CLAHE_LIMIT_NORMAL = 3.0              # 标准纹理 clip
CLAHE_LIMIT_LOW_TEXTURE = 6.0         # 低纹理 clip (连续插值到此)

# --- 搜索策略 ---
SEARCH_RADIUS = 400
LOCAL_FAIL_LIMIT = 3
SIFT_JUMP_THRESHOLD = 500
NEARBY_SEARCH_RADIUS = 600            # 匹配失败时邻近搜索范围
LOCAL_REVALIDATE_INTERVAL = 3         # 连续局部命中 N 次后强制做一次全局复核（降低：防止低纹理假成功自我强化）
LOCAL_REVALIDATE_MIN_QUALITY = 0.45   # 局部质量偏低时提前触发全局复核
LOCAL_REVALIDATE_MARGIN = 0.08        # 全局质量至少比局部高出该冗余才覆盖局部结果
LOCAL_REVALIDATE_DIFF = 220           # 全局/局部差异超过此值视为冲突，防局部错误自我强化

# --- LK 光流加速（每帧 ~2ms，降低 SIFT 调用频率）---
LK_ENABLED = True
LK_SIFT_INTERVAL = 4                  # 每 N 帧强制跑一次 SIFT 做漂移校正
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
WATCHDOG_SUSPECT_LIMIT = 3           # 连续 N 次 SIFT 帧检测到位移不一致则强制全局重定位
WATCHDOG_TRIGGER_COOLDOWN = 24       # 看门狗解锁后冷却帧数；冷却期内不再次触发硬解锁，避免短时间连环炸

# --- 状态冻结 ---
FREEZE_RESUME_SEARCH_RADIUS = 900      # 冻结恢复首帧的轻量提示搜索半径（先尝试旧坐标附近）

# --- 小地图圆形检测 ---
MINIMAP_CAPTURE_MARGIN = 1.4
MINIMAP_CIRCLE_CALIBRATION_FRAMES = 8
MINIMAP_CIRCLE_R_TOLERANCE = 8
MINIMAP_CIRCLE_CENTER_TOLERANCE = 15
MINIMAP_CIRCLE_RECALIBRATE_MISS = 30
MINIMAP_LOCAL_MIN_RADIUS_RATIO = 0.22   # 局部方形截取中，小地图半径占短边的最小比例
MINIMAP_LOCAL_MAX_RADIUS_RATIO = 0.48   # 局部方形截取中，小地图半径占短边的最大比例
MINIMAP_LOCAL_MIN_SCORE = 0.22          # 局部圆检测最低接受分（冻结态重建时防误判）
MINIMAP_CENTER_EXCLUDE_RATIO = 0.16        # 高纹理场景（urban）挖空比例
MINIMAP_CENTER_EXCLUDE_RATIO_MIXED = 0.17  # 混合纹理场景（mixed）挖空比例
MINIMAP_CENTER_EXCLUDE_RATIO_HARD = 0.18  # 低纹理/海洋场景（low_texture/ocean）挖空比例

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
# 饱和度(S)通道辅助 SIFT（低纹理/海洋场景）
# ==========================================
# 是否启用 S 通道辅助特征索引（启动时额外耗时 ~10-20s，内存 +30-50MB）
SIFT_SAT_ENABLED = True
# S 通道分块提取边长（同灰度通道）
SIFT_SAT_TILE_SIZE = 1536
# 每块最大特征数：控制总特征量和内存，0 表示不限（建议 500-800）
SIFT_SAT_MAX_FEATURES_PER_TILE = 600
# S 通道匹配 Lowe ratio（宽松，海洋特征重复性高）
SIFT_SAT_MATCH_RATIO = 0.90
# S 通道最低有效匹配点数
SIFT_SAT_MIN_MATCH = 3
# S 通道 ECC 相关系数阈值（比灰度 ECC 稍低，因 S 通道对比度更弱）
SIFT_SAT_ECC_MIN_CC = 0.28

# ==========================================
# 频域归一化互相关（phaseCorrelate）—— 低纹理/海洋兜底
# ==========================================
# 是否启用（依赖 last_x/last_y 位置提示）
PHASE_CORRELATE_ENABLED = True
# 响应值阈值：海洋场景实测约 0.85-0.95（+5px 偏移），保守阈值设 0.05
PHASE_CORRELATE_MIN_RESPONSE = 0.05
# 搜索窗口放大比：crop = minimap × scale × ratio；2.0 → ±150px 搜索范围（scale=4.0 时）
PHASE_CORRELATE_CROP_RATIO = 2.0

# --- 箭头方向 ---
ARROW_ANGLE_SMOOTH_ALPHA = 0.35
ARROW_MOVE_MIN_DISPLACEMENT = 6
ARROW_POS_HISTORY_LEN = 4
ARROW_STOPPED_DEBOUNCE = 20
ARROW_SNAP_THRESHOLD = 90

# --- 线性过滤 ---
LINEAR_FILTER_ENABLED = True
LINEAR_FILTER_WINDOW = 10
LINEAR_FILTER_MAX_DEVIATION = 120
LINEAR_FILTER_MAX_CONSECUTIVE = 10
RENDER_OFFSET_X = 0
RENDER_OFFSET_Y = 0

# --- 渲染平滑 ---
RENDER_STILL_THRESHOLD = 5
RENDER_EMA_ALPHA = 0.35          # 慢速时的最低 alpha（最平滑）
RENDER_EMA_ALPHA_MAX = 0.92      # 快速时的最高 alpha（立即跟随，无视觉滞后）
RENDER_EMA_SLOW_DIST = 6         # 低于此 px 差距使用 ALPHA 最小值
RENDER_EMA_FAST_DIST = 45        # 高于此 px 差距使用 ALPHA 最大值

# --- 传送检测 ---
TP_JUMP_THRESHOLD = 300
TP_CONFIRM_FRAMES = 3
TP_CLUSTER_RADIUS = 150

