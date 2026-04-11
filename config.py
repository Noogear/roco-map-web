# ==========================================
# 游戏地图跟点助手 - 全局配置文件
# ==========================================
# --- 0. 服务器设置 ---
PORT = 8686

# --- 1. 屏幕截图区域 (Minimap Region) ---
# 请根据你的显示器分辨率和游戏 UI 调整这些值
# 可以通过微信/QQ截图工具查看具体像素坐标
MINIMAP = {
    "top": 292,
    "left": 1853,
    "width": 150,
    "height": 150
}

# --- 2. 悬浮窗 UI 设置 ---
WINDOW_GEOMETRY = "400x400+1500+100"  # 悬浮窗宽x高+X坐标+Y坐标
VIEW_SIZE = 400                       # 悬浮窗内显示的地图视野大小

# --- 3. 地图文件路径 ---
LOGIC_MAP_PATH = "web/big_map.png"        # 用于特征提取的纯净底图
DISPLAY_MAP_PATH = "web/big_map-1.png"    # 用于显示的带标记 UI 图

# --- 4. 惯性导航设置 (防跟丢兜底) ---
MAX_LOST_FRAMES = 50                  # 最大容忍丢失帧数 (约 10 秒)

# ==========================================
# SIFT 传统视觉算法专属配置 (main_sift.py)
# ==========================================
SIFT_REFRESH_RATE = 50                # 刷新延迟 (毫秒)，50ms 约等于 20fps
SIFT_CLAHE_LIMIT = 3.0                # CLAHE 对比度增强极限 (用于榨取海水/草地纹理)
SIFT_MATCH_RATIO = 0.82               # Lowe's Ratio 阈值 (越大越容易匹配，但错判率越高)
SIFT_MIN_MATCH_COUNT = 5              # 判定成功所需的最低匹配点数
SIFT_RANSAC_THRESHOLD = 8.0           # 允许的空间误差阈值

# === 局部优先搜索设置 (main123 优化) ===
SEARCH_RADIUS = 400                  # 局部搜索半径（像素）
LOCAL_FAIL_LIMIT = 3                 # 局部搜索连续失败 N 帧后回退全局
SIFT_JUMP_THRESHOLD = 500            # 局部模式下坐标最大允许跳变距离（像素）
SIFT_MAX_HOMOGRAPHY_SCALE = 8.0      # Homography 最大允许缩放 (防UI误匹配，正常≈1-3)

# === 自适应 CLAHE（按纹理复杂度切换增强档位）===
CLAHE_LOW_TEXTURE_THRESHOLD = 30       # 灰度标准差 < 此值 → 低纹理（海水/草地）
CLAHE_HIGH_TEXTURE_THRESHOLD = 60      # 灰度标准差 > 此值 → 高纹理（城镇/建筑）
CLAHE_LIMIT_LOW_TEXTURE = 5.0          # 低纹理区域 CLAHE clip limit（更激进）
CLAHE_LIMIT_HIGH_TEXTURE = 2.0         # 高纹理区域 CLAHE clip limit（更温和）

# === SIFT 检测参数 ===
SIFT_CONTRAST_THRESHOLD = 0.02        # SIFT 特征对比度阈值（默认0.04，越低特征越多）

# === ORB 快速备份引擎 ===
ORB_BACKUP_ENABLED = True              # SIFT 失败时启用 ORB 快速备份匹配
ORB_NFEATURES = 500                    # ORB 特征数上限

# === BLOCK 状态冻结 ===
FREEZE_TIMEOUT = 30.0                  # 冻结状态超时（秒），超时后丢弃冻结数据

# === 小地图圆形检测 (方形截取模式) ===
MINIMAP_CAPTURE_MARGIN = 1.4            # 浏览器方形截取扩展系数 (相对于圆直径)
MINIMAP_CIRCLE_CALIBRATION_FRAMES = 8   # 自校准所需连续成功帧数
MINIMAP_CIRCLE_R_TOLERANCE = 8          # 校准后允许半径偏差 (像素)
MINIMAP_CIRCLE_CENTER_TOLERANCE = 15    # 校准后允许圆心偏移 (像素)
MINIMAP_CIRCLE_RECALIBRATE_MISS = 30    # 连续多少帧未检测到圆 → 重新校准

# === 箭头检测优化 ===
ARROW_TEMPLATE_MATCH_THRESHOLD = 0.65   # 模板匹配相关系数阈值 (快速路径)
ARROW_ROI_PADDING = 15                  # 模板匹配 ROI 扩展像素
ARROW_HSV_LEARN_ALPHA = 0.1             # HSV 范围自适应学习率
ARROW_ANGLE_SMOOTH_ALPHA = 0.4          # 角度 EMA 平滑系数 (0=不平滑, 1=无惯性)

# === 坐标锁定模式设置 ===
COORD_LOCK_ENABLED = False             # 是否启用坐标锁定（运行时动态切换）
COORD_LOCK_HISTORY_SIZE = 10           # 锚点计算：取最近 N 个历史坐标的平均值
COORD_LOCK_SEARCH_RADIUS = 400         # 锁定后允许的搜索半径（像素）
COORD_LOCK_MAX_RETRIES = 5             # 单帧最大重试次数
COORD_LOCK_MIN_HISTORY_TO_ACTIVATE = 10   # 至少积累多少个历史坐标才允许开启锁定

# === 线性速度一致性过滤（丢弃非线性跳变）===
LINEAR_FILTER_ENABLED = True           # 是否启用线性过滤（锁定模式自动生效）
LINEAR_FILTER_WINDOW = 10              # 取最近多少帧计算平均速度
LINEAR_FILTER_MAX_DEVIATION = 120      # 允许偏离预测位置的最大距离（像素）
LINEAR_FILTER_MAX_CONSECUTIVE = 10     # 连续丢弃多少帧后强制接受真实坐标（防死锁）
# 正值 = 向右/下偏移，负值 = 向左/上偏移。用于微调定位点位置。
RENDER_OFFSET_X = 0                  # 像素偏移（如果定位点偏左，改为正值如 +10）
RENDER_OFFSET_Y = 0                  # 像素偏移

# ==========================================
# LoFTR AI 深度学习算法专属配置 (main_ai.py)
# ==========================================
AI_REFRESH_RATE = 200                 # AI 推理耗时较高，建议 200ms (5fps)
AI_CONFIDENCE_THRESHOLD = 0.25        # AI 置信度阈值 (越低越容易妥协)
AI_MIN_MATCH_COUNT = 6                # 判定成功所需的最低匹配点数
AI_RANSAC_THRESHOLD = 8.0             # 允许的空间误差阈值
# 雷达扫描参数
AI_SCAN_SIZE = 1600                   # 全局搜索时的区块大小
AI_SCAN_STEP = 1400                   # 全局搜索的步长
AI_TRACK_RADIUS = 500                 # 局部追踪时，向外扩展的半径 (400即截取800x800)

# ==========================================
# 混合引擎模式 (SIFT 主引擎 + LoFTR 后台重定位)
# ==========================================
HYBRID_ENABLED = True                  # 非 sift-only 模式下自动启用混合引擎
HYBRID_TRIGGER_LOST_FRAMES = 5         # SIFT 连续惯性 N 帧后触发 LoFTR 重定位
HYBRID_CONFUSED_IMMEDIATE = True       # SIFT 几何混乱时立即触发 LoFTR（不等 N 帧）
HYBRID_COOLDOWN = 3.0                  # LoFTR 两次触发最小间隔（秒）
HYBRID_COARSE_TILE = 400               # 粗扫阶段的 tile 目标尺寸（像素，缩放后）
HYBRID_COARSE_STEP = 350               # 粗扫步长（缩放前 = step*4）
HYBRID_FINE_RADIUS = 300               # 精定位阶段的 crop 半径（像素）
HYBRID_MINI_SIZE = 128                 # LoFTR 输入的 minimap 统一尺寸