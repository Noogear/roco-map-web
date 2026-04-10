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
SIFT_MATCH_RATIO = 0.9                # Lowe's Ratio 阈值 (越大越容易匹配，但错判率越高)
SIFT_MIN_MATCH_COUNT = 5              # 判定成功所需的最低匹配点数
SIFT_RANSAC_THRESHOLD = 8.0           # 允许的空间误差阈值

# === 局部优先搜索设置 (main123 优化) ===
SEARCH_RADIUS = 400                  # 局部搜索半径（像素）
LOCAL_FAIL_LIMIT = 3                 # 局部搜索连续失败 N 帧后回退全局
SIFT_JUMP_THRESHOLD = 500            # 局部模式下坐标最大允许跳变距离（像素）

# === 渲染偏移校正 ===
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