"""项目路径集中配置（根目录版本）。

说明：
- 所有跨模块共享的文件/目录路径从这里读取。
- 统一使用 pathlib.Path，调用方需要字符串时再 str(path)。
"""

from __future__ import annotations

from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# 项目根目录
# ════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT_DIR = Path(__file__).resolve().parent

# ════════════════════════════════════════════════════════════════════════════
# 前端相关路径
# ════════════════════════════════════════════════════════════════════════════
FRONTEND_DIR = PROJECT_ROOT_DIR / "frontend"
FRONTEND_IMG_DIR = FRONTEND_DIR / "img"
FRONTEND_BUILD_ACTIVE_DIR = PROJECT_ROOT_DIR / "frontend_build" / "active"

# ════════════════════════════════════════════════════════════════════════════
# 资源目录
# ════════════════════════════════════════════════════════════════════════════
ASSETS_ROOT_DIR = PROJECT_ROOT_DIR / "assets"
ASSETS_ICON_DIR = ASSETS_ROOT_DIR / "icon"
ASSETS_NPY_DIR = ASSETS_ROOT_DIR / "npy"
ASSETS_NPY_ICON_DIR = ASSETS_NPY_DIR / "icon"
ASSETS_MAP_DIR = ASSETS_ROOT_DIR / "map"
ASSETS_MARKER_DATA_DIR = ASSETS_ROOT_DIR / "map_data"

# ════════════════════════════════════════════════════════════════════════════
# 地图文件路径
# ════════════════════════════════════════════════════════════════════════════
LOGIC_MAP_PATH = ASSETS_MAP_DIR / "map_z7.webp"
DISPLAY_MAP_PATH = LOGIC_MAP_PATH

# ════════════════════════════════════════════════════════════════════════════
# 地图标记数据文件（供后端 API 与构建脚本共享）
# ════════════════════════════════════════════════════════════════════════════
MARKER_CATEGORIES_JSON_PATH = ASSETS_MARKER_DATA_DIR / "categories.json"
MARKER_LITE_JSON_PATH = ASSETS_MARKER_DATA_DIR / "rocom_markers_lite.json"
MARKER_DETAIL_JSON_PATH = ASSETS_MARKER_DATA_DIR / "rocom_markers_detail.json"

# ════════════════════════════════════════════════════════════════════════════
# 玩家朝向箭头模板（仅对外暴露 .npy 产物；源 PNG 保持 map_builder 私有）
# ════════════════════════════════════════════════════════════════════════════
PLAYER_ARROW_TEMPLATE_NPY_PATH = ASSETS_NPY_ICON_DIR / "arrow_up.npy"
