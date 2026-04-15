"""
坐标验证脚本：在地图上叠加从 17173 API 抓取的标记点，
与 draw_markers.py 的内部坐标系做视觉比对。

依赖：需要已下载的 test_map_z12.png（先运行 download_map.py）
输出：verify_markers.png — 仅绘制「魔力之源（传送点）」以便快速验证

运行：python verify_markers.py
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_MAP = os.path.join(_ROOT, "output", "map_z7.webp")
OUTPUT = os.path.join(_ROOT, "output", "verify_markers.png")
MARKERS_JSON = os.path.join(_HERE, "rocom_markers.json")
ICON_DIR = os.path.join(_ROOT, "frontend", "img")

# ── 坐标常数（与 draw_markers.py 一致，z=7） ─────────────────────────────────────
X_MIN = -12
Y_MIN = -11
TILE_SIZE = 256
ORIGIN_X = -X_MIN * TILE_SIZE   # 3072
ORIGIN_Y = -Y_MIN * TILE_SIZE   # 2816

# ── 用于对比的现有硬编码样本（draw_markers.py 中已知正确的点） ──────────────
# 格式：(title, lat, lng)  ← internal 坐标
KNOWN_POINTS = [
    # 从原 points_data 中摘取几个代表性的点（category 701 = 魔力之源/传送点）
    # 如果你的 markType 701 数据里有这些，可在这里列出做对比
    # 例：("城区中央广场", -100, 200),
]

# ── 要单独验证的 category_id（传送点类） ───────────────────────────────────
VERIFY_CAT = 17310030038   # 魔力之源（传送点）


def load_font(size=18):
    for f in ["C:/Windows/Fonts/msyh.ttc",
              "C:/Windows/Fonts/simhei.ttf",
              "C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_circle(draw, cx, cy, r, fill, outline="white"):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=2)


def main():
    if not os.path.exists(INPUT_MAP):
        print(f"[错误] 找不到 {INPUT_MAP}，请先运行 download_map.py 下载地图切片。")
        return

    if not os.path.exists(MARKERS_JSON):
        print(f"[错误] 找不到 {MARKERS_JSON}，请先运行 fetch_markers.py。")
        return

    with open(MARKERS_JSON, encoding="utf-8") as f:
        markers = json.load(f)

    img = Image.open(INPUT_MAP).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(16)
    font_small = load_font(12)

    # ── 1. 绘制从 API 抓取的「传送点」（蓝色圆点）──────────────────────────
    api_points = [m for m in markers if m.get("markType") == VERIFY_CAT]
    print(f"找到 {len(api_points)} 个「魔力之源（传送点）」（category {VERIFY_CAT}）")

    # 尝试加载图标
    icon_path = os.path.join(ICON_DIR, f"{VERIFY_CAT}.png")
    icon = None
    if os.path.exists(icon_path):
        try:
            icon = Image.open(icon_path).convert("RGBA").resize((48, 48), Image.Resampling.LANCZOS)
        except Exception:
            pass

    for m in api_points:
        lat = m["point"]["lat"]
        lng = m["point"]["lng"]
        px = lng + ORIGIN_X
        py = lat + ORIGIN_Y
        title = m.get("title", "")

        if icon:
            paste_x = px - icon.width // 2
            paste_y = py - icon.height // 2
            overlay.paste(icon, (paste_x, paste_y), mask=icon)
        # 在图标/坐标点上叠加高亮圆环，方便在大图中快速定位
        draw_circle(draw, px, py, 22, fill=(0, 0, 0, 0), outline=(255, 80, 0, 230))
        draw_circle(draw, px, py, 4, fill=(255, 80, 0, 200))

        # 简短标注（每隔 5 个才显示，避免文字堆叠）
        if api_points.index(m) % 5 == 0 and title:
            draw.text((px + 8, py - 8), title, font=font_small, fill=(255, 255, 80, 220),
                      stroke_width=1, stroke_fill=(0, 0, 0, 180))

    # ── 2. 绘制原有 draw_markers.py 已知正确点（红色叉标记，用于比对）────────
    for title, lat, lng in KNOWN_POINTS:
        px = lng + ORIGIN_X
        py = lat + ORIGIN_Y
        r = 10
        draw.line((px - r, py - r, px + r, py + r), fill=(255, 50, 50, 255), width=3)
        draw.line((px - r, py + r, px + r, py - r), fill=(255, 50, 50, 255), width=3)
        draw.text((px + 12, py - 10), title, font=font, fill=(255, 50, 50, 230),
                  stroke_width=1, stroke_fill=(0, 0, 0, 200))

    # ── 3. 叠加图层并输出全分辨率 ──────────────────────────────────────────
    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(OUTPUT)
    print(f"[OK] 验证图已保存：{OUTPUT}")
    print(f"     尺寸：{combined.width}x{combined.height}")
    print(f"     找到 {len(api_points)} 个标记点，已用橙色圆环高亮标出。")


if __name__ == "__main__":
    main()
