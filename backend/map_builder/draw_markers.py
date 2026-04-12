from PIL import Image
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))

# ─── 坐标常数（与 fetch_markers.py 一致，z=7） ────────────────────────────────────────────
X_MIN = -12
Y_MIN = -11
TILE_SIZE = 256
ORIGIN_X = -X_MIN * TILE_SIZE   # 3072
ORIGIN_Y = -Y_MIN * TILE_SIZE   # 2816

ICON_SIZE    = 36
ICON_DIR     = os.path.join(_ROOT, "frontend", "img")
INPUT_MAP    = os.path.join(_ROOT, "output", "map_z7.webp")
OUTPUT_MAP   = os.path.join(_ROOT, "output", "map_with_markers.jpg")
MARKERS_JSON = os.path.join(_HERE, "rocom_markers.json")


def draw_markers():
    if not os.path.exists(INPUT_MAP):
        print(f"[错误] 找不到地图文件: {INPUT_MAP}，请先运行 download_map.py")
        return

    if not os.path.exists(MARKERS_JSON):
        print(f"[错误] 找不到标记数据: {MARKERS_JSON}，请先运行 fetch_markers.py")
        return

    with open(MARKERS_JSON, encoding="utf-8") as f:
        points_data = json.load(f)

    print(f"已加载 {len(points_data)} 个标记点")

    img = Image.open(INPUT_MAP).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    icon_cache = {}
    missing_types = set()

    for item in points_data:
        lat = item["point"]["lat"]
        lng = item["point"]["lng"]
        mark_type = item.get("markType")
        px = lng + ORIGIN_X
        py = lat + ORIGIN_Y

        # 图标（带缓存，避免重复读盘）
        if mark_type not in icon_cache:
            icon_path = os.path.join(ICON_DIR, f"{mark_type}.png")
            if os.path.exists(icon_path):
                raw = Image.open(icon_path).convert("RGBA")
                icon_cache[mark_type] = raw.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
            else:
                icon_cache[mark_type] = None
                missing_types.add(mark_type)

        icon = icon_cache[mark_type]
        if icon:
            paste_x = px - ICON_SIZE // 2
            paste_y = py - ICON_SIZE // 2
            overlay.paste(icon, (paste_x, paste_y), mask=icon)

    if missing_types:
        print(f"[警告] 以下 markType 缺少图标文件: {sorted(missing_types)}")

    combined = Image.alpha_composite(img, overlay)
    result = combined.convert("RGB")
    result.save(OUTPUT_MAP, quality=92, optimize=True)
    print(f"[OK] 已保存: {OUTPUT_MAP}  ({len(points_data)} 个标记点，尺寸 {result.width}x{result.height})")


if __name__ == "__main__":
    draw_markers()
