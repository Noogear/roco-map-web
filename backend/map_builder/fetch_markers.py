"""
从 17173.com 互动地图 API 抓取洛克王国世界所有标记点，
转换为 draw_markers.py 所使用的内部坐标格式，
并将图标下载到 web/img/ 目录。

运行完成后会输出 rocom_markers.json，
可直接替换 draw_markers.py 中的 points_data 列表。
"""

import json
import os
import sys
import requests

# ── 配置 ──────────────────────────────────────────────────────────────────────
MAP_ID = 4010
API_URL = f"https://map.17173.com/app/location/list?mapIds={MAP_ID}"
ICON_CDN = "https://ue.17173cdn.com/a/terra/icon/rocom/{cat_id}.png"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(_ROOT, "frontend", "img")   # 图标保存路径
OUTPUT_JSON = os.path.join(_HERE, "rocom_markers.json")

# 地图 Mapbox GL bounds：[west_lng, south_lat, east_lng, north_lat]
BOUNDS_W, BOUNDS_S, BOUNDS_E, BOUNDS_N = -1.4, 0.0, 0.0, 1.4

# 瓦片网格范围（与 download_map.py 保持一致，z=7）
X_MIN, X_MAX = -12, 11   # 共 24 列
Y_MIN, Y_MAX = -11, 11   # 共 23 行
TILE_SIZE = 256
MAP_W = (X_MAX - X_MIN + 1) * TILE_SIZE   # 24 * 256 = 6144 px
MAP_H = (Y_MAX - Y_MIN + 1) * TILE_SIZE   # 23 * 256 = 5888 px
# 内部坐标原点 = 瓦片 (0,0) 的左上角像素位置
ORIGIN_X = -X_MIN * TILE_SIZE   # 12 * 256 = 3072
ORIGIN_Y = -Y_MIN * TILE_SIZE   # 11 * 256 = 2816


# ── 坐标转换 ──────────────────────────────────────────────────────────────────
def api_to_point(latitude: float, longitude: float) -> dict:
    """将 17173 API 的 latitude/longitude 转换为 draw_markers.py 的 {lat, lng}。"""
    px = (longitude - BOUNDS_W) / (BOUNDS_E - BOUNDS_W) * MAP_W
    py = (BOUNDS_N - latitude) / (BOUNDS_N - BOUNDS_S) * MAP_H
    return {
        "lat": round(py) - ORIGIN_Y,
        "lng": round(px) - ORIGIN_X,
    }


# ── 图标下载 ──────────────────────────────────────────────────────────────────
def download_icon(cat_id: int, session: requests.Session) -> bool:
    """下载单个分类图标到 ICON_DIR，已存在则跳过。返回是否成功。"""
    dest = os.path.join(ICON_DIR, f"{cat_id}.png")
    if os.path.exists(dest):
        return True
    url = ICON_CDN.format(cat_id=cat_id)
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  [警告] 图标下载失败 {url}: {e}", file=sys.stderr)
        return False


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/136.0.0.0 Safari/537.36",
        "Referer": "https://map.17173.com/",
    }

    print(f"正在拉取标记点数据（mapId={MAP_ID}）…")
    session = requests.Session()
    session.headers.update(headers)

    resp = session.get(API_URL, timeout=20)
    resp.raise_for_status()
    raw_list = resp.json()

    # API 返回 {"code": 200, "data": [...]} 格式
    if isinstance(raw_list, dict):
        raw_list = raw_list.get("data", raw_list)
    if not isinstance(raw_list, list):
        print(f"[错误] API 返回格式异常: {type(raw_list)}", file=sys.stderr)
        sys.exit(1)

    print(f"共获取到 {len(raw_list)} 个标记点，开始转换坐标并下载图标…")
    os.makedirs(ICON_DIR, exist_ok=True)

    icon_ids_seen: set[int] = set()
    points_data = []

    for item in raw_list:
        lat_raw = item.get("latitude", 0)
        lng_raw = item.get("longitude", 0)
        cat_id = item.get("category_id") or item.get("categoryId")

        try:
            lat_f = float(lat_raw)
            lng_f = float(lng_raw)
        except (TypeError, ValueError):
            continue

        point = api_to_point(lat_f, lng_f)

        entry = {
            "markType": cat_id,
            "title": item.get("title", ""),
            "id": str(item.get("id", "")),
            "point": point,
        }
        # 保留 description 字段，如果有的话
        if item.get("description"):
            entry["description"] = item["description"]

        points_data.append(entry)

        # 下载图标（每种类型只下一次）
        if cat_id and cat_id not in icon_ids_seen:
            icon_ids_seen.add(cat_id)
            ok = download_icon(cat_id, session)
            status = "OK" if ok else "!!"
            print(f"  {status} 图标 {cat_id} ({item.get('title', '')})")

    # 输出 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(points_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共转换 {len(points_data)} 个标记点。")
    print(f"结果已保存到 {OUTPUT_JSON}")
    print(f"图标已下载到 {ICON_DIR}/")
    print(f"\n使用方式：在 draw_markers.py 顶部添加：")
    print(f"  import json")
    print(f"  points_data = json.load(open('rocom_markers.json', encoding='utf-8'))")


if __name__ == "__main__":
    main()
