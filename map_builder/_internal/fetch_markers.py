"""抓取 17173 互动地图标记点，生成 markers 相关 JSON 与图标资源。"""

import json
import math
import os
import sys
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from path_config import ASSETS_MARKER_DATA_DIR, FRONTEND_IMG_DIR
from map_builder._internal.common import OUTPUT_DIR

# ── 配置 ─────────────────────────────────────────────────
MAP_ID = 4010
API_URL = f"https://map.17173.com/app/location/list?mapIds={MAP_ID}"
ICON_CDN = "https://ue.17173cdn.com/a/terra/icon/rocom/{cat_id}.png"
ICON_DIR = str(FRONTEND_IMG_DIR)   # 图标保存路径
OUTPUT_LITE_JSON = os.path.join(str(ASSETS_MARKER_DATA_DIR), "rocom_markers_lite.json")
OUTPUT_DETAIL_JSON = os.path.join(str(ASSETS_MARKER_DATA_DIR), "rocom_markers_detail.json")
OUTPUT_FULL_JSON = str(OUTPUT_DIR / "rocom_markers.json")


# ── 坐标转换 ──────────────────────────────────────────────────────────────────
# 17173 地图使用 Mapbox GL + EPSG:3857 (Web Mercator) 投影，最大瓦片层级 z=13。
# 拼接大图（wiki-dev 瓦片 24×23 网格）等效于 z=13 世界像素空间的一部分。
# bounds [-1.4,0,0,1.4] 仅约束相机平移范围，瓦片网格只覆盖 bounds 范围的 ~75%。
# 因此简单线性 (lng-W)/(E-W)*MAP_W 会把标记向中心方向压缩约 25%。
#
# 正确投影：先用 Web Mercator 算出 z=13 世界像素，再减去拼接图左上角的世界偏移。
# 偏移量由标定点 (已知 API 经纬度 + 对应图像像素) 反推: offset = world - pixel。
_WORLD_SCALE = 256 * (2 ** 13)     # 2097152  —— z=13 全球像素尺寸


def _lngX(lng: float) -> float:
    """Mapbox GL 标准 lngX: 经度 → 归一化 x [0,1]"""
    return (180.0 + lng) / 360.0


def _latY(lat: float) -> float:
    """Mapbox GL 标准 latY: 纬度 → 归一化 Mercator y [0,1]"""
    return (180.0 - 180.0 / math.pi * math.log(
        math.tan(math.pi / 4.0 + lat * math.pi / 360.0))) / 360.0


# 标定点：(api_lng, api_lat, pixel_x, pixel_y)
_CALIBRATION = [
    (-0.9614146817854703,  0.4290267288905767, 1616, 4097),  # 商店
    (-1.192595986847067,   0.9266004279475766,  274, 1205),  # 岚语雪山
]
_OFFSET_X = sum(_lngX(lng) * _WORLD_SCALE - px for lng, _, px, _ in _CALIBRATION) / len(_CALIBRATION)
_OFFSET_Y = sum(_latY(lat) * _WORLD_SCALE - py for _, lat, _, py in _CALIBRATION) / len(_CALIBRATION)


def api_to_pixel(latitude: float, longitude: float) -> tuple[int, int]:
    """将 17173 API 的 latitude/longitude 转换为拼接大图像素坐标 (x, y)。"""
    px = _lngX(longitude) * _WORLD_SCALE - _OFFSET_X
    py = _latY(latitude) * _WORLD_SCALE - _OFFSET_Y
    return int(round(px)), int(round(py))


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
    os.makedirs(str(ASSETS_MARKER_DATA_DIR), exist_ok=True)
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    icon_ids_seen: set[int] = set()
    points_lite = []
    points_detail = {}

    for item in raw_list:
        lat_raw = item.get("latitude", 0)
        lng_raw = item.get("longitude", 0)
        cat_id = item.get("category_id")
        
        # 统一把 id 转成字符串
        item_id = str(item.get("id", ""))
        item["id"] = item_id

        try:
            lat_f = float(lat_raw)
            lng_f = float(lng_raw)
        except (TypeError, ValueError):
            continue

        # 使用 Mercator + Homography 校正投影
        abs_x, abs_y = api_to_pixel(lat_f, lng_f)

        # 构建轻量化数据，只供大地图过滤/撒点用
        lite_entry = {
            "id": item_id,
            "markType": str(cat_id) if cat_id else "",
            "x": abs_x,
            "y": abs_y
        }
        points_lite.append(lite_entry)
        
        # 构建详情数据：目前仅裁剪保留 title 和 description
        detail_entry = {
            "title": item.get("title", "")
        }
        
        # 只有在 description 存在且不为空时才保留，再次减小体积
        desc = item.get("description", "")
        if desc:
            detail_entry["description"] = desc
            
        points_detail[item_id] = detail_entry

        # 下载图标（每种类型只下一次）
        if cat_id and cat_id not in icon_ids_seen:
            icon_ids_seen.add(cat_id)
            ok = download_icon(cat_id, session)
            status = "OK" if ok else "!!"
            print(f"  {status} 图标 {cat_id} ({item.get('title', '')})")

    # 输出 JSON
    with open(OUTPUT_LITE_JSON, "w", encoding="utf-8") as f:
        json.dump(points_lite, f, ensure_ascii=False, separators=(',', ':'))

    with open(OUTPUT_DETAIL_JSON, "w", encoding="utf-8") as f:
        json.dump(points_detail, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FULL_JSON, "w", encoding="utf-8") as f:
        json.dump(raw_list, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共转换 {len(points_lite)} 个标记点。")
    print(f"极简数据已保存到 {OUTPUT_LITE_JSON}")
    print(f"详情数据已保存到 {OUTPUT_DETAIL_JSON}")
    print(f"完整原始数据已保存到 {OUTPUT_FULL_JSON}")
    print(f"图标已下载到 {ICON_DIR}/")
    print(f"\n使用方式：")
    print(f"  points_lite  = json.load(open('{OUTPUT_LITE_JSON}'))    # 前端仅需要 id, markType, x, y")
    print(f"  points_detail = json.load(open('{OUTPUT_DETAIL_JSON}'))  # 弹窗所需的图片文本")


if __name__ == "__main__":
    main()
