"""从已有的 rocom_markers.json 重新生成 rocom_markers_lite.json（使用 z=13 Mercator 投影）。"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from map_builder.fetch_markers import api_to_pixel

INPUT = os.path.join(_HERE, 'rocom_markers.json')
OUTPUT = os.path.join(_HERE, 'rocom_markers_lite.json')

with open(INPUT, encoding='utf-8') as f:
    full = json.load(f)

lite = []
for mk in full:
    lat = mk.get('latitude')
    lng = mk.get('longitude')
    if lat is None or lng is None:
        continue
    try:
        lat_f, lng_f = float(lat), float(lng)
    except:
        continue
    x, y = api_to_pixel(lat_f, lng_f)
    lite.append({
        'id': str(mk.get('id', '')),
        'markType': str(mk.get('category_id') or mk.get('markType', '')),
        'x': x,
        'y': y
    })

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(lite, f, ensure_ascii=False, separators=(',', ':'))

print(f"Regenerated {OUTPUT}: {len(lite)} markers")
xs = [p['x'] for p in lite]
ys = [p['y'] for p in lite]
print(f"X: {min(xs)}~{max(xs)}, Y: {min(ys)}~{max(ys)}")
