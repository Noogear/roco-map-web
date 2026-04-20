#!/usr/bin/env python3
"""
调试候选框提取，用于理解为什么某些图标没有被检测到
"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from backend.map import detect_minimap_circle
from picture2.icon_matcher_v2 import preprocess_minimap, extract_icon_candidates

# 处理特定图片
img_path = 'picture/QQ20260412-010841.png'
img = cv2.imread(img_path)

# 检测小地图
result = detect_minimap_circle(img, debug=False)
if not result:
    print(f"无法检测小地图: {img_path}")
    sys.exit(1)

# 提取参数
result_dict = result
h, w = img.shape[:2]
cx_norm = float(result_dict['cx'])
cy_norm = float(result_dict['cy'])
r_norm = float(result_dict['r'])

cx = int(cx_norm * w)
cy = int(cy_norm * h)
r = int(r_norm * min(h, w) / 2)

print(f"图片: {w}x{h}")
print(f"小地图检测: center=({cx}, {cy}), radius={r}")

# 提取小地图
y_start = max(0, cy - r)
y_end = min(h, cy + r)
x_start = max(0, cx - r)
x_end = min(w, cx + r)
minimap = img[y_start:y_end, x_start:x_end]

print(f"小地图尺寸: {minimap.shape}")

# 预处理
processed, circle_mask, static_mask = preprocess_minimap(minimap, (r, r), float(r))
print(f"预处理完成: circle_mask {circle_mask.shape}, static_mask {static_mask.shape}")

# 提取候选
candidates = extract_icon_candidates(minimap, circle_mask)
print(f"\n提取到 {len(candidates)} 个候选框:")

for i, (x, y, w, h) in enumerate(candidates):
    # 在小地图坐标中的位置
    cx_cand = x + w // 2
    cy_cand = y + h // 2
    
    # 计算到小地图中心的距离
    dist_to_center = np.sqrt((cx_cand - r)**2 + (cy_cand - r)**2)
    ratio = dist_to_center / r
    
    print(f"  [{i:2d}] pos=({x:3d},{y:3d}) size=({w:3d}x{h:3d}) "
          f"center=({cx_cand:3d},{cy_cand:3d}) dist_ratio={ratio:.2f}")

# 可视化候选框
viz_img = minimap.copy()
for i, (x, y, w, h) in enumerate(candidates):
    # 绘制候选框
    cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(viz_img, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 绘制小地图圆形
cv2.circle(viz_img, (r, r), r, (255, 0, 0), 2)

cv2.imwrite('_debug_candidates_viz.png', viz_img)
print(f"\n可视化已保存: _debug_candidates_viz.png")
