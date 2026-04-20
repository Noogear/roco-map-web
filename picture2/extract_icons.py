"""
提取 CompassIcon.png 精灵图中的每个图标，按行列顺序编号保存到 output/ 目录。
"""
import os
import numpy as np
from PIL import Image
import cv2

INPUT = os.path.join(os.path.dirname(__file__), "CompassIcon.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载图像（保留 alpha 通道）
img = Image.open(INPUT).convert("RGBA")
arr = np.array(img)

# 用较低阈值检测所有像素（含半透明描边/发光效果）
alpha = arr[:, :, 3]
mask_loose = (alpha > 10).astype(np.uint8) * 255   # 宽松掩码：用于检测
mask_solid = (alpha > 50).astype(np.uint8) * 255   # 严格掩码：用于过滤噪点

# 分区处理：顶部（y<640）用小闭合核连接碎片；底部（y>=640）不膨胀避免合并密集圆形
BOTTOM_Y = 640
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

closed = mask_loose.copy()
# 仅对顶部区域做形态学闭合
top_mask = mask_loose[:BOTTOM_Y, :]
top_closed = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kernel5)
closed[:BOTTOM_Y, :] = top_closed

# 连通域分析（基于分区处理后的掩码）
n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

# 双重过滤：边界框面积 >= 400 且在宽松掩码中实际像素 >= 100
MIN_BBOX_AREA = 400
MIN_PIX_AREA  = 100
boxes = []
for i in range(1, n_labels):
    x, y, w, h, area = stats[i]
    if w * h < MIN_BBOX_AREA:
        continue
    # 统计该连通域在宽松掩码中的实际像素数
    comp_mask = (labels == i)
    pix_count = int(mask_loose[comp_mask].sum() // 255)
    if pix_count < MIN_PIX_AREA:
        continue
    boxes.append((y, x, w, h))  # 先按行(y)排，再按列(x)排

# 按行列顺序排序（行优先）
boxes.sort(key=lambda b: (b[0], b[1]))

print(f"共检测到 {len(boxes)} 个图标")

# 提取并保存
padding = 2  # 各方向额外留白
H, W = arr.shape[:2]

for idx, (y, x, w, h) in enumerate(boxes, start=1):
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)
    crop = img.crop((x1, y1, x2, y2))
    out_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.png")
    crop.save(out_path)
    print(f"  [{idx:03d}] ({x1},{y1})-({x2},{y2})  size={x2-x1}x{y2-y1}")

print(f"\n完成！所有图标已保存到: {OUTPUT_DIR}")
