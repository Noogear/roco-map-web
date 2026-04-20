"""
提取 BigMapStatic.png 精灵图中的每个图标，按行列顺序编号保存到 output2/ 目录。
"""
import os
import numpy as np
from PIL import Image
import cv2

INPUT = os.path.join(os.path.dirname(__file__), "BigMapStatic.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载图像（保留 alpha 通道）
img = Image.open(INPUT).convert("RGBA")
arr = np.array(img)
H, W = arr.shape[:2]

alpha = arr[:, :, 3]
mask_loose = (alpha > 60).astype(np.uint8) * 255   # 检测掩码

# 两阶段：
# 1) 无形态学的宽松连通域，避免把小图标腐蚀切断；
# 2) 仅对“超大连通域”做局部细分（腐蚀+膨胀）。
n_loose, labels_loose, stats_loose, _ = cv2.connectedComponentsWithStats(mask_loose, connectivity=8)

MIN_BBOX_AREA = 400
MIN_PIX_AREA = 100
LARGE_SIDE = 200

k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

boxes = []
for i in range(1, n_loose):
    x, y, w, h, area = stats_loose[i]
    if w * h < MIN_BBOX_AREA or area < MIN_PIX_AREA:
        continue

    # 小/中等组件直接保留（避免不必要切割）
    if max(w, h) <= LARGE_SIDE:
        boxes.append((y, x, w, h))
        continue

    # 仅对超大组件做局部细分
    roi = (labels_loose[y:y + h, x:x + w] == i).astype(np.uint8) * 255
    roi_proc = cv2.dilate(cv2.erode(roi, k3, iterations=1), k2, iterations=1)
    n_sub, labels_sub, stats_sub, _ = cv2.connectedComponentsWithStats(roi_proc, connectivity=8)

    sub_added = 0
    for j in range(1, n_sub):
        sx, sy, sw, sh, sarea = stats_sub[j]
        if sw * sh < MIN_BBOX_AREA or sarea < MIN_PIX_AREA:
            continue
        boxes.append((y + sy, x + sx, sw, sh))
        sub_added += 1

    # 若局部分割失败（仍是一整块），回退保留原框，避免漏图标
    if sub_added == 0:
        boxes.append((y, x, w, h))


def _bbox_intersection(a, b):
    ay, ax, aw, ah = a
    by, bx, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


def _merge_bbox(a, b):
    ay, ax, aw, ah = a
    by, bx, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (y1, x1, x2 - x1, y2 - y1)


def merge_split_fragments(boxes):
    """合并被腐蚀切开的小图标碎片。

    规则：
    1) 重叠分裂：两个小框有较大重叠（常见上下断裂）；
    2) 瘦长左右分裂：两个瘦长框同高、近邻、中间有小缝（常见左右断裂）。
    """
    changed = True
    merged_boxes = list(boxes)
    while changed:
        changed = False
        used = [False] * len(merged_boxes)
        new_boxes = []
        for i in range(len(merged_boxes)):
            if used[i]:
                continue
            cur = merged_boxes[i]
            cy, cx, cw, ch = cur
            c_area = cw * ch
            for j in range(i + 1, len(merged_boxes)):
                if used[j]:
                    continue
                oth = merged_boxes[j]
                oy, ox, ow, oh = oth
                o_area = ow * oh

                # 仅在小图标范围内尝试回并，避免影响大 UI 框
                if max(cw, ch, ow, oh) > 85:
                    continue

                inter = _bbox_intersection(cur, oth)
                min_area = min(c_area, o_area)

                # 规则 1：重叠分裂（上下断裂）
                overlap_ratio = inter / max(1, min_area)
                if overlap_ratio >= 0.35:
                    cur = _merge_bbox(cur, oth)
                    cy, cx, cw, ch = cur
                    c_area = cw * ch
                    used[j] = True
                    changed = True
                    continue

                # 规则 1.5：短片上下拼接（大量“上半+下半”切割）
                # 条件：x 高重叠、纵向几乎相接、至少一片高度较短
                cy2, cx2 = cx + cw, cy + ch
                oy2, ox2 = ox + ow, oy + oh
                x_overlap = max(0, min(cx2, ox2) - max(cx, ox))
                y_gap = max(cy, oy) - min(cy2, oy2)  # <0 表示有少量重叠
                x_overlap_ratio = x_overlap / max(1, min(cw, ow))
                short_piece = min(ch, oh) <= 35
                if x_overlap_ratio >= 0.72 and -6 <= y_gap <= 10 and short_piece:
                    merged = _merge_bbox(cur, oth)
                    # 防止把独立的大图标列纵向串起来
                    if merged[2] <= 84 and merged[3] <= 76:
                        cur = merged
                        cy, cx, cw, ch = cur
                        c_area = cw * ch
                        used[j] = True
                        changed = True
                        continue

                # 规则 2：瘦长左右分裂（例如 009+010）
                # 条件：高度相近、宽度较窄、水平间距小
                h_similar = abs(ch - oh) <= 8
                skinny = cw <= 28 and ow <= 28 and ch >= 28 and oh >= 28
                y_overlap = max(0, min(cy2, oy2) - max(cy, oy))
                x_gap = max(cx, ox) - min(cx2, ox2)
                if h_similar and skinny and y_overlap >= min(ch, oh) * 0.7 and 0 <= x_gap <= 8:
                    merged = _merge_bbox(cur, oth)
                    # 防止把两个独立图标合得过宽
                    if merged[2] <= 64 and merged[3] <= 64:
                        cur = merged
                        cy, cx, cw, ch = cur
                        c_area = cw * ch
                        used[j] = True
                        changed = True

            new_boxes.append(cur)
        merged_boxes = new_boxes
    return merged_boxes


boxes = merge_split_fragments(boxes)


def merge_short_stacks(boxes):
    """二次回并：把仍然残留的短条碎片（如 13~26px 高）向上下邻近主体拼接。"""
    items = list(boxes)
    changed = True
    while changed:
        changed = False
        used = [False] * len(items)
        out = []
        for i in range(len(items)):
            if used[i]:
                continue
            a = items[i]
            ay, ax, aw, ah = a
            ax2, ay2 = ax + aw, ay + ah
            best_j = -1
            best_score = -1.0

            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                b = items[j]
                by, bx, bw, bh = b
                bx2, by2 = bx + bw, by + bh

                # 只针对至少一边“很短”的碎片
                if min(ah, bh) > 26:
                    continue

                x_overlap = max(0, min(ax2, bx2) - max(ax, bx))
                x_overlap_ratio = x_overlap / max(1, min(aw, bw))
                y_gap = max(ay, by) - min(ay2, by2)

                if x_overlap_ratio < 0.75:
                    continue
                if not (-6 <= y_gap <= 10):
                    continue

                merged = _merge_bbox(a, b)
                # 防止跨图标拼接
                if merged[2] > 84 or merged[3] > 76:
                    continue

                # 更偏好 x 重叠高、间距小的配对
                score = x_overlap_ratio * 10 - abs(y_gap)
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0:
                a = _merge_bbox(a, items[best_j])
                used[best_j] = True
                changed = True
            out.append(a)

        items = out
    return items


boxes = merge_short_stacks(boxes)

# 过滤细长噪声框：最短边很小且最长边很长，通常是大面板分割残留细线
boxes = [b for b in boxes if not (min(b[2], b[3]) < 10 and max(b[2], b[3]) > 120)]

# 按行列顺序排序（行优先）
boxes.sort(key=lambda b: (b[0], b[1]))

print(f"共检测到 {len(boxes)} 个图标")

# 提取并保存
padding = 2

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
