#!/usr/bin/env python3
"""
诊断工具：直接测试库中的图标与小地图中的补丁的匹配度
帮助理解为什么某些图标无法被识别
"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from pathlib import Path
from backend.map import detect_minimap_circle
from picture2.icon_matcher_v2 import load_icon_library, _match_single_template

# 用户可配置的参数
target_icons = ['0002', '055', '021']  # 要诊断的图标
test_image = 'picture/QQ20260412-010841.png'

print(f"诊断工具: 测试 {test_image} 中是否包含 {target_icons}")
print("=" * 70)

# 1. 加载库
library = load_icon_library(Path('picture2/icon'))
print(f"✓ 加载 {len(library.icons)} 个图标模板")

# 2. 读取小地图
img = cv2.imread(test_image)
if img is None:
    print(f"✗ 无法读取 {test_image}")
    sys.exit(1)

# 3. 检测小地图
result = detect_minimap_circle(img, debug=False)
if not result:
    print(f"✗ 无法检测小地图")
    sys.exit(1)

result_dict = result
h, w = img.shape[:2]
cx_norm = float(result_dict['cx'])
cy_norm = float(result_dict['cy'])
r_norm = float(result_dict['r'])

cx = int(cx_norm * w)
cy = int(cy_norm * h)
r = int(r_norm * min(h, w) / 2)

y_start = max(0, cy - r)
y_end = min(h, cy + r)
x_start = max(0, cx - r)
x_end = min(w, cx + r)
minimap = img[y_start:y_end, x_start:x_end]

print(f"✓ 小地图提取: {minimap.shape[1]}x{minimap.shape[0]} @ ({cx}, {cy}), r={r}")
print()

# 4. 对于每个目标图标，尝试从小地图的不同位置匹配
for icon_id in target_icons:
    # 找到库中的该图标
    template_idx = None
    for idx, icon_meta in enumerate(library.icons):
        if str(idx).zfill(4) == icon_id or f"{idx:04d}" == icon_id:
            template_idx = idx
            break
    
    if template_idx is None:
        # 尝试直接查找
        for idx, icon_meta in enumerate(library.icons):
            icon_name = Path(icon_meta.name).stem
            if icon_name == icon_id:
                template_idx = idx
                break
    
    if template_idx is None:
        print(f"✗ {icon_id}: 库中未找到")
        continue
    
    template_meta = library.icons[template_idx]
    print(f"📦 {icon_id} (库编号 {template_idx}):")
    print(f"   原始尺寸: {template_meta.original_width}x{template_meta.original_height}")
    print(f"   色调: {template_meta.dominant_color}, 色彩相似度权重: RGB均值 {template_meta.mean_bgr}")
    
    # 尝试在小地图的网格上匹配
    grid_size = 20
    matches = []
    
    for y_pos in range(0, minimap.shape[0], grid_size):
        for x_pos in range(0, minimap.shape[1], grid_size):
            # 提取候选补丁（多种尺寸）
            for patch_size in [20, 25, 30, 35, 40, 45]:
                if y_pos + patch_size > minimap.shape[0] or x_pos + patch_size > minimap.shape[1]:
                    continue
                
                patch_bgr = minimap[y_pos:y_pos+patch_size, x_pos:x_pos+patch_size]
                if patch_bgr.size == 0:
                    continue
                
                # 尝试多尺度匹配
                best_ncc = -1.0
                best_scale = None
                
                for scale in [0.8, 1.0, 1.2]:
                    # 获取库中该尺度的模板
                    scale_key = {0.8: "80pct", 1.0: "100pct", 1.2: "120pct"}[scale]
                    
                    template_gray = template_meta.variants[f"gray_{scale_key}"]
                    template_mask = template_meta.variants[f"mask_{scale_key}"]
                    
                    # 调整补丁尺寸匹配模板
                    if patch_size < template_gray.shape[1]:
                        # 补丁太小，需要上采样
                        scale_factor = template_gray.shape[1] / patch_size
                        patch_resized = cv2.resize(patch_bgr, None, fx=scale_factor, fy=scale_factor, 
                                                  interpolation=cv2.INTER_LINEAR)
                    else:
                        patch_resized = cv2.resize(patch_bgr, (template_gray.shape[1], template_gray.shape[0]))
                    
                    patch_gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
                    
                    # 简单 NCC
                    patch_mean = patch_gray.mean()
                    template_mean = template_gray.mean()
                    
                    patch_norm = patch_gray - patch_mean
                    template_norm = template_gray - template_mean
                    
                    numerator = np.sum(patch_norm * template_norm * (template_mask > 0))
                    denom = np.sqrt(np.sum(patch_norm**2 * (template_mask > 0)) * np.sum(template_norm**2 * (template_mask > 0)))
                    
                    if denom > 1e-6:
                        ncc = numerator / denom
                    else:
                        ncc = -1.0
                    
                    if ncc > best_ncc:
                        best_ncc = ncc
                        best_scale = scale
                
                if best_ncc > 0.4:  # 低阈值，收集所有合理的匹配
                    matches.append({
                        'pos': (x_pos, y_pos),
                        'size': patch_size,
                        'scale': best_scale,
                        'ncc': best_ncc
                    })
    
    if matches:
        print(f"   ✓ 找到 {len(matches)} 个潜在匹配:")
        for m in sorted(matches, key=lambda x: x['ncc'], reverse=True)[:5]:
            print(f"      - 位置({m['pos'][0]:3d}, {m['pos'][1]:3d}), 补丁{m['size']}x{m['size']}, "
                  f"尺度{m['scale']:.1f}, NCC={m['ncc']:.4f}")
    else:
        print(f"   ✗ 未找到匹配 (最高 NCC < 0.4)")
    
    print()

print("=" * 70)
print("诊断完成")
print("\n解释:")
print("- 若找到多个匹配且 NCC > 0.5: 库和小地图兼容，问题在候选提取或精排阈值")
print("- 若找到匹配但 NCC < 0.4: 库和小地图特征差异大（缩放/旋转/风格）")
print("- 若完全未找到: 图标不存在于此小地图，或位置与期望不符")
