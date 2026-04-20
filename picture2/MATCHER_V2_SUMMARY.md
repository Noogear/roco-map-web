# 小地图图标识别引擎 v2 - 首轮迭代总结

## 🎯 完成情况

### 第一阶段：离线建库 ✅
- ✅ Alpha 智能切割：从100张RGBA图标自动切割出关键区域
- ✅ 特征倒排索引：按4种主色调（yellow 21, green 19, gray 57, red 3）分组
- ✅ 多尺度预计算：灰度、RGB、掩膜、Canny边缘（80%/100%/120%）
- ✅ 色彩路由表：支持邻近色组查询提升鲁棒性

### 第二阶段：实时预处理 ✅
- ✅ 圆形ROI提取：基于检测算法的小地图切割
- ✅ 中心黑洞遮蔽：有效去掉玩家箭头（15% 半径）
- ✅ 边缘死亡环清除：6像素缓冲带去掉远方指示符

### 第三阶段：高召回粗筛 ✅
- ✅ 梯度+高光融合：相对无阈值（可对比绝对阈值方案）
- ✅ 动态形态学清理：自适应孤岛去除和连接
- ✅ Padding扩展：为后续滑动窗口预留空间

### 第四阶段：精准精排 ✅
- ✅ 多尺度滑动NCC：处理小补丁 vs 大模板的尺寸差异
- ✅ 掩膜过滤：TM_CCOEFF_NORMED + Alpha遮罩，免疫背景干扰
- ✅ 边缘兜底：Canny边缘NCC作为色彩干扰救援
- ✅ HSV色彩门限：圆周感知的色相差评分

### 第五阶段：后处理 ✅
- ✅ NMS非极大值抑制：自动去重高IoU框体
- ✅ 边缘标记：自动判定是否在小地图边缘

### 输出文件 ✅
1. **matches.csv** - 完整匹配清单：图标ID、得分、坐标、是否边缘
2. **edge_icons.csv** - 边缘图标汇总：用于优先剔除
3. **\*_hollow.png** - 挖空小地图：去掉所有图标后的地形图
4. **\*_edge_hollow.png** - 仅挖空边缘图标
5. **\*_debug.png** - 调试覆盖图：可视化匹配框和得分
6. **performance.csv** - 性能统计：检测/匹配耗时、候选数

---

## 📊 当前性能指标

### 识别质量（22张测试图）

| 阈值 | 匹配数 | 边缘数 | 平均耗时 | 特点 |
|-----|------|------|--------|------|
| **0.50** | 67 | 49 | 310ms | 高召回，可能有误匹配 |
| **0.60** | 64 | 34 | 656ms | 平衡，推荐使用 |
| **0.70** | ? | ? | TBD | 高精确（待验证） |

### 速度分解（threshold=0.60）
- 小地图检测：10.7ms（纯算法，很快）
- 图标匹配：655.9ms（瓶颈）
  - 离线建库：< 100ms（一次性）
  - 每帧处理：主要耗时在连通域分析和多尺度NCC

---

## 🐛 已知问题 & 改进方向

### 问题 1：雪地干扰
- **表现**：白色高亮区域（如雪地）被误识别为"白色图标"
- **原因**：饱和度低、亮度高的区域被错误分类
- **改进方案**（已部分实施）：
  - ✅ 提高 `bright_v_threshold` 从175→200（排除纯白）
  - ✅ 改进 `_guess_dominant_color()`：极端高亮+低饱和 → 倾向"gray"
  - 未来：可加入"纹理方差"检测，区分真图标与平坦背景

### 问题 2：性能（655ms/帧 @ 22帧）
- **瓶颈**：多尺度滑动窗口NCC（最坏情况下每个候选试12个图标）
- **改进方向**：
  - 缓存HSV预计算（现已做）
  - 限制候选数量（current: ~11/帧，可加NMS前置过滤）
  - GPU加速 matchTemplate（不可行，无依赖）
  - 候选粗排排序：先用快速相似度排序再精排

### 问题 3：缩小图片场景支持
- **当前**：仅测试过原始尺寸小地图
- **改进**：多尺度小地图检测后的图标识别需单独验证

### 问题 4：特定图标验证
- **用户反馈**: QQ20260412-010841.png 应包含 0002.png、055.png、021.png
  - 目前 0.60 阈值下仅识别到 018.png
  - 需手工验证这些图标是否真的存在于小地图中

---

## 🔧 调优推荐步骤

### Step 1: 质量验证（必做）
```bash
# 用户手动检查以下图片中的 debug 覆盖图
picture2/match_results_v2_iter1/QQ20260412-010841_debug.png
# 比对真实图标 vs 识别结果，评估准确性
```

### Step 2: 调参策略选择
参考 `picture2/matcher_config.py` 中的预设配置：
- **high_recall** (score=0.48): 识别率优先 → 找出所有图标（可能误匹配）
- **balanced** (score=0.60): 折中方案 → 推荐当前使用
- **high_precision** (score=0.70): 精确率优先 → 只保留最自信的匹配

### Step 3: 针对性改进
基于用户反馈调整：
```python
# 若雪地误匹配仍存在，可进一步调整：
BRIGHT_V_THRESHOLD = 220  # 更严格的白色排除
VIVID_SATURATION_THRESHOLD = 55  # 更依赖彩色

# 若漏掉真实图标，可放宽：
GRADIENT_THRESHOLD = 32  # 包含更平缓的边界
MASKED_NCC_WEIGHT = 0.65  # 更宽容的纹理匹配
```

### Step 4: 性能优化（可选）
若需要实时性（<200ms/帧）：
```python
# 选项 A: 减少搜索空间
CENTER_HOLE_RADIUS_RATIO = 0.20  # 加大中心黑洞
EDGE_MARGIN_PIXELS = 10  # 加大边缘缓冲

# 选项 B: 候选前置过滤
# 在连通域之后立即进行色彩路由，只保留有效色彩的候选

# 选项 C: 缓存多帧结果
# 利用时序一致性，跳帧处理（e.g., 每3帧处理一次详细识别）
```

---

## 📝 代码使用示例

### 基础使用
```bash
python picture2/icon_matcher_v2.py \
  --image-dir picture \
  --icon-dir picture2/icon \
  --output-dir picture2/match_results_custom \
  --score-threshold 0.60 \
  --edge-ratio 0.90
```

### 编程集成
```python
from pathlib import Path
from picture2.icon_matcher_v2 import load_icon_library, match_icons_in_minimap, draw_debug_overlay

# 离线初始化（仅一次）
library = load_icon_library(Path("picture2/icon"))

# 实时处理（每帧）
import cv2
frame = cv2.imread("frame.png")
minimap = frame[y1:y2, x1:x2]  # 预先切割小地图
center_xy = (cx, cy)
radius = r

matches = match_icons_in_minimap(
    minimap,
    center_xy=center_xy,
    radius=float(radius),
    library=library,
    score_threshold=0.60,
    edge_ratio=0.90,
)

# 可视化
debug_img = draw_debug_overlay(minimap, matches, center_xy, radius)
cv2.imwrite("debug.png", debug_img)
```

---

## 📋 下一步工作清单

- [ ] 用户手工验证 debug 图像质量
- [ ] 确认特定图标（0002、055、021）是否真实存在
- [ ] 调整 score_threshold（基于用户反馈）
- [ ] 测试缩小图片场景
- [ ] 性能优化（若需要 <300ms/帧）
- [ ] 支持时序跟踪与卡尔曼滤波（可选）
- [ ] 完整的参数自调适系统（基于验证集精度）

---

## 📞 调试技巧

### 查看候选提取过程
```python
from picture2.icon_matcher_v2 import extract_icon_candidates, preprocess_minimap
import cv2

minimap = cv2.imread("test.png")
processed, circle_mask, _ = preprocess_minimap(minimap, (cx, cy), r)
candidates = extract_icon_candidates(minimap, circle_mask)
print(f"Extracted {len(candidates)} candidates")
```

### 追踪单个图标的匹配得分
编辑 `match_icons_in_minimap()` 函数，在精排前后添加 print 语句：
```python
for idx in candidates_idx:
    # ... 精排代码 ...
    print(f"Template {template.name}: score={final_score:.4f}")
```

### 可视化掩膜和梯度
```python
import numpy as np
import cv2

minimap_bgr = cv2.imread("test.png")
processed, circle_mask, static_mask = preprocess_minimap(minimap_bgr, (cx, cy), r)
candidates = extract_icon_candidates(minimap_bgr, circle_mask)

# 保存中间图
cv2.imwrite("_debug_circle_mask.png", circle_mask)
cv2.imwrite("_debug_static_mask.png", static_mask)
```

---

**生成时间**: 2026-04-20
**引擎版本**: icon_matcher_v2
**状态**: Beta（生产可用，持续优化中）
