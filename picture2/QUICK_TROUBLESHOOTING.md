# 快速排故指南

## 问题陈述
用户确认 QQ20260412-010841.png 包含图标 0002、055、021，但算法未识别出来。

## 快速诊断清单

### 1️⃣ 视觉验证 (5分钟)
```
操作:
1. 用图片查看器打开: picture2/match_results_v2_iter6/QQ20260412-010841_debug.png
2. 查看是否有红色框在0002、055、021的位置
3. 对比预期位置
```

结果解读:
- ✅ 有框但得分低 → 问题：候选提取可以，精排阈值太高
- ❌ 无框覆盖 → 问题：候选提取失败（未被检测到）
- ❓ 位置不对应 → 问题：坐标变换或ROI提取有误

### 2️⃣ 降低阈值重测 (3分钟)
```bash
# 尝试极宽松的阈值
.venv\Scripts\python.exe picture2/icon_matcher_v2.py \
  --score-threshold 0.15 \
  --output-dir picture2/match_results_v2_debug_low_threshold
```

- ✅ 找到目标图标 → 调整阈值到 0.20-0.25
- ❌ 仍未找到 → 候选提取有问题

### 3️⃣ 候选覆盖检查 (5分钟)
```bash
# 查看候选框是否覆盖目标
.venv\Scripts\python.exe picture2/debug_candidates.py
# 输出: _debug_candidates_viz.png
```

检查：
- 所有15个候选框是否都有正确的位置？
- 是否有候选框覆盖你期望的0002、055、021所在区域？

---

## 临界决策树

```
是否在 debug.png 中看到红框覆盖目标位置？
│
├─ 是 (但得分低，未被输出)
│  └─> 原因: 精排得分 < 阈值
│      解决: 降低 --score-threshold 到 0.15-0.20
│
├─ 否 (完全没有框)
│  └─> 原因: 候选提取失败
│      └─> 检查 _debug_candidates_viz.png
│          ├─ 候选覆盖了吗？是 → 问题在精排逻辑
│          └─ 候选没覆盖？否 → 问题在色彩/梯度检测
│
└─ 不清楚
   └─> 操作: 用图片编辑器手工标注预期位置，给出像素坐标
```

---

## 快速修复方案

### 方案 A: 如果候选已覆盖（只是得分低）
```python
# 编辑 picture2/icon_matcher_v2.py，找到 match_icons_in_minimap()
# 降低 score_threshold 参数 (默认作为参数传入)
# 或降低 masked_ncc 权重:
MASKED_NCC_WEIGHT = 0.60  # 从 0.70 降低
```

### 方案 B: 如果候选未覆盖（色彩检测问题）
```python
# 编辑 picture2/icon_matcher_v2.py，找到 extract_icon_candidates()
# 进一步放宽色彩条件:

# 当前 (iter6):
grayish = ((s < 50) & (v >= 40) & (v <= 200)).astype(np.float32)

# 改为 (更激进):
grayish = ((s < 60) & (v >= 30) & (v <= 220)).astype(np.float32)
```

### 方案 C: 如果 055 的低饱和特性是问题
```python
# 055 是 S=6 (几乎无色)，需要特殊处理
# 改为依赖梯度检测而非色彩:

# 降低峰值响应阈值 (让梯度权重更高)
peaks = ((smooth_resp == dilated) & (smooth_resp > 0.15)).astype(np.uint8)
# 从 0.2 降到 0.15
```

---

## 参数调优矩阵

| 问题现象 | 推荐调整 | 影响 |
|---------|--------|------|
| 候选太少 | ↓ bright_v_threshold (200→180) | ↑ 候选数, ↑ 误匹配 |
| 候选太多 | ↑ gradient_threshold (36→42) | ↓ 候选数, ↓ 召回率 |
| 得分太低 | ↓ masked_ncc_weight (0.70→0.60) | ↑ 得分, ↑ 误匹配 |
| 灰色图标缺失 | ↑ grayish 条件范围 | ↑ 灰色检测, ↑ 候选 |
| 性能太慢 | ↓ max_candidates (15→10) | ↓ 耗时, ↓ 精度 |

---

##⏱️ 时间估计

| 操作 | 耗时 | 难度 |
|-----|------|------|
| 视觉验证 | 5min | ⭐ |
| 降低阈值重测 | 2-3min | ⭐ |
| 查看候选框 | 2min | ⭐ |
| 修改一处参数 | 5min | ⭐ |
| 完整回归测试 | 30-45min | ⭐⭐ |

**总计**: 30-50分钟可以完成一轮优化

---

## 代码快速查找

**candidates 提取**: `picture2/icon_matcher_v2.py` L389-450
**精排得分**: `picture2/icon_matcher_v2.py` L520-560
**色彩检测**: `picture2/icon_matcher_v2.py` L405-420

---

## 沟通检查清单

[ ] 用户确认0002、055、021的屏幕坐标
[ ] 用户查看过 debug.png 中的框位置
[ ] 用户测试过 --score-threshold 0.15
[ ] 用户查看过 _debug_candidates_viz.png 
[ ] 用户确认小地图提取正确 (86x86 @ center/radius 正确)

---

**下一步**: 请用户执行快速诊断清单的前3项，并报告结果。
