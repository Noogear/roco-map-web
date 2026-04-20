"""
compare_arrow_icon.py - 箭头方向检测 vs 图标位移方向 对比测试

对比两个独立信号的移动方向一致性：
  1. 箭头检测 (backend/map/arrow.py) → angle_deg（罗盘角，上=0°顺时针）
  2. 图标位移追踪 (icon_displacement_tracker.py) → consensus_dx, consensus_dy

对齐逻辑：
  - 箭头 angle_deg：0°=上 90°=右 180°=下 270°=左（罗盘角）
  - 图标位移：玩家向右走 → 地图向左滚 → 图标 dx<0
  - 因此图标位移方向取反后应与箭头方向一致

用法:
    python algorithm/compare_arrow_icon.py --video test_file/2.mp4
    python algorithm/compare_arrow_icon.py --video test_file/2.mp4 --max-frames 500 --step 2
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.map import arrow as ar
from backend.map.autodetect import detect_minimap_circle
from backend.map.icon_tracker import (
    OpticalFlowTracker, detect_initial_icons, load_icon_templates,
)


def angle_diff(a: float, b: float) -> float:
    """计算两个角度之间的最小差值（0~180）。"""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def displacement_to_compass(dx: float, dy: float) -> float:
    """将图标位移 (dx, dy) 转换为玩家移动的罗盘角。

    图标位移与玩家移动反向，所以取反。
    罗盘角：0°=上, 90°=右, 180°=下, 270°=左。
    """
    # 取反（图标位移方向与玩家移动方向相反）
    px, py = -dx, -dy
    # atan2(x, -y) 将数学坐标转为罗盘角（上=0°）
    angle = math.degrees(math.atan2(px, -py))
    return angle % 360.0


def main():
    parser = argparse.ArgumentParser(description="箭头检测 vs 图标位移 方向对比")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--icon-dir", type=Path,
                        default=ROOT / "picture2" / "icon")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-disp", type=float, default=0.5,
                        help="最小位移阈值，低于此值跳过比较")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[VIDEO] {args.video.name}: {total_frames} frames, {fps:.1f} fps, {vw}x{vh}")

    # ---- 小地图定标 ----
    sample_indices = np.linspace(0, max(0, total_frames - 2), min(12, total_frames)).astype(int).tolist()
    dets = []
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            d = detect_minimap_circle(frame)
        except Exception:
            d = None
        if d:
            dets.append((int(d['px']), int(d['py']), int(d['pr'])))

    if not dets:
        print("[ERROR] 无法检测到小地图")
        cap.release()
        sys.exit(1)

    mx = int(np.median([d[0] for d in dets]))
    my = int(np.median([d[1] for d in dets]))
    mr = int(np.median([d[2] for d in dets]))
    print(f"[MINIMAP] 共识圆心=({mx},{my}), 半径={mr}")

    # ---- 加载图标模板 ----
    templates = load_icon_templates(args.icon_dir, target_radius=float(mr), max_templates=15)
    print(f"[ICONS] 加载了 {len(templates)} 个模板")

    # ---- 创建光流追踪器 ----
    local_center = (float(mr), float(mr))
    tracker = OpticalFlowTracker(
        center_xy=local_center,
        radius=float(mr),
        stationary_threshold=0.3,
        max_jump=float(mr) * 0.4,
    )

    # ---- 输出 ----
    if args.output_dir is None:
        args.output_dir = ROOT / "algorithm" / "output" / f"compare_{args.video.stem}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐帧处理 ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    processed = 0
    limit = args.max_frames if args.max_frames > 0 else total_frames

    prev_stable_angle = None
    last_init_frame = -999
    MAX_TRACKED = 6
    MIN_TRACKED = 2

    # 渐变翻转检测：保留最近 N 个稳定角度作为长期参考
    DRIFT_HISTORY_LEN = 15       # 滑窗长度（step=3 时约 1.5 秒）
    DRIFT_OLD_THRESH = 130.0     # 与历史参考的偏差阈值
    DRIFT_PREV_THRESH = 40.0     # 与 prev_stable 的偏差上限（确认是渐变非突变）
    stable_angle_history: list[float] = []

    records = []
    t_start = time.perf_counter()

    while frame_idx < total_frames and processed < limit:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.step != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter()

        # ---- 箭头检测 ----
        arrow_result = ar.detect_arrow_from_frame(
            frame, mx, my, mr, prev_stable_angle=prev_stable_angle
        )
        arrow_angle = None
        if arrow_result is not None:
            arrow_angle = float(arrow_result["angle_deg"])
            skew = abs(float(arrow_result.get("skewness", 0.0)))
            asp = float(arrow_result.get("aspect", 0.0))

            # 即时反转保护：与 prev_stable 突变 >160° 时翻转
            if prev_stable_angle is not None:
                delta = angle_diff(arrow_angle, prev_stable_angle)
                if delta > ar.ANTIFLIP_ANGLE_THRESH:
                    arrow_angle = (arrow_angle + 180.0) % 360.0

            # 渐变反转保护：与滑窗历史参考偏差 >130° 但与 prev 很近（逐帧漂移）
            if (len(stable_angle_history) >= DRIFT_HISTORY_LEN
                    and prev_stable_angle is not None):
                old_ref = stable_angle_history[0]
                delta_old = angle_diff(arrow_angle, old_ref)
                delta_prev = angle_diff(arrow_angle, prev_stable_angle)
                if delta_old > DRIFT_OLD_THRESH and delta_prev < DRIFT_PREV_THRESH:
                    arrow_angle = (arrow_angle + 180.0) % 360.0
                    # 用纠正值填充历史，防止再次被错误值快速污染
                    stable_angle_history[:] = [arrow_angle] * DRIFT_HISTORY_LEN

            is_amb = (skew < ar.AMB_SKEW_ABS
                      and ar.AMB_ASPECT_MIN <= asp <= ar.AMB_ASPECT_MAX)
            if not is_amb:
                prev_stable_angle = arrow_angle
                stable_angle_history.append(arrow_angle)
                if len(stable_angle_history) > DRIFT_HISTORY_LEN:
                    stable_angle_history.pop(0)

        # ---- 图标位移追踪 ----
        fh, fw = frame.shape[:2]
        x1 = max(0, mx - mr)
        y1 = max(0, my - mr)
        x2 = min(fw, mx + mr)
        y2 = min(fh, my + mr)
        minimap = frame[y1:y2, x1:x2]
        minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        # 重新初始化
        if tracker.n_tracked < MIN_TRACKED and (frame_idx - last_init_frame) >= 5:
            icons = detect_initial_icons(
                minimap_gray, local_center, float(mr), templates,
                score_threshold=0.50, edge_ratio=0.82, max_icons=MAX_TRACKED,
            )
            if icons:
                tracker.init_points(minimap_gray, icons)
            if tracker.n_tracked < MIN_TRACKED:
                tracker.add_feature_points(minimap_gray, frame_idx,
                                           max_extra=MAX_TRACKED - tracker.n_tracked)
            last_init_frame = frame_idx

        icon_result = tracker.update(minimap_gray, frame_idx)

        latency = (time.perf_counter() - t0) * 1000.0

        # ---- 对比 ----
        icon_angle = None
        disp = icon_result.consensus_dist
        diff = None

        if disp >= args.min_disp:
            icon_angle = displacement_to_compass(
                icon_result.consensus_dx, icon_result.consensus_dy
            )

        if arrow_angle is not None and icon_angle is not None:
            diff = angle_diff(arrow_angle, icon_angle)

        records.append({
            "frame_idx": frame_idx,
            "arrow_angle": arrow_angle,
            "icon_dx": icon_result.consensus_dx,
            "icon_dy": icon_result.consensus_dy,
            "icon_disp": disp,
            "icon_angle": icon_angle,
            "angle_diff": diff,
            "n_tracked": icon_result.n_tracked,
            "n_moving": icon_result.n_moving,
            "latency_ms": latency,
        })

        processed += 1

        if processed <= 5 or processed % 100 == 0 or (diff is not None and diff > 30):
            arrow_str = f"{arrow_angle:6.1f}°" if arrow_angle is not None else "  MISS"
            icon_str = f"{icon_angle:6.1f}°" if icon_angle is not None else "  SKIP"
            diff_str = f"{diff:5.1f}°" if diff is not None else "   --"
            print(
                f"  F{frame_idx:5d} | 箭头={arrow_str} | 图标={icon_str} "
                f"(d={disp:4.1f}px) | diff={diff_str} | "
                f"trk={icon_result.n_tracked} | {latency:5.1f}ms"
            )

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    # ---- 汇总统计 ----
    print(f"\n{'='*70}")
    print(f"[SUMMARY] 处理 {processed} 帧, 耗时 {elapsed:.1f}s ({processed/max(elapsed,0.01):.1f} fps)")

    diffs = [r["angle_diff"] for r in records if r["angle_diff"] is not None]
    both_ok = len(diffs)
    arrow_ok = sum(1 for r in records if r["arrow_angle"] is not None)
    icon_ok = sum(1 for r in records if r["icon_angle"] is not None)

    print(f"  箭头检测成功: {arrow_ok}/{processed} ({100*arrow_ok/max(processed,1):.1f}%)")
    print(f"  图标位移有效: {icon_ok}/{processed} ({100*icon_ok/max(processed,1):.1f}%)")
    print(f"  可对比帧数: {both_ok}/{processed} ({100*both_ok/max(processed,1):.1f}%)")

    if diffs:
        arr = np.array(diffs)
        within_30 = np.sum(arr <= 30)
        within_45 = np.sum(arr <= 45)
        within_90 = np.sum(arr <= 90)
        print(f"\n  方向差异分布 ({both_ok} 帧):")
        print(f"    mean={np.mean(arr):.1f}°, median={np.median(arr):.1f}°, "
              f"p95={np.percentile(arr, 95):.1f}°")
        print(f"    ≤30°: {within_30}/{both_ok} ({100*within_30/both_ok:.1f}%)")
        print(f"    ≤45°: {within_45}/{both_ok} ({100*within_45/both_ok:.1f}%)")
        print(f"    ≤90°: {within_90}/{both_ok} ({100*within_90/both_ok:.1f}%)")

    # ---- 保存 CSV ----
    csv_path = args.output_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_idx", "arrow_angle", "icon_dx", "icon_dy",
            "icon_disp", "icon_angle", "angle_diff", "n_tracked",
            "n_moving", "latency_ms",
        ])
        writer.writeheader()
        for r in records:
            row = {k: (f"{v:.3f}" if isinstance(v, float) and v is not None else v)
                   for k, v in r.items()}
            writer.writerow(row)

    print(f"\n[OUTPUT] CSV 已保存到 {csv_path}")


if __name__ == "__main__":
    main()
