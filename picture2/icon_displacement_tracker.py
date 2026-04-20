"""
icon_displacement_tracker.py - 基于小地图图标位移的玩家移动距离辅助估算

CLI 包装脚本 —— 核心算法见 backend.map.icon_tracker

用法:
    python picture2/icon_displacement_tracker.py --video test_file/2.mp4
    python picture2/icon_displacement_tracker.py --video test_file/2.mp4 --icons 055,0002,021
    python picture2/icon_displacement_tracker.py --video test_file/2.mp4 --step 2 --max-frames 500
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.map.autodetect import detect_minimap_circle
from backend.map.icon_tracker import (
    FrameResult,
    IconTemplate,
    OpticalFlowTracker,
    TrackedPoint,
    detect_initial_icons,
    load_icon_templates,
)


# ============================================================================
# 主流程
# ============================================================================

def run_video_tracking(
    video_path: Path,
    icon_dir: Path,
    icon_names: set[str] | None = None,
    step: int = 1,
    max_frames: int = 0,
    score_threshold: float = 0.50,
    edge_ratio: float = 0.82,
    stationary_threshold: float = 0.8,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> OpticalFlowTracker:
    """对视频逐帧运行图标位移追踪（模板初始化 + LK 光流追踪）。"""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"[VIDEO] {video_path.name}: {total_frames} frames, {fps:.1f} fps, {vw}x{vh}")

    # ---- 1. 小地图定标：采样若干帧取共识 ----
    sample_indices = np.linspace(0, max(0, total_frames - 2), min(12, total_frames)).astype(int).tolist()
    dets_raw = []
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
            dets_raw.append((int(d['px']), int(d['py']), int(d['pr'])))

    if not dets_raw:
        print("[ERROR] 无法在视频中检测到小地图")
        cap.release()
        sys.exit(1)

    px = int(np.median([d[0] for d in dets_raw]))
    py = int(np.median([d[1] for d in dets_raw]))
    pr = int(np.median([d[2] for d in dets_raw]))
    if verbose:
        print(f"[MINIMAP] 共识圆心=({px},{py}), 半径={pr} (来自 {len(dets_raw)}/{len(sample_indices)} 采样)")

    # ---- 2. 加载图标模板（只取 top-15 高对比度） ----
    templates = load_icon_templates(icon_dir, icon_names, target_radius=float(pr), max_templates=15)
    if verbose:
        print(f"[ICONS] 加载了 {len(templates)} 个模板: {[t.name for t in templates]}")

    # ---- 3. 创建光流追踪器 ----
    local_center = (float(pr), float(pr))  # 小地图裁切后圆心
    tracker = OpticalFlowTracker(
        center_xy=local_center,
        radius=float(pr),
        stationary_threshold=stationary_threshold,
        max_jump=float(pr) * 0.4,
    )

    REINIT_INTERVAL = 120   # 追踪点不足时最多每 N 帧重新初始化
    MIN_TRACKED = 2         # 追踪点少于此值时立即重新初始化
    MAX_TRACKED = 6         # 追踪点上限

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    processed = 0
    t_start = time.perf_counter()
    limit = max_frames if max_frames > 0 else total_frames
    last_init_frame = -REINIT_INTERVAL

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(exist_ok=True)

    while frame_idx < total_frames and processed < limit:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter()

        # 提取小地图
        fh, fw = frame.shape[:2]
        x1 = max(0, px - pr)
        y1 = max(0, py - pr)
        x2 = min(fw, px + pr)
        y2 = min(fh, py + pr)
        minimap = frame[y1:y2, x1:x2]
        minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        # 判断是否需要重新初始化（追踪点不足时）
        need_init = (
            tracker.n_tracked < MIN_TRACKED
            and (frame_idx - last_init_frame) >= 5  # 避免连续重试
        )

        if need_init and templates:
            icons = detect_initial_icons(
                minimap_gray, local_center, float(pr), templates,
                score_threshold=score_threshold,
                edge_ratio=edge_ratio,
                max_icons=MAX_TRACKED,
            )
            if icons:
                tracker.init_points(minimap_gray, icons)
                if verbose and processed == 0:
                    print(f"[INIT] 初始检测到 {len(icons)} 个图标: "
                          f"{[(n, f'{s:.2f}') for n, _, _, s in icons]}")
            # 补充角点（只在总追踪点仍不足时）
            if tracker.n_tracked < MIN_TRACKED:
                tracker.add_feature_points(minimap_gray, frame_idx, max_extra=MAX_TRACKED - tracker.n_tracked)
            last_init_frame = frame_idx

        # 定期模板重校验：即使追踪点足够，也检查是否偏离已知图标位置
        elif (templates
              and tracker.n_tracked >= MIN_TRACKED
              and (frame_idx - last_init_frame) >= REINIT_INTERVAL):
            verify_icons = detect_initial_icons(
                minimap_gray, local_center, float(pr), templates,
                score_threshold=score_threshold,
                edge_ratio=edge_ratio,
                max_icons=MAX_TRACKED,
            )
            if verify_icons:
                icon_pts = [(cx, cy) for _, cx, cy, _ in verify_icons]
                # 保留距某个已知图标 < 15px 的追踪点
                verified = []
                for tp in tracker._tracked_points:
                    min_d = min(math.hypot(tp.pt[0] - ix, tp.pt[1] - iy)
                               for ix, iy in icon_pts)
                    if min_d < 15.0:
                        verified.append(tp)
                if verified:
                    tracker._tracked_points = verified
                # 补充新检测到的图标
                tracker.init_points(minimap_gray, verify_icons)
            last_init_frame = frame_idx

        # 光流追踪
        result = tracker.update(minimap_gray, frame_idx)

        latency = (time.perf_counter() - t0) * 1000.0
        result.latency_ms = latency

        processed += 1

        # 日志
        if verbose and (processed <= 5 or processed % 100 == 0 or result.consensus_dist > 3.0):
            print(
                f"  F{frame_idx:5d} | trk={result.n_tracked:2d} | "
                f"mov={result.n_moving}/{result.n_tracked} | "
                f"disp=({result.consensus_dx:+5.1f},{result.consensus_dy:+5.1f}) "
                f"d={result.consensus_dist:5.1f}px | "
                f"{latency:5.1f}ms"
            )

        # 可视化
        if output_dir and (processed <= 10 or processed % 50 == 0):
            viz = minimap.copy()
            cv2.circle(viz, (int(local_center[0]), int(local_center[1])), int(pr), (255, 180, 80), 1)
            for tp in tracker._tracked_points:
                color = (0, 255, 0) if tp.age > 3 else (0, 255, 255)
                cv2.circle(viz, (int(tp.pt[0]), int(tp.pt[1])), 4, color, -1)
                cv2.putText(viz, tp.name[:6], (int(tp.pt[0]) + 5, int(tp.pt[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
            if result.consensus_dist > 0.5:
                ax, ay = int(local_center[0]), int(local_center[1])
                bx = int(ax + result.consensus_dx * 5)
                by = int(ay + result.consensus_dy * 5)
                cv2.arrowedLine(viz, (ax, ay), (bx, by), (0, 0, 255), 2, tipLength=0.3)
            cv2.imwrite(str(viz_dir / f"frame_{frame_idx:05d}.png"), viz)

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    # ---- 汇总 ----
    if verbose:
        print(f"\n{'='*60}")
        print(f"[SUMMARY] 处理 {processed} 帧, 耗时 {elapsed:.1f}s ({processed / max(elapsed, 0.01):.1f} fps)")
        print(f"  累计路径长度: {tracker.cumulative_distance:.1f} px")
        print(f"  累计直线位移: {tracker.cumulative_displacement:.1f} px")
        print(f"  方向: dx={tracker._cumulative_dx:.1f}, dy={tracker._cumulative_dy:.1f}")

        dists = [r.consensus_dist for r in tracker.frame_results if r.consensus_dist > 0]
        if dists:
            arr = np.array(dists)
            print(f"\n  帧位移分布 (非零帧 {len(dists)}/{processed}):")
            print(f"    mean={np.mean(arr):.2f}, median={np.median(arr):.2f}, "
                  f"p95={np.percentile(arr, 95):.2f}, max={np.max(arr):.2f}")

        # 延迟分布
        lats = [r.latency_ms for r in tracker.frame_results]
        if lats:
            lats_arr = np.array(lats)
            print(f"\n  延迟分布:")
            print(f"    mean={np.mean(lats_arr):.1f}ms, median={np.median(lats_arr):.1f}ms, "
                  f"p95={np.percentile(lats_arr, 95):.1f}ms, max={np.max(lats_arr):.1f}ms")

    # 保存 CSV
    if output_dir:
        _save_results(output_dir, tracker, processed, elapsed)

    return tracker


def _save_results(output_dir: Path, tracker: OpticalFlowTracker, total_processed: int, elapsed: float):
    """保存追踪结果到 CSV。"""
    frame_csv = output_dir / "frame_displacements.csv"
    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "n_tracked", "n_moving",
            "consensus_dx", "consensus_dy", "consensus_dist",
            "latency_ms",
        ])
        for r in tracker.frame_results:
            writer.writerow([
                r.frame_idx, r.n_tracked, r.n_moving,
                f"{r.consensus_dx:.3f}", f"{r.consensus_dy:.3f}",
                f"{r.consensus_dist:.3f}", f"{r.latency_ms:.1f}",
            ])

    summary_path = output_dir / "tracking_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"total_processed_frames: {total_processed}\n")
        f.write(f"elapsed_sec: {elapsed:.2f}\n")
        f.write(f"cumulative_path_length_px: {tracker.cumulative_distance:.2f}\n")
        f.write(f"cumulative_displacement_px: {tracker.cumulative_displacement:.2f}\n")
        f.write(f"cumulative_dx: {tracker._cumulative_dx:.2f}\n")
        f.write(f"cumulative_dy: {tracker._cumulative_dy:.2f}\n")

        dists = [r.consensus_dist for r in tracker.frame_results if r.consensus_dist > 0]
        if dists:
            arr = np.array(dists)
            f.write(f"moving_frames: {len(dists)}\n")
            f.write(f"mean_frame_disp: {np.mean(arr):.2f}\n")
            f.write(f"median_frame_disp: {np.median(arr):.2f}\n")
            f.write(f"p95_frame_disp: {np.percentile(arr, 95):.2f}\n")
            f.write(f"max_frame_disp: {np.max(arr):.2f}\n")

        lats = [r.latency_ms for r in tracker.frame_results]
        if lats:
            arr = np.array(lats)
            f.write(f"mean_latency_ms: {np.mean(arr):.1f}\n")
            f.write(f"median_latency_ms: {np.median(arr):.1f}\n")
            f.write(f"p95_latency_ms: {np.percentile(arr, 95):.1f}\n")

    print(f"\n[OUTPUT] 结果已保存到 {output_dir}")


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="小地图图标位移追踪器 - 玩家移动距离辅助估算")
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--icon-dir", type=Path,
                        default=Path(__file__).resolve().parent / "icon",
                        help="图标模板目录 (默认: picture2/icon)")
    parser.add_argument("--icons", type=str, default="",
                        help="指定追踪的图标ID，逗号分隔 (为空=全部)")
    parser.add_argument("--step", type=int, default=1,
                        help="跳帧步长 (1=每帧)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最大处理帧数 (0=全部)")
    parser.add_argument("--score-threshold", type=float, default=0.50,
                        help="图标匹配分数阈值")
    parser.add_argument("--edge-ratio", type=float, default=0.82,
                        help="边缘排除比例 (dist/r > 此值则排除)")
    parser.add_argument("--stationary-threshold", type=float, default=0.8,
                        help="静止判定阈值 (px)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录 (默认: picture2/tracking_results/<视频名>)")
    parser.add_argument("--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    icon_names = None
    if args.icons:
        icon_names = {x.strip() for x in args.icons.split(",") if x.strip()}

    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent / "tracking_results" / args.video.stem
        )

    run_video_tracking(
        video_path=args.video,
        icon_dir=args.icon_dir,
        icon_names=icon_names,
        step=args.step,
        max_frames=args.max_frames,
        score_threshold=args.score_threshold,
        edge_ratio=args.edge_ratio,
        stationary_threshold=args.stationary_threshold,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
