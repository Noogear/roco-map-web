"""
可视化箭头检测结果 — 截取指定帧的小地图区域并绘制检测到的箭头方向。
用于人工核查箭头方向正确性。
"""
import sys, math, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.map import arrow as ar
from backend.map.autodetect import detect_minimap_circle


def draw_arrow_on_crop(crop_bgr, result, scale=4):
    """放大 crop 并绘制检测到的箭头方向线。"""
    h, w = crop_bgr.shape[:2]
    big = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    if result is None:
        cv2.putText(big, "MISS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return big

    cx, cy = result["centroid"]
    hx, hy = result["head_xy"]
    tx, ty = result["tail_xy"]

    # 缩放坐标
    cx, cy = int(cx * scale), int(cy * scale)
    hx, hy = int(hx * scale), int(hy * scale)
    tx, ty = int(tx * scale), int(ty * scale)

    # 箭头线（绿色 = 头部方向）
    cv2.arrowedLine(big, (tx, ty), (hx, hy), (0, 255, 0), 2, tipLength=0.3)
    # 质心
    cv2.circle(big, (cx, cy), 3, (0, 255, 255), -1)

    return big


def make_info_panel(result, frame_idx, panel_h, panel_w=200):
    """生成信息面板。"""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    lines = [f"Frame: {frame_idx}"]
    if result:
        lines += [
            f"Angle: {result['angle_deg']:.1f}",
            f"Skew: {result['skewness']:.3f}",
            f"Aspect: {result['aspect']:.2f}",
            f"Pixels: {result['mask_pixels']}",
            f"Warn: {result.get('warning') or '-'}",
        ]
    else:
        lines.append("MISS")

    for i, line in enumerate(lines):
        cv2.putText(panel, line, (5, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return panel


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=str, required=True, help="逗号分隔的帧号列表")
    parser.add_argument("--output", default="algorithm/output/arrow_vis")
    parser.add_argument("--context", type=int, default=5, help="每个目标帧前后各取几帧")
    args = parser.parse_args()

    target_frames = [int(x.strip()) for x in args.frames.split(",")]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 检测小地图
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    circ = detect_minimap_circle(frame0)
    if circ is None:
        print("无法检测小地图")
        return
    mx = int(circ["px"])
    my = int(circ["py"])
    mr = int(circ["pr"])
    print(f"小地图: center=({mx},{my}), r={mr}")

    crop_r = max(ar.ARROW_CROP_RADIUS, int(mr * 0.25))
    scale = 4

    prev = None
    for tf in target_frames:
        # 收集 context 帧
        start_f = max(0, tf - args.context)
        end_f = min(total - 1, tf + args.context)

        images = []
        for fi in range(start_f, end_f + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            r = ar.detect_arrow_from_frame(frame, mx, my, mr, prev_stable_angle=prev)

            # 裁剪小地图区域
            h_img, w_img = frame.shape[:2]
            x1 = max(0, mx - crop_r)
            y1 = max(0, my - crop_r)
            x2 = min(w_img, mx + crop_r)
            y2 = min(h_img, my + crop_r)
            crop = frame[y1:y2, x1:x2].copy()

            vis = draw_arrow_on_crop(crop, r, scale)
            info = make_info_panel(r, fi, vis.shape[0])
            combined = np.hstack([vis, info])

            # 高亮目标帧
            if fi == tf:
                cv2.rectangle(combined, (0, 0), (combined.shape[1]-1, combined.shape[0]-1), (0, 0, 255), 3)

            images.append(combined)

            # 更新 prev
            if r:
                sk_abs = abs(r["skewness"])
                if not (sk_abs < ar.AMB_SKEW_ABS and ar.AMB_ASPECT_MIN <= r["aspect"] <= ar.AMB_ASPECT_MAX):
                    prev = r["angle_deg"]

        if images:
            # 竖向拼接所有帧
            max_w = max(img.shape[1] for img in images)
            padded = []
            for img in images:
                if img.shape[1] < max_w:
                    pad = np.zeros((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8)
                    img = np.hstack([img, pad])
                padded.append(img)
            grid = np.vstack(padded)
            out_path = out_dir / f"arrow_F{tf:04d}.png"
            cv2.imwrite(str(out_path), grid)
            print(f"[SAVED] {out_path}")

    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()
