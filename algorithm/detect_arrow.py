"""批量图片测试入口：调用 backend/map 中的算法。"""
from __future__ import annotations

import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.map import arrow as ar
from backend.map.minimap import consensus_from_images
from backend.map.autodetect import detect_minimap_circle

PICTURE_DIR = ROOT / "picture"
OUTPUT_DIR = ROOT / "algorithm" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _crop_center(img_bgr, cx, cy, radius):
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    return img_bgr[y1:y2, x1:x2].copy(), x1, y1


def _visualize(img_bgr, img_path, crop_bgr, result, mx, my, mr, crop_r, seq_idx):
    prefix = f"{seq_idx:02d}"
    vis = img_bgr.copy()
    cv2.circle(vis, (mx, my), mr, (0, 255, 0), 2)
    cv2.circle(vis, (mx, my), crop_r, (255, 255, 0), 1)

    if result is not None:
        x1 = max(0, mx - crop_r)
        y1 = max(0, my - crop_r)
        hx = int(result["head_xy"][0]) + x1
        hy = int(result["head_xy"][1]) + y1
        tx = int(result["tail_xy"][0]) + x1
        ty = int(result["tail_xy"][1]) + y1
        cv2.arrowedLine(vis, (tx, ty), (hx, hy), (0, 0, 255), 2, tipLength=0.4)
        cv2.putText(vis, f"{result['angle_deg']:.1f}", (mx + mr + 4, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(vis, "MISS", (mx + mr + 4, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

    cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_full.jpg"), vis)

    crop_vis = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_NEAREST)
    if result is not None:
        s = 160 / max(crop_r * 2, 1)
        hx_s = int(result["head_xy"][0] * s)
        hy_s = int(result["head_xy"][1] * s)
        tx_s = int(result["tail_xy"][0] * s)
        ty_s = int(result["tail_xy"][1] * s)
        cv2.arrowedLine(crop_vis, (tx_s, ty_s), (hx_s, hy_s), (0, 0, 255), 2, tipLength=0.4)
    cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_crop.jpg"), crop_vis)


def _process_one(img_path, idx, ccx, ccy, cr, prev_stable_angle, det_cache):
    t0 = time.perf_counter()
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    cached = det_cache.get(img_path.name)
    if cached is not None:
        mx, my, mr = cached
    else:
        d = detect_minimap_circle(img_bgr)
        if d is None:
            return None
        mx, my, mr = int(d["px"]), int(d["py"]), int(d["pr"])
        if ccx is not None and (abs(mx - ccx) + abs(my - ccy) > 60 or abs(mr - cr) > 30):
            mx, my, mr = ccx, ccy, cr

    result = ar.detect_arrow_from_frame(img_bgr, mx, my, mr, prev_stable_angle=prev_stable_angle)
    crop_r = max(ar.ARROW_CROP_RADIUS, int(mr * 0.25))
    crop_bgr, _, _ = _crop_center(img_bgr, mx, my, crop_r)
    _visualize(img_bgr, img_path, crop_bgr, result, mx, my, mr, crop_r, idx)

    if result is not None:
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        warn = result.get("warning") or ""
        extra = f" [!]{warn}" if warn else ""
        print(f"  -> [{idx:02d}] {img_path.name}: angle={result['angle_deg']}{extra}")
    else:
        print(f"  -> [{idx:02d}] {img_path.name}: NO ARROW")

    return result


def main():
    images = sorted(PICTURE_DIR.glob("*.png")) + sorted(PICTURE_DIR.glob("*.jpg"))
    if not images:
        print(f"[ERROR] no images in {PICTURE_DIR}")
        return

    print(f"Total: {len(images)} images\n")
    ccx, ccy, cr, det_cache = consensus_from_images(images)
    if ccx is None:
        print("[ERROR] consensus failed")
        return
    print(f"[Consensus] center=({ccx},{ccy}) r={cr} valid={len(det_cache)}/{len(images)}")

    results = {}
    prev_stable = None
    for i, p in enumerate(images, 1):
        r = _process_one(p, i, ccx, ccy, cr, prev_stable, det_cache)
        results[p.name] = (i, r)
        if r is not None:
            skew = abs(float(r.get("skewness", 0.0)))
            asp = float(r.get("aspect", 0.0))
            if not (skew < ar.AMB_SKEW_ABS and ar.AMB_ASPECT_MIN <= asp <= ar.AMB_ASPECT_MAX):
                prev_stable = float(r["angle_deg"])

    print(f"\n{'#':<4} {'filename':<32} {'angle':>8} {'px':>6} {'asp':>6} {'skew':>7} {'ms':>8} status")
    print("-" * 92)

    times = []
    times, tmpl_hits = [], 0
    for name, (idx, r) in results.items():
        if r is None:
            print(f"{idx:<4} {name:<32} {'--':>8} {'--':>6} {'--':>6} {'--':>7} {'--':>8} FAIL")
            continue
        ms = float(r.get("elapsed_ms", 0.0))
        times.append(ms)
        if r.get("warning") and "[amb_tmpl]" in (r.get("warning") or ""):
            tmpl_hits += 1
        st = "WARN" if r.get("warning") else "OK"
        print(f"{idx:<4} {name:<32} {r['angle_deg']:>8.1f} {r['mask_pixels']:>6} {r['aspect']:>6.2f} {r['skewness']:>7.3f} {ms:>8.2f} {st}")

    if times:
        arr = np.asarray(times, dtype=np.float64)
        print(f"\n[Timing] frames={len(arr)} min={arr.min():.2f}ms avg={arr.mean():.2f}ms p95={np.percentile(arr,95):.2f}ms max={arr.max():.2f}ms")
        if tmpl_hits:
            print(f"[Template] amb_tmpl triggered={tmpl_hits}/{len(arr)} ({100*tmpl_hits/len(arr):.1f}%)")


if __name__ == "__main__":
    main()
