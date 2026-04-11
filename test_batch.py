"""
批量测试脚本：从 picture/ 文件夹加载截图，模拟方形截取 + HoughCircles 圆检测流程。
测试圆形小地图自动检测、自校准、SIFT 匹配等特性。
"""
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from tracker_engines import SIFTMapTracker
from tracker_core import AIMapTrackerWeb


def detect_minimap_circle(img_bgr):
    """从全屏截图的右上角自动检测圆形小地图（仅用于确定参考位置）"""
    h, w = img_bgr.shape[:2]
    roi_size = 300
    roi = img_bgr[0:roi_size, w - roi_size:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 50,
        param1=80, param2=40, minRadius=40, maxRadius=140
    )
    if circles is not None and len(circles[0]) > 0:
        c = circles[0][0]
        cx = int(w - roi_size + c[0])
        cy = int(c[1])
        r = int(c[2])
        return cx, cy, r
    return None


def extract_square(img_bgr, cx, cy, r, margin=1.4):
    """模拟浏览器方形截取（中心+扩展 margin）"""
    h, w = img_bgr.shape[:2]
    half = int(r * margin)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return img_bgr[y1:y2, x1:x2].copy()


def sift_match_detail(engine, minimap_bgr):
    """手动走一遍 SIFT 匹配流程，获取详细统计"""
    minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
    minimap_gray = engine.clahe.apply(minimap_gray)
    h_mm, w_mm = minimap_gray.shape[:2]
    circ_mask = SIFTMapTracker._make_circular_mask(h_mm, w_mm)
    kp_mini, des_mini = engine.sift.detectAndCompute(minimap_gray, circ_mask)

    detail = {
        'n_features': len(kp_mini) if kp_mini else 0,
        'n_good': 0, 'n_inliers': 0, 'det': 0.0, 'scale': 0.0,
    }
    if des_mini is None or len(kp_mini) < 2:
        return detail

    kp_all = list(engine.kp_big_all)
    try:
        matches = engine.flann_global.knnMatch(des_mini, k=2)
    except cv2.error:
        return detail

    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                good.append(m)
    detail['n_good'] = len(good)
    if len(good) < config.SIFT_MIN_MATCH_COUNT:
        return detail

    src = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_all[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)
    if M is not None and mask is not None:
        detail['n_inliers'] = int(mask.sum())
        detail['det'] = abs(np.linalg.det(M[:2, :2]))
        sx = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        sy = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
        detail['scale'] = (sx + sy) / 2
    return detail


def run_test():
    pic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'picture')
    files = sorted([f for f in os.listdir(pic_dir) if f.lower().endswith('.png')])
    print(f"找到 {len(files)} 张截图\n")

    print("初始化追踪器 (SIFT-only)...")
    tracker = AIMapTrackerWeb(sift_only=True)
    engine = tracker.sift_engine
    print(f"地图: {engine.map_width}x{engine.map_height}\n")

    # 确定参考位置（用于模拟浏览器截取中心）
    ref_cx, ref_cy, ref_r = None, None, None
    for fname in files:
        img = cv2.imread(os.path.join(pic_dir, fname))
        if img is None:
            continue
        circle = detect_minimap_circle(img)
        if circle:
            ref_cx, ref_cy, ref_r = circle
            print(f"参考圆: ({ref_cx},{ref_cy},r={ref_r}) from {fname}")
            break

    if ref_cx is None:
        print("错误：无法从任何截图中检测到小地图位置")
        return

    margin = getattr(config, 'MINIMAP_CAPTURE_MARGIN', 1.4)

    SEP = "=" * 100
    print(SEP)
    print(f"{'文件':<22} {'圆检测':<18} {'feat':>5} {'good':>5} {'inlr':>5} "
          f"{'scale':>6} {'质量':>5} {'结果':<8} {'坐标':<14} {'ms':>5}")
    print(SEP)

    stats = {'total': 0, 'circle_ok': 0, 'circle_miss': 0,
             'found': 0, 'miss': 0, 'low_quality': []}

    for fname in files:
        fpath = os.path.join(pic_dir, fname)
        short = fname[-18:]
        img = cv2.imread(fpath)
        if img is None:
            continue
        stats['total'] += 1

        # 模拟浏览器方形截取
        square = extract_square(img, ref_cx, ref_cy, ref_r, margin)

        # 圆形小地图检测 + 提取
        minimap = tracker._detect_and_extract_minimap(square)

        if minimap is None:
            stats['circle_miss'] += 1
            cal = tracker._circle_cal
            cal_info = f"(exp_r={cal.expected_r})" if cal._calibrated else "(校准中)"
            print(f"{short:<22} {'✗ 无圆':<18} {'--':>5} {'--':>5} {'--':>5} "
                  f"{'--':>6} {'--':>5} {'BLOCK':<8} {'--':<14} {'--':>5} {cal_info}")
            continue

        stats['circle_ok'] += 1
        cal = tracker._circle_cal
        circ_str = f"r={cal.expected_r}" if cal._calibrated else "校准中"

        # SIFT 匹配
        d = sift_match_detail(engine, minimap)
        t0 = time.perf_counter()
        result = engine.match(minimap)
        ms = (time.perf_counter() - t0) * 1000

        found = result['found'] and not result.get('is_inertial', False)
        q = result.get('match_quality', 0)
        mx = result.get('center_x') or 0
        my = result.get('center_y') or 0

        if found:
            stats['found'] += 1
            status = "FOUND"
        else:
            stats['miss'] += 1
            status = "MISS"

        if q < 0.2 and found:
            stats['low_quality'].append((short, q, mx, my))

        coord = f"({mx},{my})" if found else "--"
        print(f"{short:<22} {'✓ ' + circ_str:<18} {d['n_features']:>5} {d['n_good']:>5} "
              f"{d['n_inliers']:>5} {d['scale']:>6.1f} {q:>5.2f} {status:<8} {coord:<14} {ms:>5.0f}")

    print(SEP)
    print(f"\n总计: {stats['total']}")
    print(f"圆检测成功: {stats['circle_ok']}  圆检测失败(场景拦截): {stats['circle_miss']}")
    print(f"匹配成功: {stats['found']}  匹配失败: {stats['miss']}")
    if stats['low_quality']:
        print(f"低质量匹配({len(stats['low_quality'])}个):")
        for s, q, x, y in stats['low_quality']:
            print(f"  {s}: q={q:.2f} ({x},{y})")

    # 校准器最终状态
    cal = tracker._circle_cal
    if cal._calibrated:
        print(f"\n校准器: center=({cal.expected_cx},{cal.expected_cy}), r={cal.expected_r}")
    else:
        print(f"\n校准器: 未完成校准 (历史: {len(cal._history)}帧)")


if __name__ == '__main__':
    run_test()
