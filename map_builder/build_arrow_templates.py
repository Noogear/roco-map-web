import os
import cv2
import numpy as np

from backend import config
from backend.tracker_engines import get_shared_sift
from backend.tracking.minimap import detect_and_extract, CircleCalibrator
from backend.tracking.autodetect import detect_minimap_circle
from backend.core.enhance import correct_color_temperature, make_clahe_pair, enhance_minimap
from backend.core.features import extract_minimap_features, CircularMaskCache, sift_match_region

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICTURE_DIR = os.path.join(ROOT, 'picture')
OUT_PATH = os.path.join(ROOT, 'assets', 'arrow_template_bank.npz')

TEMPLATE_SIZE = 31
MIN_INLIERS = 20
MIN_QUALITY = 0.70

shared = get_shared_sift()
clahe_n, clahe_l = make_clahe_pair()


def preprocess(src):
    h0, w0 = src.shape[:2]
    mm = detect_and_extract(src, CircleCalibrator(), engine_frozen=False)
    if mm is None and min(h0, w0) >= 320:
        det = detect_minimap_circle(src)
        if det is not None:
            bs = min(w0, h0)
            cx = float(det['cx']) * w0
            cy = float(det['cy']) * h0
            r = float(det['r']) * bs
            margin = max(1.0, float(getattr(config, 'MINIMAP_CAPTURE_MARGIN', 1.4)))
            sz = max(10, int(round(r * 2 * margin)))
            rx = max(0, min(int(round(cx - sz / 2)), w0 - sz))
            ry = max(0, min(int(round(cy - sz / 2)), h0 - sz))
            crop = src[ry:ry + sz, rx:rx + sz].copy()
            mm2 = detect_and_extract(crop, CircleCalibrator(), engine_frozen=False)
            mm = mm2 if mm2 is not None else crop
    if mm is None:
        return None
    mm = correct_color_temperature(mm)
    graw = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
    g = enhance_minimap(graw, float(graw.std()), clahe_n, clahe_l, getattr(config, 'CLAHE_LOW_TEXTURE_THRESHOLD', 30))
    return mm, graw, g


def extract_center_patch(gray):
    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    patch = cv2.getRectSubPix(gray, (TEMPLATE_SIZE, TEMPLATE_SIZE), (cx, cy))
    if patch is None or patch.size == 0:
        return None
    blur = cv2.GaussianBlur(patch, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vmax = float(mag.max())
    if vmax <= 1e-6:
        return None
    mag /= vmax
    return mag


def build_mask(template):
    vals = template[template > 0]
    if vals.size == 0:
        return None
    thr = float(np.percentile(vals, 78.0))
    mask = (template >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)
    if int(np.count_nonzero(mask)) < 6:
        return None
    return mask


def main():
    files = sorted([f for f in os.listdir(PICTURE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    old_arrow_enabled = bool(getattr(config, 'MINIMAP_ARROW_HOLE_ENABLED', True))
    config.MINIMAP_ARROW_HOLE_ENABLED = False
    templates = []
    masks = []
    picked = []
    try:
        for fn in files:
            src = cv2.imread(os.path.join(PICTURE_DIR, fn))
            if src is None:
                continue
            prep = preprocess(src)
            if prep is None:
                continue
            mm, graw, g = prep
            kp, des = extract_minimap_features(g, shared.sift, CircularMaskCache())
            if des is None or kp is None or len(kp) < 3:
                continue
            r = sift_match_region(kp, des, g.shape, list(shared.kp_big_all), shared.flann_global,
                                  getattr(config, 'SIFT_MATCH_RATIO', 0.82), getattr(config, 'SIFT_MIN_MATCH_COUNT', 5),
                                  shared.map_width, shared.map_height)
            if r is None:
                r = sift_match_region(kp, des, g.shape, list(shared.kp_big_all), shared.flann_global,
                                      0.88, 3, shared.map_width, shared.map_height)
            if r is None:
                continue
            tx, ty, inl, q, sc = r
            if int(inl) < MIN_INLIERS or float(q) < MIN_QUALITY:
                continue
            tmpl = extract_center_patch(graw)
            if tmpl is None:
                continue
            mask = build_mask(tmpl)
            if mask is None:
                continue
            s = float(tmpl.sum())
            if s <= 1e-6:
                continue
            tmpl = tmpl / s
            templates.append(tmpl.astype(np.float32))
            masks.append(mask.astype(np.uint8))
            picked.append((fn, int(inl), float(q), mm.shape[:2]))
    finally:
        config.MINIMAP_ARROW_HOLE_ENABLED = old_arrow_enabled

    if not templates:
        raise SystemExit('No strong arrow templates extracted from screenshots.')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        templates=np.stack(templates, axis=0),
        masks=np.stack(masks, axis=0),
    )
    print('saved', OUT_PATH)
    print('count', len(templates))
    for item in picked:
        print(item)


if __name__ == '__main__':
    main()
