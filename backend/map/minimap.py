"""
backend/map/minimap.py - 小地图圆形共识检测

将脚本中的重复“多样本聚类取共识中心”逻辑集中到 map 包。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from backend.map.autodetect import detect_minimap_circle
from backend.map.minimap_runtime import CircleCalibrator, MinimapCrop, detect_and_extract_with_meta


def _cluster_best(dets: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    if not dets:
        return []
    best = []
    for ref in dets:
        cl = [d for d in dets if math.hypot(d[0] - ref[0], d[1] - ref[1]) < 80 and abs(d[2] - ref[2]) < 40]
        if len(cl) > len(best):
            best = cl
    return best


def consensus_from_detections(dets: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    """从 (px,py,pr) 列表计算共识小地图圆心。"""
    best = _cluster_best(dets)
    if not best:
        return None
    cx = int(round(float(np.median([d[0] for d in best]))))
    cy = int(round(float(np.median([d[1] for d in best]))))
    cr = int(round(float(np.median([d[2] for d in best]))))
    return cx, cy, cr


def consensus_from_images(image_paths: list[Path]) -> tuple[int, int, int, dict[str, tuple[int, int, int]]]:
    """
    从图片路径列表检测并聚合共识。

    Returns
    -------
    (ccx, ccy, cr, raw_cache)
      - ccx, ccy, cr: 共识中心（无结果时为 None）
      - raw_cache: {filename: (px,py,pr)}
    """
    raw: dict[str, tuple[int, int, int]] = {}
    dets: list[tuple[int, int, int]] = []

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            d = detect_minimap_circle(img)
        except Exception:
            d = None
        if d:
            t = (int(d["px"]), int(d["py"]), int(d["pr"]))
            raw[p.name] = t
            dets.append(t)

    cc = consensus_from_detections(dets)
    if cc is None:
        return None, None, None, raw
    return cc[0], cc[1], cc[2], raw


def consensus_from_video_capture(cap: cv2.VideoCapture, sample_indices: list[int]) -> tuple[int, int, int] | None:
    """从视频指定帧索引采样检测并聚合共识。调用后会重置到第0帧。"""
    dets: list[tuple[int, int, int]] = []
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            d = detect_minimap_circle(frame)
        except Exception:
            d = None
        if d:
            dets.append((int(d["px"]), int(d["py"]), int(d["pr"])))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return consensus_from_detections(dets)


__all__ = [
    "CircleCalibrator",
    "MinimapCrop",
    "detect_and_extract_with_meta",
    "consensus_from_detections",
    "consensus_from_images",
    "consensus_from_video_capture",
]
