"""
backend/map - 小地图相关算法包。

主要导出：
  arrow.detect_arrow / detect_arrow_from_frame - 箭头方向检测
  icon_tracker.OpticalFlowTracker - LK 光流多点追踪器
  icon_tracker.load_icon_templates / detect_initial_icons - 图标模板加载与检测
"""
from .arrow import detect_arrow, detect_arrow_from_frame
from .autodetect import detect_minimap_circle, detect_minimap_circle_batch
from .icon_tracker import (
  IconTemplate,
  TrackedPoint,
  FrameResult,
  OpticalFlowTracker,
  load_icon_templates,
  detect_initial_icons,
)
from .minimap import (
  CircleCalibrator,
  MinimapCrop,
  detect_and_extract_with_meta,
  consensus_from_detections,
  consensus_from_images,
  consensus_from_video_capture,
)

__all__ = [
  "detect_arrow",
  "detect_arrow_from_frame",
  "detect_minimap_circle",
  "detect_minimap_circle_batch",
  "IconTemplate",
  "TrackedPoint",
  "FrameResult",
  "OpticalFlowTracker",
  "load_icon_templates",
  "detect_initial_icons",
  "CircleCalibrator",
  "MinimapCrop",
  "detect_and_extract_with_meta",
  "consensus_from_detections",
  "consensus_from_images",
  "consensus_from_video_capture",
]
