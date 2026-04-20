from __future__ import annotations

import base64
from threading import Lock

import cv2
import numpy as np

from backend import config
from backend.common.model.data_standards import DataScope, bind_scope
from backend.vision.engine.feature_map_tracker import FeatureMapTracker
from backend.vision.engine.shared_feature_factory import get_shared_feature
from backend.vision.engine.shared_feature_resources import SharedFeatureResources


class MapTrackerWeb:
    def __init__(self, session_id: str = 'default', shared: SharedFeatureResources | None = None):
        bind_scope(self, DataScope.SESSION_SCOPED)
        self.session_id = session_id
        self._shared = shared or get_shared_feature()
        self.feature_engine = FeatureMapTracker(self._shared)
        self.map_height = self._shared.map_height
        self.map_width = self._shared.map_width
        self.lock = Lock()
        self.current_frame_bgr = None
        self.latest_status = {'mode': 'clean', 'state': '--', 'position': {'x': 0, 'y': 0}, 'found': False, 'matches': 0}

    def set_minimap(self, minimap_bgr, token: str = ''):
        _ = token
        with self.lock:
            self.current_frame_bgr = minimap_bgr.copy()

    def process_frame(self, need_base64=True, need_jpeg=True):
        with self.lock:
            if self.current_frame_bgr is None:
                return None
            frame = self.current_frame_bgr.copy()
        result = self.feature_engine.match(frame)
        found = bool(result.get('found', False))
        out = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
        cv2.putText(out, 'idle', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        img_base64 = None
        jpeg_bytes = None
        if need_base64 or need_jpeg:
            ok, jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg_bytes = jpeg.tobytes()
                if need_base64:
                    img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode('utf-8')
        self.latest_status = {'mode': 'clean', 'state': 'FOUND' if found else 'IDLE', 'position': {'x': int(result['center_x'] or 0), 'y': int(result['center_y'] or 0)}, 'found': found, 'matches': 0, 'match_quality': float(result.get('match_quality', 0.0)), 'source': str(result.get('source', ''))}
        return img_base64, jpeg_bytes

    def get_latest_result_jpeg(self):
        return None

    def get_latest_result_base64(self):
        return None
