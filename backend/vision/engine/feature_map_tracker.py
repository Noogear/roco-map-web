from __future__ import annotations

from threading import Lock

from backend.common.model.data_standards import DataScope, bind_scope
from backend.vision.engine.shared_feature_resources import SharedFeatureResources


class FeatureMapTracker:
    def __init__(self, shared: SharedFeatureResources):
        bind_scope(self, DataScope.SESSION_SCOPED)
        self.map_height = shared.map_height
        self.map_width = shared.map_width
        self.last_x: int | None = None
        self.last_y: int | None = None
        self._lock = Lock()

    def match(self, _minimap_bgr, *, minimap_center=None, minimap_radius=None):
        _ = minimap_center, minimap_radius
        with self._lock:
            return {'found': False, 'center_x': None, 'center_y': None, 'match_quality': 0.0, 'source': 'EMPTY', 'map_width': self.map_width, 'map_height': self.map_height}

    def mark_measurement_rejected(self) -> None:
        return None

    def reset(self) -> None:
        with self._lock:
            self.last_x = None
            self.last_y = None

    @property
    def last_position(self) -> tuple[int, int] | None:
        if self.last_x is None or self.last_y is None:
            return None
        return self.last_x, self.last_y
