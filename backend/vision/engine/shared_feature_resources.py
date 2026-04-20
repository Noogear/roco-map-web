from __future__ import annotations

import cv2

from backend import config
from backend.common.model.data_standards import DataScope, bind_scope


class SharedFeatureResources:
    def __init__(self) -> None:
        bind_scope(self, DataScope.GLOBAL_SHARED)
        logic_map_gray = cv2.imread(config.LOGIC_MAP_PATH, cv2.IMREAD_GRAYSCALE)
        if logic_map_gray is None:
            raise FileNotFoundError(f'not found: {config.LOGIC_MAP_PATH}')
        self.map_height, self.map_width = logic_map_gray.shape[:2]
        self.logic_map_gray = logic_map_gray
