from __future__ import annotations

from backend.vision.engine.shared_feature_resources import SharedFeatureResources

_shared_feature_resources: SharedFeatureResources | None = None


def get_shared_feature() -> SharedFeatureResources:
    global _shared_feature_resources
    if _shared_feature_resources is None:
        _shared_feature_resources = SharedFeatureResources()
    return _shared_feature_resources
