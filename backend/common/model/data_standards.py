from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataScope(str, Enum):
    GLOBAL_SHARED = 'global_shared'
    SESSION_SCOPED = 'session_scoped'
    REQUEST_SCOPED = 'request_scoped'


@dataclass(frozen=True)
class DataContract:
    name: str
    scope: DataScope
    mutable: bool
    description: str


SCOPE_ATTR = '__data_scope__'


def bind_scope(obj, scope: DataScope) -> None:
    setattr(obj, SCOPE_ATTR, scope)


def get_scope(obj) -> DataScope | None:
    scope = getattr(obj, SCOPE_ATTR, None)
    if isinstance(scope, DataScope):
        return scope
    return None


def ensure_scope(obj, expected: DataScope, name: str = 'object') -> None:
    actual = get_scope(obj)
    if actual != expected:
        raise RuntimeError(f'{name} scope mismatch: expected={expected.value}, actual={getattr(actual, "value", actual)}')


def audit_tracker_scope(tracker_obj) -> None:
    ensure_scope(tracker_obj, DataScope.SESSION_SCOPED, 'MapTrackerWeb')
    if hasattr(tracker_obj, 'feature_engine'):
        ensure_scope(getattr(tracker_obj, 'feature_engine'), DataScope.SESSION_SCOPED, 'FeatureMapTracker')
    if hasattr(tracker_obj, '_shared'):
        ensure_scope(getattr(tracker_obj, '_shared'), DataScope.GLOBAL_SHARED, 'SharedFeatureResources')


GLOBAL_SHARED_CONTRACTS: tuple[DataContract, ...] = (
    DataContract(name='SharedFeatureResources', scope=DataScope.GLOBAL_SHARED, mutable=False, description='shared readonly data'),
)
SESSION_SCOPED_CONTRACTS: tuple[DataContract, ...] = (
    DataContract(name='MapTrackerWeb', scope=DataScope.SESSION_SCOPED, mutable=True, description='session state'),
)
REQUEST_SCOPED_CONTRACTS: tuple[DataContract, ...] = (
    DataContract(name='RequestData', scope=DataScope.REQUEST_SCOPED, mutable=True, description='request temp state'),
)
