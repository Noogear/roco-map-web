"""
core/data_standards.py - 多用户数据域标准

用于统一约束后端数据归属：
  - GLOBAL_SHARED: 全局只读共享（跨会话复用）
  - SESSION_SCOPED: 按 session_token 隔离
  - REQUEST_SCOPED: 单请求临时数据
"""

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
    """给运行时对象打上数据域标签，用于审计。"""
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
    """
    运行时审计：
      - tracker 必须是 SESSION_SCOPED
      - tracker.feature_engine 必须是 SESSION_SCOPED
      - tracker._shared 必须是 GLOBAL_SHARED
    """
    ensure_scope(tracker_obj, DataScope.SESSION_SCOPED, 'MapTrackerWeb')
    if hasattr(tracker_obj, 'feature_engine'):
        ensure_scope(getattr(tracker_obj, 'feature_engine'), DataScope.SESSION_SCOPED, 'FeatureMapTracker')
    if hasattr(tracker_obj, '_shared'):
        ensure_scope(getattr(tracker_obj, '_shared'), DataScope.GLOBAL_SHARED, 'SharedFeatureResources')


GLOBAL_SHARED_CONTRACTS: tuple[DataContract, ...] = (
    DataContract(
        name='SharedFeatureResources',
        scope=DataScope.GLOBAL_SHARED,
        mutable=False,
        description='大地图 ORB 特征、全局 FLANN、Hash 索引，只读共享。',
    ),
    DataContract(
        name='MapDataStore',
        scope=DataScope.GLOBAL_SHARED,
        mutable=False,
        description='markers/categories/searchIndex 缓存，按版本刷新。',
    ),
)


SESSION_SCOPED_CONTRACTS: tuple[DataContract, ...] = (
    DataContract(
        name='MapTrackerWeb',
        scope=DataScope.SESSION_SCOPED,
        mutable=True,
        description='每个 session_token 独立追踪状态（frame/status/smoother）。',
    ),
    DataContract(
        name='PushSession',
        scope=DataScope.SESSION_SCOPED,
        mutable=True,
        description='每个 SID/JPEG 推送节流状态，仅作用于所属会话。',
    ),
)


REQUEST_SCOPED_CONTRACTS: tuple[DataContract, ...] = (
    DataContract(
        name='RecognizeSingleRequestData',
        scope=DataScope.REQUEST_SCOPED,
        mutable=True,
        description='recognize_single 里的临时图像/特征/候选，仅当前请求可见。',
    ),
)
