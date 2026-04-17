"""
core/temporal_bridge.py - 低纹理时序桥接器（多假设轨迹）

用途：
    在 ocean / low_texture 场景中，融合弱观测（ORB/ECC/HASH）与运动先验，
    用 top-k 多假设做时序打分，避免逐帧贪心导致的错误吸附。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BridgeObservation:
    """单帧候选观测。"""

    x: int
    y: int
    quality: float
    source: str


@dataclass
class _Hypothesis:
    x: float
    y: float
    score: float
    source: str


class LowTextureTemporalBridge:
    """低纹理段的时序桥接状态机。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        top_k: int = 8,
        decay: float = 0.92,
        transition_penalty: float = 0.002,
        min_obs_quality: float = 0.08,
        strong_source_bonus: float = 0.12,
        exit_min_quality: float = 0.55,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_k = max(2, int(top_k))
        self.decay = float(decay)
        self.transition_penalty = float(transition_penalty)
        self.min_obs_quality = float(min_obs_quality)
        self.strong_source_bonus = float(strong_source_bonus)
        self.exit_min_quality = float(exit_min_quality)

        self.active: bool = False
        self._hypotheses: list[_Hypothesis] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.active = False
        self._hypotheses.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _is_strong_source(source: str) -> bool:
        if not source:
            return False
        return (
            source == 'ORB_LOCAL'
            or source.startswith('ORB_GLOBAL')
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _bucket_key(x: float, y: float, grid: int = 40) -> tuple[int, int]:
        return int(round(x / grid)), int(round(y / grid))

    # ------------------------------------------------------------------
    def _dedupe_and_keep_topk(self, hyps: Iterable[_Hypothesis]) -> None:
        best_by_bucket: dict[tuple[int, int], _Hypothesis] = {}
        for h in hyps:
            key = self._bucket_key(h.x, h.y)
            prev = best_by_bucket.get(key)
            if prev is None or h.score > prev.score:
                best_by_bucket[key] = h
        merged = sorted(best_by_bucket.values(), key=lambda item: item.score, reverse=True)
        self._hypotheses = merged[: self.top_k]

    # ------------------------------------------------------------------
    def step(
        self,
        *,
        scene: str,
        observations: list[BridgeObservation],
        motion_hint: tuple[float, float] = (0.0, 0.0),
        map_width: int,
        map_height: int,
    ) -> BridgeObservation | None:
        """推进一帧，返回桥接后的最佳观测（可能为 None）。"""
        if not self.enabled:
            return None

        low_scene = scene in {'ocean', 'low_texture'}
        if not low_scene:
            # 离开低纹理段：等待高质量强观测后自动退出
            if self.active and observations:
                strongest = max(observations, key=lambda o: o.quality)
                if self._is_strong_source(strongest.source) and strongest.quality >= self.exit_min_quality:
                    self.reset()
            return None

        valid_obs = [o for o in observations if o.quality >= self.min_obs_quality]
        if not valid_obs and not self.active:
            return None

        dx, dy = motion_hint

        # 激活：从首批观测初始化 top-k
        if not self.active:
            seeds = sorted(valid_obs, key=lambda o: o.quality, reverse=True)[: self.top_k]
            if not seeds:
                return None
            self._hypotheses = [
                _Hypothesis(x=float(o.x), y=float(o.y), score=float(o.quality), source=o.source)
                for o in seeds
            ]
            self.active = True

        # 若无观测，仅做预测扩散 + 衰减
        if not valid_obs:
            propagated: list[_Hypothesis] = []
            for h in self._hypotheses:
                nx = min(max(0.0, h.x + dx), float(map_width - 1))
                ny = min(max(0.0, h.y + dy), float(map_height - 1))
                propagated.append(_Hypothesis(nx, ny, h.score * self.decay, h.source))
            self._dedupe_and_keep_topk(propagated)
        else:
            updated: list[_Hypothesis] = []
            for obs in valid_obs:
                best_prev_score = 0.0
                for prev in self._hypotheses:
                    pred_x = prev.x + dx
                    pred_y = prev.y + dy
                    dist = abs(obs.x - pred_x) + abs(obs.y - pred_y)
                    trans_score = prev.score * self.decay - self.transition_penalty * dist
                    if trans_score > best_prev_score:
                        best_prev_score = trans_score

                base = float(obs.quality)
                if self._is_strong_source(obs.source):
                    base += self.strong_source_bonus
                score = max(base, base + best_prev_score)
                updated.append(_Hypothesis(float(obs.x), float(obs.y), score, obs.source))

            # 保留少量旧轨迹，防止单帧观测抖动导致轨迹瞬断
            updated.extend(_Hypothesis(h.x + dx, h.y + dy, h.score * self.decay * 0.95, h.source)
                           for h in self._hypotheses)
            self._dedupe_and_keep_topk(updated)

        if not self._hypotheses:
            self.reset()
            return None

        best = self._hypotheses[0]
        return BridgeObservation(
            x=int(round(best.x)),
            y=int(round(best.y)),
            quality=max(self.min_obs_quality, min(1.0, best.score)),
            source=best.source or 'BRIDGE',
        )
