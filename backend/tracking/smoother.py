"""
tracking/smoother.py - 坐标平滑器

将原来散落在 MapTrackerWeb 中的四套平滑机制合并为一个 CoordSmoother：
  1. pos_history 异常值过滤（欧氏中位数）
  2. Kalman 滤波（4 状态 x/y/vx/vy）
  3. smooth_buffer 持久化缓冲（启动时恢复 Kalman 热状态）
  4. 渲染 EMA 防抖（静止死区 + 速度自适应 alpha）

外部只需调用：
  update(cx, cy, found, is_inertial, match_quality, arrow_stopped,
         tp_far_candidate=None) → (display_x, display_y, tp_confirmed)
"""

from __future__ import annotations

import math
import json
import os
from collections import deque

import cv2
import numpy as np

from backend import config


class CoordSmoother:
    """
    统一坐标平滑器，合并原来的 4 套机制。
    """

    SMOOTH_BUFFER_SIZE: int = 60
    POS_HISTORY_SIZE: int = 20
    POS_OUTLIER_THRESHOLD: float = 200.0

    def __init__(
        self,
        smooth_buffer_path: str | None = None,
    ) -> None:
        # 1. 异常值过滤历史
        self.pos_history: deque[tuple[int, int]] = deque(maxlen=self.POS_HISTORY_SIZE)

        # 2. Kalman
        self._kalman = self._create_kalman_filter()
        self._kalman_initialized: bool = False

        # 3. 持久化平滑缓冲
        self.smooth_buffer_x: deque[int] = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self.smooth_buffer_y: deque[int] = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self._smooth_buffer_path = smooth_buffer_path
        if smooth_buffer_path:
            self._load_smooth_buffer()

        # 4. 渲染 EMA
        self._display_x: int | None = None
        self._display_y: int | None = None

        # 传送检测候选缓冲
        self._tp_candidate_buffer: deque[tuple[int, int]] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # 公开主接口
    # ------------------------------------------------------------------

    def update(
        self,
        cx: int | None,
        cy: int | None,
        found: bool,
        is_inertial: bool,
        match_quality: float,
        arrow_stopped: bool,
        tp_far_candidate: tuple[int, int, float] | None = None,
    ) -> tuple[int, int, bool]:
        """
        接收一帧坐标，经过完整流水线处理后返回 (display_x, display_y, tp_confirmed)。
        tp_confirmed=True 时 display_x/y 即为传送确认后的新位置（供调用方同步引擎）。
        found=False 时直接返回 (0, 0, False)。
        """
        if not found or cx is None:
            return 0, 0, False

        # ① 异常值过滤
        if not is_inertial:
            if len(self.pos_history) >= 3:
                hist = list(self.pos_history)
                ref_x = sorted(h[0] for h in hist)[len(hist) // 2]
                ref_y = sorted(h[1] for h in hist)[len(hist) // 2]
                dist = math.sqrt((cx - ref_x) ** 2 + (cy - ref_y) ** 2)
                if dist <= self.POS_OUTLIER_THRESHOLD:
                    self.pos_history.append((cx, cy))
                else:
                    if self.pos_history:
                        cx, cy = self.pos_history[-1]
            else:
                self.pos_history.append((cx, cy))

        # ② Kalman 平滑
        if not is_inertial:
            smooth_x, smooth_y = self._kalman_update(cx, cy, match_quality)
        else:
            predicted = self._kalman_predict()
            if predicted is not None:
                smooth_x, smooth_y = predicted
                self._kalman_update(smooth_x, smooth_y, quality=0.3)
            else:
                smooth_x, smooth_y = cx, cy

        # ③ 持久化
        self.smooth_buffer_x.append(smooth_x)
        self.smooth_buffer_y.append(smooth_y)
        if len(self.smooth_buffer_x) >= self.SMOOTH_BUFFER_SIZE:
            self._save_smooth_buffer()

        # ④ 传送检测（来源1: Kalman 跳变 / 来源2: 引擎远距离候选）
        _tp_thresh = getattr(config, 'TP_JUMP_THRESHOLD', 300)
        _tp_confirm = getattr(config, 'TP_CONFIRM_FRAMES', 3)
        _tp_radius = getattr(config, 'TP_CLUSTER_RADIUS', 150)
        _tp_new_entry = False

        # 来源2：引擎传回的远距离候选（绕过了 JUMP_THRESHOLD 但质量达标）
        if tp_far_candidate is not None and self._display_x is not None:
            fc_x, fc_y, _ = tp_far_candidate
            if math.sqrt((fc_x - self._display_x) ** 2 + (fc_y - self._display_y) ** 2) > _tp_thresh:
                self._tp_candidate_buffer.append((fc_x, fc_y))
                _tp_new_entry = True

        # 来源1：真实匹配帧的卡尔曼输出超跳变阈值
        if not is_inertial and self._display_x is not None:
            jump = math.sqrt((smooth_x - self._display_x) ** 2 + (smooth_y - self._display_y) ** 2)
            if jump > _tp_thresh:
                self._tp_candidate_buffer.append((smooth_x, smooth_y))
                _tp_new_entry = True
            else:
                # 跳变量正常 → 正常移动，清除候选缓冲
                self._tp_candidate_buffer.clear()

        if _tp_new_entry and len(self._tp_candidate_buffer) >= _tp_confirm:
            cands = list(self._tp_candidate_buffer)
            med_x = sorted(c[0] for c in cands)[len(cands) // 2]
            med_y = sorted(c[1] for c in cands)[len(cands) // 2]
            scatter = max(
                math.sqrt((c[0] - med_x) ** 2 + (c[1] - med_y) ** 2) for c in cands
            )
            if scatter <= _tp_radius:
                print(f"[传送检测] 聚类确认 scatter={scatter:.0f}px → 立即重置到 ({med_x},{med_y})")
                self.reset_to(med_x, med_y)
                return med_x, med_y, True

        # ⑤ 渲染 EMA 防抖
        display_x, display_y = self._render_ema(smooth_x, smooth_y, is_inertial, arrow_stopped)
        return display_x, display_y, False

    def reset_to(self, x: int, y: int) -> None:
        """传送确认后立即重置所有过滤器到新位置。"""
        self.pos_history.clear()
        self.smooth_buffer_x.clear()
        self.smooth_buffer_y.clear()
        self._kalman_reset(x, y)
        self._display_x = x
        self._display_y = y
        self._tp_candidate_buffer.clear()

    def clear_state(self) -> None:
        """清空所有平滑状态（历史/缓冲/Kalman/EMA/传送候选）。"""
        self.pos_history.clear()
        self.smooth_buffer_x.clear()
        self.smooth_buffer_y.clear()
        self._kalman = self._create_kalman_filter()
        self._kalman_initialized = False
        self._display_x = None
        self._display_y = None
        self._tp_candidate_buffer.clear()

    # ------------------------------------------------------------------
    # Kalman 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _create_kalman_filter() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        kf.processNoiseCov[2, 2] = 2.0
        kf.processNoiseCov[3, 3] = 2.0
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 500.0
        return kf

    def _kalman_update(self, cx: int, cy: int, quality: float = 1.0) -> tuple[int, int]:
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        # 原观测噪声 * (1/quality)，质量低 → 噪声大 → 更信任预测
        noise = max(1.0, 50.0 * (1.0 - quality))
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise
        if not self._kalman_initialized:
            self._kalman.statePre = np.array(
                [[np.float32(cx)], [np.float32(cy)], [0.0], [0.0]])
            self._kalman.statePost = self._kalman.statePre.copy()
            self._kalman_initialized = True
        self._kalman.predict()
        corrected = self._kalman.correct(measurement)
        return int(corrected[0, 0]), int(corrected[1, 0])

    def _kalman_predict(self) -> tuple[int, int] | None:
        if not self._kalman_initialized:
            return None
        predicted = self._kalman.predict()
        return int(predicted[0, 0]), int(predicted[1, 0])

    def _kalman_reset(self, x: int, y: int) -> None:
        self._kalman = self._create_kalman_filter()
        self._kalman_initialized = False
        self._kalman_update(x, y, quality=1.0)

    # ------------------------------------------------------------------
    # 渲染 EMA 防抖
    # ------------------------------------------------------------------

    def _render_ema(
        self,
        smooth_x: int,
        smooth_y: int,
        is_inertial: bool,
        arrow_stopped: bool,
    ) -> tuple[int, int]:
        still_threshold = getattr(config, 'RENDER_STILL_THRESHOLD', 2)
        ema_alpha_min = getattr(config, 'RENDER_EMA_ALPHA', 0.35)
        ema_alpha_max = getattr(config, 'RENDER_EMA_ALPHA_MAX', 0.92)
        ema_slow_dist = getattr(config, 'RENDER_EMA_SLOW_DIST', 6)
        ema_fast_dist = getattr(config, 'RENDER_EMA_FAST_DIST', 45)

        if arrow_stopped and not is_inertial:
            still_threshold = getattr(config, 'RENDER_STOPPED_STILL_THRESHOLD', 10)

        if self._display_x is None:
            self._display_x = smooth_x
            self._display_y = smooth_y
            return smooth_x, smooth_y

        # 惯性帧 + TP 候选积累中：冻结渲染坐标，防止速度外推漂向错误方向
        if is_inertial and self._tp_candidate_buffer:
            return self._display_x, self._display_y

        ddx = smooth_x - self._display_x
        ddy = smooth_y - self._display_y
        dist = math.sqrt(ddx * ddx + ddy * ddy)

        if not is_inertial and dist <= still_threshold:
            return self._display_x, self._display_y

        t = max(0.0, min(1.0, (dist - ema_slow_dist) / max(ema_fast_dist - ema_slow_dist, 1)))
        alpha = ema_alpha_min + t * (ema_alpha_max - ema_alpha_min)
        self._display_x = int(self._display_x + alpha * ddx)
        self._display_y = int(self._display_y + alpha * ddy)
        return self._display_x, self._display_y

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load_smooth_buffer(self) -> None:
        path = self._smooth_buffer_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            xs = data.get('smooth_x', [])
            ys = data.get('smooth_y', [])
            for x, y in zip(xs, ys):
                self.smooth_buffer_x.append(int(x))
                self.smooth_buffer_y.append(int(y))
            if xs and ys:
                lx, ly = int(xs[-1]), int(ys[-1])
                self._kalman_update(lx, ly, quality=0.5)
                self._display_x = lx
                self._display_y = ly
                print(f"[平滑缓冲] 已加载 {len(xs)} 个历史坐标，最后位置: ({lx},{ly})")
        except Exception as e:
            print(f"[平滑缓冲] 加载失败: {e}")

    def _save_smooth_buffer(self) -> None:
        path = self._smooth_buffer_path
        if not path:
            return
        try:
            data = {
                'smooth_x': list(self.smooth_buffer_x),
                'smooth_y': list(self.smooth_buffer_y),
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[平滑缓冲] 保存失败: {e}")
