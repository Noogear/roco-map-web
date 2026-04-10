import sys
import math
import cv2
import numpy as np
from PIL import Image as PILImage
import base64
import os
from io import BytesIO
import json
import struct
import time
import config
from flask import Flask, render_template, Response, jsonify, request, send_file, send_from_directory, copy_current_request_context
from threading import Lock
from flask_socketio import SocketIO, emit, disconnect
from collections import deque

# 坐标过滤参数
POS_HISTORY_SIZE = 20      # 保留最近 N 次坐标
POS_OUTLIER_THRESHOLD = 200 # 超过此距离视为异常帧，丢弃

# 解析启动参数: python main_ai_web.py [cpu|sift|ai|loftr]
#   cpu / sift  -> 仅 SIFT 模式（不加载 torch/kornia，秒启动）
#   ai  / loftr -> 完整双引擎模式（需要 torch + kornia）
_START_MODE = (sys.argv[1].lower() if len(sys.argv) > 1 else 'ai')
_SIFT_ONLY = _START_MODE in ('cpu', 'sift')

app = Flask(__name__)
# 使用 eventlet 或 threading 模式支持 WebSocket
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


class SIFTMapTracker:
    """SIFT 传统特征匹配引擎"""

    def __init__(self):
        print("正在初始化 SIFT 引擎...")
        self.clahe = cv2.createCLAHE(clipLimit=config.SIFT_CLAHE_LIMIT, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create()
        
        # 线程锁（防止 WebSocket 并发请求导致 kp_local / flann 竞态）
        self._lock = Lock()

        logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = logic_map_bgr.shape[:2]

        logic_map_gray = cv2.cvtColor(logic_map_bgr, cv2.COLOR_BGR2GRAY)
        logic_map_gray = self.clahe.apply(logic_map_gray)
        print("正在提取大地图 SIFT 特征点...")
        self.kp_big_all, self.des_big_all = self.sift.detectAndCompute(logic_map_gray, None)
        print(f"✅ 全局特征点: {len(self.kp_big_all)} 个")

        # 预存全局特征点坐标（用于快速局部筛选）
        self.kp_coords = np.array([kp.pt for kp in self.kp_big_all], dtype=np.float32)

        # === 全局 FLANN 索引（常驻不销毁）===
        print("正在构建全局 FLANN 索引...")
        self.flann_global = self._create_flann(self.des_big_all)

        # === 局部 FLANN（动态重建，初始=全局）===
        self.kp_local = list(self.kp_big_all)
        self.des_local = self.des_big_all.copy()
        self.flann_local = self.flann_global
        self.using_local = False
        self.local_fail_count = 0

        # === 局部搜索参数 ===
        self.SEARCH_RADIUS = getattr(config, 'SEARCH_RADIUS', 400)
        self.LOCAL_FAIL_LIMIT = getattr(config, 'LOCAL_FAIL_LIMIT', 5)
        self.JUMP_THRESHOLD = getattr(config, 'SIFT_JUMP_THRESHOLD', 500)

        # 惯性导航状态
        self.last_x = None
        self.last_y = None
        self.lost_frames = 0

        # 性能监控（与 main123 对齐）
        import time as _time
        self._perf_time = _time
        self._frame_times = []

    # ------------------------------------------
    # 构建/重建 FLANN 索引
    # ------------------------------------------
    @staticmethod
    def _create_flann(descriptors):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flann.add([descriptors])
        flann.train()
        return flann

    # ------------------------------------------
    # 切换到局部搜索模式
    # ------------------------------------------
    def _switch_to_local(self, cx, cy):
        """以 (cx, cy) 为中心，提取半径内的特征点，重建局部 FLANN"""
        r = self.SEARCH_RADIUS
        dx = np.abs(self.kp_coords[:, 0] - cx)
        dy = np.abs(self.kp_coords[:, 1] - cy)
        mask = (dx < r) & (dy < r)
        indices = np.where(mask)[0]

        if len(indices) < 20:   # 特征点太少不值得切局部
            return

        self.kp_local = [self.kp_big_all[i] for i in indices]
        self.des_local = self.des_big_all[indices]
        self.flann_local = self._create_flann(self.des_local)
        self.using_local = True
        self.local_fail_count = 0

    # ------------------------------------------
    # 回退到全局搜索模式
    # ------------------------------------------
    def _switch_to_global(self):
        """回退全局（全局 FLANN 常驻，无需重建）"""
        self.using_local = False
        self.local_fail_count = 0
        print(f"🌍 已切换到 全局 搜索模式 | 特征点: {len(self.kp_big_all)} | lost: {self.lost_frames}")

    # ------------------------------------------
    # 箭头检测：从圆形小地图中提取玩家箭头中心坐标
    # 箭头颜色: 左侧 #FEB630(黄) / 右侧 #E78B1A(橙) → BGR 格式
    # ------------------------------------------
    @staticmethod
    def _detect_arrow_center(minimap_bgr):
        """
        在小地图图像中检测黄色/橙色箭头，返回 (cx, cy, arrow_patch) 或 None。
        使用 HSV 颜色范围 + 形态学过滤。
        arrow_patch: 箭头的RGBA图像块(BGRA格式)，可直接贴到目标图上
        """
        h, w = minimap_bgr.shape[:2]
        if h < 10 or w < 10:
            return None

        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

        # 黄-橙 HSV 范围（覆盖 #FEB630 ~ #E78B1A）
        lower = np.array([15, 120, 180])   # H:黄-橙区间, S:高饱和度, V:明亮
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学去噪：先开运算去小噪点，再闭运算填空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 计算轮廓和质心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 取最大连通区域（箭头主体），计算质心
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 20:   # 太小不可信
            return None

        M = cv2.moments(largest)
        if M['m00'] < 1e-6:
            return None

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # === 提取箭头图像块（含白色描边，用于直接贴图）===
        try:
            # 检测白色/浅色描边
            hsv_full = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 80, 255])
            white_mask = cv2.inRange(hsv_full, white_lower, white_upper)

            # 膨胀黄色mask，找其外围的白色像素作为"描边"
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            yellow_dilated = cv2.dilate(mask, kernel_dilate, iterations=2)
            white_border = white_mask & yellow_dilated & ~mask

            # 合并黄色 + 白色描边 = 完整箭头形状作为alpha通道
            combined_alpha = cv2.bitwise_or(mask, white_border)

            # 形态学清理
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_alpha = cv2.morphologyEx(combined_alpha, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

            # 裁剪包围矩形区域
            x, y, bw, bh = cv2.boundingRect(combined_alpha)
            pad = 2  # 边缘留几像素余量
            x1_crop = max(0, x - pad)
            y1_crop = max(0, y - pad)
            x2_crop = min(w, x + bw + pad)
            y2_crop = min(h, y + bh + pad)

            # 提取BGR + Alpha
            bgr_crop = minimap_bgr[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            alpha_crop = combined_alpha[y1_crop:y2_crop, x1_crop:x2_crop]

            # 合成BGRA图像块
            arrow_patch = cv2.merge([bgr_crop[:, :, 0], bgr_crop[:, :, 1],
                                     bgr_crop[:, :, 2], alpha_crop])

            # 记录patch左上角相对于质心的偏移（用于贴图定位）
            patch_offset_x = int(cx) - x1_crop
            patch_offset_y = int(cy) - y1_crop
        except Exception:
            arrow_patch = None
            patch_offset_x = 0
            patch_offset_y = 0

        return (float(cx), float(cy), arrow_patch, patch_offset_x, patch_offset_y)

    @staticmethod
    def _draw_arrow_marker(img_bgr, cx, cy, size=None, angle=0):
        """
        在地图上绘制玩家方向箭头标记。
        模仿游戏小地图的箭头样式：三角形 + 渐变填充。
        angle: 旋转角度（度），0=朝上，顺时针增加
        """
        if size is None:
            size = 12

        # 定义三角形顶点（箭头尖朝上）
        pts = np.array([
            [cx, cy - size],           # 顶点（朝上/前进方向）
            [cx - size * 0.6, cy + size * 0.7],  # 左下
            [cx + size * 0.6, cy + size * 0.7],  # 右下
        ], dtype=np.float64)

        # 绕中心 (cx, cy) 旋转
        if angle != 0:
            rad = math.radians(-angle)  # OpenCV角度转弧度（负号：顺时针为正）
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            centered = pts - np.array([cx, cy])
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = (rot_mat @ centered.T).T + np.array([cx, cy])
            pts = rotated.astype(np.int32)
        else:
            pts = pts.astype(np.int32)

        # 填充渐变色（左亮右暗，模仿原版箭头）
        overlay = img_bgr.copy()
        cv2.fillPoly(overlay, [pts], (48, 182, 254))  # #FEB630 BGR
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 1)  # 白色描边
        cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)

    @staticmethod
    def _paste_arrow_patch(img_bgr, cx, cy, arrow_patch, offset_x, offset_y, scale=1.0):
        """
        将从小地图抠出的箭头图像块(BGRA)贴到目标图像上。
        cx, cy: 目标图上的贴图位置（箭头质心对齐点）
        offset_x, offset_y: patch内质心相对于patch左上角的偏移
        scale: 缩放倍数（1.0=原始大小）
        """
        if arrow_patch is None or arrow_patch.size == 0:
            return

        ph, pw = arrow_patch.shape[:2]

        if scale != 1.0:
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            arrow_patch = cv2.resize(arrow_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            offset_x = int(offset_x * scale)
            offset_y = int(offset_y * scale)
            ph, pw = new_w, new_h

        # 计算patch在目标图上的放置区域
        paste_x = cx - offset_x
        paste_y = cy - offset_y

        dh, dw = img_bgr.shape[:2]

        # 裁剪到目标图边界
        sx1 = max(0, -paste_x)
        sy1 = max(0, -paste_y)
        dx1 = max(0, paste_x)
        dy1 = max(0, paste_y)
        dx2 = min(dw, paste_x + pw)
        dy2 = min(dh, paste_y + ph)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        if dx2 <= dx1 or dy2 <= dy1 or sx2 <= sx1 or sy2 <= sy1:
            return

        roi = img_bgr[dy1:dy2, dx1:dx2]
        patch_region = arrow_patch[sy1:sy2, sx1:sx2]

        bgr_part = patch_region[:, :, :3]
        alpha_part = patch_region[:, :, 3:4].astype(np.float32) / 255.0

        blended = (bgr_part.astype(np.float32) * alpha_part +
                   roi.astype(np.float32) * (1.0 - alpha_part))
        img_bgr[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)

    # ------------------------------------------
    # 核心匹配（两轮搜索 + 跳变过滤）
    # ------------------------------------------
    def match(self, minimap_bgr):
        with self._lock:  # 线程安全：防止并发修改 kp_local / flann 导致 IndexError
            return self._match_impl(minimap_bgr)

    def _match_impl(self, minimap_bgr):
        t_start = self._perf_time.perf_counter()
        found = False
        center_x, center_y = None, None
        arrow_patch = None
        patch_offset_x = 0
        patch_offset_y = 0
        is_inertial = False

        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = self.clahe.apply(minimap_gray)
        kp_mini, des_mini = self.sift.detectAndCompute(minimap_gray, None)

        if des_mini is not None and len(kp_mini) >= 2:

            for search_round in range(2):

                # --- 选择当前轮次的 FLANN 和特征点 ---
                if search_round == 0 and self.using_local:
                    current_kp = self.kp_local
                    current_flann = self.flann_local
                elif search_round == 0 and not self.using_local:
                    current_kp = list(self.kp_big_all)
                    current_flann = self.flann_global
                else:
                    # 第2轮：局部失败或跳变过大 → 回退全局
                    if not self.using_local:
                        break
                    current_kp = list(self.kp_big_all)
                    current_flann = self.flann_global

                try:
                    matches = current_flann.knnMatch(des_mini, k=2)
                except cv2.error:
                    matches = []

                good = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                            good.append(m)

                if len(good) >= config.SIFT_MIN_MATCH_COUNT:
                    src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD)

                    if M is not None:
                        # 验证 RANSAC 内点数量（防假匹配）
                        inlier_count = int(mask.sum()) if mask is not None else 0
                        if inlier_count < config.SIFT_MIN_MATCH_COUNT:
                            continue

                        h, w = minimap_gray.shape

                        # === 箭头中心检测（替代几何中心）===
                        arrow_center = self._detect_arrow_center(minimap_bgr)
                        if arrow_center is not None:
                            ref_x, ref_y, arrow_patch, patch_offset_x, patch_offset_y = arrow_center
                        else:
                            # 回退：使用图像几何中心
                            ref_x, ref_y = w / 2, h / 2

                        center_pt = np.float32([[[ref_x, ref_y]]])
                        dst_center = cv2.perspectiveTransform(center_pt, M)
                        tx = int(dst_center[0][0][0])
                        ty = int(dst_center[0][0][1])

                        if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
                            # 跳变过滤：只要有历史坐标就检查（与 main123 对齐）
                            if search_round == 0 and self.last_x is not None:
                                jump = abs(tx - self.last_x) + abs(ty - self.last_y)
                                if jump < 500:
                                    found = True
                                    center_x, center_y = tx, ty
                                    break
                                else:
                                    continue  # 跳变过大，去第2轮用全局重搜
                            else:
                                # 首次定位 / 丢失后重新找到(last_x已被清除) → 直接接受
                                found = True
                                center_x, center_y = tx, ty
                                break

        # --- 状态更新 ---
        if found:
            self.last_x = center_x
            self.last_y = center_y
            self.lost_frames = 0
            self.local_fail_count = 0
            self._switch_to_local(center_x, center_y)

        else:
            self.lost_frames += 1

            # 局部连续失败 → 及时回退全局
            if self.using_local:
                self.local_fail_count += 1
                if self.local_fail_count >= self.LOCAL_FAIL_LIMIT:
                    self._switch_to_global()

            # 惯性兜底
            if self.last_x is not None and self.lost_frames <= config.MAX_LOST_FRAMES:
                found = True
                center_x = self.last_x
                center_y = self.last_y
                is_inertial = True
            elif self.lost_frames > config.MAX_LOST_FRAMES:
                # 超时清除旧坐标，避免下次全局定位被跳变过滤误杀
                self.last_x = None
                self.last_y = None
                if self.using_local:
                    self._switch_to_global()

        # === 性能统计（与 main123 对齐，每60帧输出一次）===
        t_elapsed = (self._perf_time.perf_counter() - t_start) * 1000
        self._frame_times.append(t_elapsed)
        if len(self._frame_times) >= 60:
            avg = sum(self._frame_times) / len(self._frame_times)
            mode = "局部" if self.using_local else "全局"
            feat = len(self.kp_local) if self.using_local else len(self.kp_big_all)
            print(f"[{mode}模式] 平均耗时: {avg:.1f}ms | "
                  f"特征点: {feat} | lost: {self.lost_frames} | "
                  f"fps: {1000/avg:.1f}")
            self._frame_times.clear()

        return {
            'found': found,
            'center_x': center_x,
            'center_y': center_y,
            'arrow_patch': arrow_patch,
            'patch_offset_x': patch_offset_x,
            'patch_offset_y': patch_offset_y,
            'is_inertial': is_inertial,
            'match_count': 0,
            'map_width': self.map_width,
            'map_height': self.map_height,
        }


class LoFTRMapTracker:
    """LoFTR AI 深度学习匹配引擎（仅在需要时才 import torch/kornia）"""
    def __init__(self):
        # 延迟导入，SIFT 模式下完全不会触发
        import torch as _torch
        import kornia as _K
        from kornia.feature import LoFTR as _LoFTR

        print("正在加载 LoFTR AI 模型...")
        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        print(f"当前计算设备: {self.device}")

        # 优先从本地 web/ 目录加载模型权重，避免联网下载
        local_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'loftr_outdoor.ckpt')
        if os.path.exists(local_ckpt):
            print(f"从本地加载模型: {local_ckpt}")
            self.matcher = _LoFTR(pretrained=None).to(self.device)
            raw = _torch.load(local_ckpt, map_location=self.device, weights_only=False)
            sd = raw.get('state_dict', raw)
            self.matcher.load_state_dict(sd)
        else:
            print("本地未找到模型，尝试联网下载...")
            self.matcher = _LoFTR(pretrained='outdoor').to(self.device)

        self.matcher.eval()
        self.torch = _torch
        self.kornia = _K
        print("AI 模型加载完成！")

        self.logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if self.logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = self.logic_map_bgr.shape[:2]

        self.display_map_bgr = cv2.imread(config.DISPLAY_MAP_PATH)
        if self.display_map_bgr is None:
            raise FileNotFoundError(f"找不到显示地图: {config.DISPLAY_MAP_PATH}！")

        self.state = "GLOBAL_SCAN"
        self.last_x = 0
        self.last_y = 0
        self.scan_size = config.AI_SCAN_SIZE
        self.scan_step = config.AI_SCAN_STEP
        self.scan_x = 0
        self.scan_y = 0
        self.search_radius = config.AI_TRACK_RADIUS
        self.lost_frames = 0
        self.max_lost_frames = 3

    def preprocess_image(self, img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        new_h = h - (h % 8)
        new_w = w - (w % 8)
        img_gray = cv2.resize(img_gray, (new_w, new_h))
        tensor = self.kornia.image_to_tensor(img_gray, False).float() / 255.0
        return tensor.to(self.device)

    def match(self, minimap_bgr):
        with self._lock:  # 线程安全：防止并发修改 kp_local / flann 导致 IndexError
            return self._match_impl(minimap_bgr)

    def _match_impl(self, minimap_bgr):
        t_start = self._perf_time.perf_counter()
        found = False
        display_crop = None
        half_view = config.VIEW_SIZE // 2
        match_count = 0

        if self.state == "GLOBAL_SCAN":
            x1 = self.scan_x
            y1 = self.scan_y
            x2 = min(self.map_width, x1 + self.scan_size)
            y2 = min(self.map_height, y1 + self.scan_size)
            display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
            display_crop = cv2.resize(display_crop, (config.VIEW_SIZE, int(config.VIEW_SIZE * max(1, y2 - y1) / max(1, x2 - x1))))
            cv2.putText(display_crop, f"Global Scan: X:{x1} Y:{y1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        else:
            x1 = max(0, self.last_x - self.search_radius)
            y1 = max(0, self.last_y - self.search_radius)
            x2 = min(self.map_width, self.last_x + self.search_radius)
            y2 = min(self.map_height, self.last_y + self.search_radius)

        local_logic_map = self.logic_map_bgr[y1:y2, x1:x2]

        if local_logic_map.shape[0] >= 16 and local_logic_map.shape[1] >= 16:
            tensor_mini = self.preprocess_image(minimap_bgr)
            tensor_big_local = self.preprocess_image(local_logic_map)
            input_dict = {"image0": tensor_mini, "image1": tensor_big_local}

            with self.torch.no_grad():                correspondences = self.matcher(input_dict)

            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()

            valid_idx = confidence > config.AI_CONFIDENCE_THRESHOLD
            mkpts0 = mkpts0[valid_idx]
            mkpts1 = mkpts1[valid_idx]
            match_count = len(mkpts0)

            if len(mkpts0) >= config.AI_MIN_MATCH_COUNT:
                M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, config.AI_RANSAC_THRESHOLD)
                if M is not None:
                    h, w = minimap_bgr.shape[:2]
                    center_pt = np.float32([[[w / 2, h / 2]]])
                    dst_center_local = cv2.perspectiveTransform(center_pt, M)
                    center_x = int(dst_center_local[0][0][0] + x1)
                    center_y = int(dst_center_local[0][0][1] + y1)

                    if 0 <= center_x < self.map_width and 0 <= center_y < self.map_height:
                        found = True
                        self.last_x = center_x
                        self.last_y = center_y
                        self.state = "LOCAL_TRACK"
                        self.lost_frames = 0

                        vy1 = max(0, center_y - half_view)
                        vy2 = min(self.map_height, center_y + half_view)
                        vx1 = max(0, center_x - half_view)
                        vx2 = min(self.map_width, center_x + half_view)
                        display_crop = self.display_map_bgr[vy1:vy2, vx1:vx2].copy()
                        local_cx = center_x - vx1
                        local_cy = center_y - vy1
                        cv2.circle(display_crop, (local_cx, local_cy), radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.circle(display_crop, (local_cx, local_cy), radius=7, color=(255, 255, 255), thickness=1)

        if not found:
            if self.state == "LOCAL_TRACK":
                self.lost_frames += 1
                if self.lost_frames <= self.max_lost_frames:
                    vy1 = max(0, self.last_y - half_view)
                    vy2 = min(self.map_height, self.last_y + half_view)
                    vx1 = max(0, self.last_x - half_view)
                    vx2 = min(self.map_width, self.last_x + half_view)
                    display_crop = self.display_map_bgr[vy1:vy2, vx1:vx2].copy()
                    local_cx = self.last_x - vx1
                    local_cy = self.last_y - vy1
                    cv2.circle(display_crop, (local_cx, local_cy), radius=5, color=(0, 255, 255), thickness=-1)
                else:
                    print("彻底丢失目标，启动全局雷达扫描...")
                    self.state = "GLOBAL_SCAN"
                    self.scan_x = 0
                    self.scan_y = 0
                    display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                    cv2.putText(display_crop, "Radar Initializing...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
            elif self.state == "GLOBAL_SCAN":
                self.scan_x += self.scan_step
                if self.scan_x >= self.map_width:
                    self.scan_x = 0
                    self.scan_y += self.scan_step
                    if self.scan_y >= self.map_height:
                        self.scan_x = 0
                        self.scan_y = 0

        return {
            'found': found,
            'display_crop': display_crop,
            'state': self.state,
            'last_x': self.last_x,
            'last_y': self.last_y,
            'match_count': match_count,
            'map_width': self.map_width,
            'map_height': self.map_height,
        }


class AIMapTrackerWeb:
    def __init__(self):
        # --- 加载引擎 ---
        print("=" * 50)
        self.sift_engine = SIFTMapTracker()
        if _SIFT_ONLY:
            print("  [SIFT-only 模式] 已跳过 torch/kornia/LoFTR 加载")
            self.loftr_engine = None
            self.current_mode = 'sift'
        else:
            self.loftr_engine = LoFTRMapTracker()
            # 默认 SIFT 模式（用户可在前端切换）
            self.current_mode = 'sift'
        print("=" * 50)

        # 显示地图（两种模式共用）
        self.display_map_bgr = cv2.imread(config.DISPLAY_MAP_PATH)
        self.map_height = self.sift_engine.map_height
        self.map_width = self.sift_engine.map_width

        # 线程安全锁
        self.lock = Lock()

        # 当前帧数据
        self.current_frame_bgr = None
        self.latest_result_image = None
        self.latest_result_jpeg = None

        # 坐标历史（用于异常值过滤）
        self.pos_history = deque(maxlen=POS_HISTORY_SIZE)

        # 坐标平滑缓冲池（60帧滑动窗口 + 中位数滤波，消除 SIFT 抖动）
        self.SMOOTH_BUFFER_SIZE = 60
        self.smooth_buffer_x = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self.smooth_buffer_y = deque(maxlen=self.SMOOTH_BUFFER_SIZE)
        self._smooth_median_window = 15  # 取最近15帧的中位数作为输出（抗抖动）

        # 持久化文件路径（与 main_web.py 同级）
        import os as _os
        self._smooth_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.smooth_coords.json')
        self._load_smooth_buffer()  # 启动时恢复历史坐标

        self.latest_status = {
            'mode': 'sift',
            'state': '--',
            'position': {'x': 0, 'y': 0},
            'found': False,
            'matches': 0,
        }

    def _smooth_coord(self, raw_x, raw_y):
        """
        坐标平滑：60帧滑动窗口 + 最近 N 帧中位数滤波。
        返回 (smooth_x, smooth_y)，消除 SIFT 单帧随机抖动。
        每60帧自动持久化到本地文件。
        """
        if raw_x is not None and raw_y is not None:
            self.smooth_buffer_x.append(raw_x)
            self.smooth_buffer_y.append(raw_y)
            # 缓冲区满时自动保存
            if len(self.smooth_buffer_x) >= self.SMOOTH_BUFFER_SIZE:
                self._save_smooth_buffer()

        n = min(len(self.smooth_buffer_x), self._smooth_median_window)
        if n < 3:
            return raw_x or 0, raw_y or 0

        # 取最近 N 帧的中位数（抗极端抖动）
        recent_x = list(self.smooth_buffer_x)[-n:]
        recent_y = list(self.smooth_buffer_y)[-n:]
        recent_x.sort()
        recent_y.sort()
        mid = n // 2
        if n % 2 == 0:
            sx = (recent_x[mid - 1] + recent_x[mid]) // 2
            sy = (recent_y[mid - 1] + recent_y[mid]) // 2
        else:
            sx = recent_x[mid]
            sy = recent_y[mid]
        return int(sx), int(sy)

    def _load_smooth_buffer(self):
        """启动时从本地文件恢复最近60个坐标点"""
        import os as _os
        if not _os.path.isfile(self._smooth_file):
            return
        try:
            with open(self._smooth_file, 'r') as f:
                data = json.load(f)
            xs = data.get('x', [])
            ys = data.get('y', [])
            if len(xs) == len(ys) and len(xs) > 0:
                self.smooth_buffer_x.extend(xs[-self.SMOOTH_BUFFER_SIZE:])
                self.smooth_buffer_y.extend(ys[-self.SMOOTH_BUFFER_SIZE:])
                print(f"📍 已从本地恢复 {len(xs)} 个历史坐标点 (最新: {xs[-1]}, {ys[-1]})")
        except Exception as e:
            print(f"[警告] 加载坐标历史失败: {e}")

    def _save_smooth_buffer(self):
        """将当前缓冲池的坐标写入本地 JSON 文件"""
        try:
            data = {
                'x': list(self.smooth_buffer_x),
                'y': list(self.smooth_buffer_y),
                'ts': time.time(),
            }
            with open(self._smooth_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            pass  # 静默失败，不影响主流程

    def set_minimap(self, minimap_bgr):
        with self.lock:
            self.current_frame_bgr = minimap_bgr.copy()

    def set_mode(self, mode):
        """切换识别模式: 'sift' 或 'loftr'"""
        if mode not in ('sift', 'loftr'):
            return False
        if mode == 'loftr' and self.loftr_engine is None:
            print("当前为 SIFT-only 模式，无法切换到 LoFTR（请不带参数启动以启用 AI）")
            return False
        self.current_mode = mode
        # 切换模式时重置状态，避免残留
        self.smooth_buffer_x.clear()
        self.smooth_buffer_y.clear()
        if mode == 'loftr':
            self.loftr_engine.state = "GLOBAL_SCAN"
            self.loftr_engine.scan_x = 0
            self.loftr_engine.scan_y = 0
        return True

    def process_frame(self):
        with self.lock:
            if self.current_frame_bgr is None:
                return None
            minimap_bgr = self.current_frame_bgr.copy()

        half_view = config.VIEW_SIZE // 2

        if self.current_mode == 'sift':
            result = self.sift_engine.match(minimap_bgr)
            found = result['found']
            cx, cy = result['center_x'], result['center_y']
            arrow_patch = result.get('arrow_patch')
            patch_ox = result.get('patch_offset_x', 0)
            patch_oy = result.get('patch_offset_y', 0)
            is_inertial = result.get('is_inertial', False)

            # 坐标平滑：原始坐标用于状态机，平滑坐标用于渲染
            smooth_x, smooth_y = self._smooth_coord(cx, cy)

            if found and cx is not None:
                # 应用渲染偏移校正（微调定位点）
                ox = getattr(config, 'RENDER_OFFSET_X', 0)
                oy = getattr(config, 'RENDER_OFFSET_Y', 0)

                # 用平滑后坐标 + 偏移 裁剪和画点
                y1 = max(0, smooth_y + oy - half_view)
                y2 = min(self.map_height, smooth_y + oy + half_view)
                x1 = max(0, smooth_x + ox - half_view)
                x2 = min(self.map_width, smooth_x + ox + half_view)
                display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
                local_x = (smooth_x + ox) - x1
                local_y = (smooth_y + oy) - y1
                if not is_inertial:
                    if arrow_patch is not None:
                        # 直接贴小地图抠出的箭头（天然方向+颜色+描边）
                        self.sift_engine._paste_arrow_patch(display_crop, int(local_x), int(local_y),
                                                            arrow_patch, patch_ox, patch_oy, scale=1.5)
                    else:
                        # fallback：画三角形箭头
                        self.sift_engine._draw_arrow_marker(display_crop, local_x, local_y)
                    cv2.circle(display_crop, (local_x, int(local_y + 6)), radius=3, color=(255, 255, 255), thickness=-1)
                else:
                    # 惯性模式
                    if arrow_patch is not None:
                        self.sift_engine._paste_arrow_patch(display_crop, int(local_x), int(local_y),
                                                            arrow_patch, patch_ox, patch_oy, scale=1.5)
                        overlay = display_crop.copy()
                        overlay[:] = (0, 180, 180)
                        mask_arr = np.zeros_like(display_crop)
                        cv2.circle(mask_arr, (local_x, int(local_y + 4)), 10, (255, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.4, display_crop, 0.6, 0, display_crop)
                    else:
                        self.sift_engine._draw_arrow_marker(display_crop, local_x, local_y)
                        overlay = display_crop.copy()
                        overlay[:] = (0, 180, 180)  # 青色蒙版
                        mask_arr = np.zeros_like(display_crop)
                        cv2.circle(mask_arr, (local_x, int(local_y + 4)), 10, (255, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.4, display_crop, 0.6, 0, display_crop)
            else:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                cv2.putText(display_crop, "SIFT Searching...", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            status_state = 'INERTIAL' if is_inertial else ('FOUND' if found else 'SEARCHING')
            match_count = 0
            # 状态上报用平滑后坐标（前端显示更稳定）
            last_x = smooth_x
            last_y = smooth_y

        else:  # loftr
            if self.loftr_engine is None:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                cv2.putText(display_crop, "AI Engine Disabled", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                found = False
                status_state = 'DISABLED'
                match_count = 0
                last_x = last_y = 0
            else:
                result = self.loftr_engine.match(minimap_bgr)
            display_crop = result.get('display_crop')
            if display_crop is None:
                display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
            found = result['found']
            status_state = result.get('state', 'GLOBAL_SCAN')
            match_count = result.get('match_count', 0)
            last_x = result.get('last_x', 0)
            last_y = result.get('last_y', 0)

        # ====== 坐标异常值过滤 ======
        if found and (last_x or last_y):
            is_outlier = False
            if len(self.pos_history) >= 3:
                # 取历史中位数作为参考
                hist = list(self.pos_history)
                ref_x = sorted(h[0] for h in hist)[len(hist)//2]
                ref_y = sorted(h[1] for h in hist)[len(hist)//2]
                if abs(last_x - ref_x) > POS_OUTLIER_THRESHOLD or abs(last_y - ref_y) > POS_OUTLIER_THRESHOLD:
                    is_outlier = True
            if not is_outlier:
                self.pos_history.append((last_x, last_y))
            else:
                # 异常帧：丢弃，用最近一次有效坐标
                last_x = self.pos_history[-1][0]
                last_y = self.pos_history[-1][1]
        elif found:
            self.pos_history.append((last_x, last_y))

        # 渲染最终输出
        display_rgb = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(display_rgb)
        final_img = PILImage.new('RGB', (config.VIEW_SIZE, config.VIEW_SIZE), (43, 43, 43))
        final_img.paste(pil_image,
                        (max(0, half_view - pil_image.width // 2), max(0, half_view - pil_image.height // 2)))

        buffer_png = BytesIO()
        final_img.save(buffer_png, format='PNG')
        img_base64 = base64.b64encode(buffer_png.getvalue()).decode('utf-8')

        # JPEG 二进制输出（给 WebSocket 用，更小更快）
        buffer_jpeg = BytesIO()
        final_img.save(buffer_jpeg, format='JPEG', quality=85, optimize=True)
        jpeg_bytes = buffer_jpeg.getvalue()

        self.latest_status = {
            'mode': self.current_mode,
            'state': status_state,
            'position': {'x': last_x, 'y': last_y},
            'found': found,
            'matches': match_count,
        }
        self.latest_result_image = img_base64
        self.latest_result_jpeg = jpeg_bytes

        return img_base64, jpeg_bytes


# 全局实例
tracker = AIMapTrackerWeb()

# 获取 web 目录绝对路径
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

@app.route('/')
def index():
    return send_file(os.path.join(WEB_DIR, 'index.html'))


@app.route('/<path:filename>')
def serve_static(filename):
    """提供 web/ 目录下的静态资源（图片等）"""
    return send_from_directory(WEB_DIR, filename)


@app.route('/api/test_images')
def list_test_images():
    """列出 web/img/ 下可用的测试小地图"""
    img_dir = os.path.join(WEB_DIR, 'img')
    images = []
    if os.path.isdir(img_dir):
        for f in sorted(os.listdir(img_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append(f'/img/{f}')
    return jsonify(images)


@app.route('/api/upload_minimap', methods=['POST'])
def upload_minimap():
    """接收前端上传的小地图图片（支持 FormData 和 JSON base64 两种格式）"""
    img = None

    # 方式1: FormData 文件上传
    if 'image' in request.files:
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 方式2: JSON base64 上传
    elif request.is_json:
        data = request.get_json()
        b64_data = data.get('image', '')
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]  # 去掉 data:image/png;base64, 前缀
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        result = tracker.process_frame()
        if result:
            return jsonify({
                'success': True,
                'image': result[0],
                'status': tracker.latest_status
            })
    return jsonify({'error': 'Invalid image'}), 400


@app.route('/api/status')
def get_status():
    """获取当前追踪状态"""
    return jsonify(tracker.latest_status)


@app.route('/api/map_info')
def get_map_info():
    """大地图模式：返回地图尺寸和图片路径"""
    return jsonify({
        'map_width': tracker.map_width,
        'map_height': tracker.map_height,
        'display_map_url': '/big_map-1.png',
    })


@app.route('/bigmap')
def bigmap_page():
    """大地图独立页面"""
    return send_file(os.path.join(WEB_DIR, 'bigmap.html'))


@app.route('/api/mode', methods=['POST'])
def set_mode():
    """切换识别模式: sift 或 loftr"""
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'sift').lower()
    if tracker.set_mode(mode):
        return jsonify({'success': True, 'mode': tracker.current_mode})
    return jsonify({'error': f'Invalid mode: {mode}'}), 400


@app.route('/api/result')
def get_result():
    """获取最新的结果图片"""
    if tracker.latest_result_image:
        return jsonify({
            'image': tracker.latest_result_image,
            'status': tracker.latest_status
        })
    return jsonify({'error': 'No result yet'}), 404


@app.route('/api/process')
def process():
    """手动触发一次处理（用于文件模式）"""
    result = tracker.process_frame()
    if result:
        return jsonify({
            'image': result[0],
            'status': tracker.latest_status
        })
    return jsonify({'error': 'No minimap set'}), 400


# ==================== WebSocket 端点 ====================

@socketio.on('connect')
def ws_connect():
    """客户端建立 WS 连接"""
    print(f"WebSocket 客户端已连接: {request.sid}")
    emit('status', tracker.latest_status)


@socketio.on('disconnect')
def ws_disconnect():
    print(f"WebSocket 客户端断开: {request.sid}")


@socketio.on('mode')
def ws_set_mode(data):
    """通过 WS 切换模式"""
    mode = (data.get('mode', 'sift') if isinstance(data, dict) else 'sift').lower()
    ok = tracker.set_mode(mode)
    emit('mode_result', {'success': ok, 'mode': tracker.current_mode})


@socketio.on('frame')
def ws_receive_frame(raw_bytes):
    """
    接收二进制 JPEG 帧，处理并返回结果。
    协议: 客户端发送原始 JPEG bytes → 服务端返回 [JSON头(4字节长度) + JPEG图片]
    """
    # 解码图像
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        _, jpeg_result = tracker.process_frame()

        # 构造状态 JSON（紧凑格式）
        status_json = json.dumps({
            'm': tracker.current_mode,
            's': tracker.latest_status['state'],
            'x': tracker.latest_status['position']['x'],
            'y': tracker.latest_status['position']['y'],
            'f': int(tracker.latest_status['found']),
            'c': tracker.latest_status['matches'],
        }, separators=(',', ':')).encode('utf-8')

        # 发送: [4字节大端长度][JSON状态][JPEG图片]
        emit('result',
             struct.pack('>I', len(status_json)) + status_json + jpeg_result,
             binary=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


@socketio.on('frame_coords')
def ws_frame_coords(raw_bytes):
    """大地图模式：接收帧但只返回坐标（不返回裁剪图片），大幅减少带宽"""
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        tracker.set_minimap(img)
        tracker.process_frame()  # 处理，但不需要图片输出

        status_json = json.dumps({
            'm': tracker.current_mode,
            's': tracker.latest_status['state'],
            'x': tracker.latest_status['position']['x'],
            'y': tracker.latest_status['position']['y'],
            'f': int(tracker.latest_status['found']),
            'c': tracker.latest_status['matches'],
        }, separators=(',', ':')).encode('utf-8')

        emit('coords',
             struct.pack('>I', len(status_json)) + status_json,
             binary=True)
    else:
        err = b'{"error":"decode_fail"}'
        emit('error', struct.pack('>I', len(err)) + err, binary=True)


if __name__ == "__main__":
    print("=" * 50)
    mode_label = "SIFT-only (快速模式)" if _SIFT_ONLY else "双引擎 (SIFT + LoFTR AI)"
    print(f"  地图跟点 - 网页版 [{mode_label}]")
    if _SIFT_ONLY:
        print("  用法: python main_ai_web.py cpu   → 仅 SIFT (无需 torch)")
        print("       python main_ai_web.py ai    → SIFT + LoFTR AI")
    print("  打开浏览器访问: http://0.0.0.0:" + str(config.PORT))
    print("  WebSocket: ws://0.0.0.0:" + str(config.PORT) + "/socket.io/?transport=websocket")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=config.PORT, debug=False, allow_unsafe_werkzeug=True)
