from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from path_config import PLAYER_ARROW_TEMPLATE_NPY_PATH


DEFAULT_TEMPLATE_SIZE = 40
PNG_SELF_DIFF_MIN_RATIO = 0.10
_PRIVATE_ARROW_TEMPLATE_PNG_PATH = Path(__file__).resolve().parent.parent / "_private" / "arrow_up.png"


def build_synthetic_arrow_template(size: int = DEFAULT_TEMPLATE_SIZE) -> np.ndarray:
    """合成一个朝上的箭头模板（float32, [0,1]）。"""
    img = np.zeros((size, size), dtype=np.float32)
    c = size // 2

    head_bot = int(size * 0.57)
    head_hw = int(size * 0.46)
    cv2.fillPoly(
        img,
        [
            np.array(
                [
                    [c, max(1, int(size * 0.04))],
                    [c - head_hw, head_bot],
                    [c + head_hw, head_bot],
                ],
                dtype=np.int32,
            )
        ],
        1.0,
    )

    tail_hw = int(size * 0.15)
    cv2.fillPoly(
        img,
        [
            np.array(
                [
                    [c, int(size * 0.60)],
                    [c - tail_hw, int(size * 0.78)],
                    [c, int(size * 0.96)],
                    [c + tail_hw, int(size * 0.78)],
                ],
                dtype=np.int32,
            )
        ],
        0.75,
    )
    return img


def _load_npy_template(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        raw = np.load(str(path)).astype(np.float32)
    except Exception:
        return None

    if raw.ndim != 2:
        return None
    mx = float(raw.max())
    if mx <= 0:
        return None
    raw /= mx
    return raw


def _extract_template_from_png(path: Path, size: int = DEFAULT_TEMPLATE_SIZE) -> np.ndarray | None:
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    if raw.ndim == 3 and raw.shape[2] == 4:
        mask_f = raw[:, :, 3].astype(np.float32) / 255.0
    else:
        hsv = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2HSV)
        m = cv2.inRange(
            hsv,
            np.array([8, 80, 100], dtype=np.uint8),
            np.array([36, 255, 255], dtype=np.uint8),
        )
        mask_f = m.astype(np.float32) / 255.0

    if float(mask_f.max()) <= 0:
        return None

    ys, xs = np.where(mask_f > 0.1)
    if len(ys) == 0:
        return None

    cropped = mask_f[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    candidate = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    candidate = (candidate > 0.5).astype(np.float32)

    self_flip = np.flipud(candidate)
    s_s = float(np.dot(candidate.ravel(), candidate.ravel()))
    s_f = float(np.dot(candidate.ravel(), self_flip.ravel()))
    total_sf = s_s + s_f
    if total_sf <= 0:
        return None

    if abs(s_s - s_f) / total_sf < PNG_SELF_DIFF_MIN_RATIO:
        return None
    return candidate


def _save_template(path: Path, template: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), template.astype(np.float32))


def ensure_arrow_template_npy() -> Path:
    """
    保证箭头模板 .npy 可用：优先读取既有 npy，其次尝试私有 PNG，最后回退合成模板。
    返回最终 .npy 路径。
    """
    template = _load_npy_template(PLAYER_ARROW_TEMPLATE_NPY_PATH)
    if template is None:
        template = _extract_template_from_png(_PRIVATE_ARROW_TEMPLATE_PNG_PATH)
    if template is None:
        template = build_synthetic_arrow_template(DEFAULT_TEMPLATE_SIZE)

    if not PLAYER_ARROW_TEMPLATE_NPY_PATH.exists():
        _save_template(PLAYER_ARROW_TEMPLATE_NPY_PATH, template)
    return PLAYER_ARROW_TEMPLATE_NPY_PATH


def get_arrow_template(size: int = DEFAULT_TEMPLATE_SIZE) -> np.ndarray:
    """读取（并在必要时生成）朝上箭头模板，输出 float32 [0,1]。"""
    npy_path = ensure_arrow_template_npy()
    template = _load_npy_template(npy_path)
    if template is None:
        template = build_synthetic_arrow_template(DEFAULT_TEMPLATE_SIZE)

    if template.shape != (size, size):
        template = cv2.resize(template, (size, size), interpolation=cv2.INTER_AREA)
        mx = float(template.max())
        if mx > 0:
            template /= mx
    return template.astype(np.float32)
