from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from backend.map.autodetect import detect_minimap_circle


@dataclass
class StressRow:
    file: str
    case: str
    expect_found: bool
    found: bool
    expected_cx: float | None
    expected_cy: float | None
    confidence: float | None
    cx: float | None
    cy: float | None
    r: float | None
    center_err: float | None
    pass_case: bool
    note: str


def _iter_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _scale_about(img: np.ndarray, cx: float, cy: float, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.array([[scale, 0.0, (1.0 - scale) * cx], [0.0, scale, (1.0 - scale) * cy]], dtype=np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _build_cases(img: np.ndarray, det: dict) -> list[tuple[str, np.ndarray, bool, float | None, float | None, str]]:
    h, w = img.shape[:2]
    cx, cy = int(det["px"]), int(det["py"])
    cx_n, cy_n = float(det["cx"]), float(det["cy"])

    # 1) 原图：有小地图
    orig = ("orig", img, True, cx_n, cy_n, "")

    # 2) 有小地图裁剪：裁取右上区域（保留小地图）
    x1, y1 = max(0, cx - int(0.7 * w)), 0
    x2, y2 = w, min(h, cy + int(0.55 * h))
    crop_with = img[y1:y2, x1:x2].copy()
    ch, cw = crop_with.shape[:2]
    exp_cx_with = (cx - x1) / max(cw, 1)
    exp_cy_with = (cy - y1) / max(ch, 1)
    case_with = ("crop_with_minimap", crop_with, True, float(exp_cx_with), float(exp_cy_with), "")

    # 3) 无小地图裁剪：裁取左下区域（规避小地图）
    x1n, y1n = 0, min(h - 2, int(h * 0.35))
    x2n, y2n = min(w, int(w * 0.70)), h
    crop_without = img[y1n:y2n, x1n:x2n].copy()
    case_without = ("crop_without_minimap", crop_without, False, None, None, "")

    # 4) 小地图过大：以小地图中心放大整帧（中心不变）
    large = _scale_about(img, float(cx), float(cy), 1.35)
    case_large = ("minimap_large", large, True, cx_n, cy_n, "")

    # 5) 小地图过小：以小地图中心缩小整帧（中心不变）
    small = _scale_about(img, float(cx), float(cy), 0.94)
    case_small = ("minimap_small", small, True, cx_n, cy_n, "")

    # 6) 180翻转：小地图从右上翻到左下
    flip180 = cv2.flip(img, -1)
    case_flip = ("flip_180_to_left_bottom", flip180, True, 1.0 - cx_n, 1.0 - cy_n, "")

    return [orig, case_with, case_without, case_large, case_small, case_flip]


def run(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _iter_images(input_dir)
    if not files:
        raise RuntimeError(f"No images in {input_dir}")

    rows: list[StressRow] = []
    fails: list[str] = []

    for fp in files:
        img = cv2.imread(str(fp))
        if img is None or img.size == 0:
            fails.append(f"{fp.name}: unreadable")
            continue

        base = detect_minimap_circle(img)
        if base is None:
            fails.append(f"{fp.name}: base_not_found")
            continue

        for case_name, case_img, expect_found, exp_cx, exp_cy, note in _build_cases(img, base):
            det = detect_minimap_circle(case_img)
            found = det is not None
            confidence = None if det is None else float(det.get("confidence", 0.0))
            cx = None if det is None else float(det.get("cx", 0.0))
            cy = None if det is None else float(det.get("cy", 0.0))
            rr = None if det is None else float(det.get("r", 0.0))

            pass_case = (found == expect_found)
            center_err = None

            if expect_found and found and exp_cx is not None and exp_cy is not None:
                center_err = float(np.hypot(cx - exp_cx, cy - exp_cy))
                # 裁剪/翻转场景允许更宽容的位置误差
                tol = 0.10 if case_name in {"crop_with_minimap", "flip_180_to_left_bottom"} else 0.08
                if center_err > tol:
                    pass_case = False
                    note = f"center_err={center_err:.4f}>tol={tol:.2f}"

            # 对“无小地图”场景，任何 found 都算失败
            if (not expect_found) and found:
                pass_case = False
                note = f"false_positive_conf={confidence:.3f}"

            if not pass_case:
                fails.append(f"{fp.name}/{case_name}: expect={expect_found} found={found} conf={confidence} err={center_err}")

            rows.append(
                StressRow(
                    file=fp.name,
                    case=case_name,
                    expect_found=expect_found,
                    found=found,
                    expected_cx=None if exp_cx is None else round(float(exp_cx), 6),
                    expected_cy=None if exp_cy is None else round(float(exp_cy), 6),
                    confidence=None if confidence is None else round(confidence, 6),
                    cx=None if cx is None else round(cx, 6),
                    cy=None if cy is None else round(cy, 6),
                    r=None if rr is None else round(rr, 6),
                    center_err=None if center_err is None else round(center_err, 6),
                    pass_case=pass_case,
                    note=note,
                )
            )

    if not rows:
        raise RuntimeError("No test rows generated")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(files),
        "cases_per_file": 6,
        "total_cases": len(rows),
        "passed": int(sum(1 for r in rows if r.pass_case)),
        "failed": int(sum(1 for r in rows if not r.pass_case)),
        "by_case": {
            c: {
                "total": int(sum(1 for r in rows if r.case == c)),
                "passed": int(sum(1 for r in rows if r.case == c and r.pass_case)),
                "failed": int(sum(1 for r in rows if r.case == c and not r.pass_case)),
            }
            for c in sorted({r.case for r in rows})
        },
        "failed_samples": fails[:120],
    }

    csv_path = output_dir / "stress_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"files={summary['files']} total_cases={summary['total_cases']}",
                f"passed={summary['passed']} failed={summary['failed']}",
            ]
            + [f"{k}: {v['passed']}/{v['total']}" for k, v in summary["by_case"].items()]
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimap stress test: crop/no-minimap/scale/flip")
    parser.add_argument("--input", default="picture", help="input folder")
    parser.add_argument("--output", required=True, help="output folder")
    args = parser.parse_args()

    summary = run(Path(args.input), Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
