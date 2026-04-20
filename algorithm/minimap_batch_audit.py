from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.map.autodetect import detect_minimap_circle


@dataclass
class DetectRow:
    file: str
    found: bool
    px: int | None
    py: int | None
    pr: int | None
    confidence: float | None
    layout: str | None
    dist_to_consensus: float | None
    radius_delta: float | None
    status: str
    reason: str
    note: str


def _iter_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _robust_consensus(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(1.0, 1.4826 * mad)
    return med, sigma


def _draw_result(img: np.ndarray, row: DetectRow, consensus: dict[str, float]) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    cxy = (int(consensus["x"]), int(consensus["y"]))
    cr = int(consensus["r"])
    cv2.circle(vis, cxy, max(3, cr), (0, 255, 255), 2)
    cv2.putText(vis, "consensus", (cxy[0] - 65, max(20, cxy[1] - cr - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    if row.found and row.px is not None and row.py is not None and row.pr is not None:
        color = (0, 255, 0) if row.status == "ok" else (0, 0, 255)
        cv2.circle(vis, (row.px, row.py), max(3, row.pr), color, 2)
        cv2.line(vis, cxy, (row.px, row.py), color, 1)

    panel = np.zeros((92, w, 3), dtype=np.uint8)
    text1 = f"status={row.status} reason={row.reason} conf={row.confidence if row.confidence is not None else 'NA'}"
    text2 = f"det=({row.px},{row.py},r={row.pr}) consensus=({int(consensus['x'])},{int(consensus['y'])},r={int(consensus['r'])})"
    text3 = f"dist={row.dist_to_consensus if row.dist_to_consensus is not None else 'NA'} radius_delta={row.radius_delta if row.radius_delta is not None else 'NA'}"
    cv2.putText(panel, text1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, text2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, text3, (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)

    return np.vstack([vis, panel])


def run_audit(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_images(input_dir)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    raw: list[tuple[Path, dict[str, Any] | None]] = []
    for fp in files:
        img = cv2.imread(str(fp))
        if img is None or img.size == 0:
            raw.append((fp, None))
            continue
        det = detect_minimap_circle(img, debug=False)
        raw.append((fp, det))

    found_rows = [d for _, d in raw if d is not None]
    if len(found_rows) < max(5, int(len(files) * 0.5)):
        raise RuntimeError("Too many missing detections in baseline, cannot estimate consensus robustly.")

    cx_med, cx_sigma = _robust_consensus([float(d["px"]) for d in found_rows])
    cy_med, cy_sigma = _robust_consensus([float(d["py"]) for d in found_rows])
    r_med, r_sigma = _robust_consensus([float(d["pr"]) for d in found_rows])

    dist_thr = max(8.0, 3.2 * ((cx_sigma + cy_sigma) / 2.0))
    # 半径在不同截图里会受线条厚度、抗锯齿、边缘对比度影响，不能作为强失败条件
    r_thr = max(12.0, 3.0 * r_sigma)

    rows: list[DetectRow] = []
    bad_files: list[str] = []

    consensus = {"x": cx_med, "y": cy_med, "r": r_med, "cx_sigma": cx_sigma, "cy_sigma": cy_sigma, "r_sigma": r_sigma}

    for fp, det in raw:
        if det is None:
            row = DetectRow(
                file=fp.name,
                found=False,
                px=None,
                py=None,
                pr=None,
                confidence=None,
                layout=None,
                dist_to_consensus=None,
                radius_delta=None,
                status="bad",
                reason="not_found",
                note="",
            )
            bad_files.append(fp.name)
        else:
            px = int(det["px"])
            py = int(det["py"])
            pr = int(det["pr"])
            dist = float(np.hypot(px - cx_med, py - cy_med))
            r_delta = float(abs(pr - r_med))
            reason_parts: list[str] = []
            if dist > dist_thr:
                reason_parts.append("center_outlier")
            radius_warn = r_delta > r_thr
            conf = float(det.get("confidence", 0.0))
            if conf < 0.26:
                reason_parts.append("low_confidence")

            if reason_parts:
                status = "bad"
                reason = "+".join(reason_parts)
                bad_files.append(fp.name)
            else:
                status = "ok"
                reason = "pass"

            note = "radius_warn" if radius_warn else ""

            row = DetectRow(
                file=fp.name,
                found=True,
                px=px,
                py=py,
                pr=pr,
                confidence=round(conf, 4),
                layout=str(det.get("layout", "")),
                dist_to_consensus=round(dist, 3),
                radius_delta=round(r_delta, 3),
                status=status,
                reason=reason,
                note=note,
            )

        rows.append(row)

        img = cv2.imread(str(fp))
        if img is not None and img.size > 0:
            vis = _draw_result(img, row, consensus)
            cv2.imwrite(str(debug_dir / fp.name), vis)

    csv_path = output_dir / "audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total": len(rows),
        "ok": sum(1 for r in rows if r.status == "ok"),
        "bad": sum(1 for r in rows if r.status == "bad"),
        "bad_files": bad_files,
        "consensus": {
            "px": round(cx_med, 3),
            "py": round(cy_med, 3),
            "pr": round(r_med, 3),
            "sigma_x": round(cx_sigma, 3),
            "sigma_y": round(cy_sigma, 3),
            "sigma_r": round(r_sigma, 3),
            "dist_threshold": round(dist_thr, 3),
            "radius_threshold": round(r_thr, 3),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        f"total={summary['total']} ok={summary['ok']} bad={summary['bad']}",
        f"consensus(px,py,pr)=({summary['consensus']['px']},{summary['consensus']['py']},{summary['consensus']['pr']})",
        f"thresholds(dist,r)=({summary['consensus']['dist_threshold']},{summary['consensus']['radius_threshold']})",
        f"radius_warn={sum(1 for r in rows if r.note == 'radius_warn')}",
        "bad files:",
    ]
    lines.extend([f"- {bf}" for bf in bad_files])
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch audit minimap detection quality on screenshots")
    parser.add_argument("--input", default="picture", help="image folder")
    parser.add_argument("--output", required=True, help="output folder")
    args = parser.parse_args()

    summary = run_audit(Path(args.input), Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
