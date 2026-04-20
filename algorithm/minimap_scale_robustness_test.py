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
class CaseRow:
    file: str
    scale: float
    found: bool
    cx: float | None
    cy: float | None
    r: float | None
    px: int | None
    py: int | None
    pr: int | None
    confidence: float | None
    abs_dcx: float | None
    abs_dcy: float | None
    abs_dr: float | None
    status: str


def _iter_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return img
    h, w = img.shape[:2]
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


def run(input_dir: Path, output_dir: Path, focus_file: str, scales: list[float]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[CaseRow] = []
    files = _iter_images(input_dir)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    failed: list[str] = []

    for fp in files:
        img = cv2.imread(str(fp))
        if img is None or img.size == 0:
            for sc in scales:
                rows.append(CaseRow(fp.name, sc, False, None, None, None, None, None, None, None, None, None, None, "bad"))
            failed.append(f"{fp.name}: unreadable")
            continue

        base_det = detect_minimap_circle(img)
        if base_det is None:
            for sc in scales:
                rows.append(CaseRow(fp.name, sc, False, None, None, None, None, None, None, None, None, None, None, "bad"))
            failed.append(f"{fp.name}: base_not_found")
            continue

        base_cx = float(base_det["cx"])
        base_cy = float(base_det["cy"])
        base_r = float(base_det["r"])

        for sc in scales:
            test_img = _resize(img, sc)
            det = detect_minimap_circle(test_img)
            if det is None:
                rows.append(CaseRow(fp.name, sc, False, None, None, None, None, None, None, None, None, None, None, "bad"))
                failed.append(f"{fp.name}: scale={sc} not_found")
                continue

            cx = float(det["cx"])
            cy = float(det["cy"])
            rr = float(det["r"])
            dcx = abs(cx - base_cx)
            dcy = abs(cy - base_cy)
            dr = abs(rr - base_r)
            status = "ok" if (dcx <= 0.008 and dcy <= 0.008 and dr <= 0.03) else "warn"
            if status != "ok":
                failed.append(f"{fp.name}: scale={sc} drift dcx={dcx:.4f} dcy={dcy:.4f} dr={dr:.4f}")

            rows.append(
                CaseRow(
                    file=fp.name,
                    scale=float(sc),
                    found=True,
                    cx=round(cx, 6),
                    cy=round(cy, 6),
                    r=round(rr, 6),
                    px=int(det["px"]),
                    py=int(det["py"]),
                    pr=int(det["pr"]),
                    confidence=round(float(det.get("confidence", 0.0)), 6),
                    abs_dcx=round(dcx, 6),
                    abs_dcy=round(dcy, 6),
                    abs_dr=round(dr, 6),
                    status=status,
                )
            )

    arr = np.array([[r.abs_dcx or 0.0, r.abs_dcy or 0.0, r.abs_dr or 0.0] for r in rows if r.found], dtype=np.float64)
    focus_rows = [asdict(r) for r in rows if r.file == focus_file]

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(files),
        "scales": scales,
        "cases": len(rows),
        "found": int(sum(1 for r in rows if r.found)),
        "ok": int(sum(1 for r in rows if r.status == "ok")),
        "warn": int(sum(1 for r in rows if r.status == "warn")),
        "bad": int(sum(1 for r in rows if r.status == "bad")),
        "metrics": {
            "mean_abs_dcx": round(float(arr[:, 0].mean()) if len(arr) else 0.0, 6),
            "mean_abs_dcy": round(float(arr[:, 1].mean()) if len(arr) else 0.0, 6),
            "mean_abs_dr": round(float(arr[:, 2].mean()) if len(arr) else 0.0, 6),
            "max_abs_dcx": round(float(arr[:, 0].max()) if len(arr) else 0.0, 6),
            "max_abs_dcy": round(float(arr[:, 1].max()) if len(arr) else 0.0, 6),
            "max_abs_dr": round(float(arr[:, 2].max()) if len(arr) else 0.0, 6),
        },
        "focus_file": focus_file,
        "focus_rows": focus_rows,
        "failed_samples": failed[:50],
    }

    csv_path = output_dir / "scale_test_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"files={summary['files']} cases={summary['cases']} found={summary['found']}",
                f"ok={summary['ok']} warn={summary['warn']} bad={summary['bad']}",
                f"mean_abs_dcx={summary['metrics']['mean_abs_dcx']} mean_abs_dcy={summary['metrics']['mean_abs_dcy']} mean_abs_dr={summary['metrics']['mean_abs_dr']}",
                f"max_abs_dcx={summary['metrics']['max_abs_dcx']} max_abs_dcy={summary['metrics']['max_abs_dcy']} max_abs_dr={summary['metrics']['max_abs_dr']}",
                f"focus={focus_file}",
            ]
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimap detector robustness test under image scaling")
    parser.add_argument("--input", default="picture", help="input image folder")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--focus", default="QQ20260412-010841.png", help="focus sample file name")
    parser.add_argument("--scales", default="0.6,0.75,0.9,1.0,1.2,1.5", help="comma-separated scales")
    args = parser.parse_args()

    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    summary = run(Path(args.input), Path(args.output), args.focus, scales)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
