from __future__ import annotations

import json
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from map_builder._internal.common import OUTPUT_DIR


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def save_last_result(payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "last_result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_action_with_log(action_name: str, action_func) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"{ts}_{action_name}.log"

    start = time.perf_counter()
    status = "ok"
    error_message = ""

    with log_path.open("w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"\n===== 执行任务: {action_name} =====")
            print(f"日志文件: {log_path}")
            try:
                action_func()
            except Exception as exc:
                status = "failed"
                error_message = str(exc)
                traceback.print_exc()

            duration = round(time.perf_counter() - start, 3)
            print(f"任务结束，状态: {status}，耗时: {duration}s")

    save_last_result(
        {
            "action": action_name,
            "status": status,
            "error": error_message,
            "logFile": str(log_path),
            "finishedAt": datetime.now().isoformat(timespec="seconds"),
        }
    )


def action_download_map() -> None:
    from map_builder._internal.download_map import download_and_stitch

    download_and_stitch()


def action_build_markers_bundle() -> None:
    """一键生成 markers 相关文件与图标。"""
    from map_builder._internal.fetch_categories import main as fetch_categories_main
    from map_builder._internal.fetch_markers import main as fetch_markers_main

    print("[bundle] Step 1/2: 抓取分类 ...")
    fetch_categories_main()
    print("[bundle] Step 2/2: 抓取标记与图标 ...")
    fetch_markers_main()


def action_build_arrow_npy() -> None:
    from map_builder._internal.arrow_template import ensure_arrow_template_npy

    p = ensure_arrow_template_npy()
    print(f"[OK] arrow template npy 已准备: {p}")


ACTIONS: dict[str, tuple[str, callable]] = {
    "1": ("build_markers_bundle", action_build_markers_bundle),
    "2": ("download_map", action_download_map),
    "3": ("build_arrow_npy", action_build_arrow_npy),
}


def print_menu() -> None:
    print("\n================ map_builder 交互菜单 ================")
    print("1) 一键生成 markers + 图标（build_markers_bundle）")
    print("2) 一键生成大地图（download_map）")
    print("3) 生成/校验箭头模板 npy（arrow_template）")
    print("0) 退出")
    print("输出日志与结果汇总位置: map_builder/out_put")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        print_menu()
        choice = input("请输入选项编号: ").strip().lower()

        if choice in {"0", "q", "quit", "exit"}:
            print("已退出 map_builder 交互菜单。")
            return

        action = ACTIONS.get(choice)
        if not action:
            print("无效选项，请重新输入。")
            continue

        action_name, action_func = action
        run_action_with_log(action_name, action_func)
        input("\n按回车继续... ")


if __name__ == "__main__":
    main()
