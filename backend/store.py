"""
store.py - 持久化辅助（圆形状态 + 路线文件）

从 server.py 中提取，集中管理所有文件 I/O，
server.py 只需从此模块调用函数，不直接操作文件。
"""

from __future__ import annotations

import json
import os
import time


# ---------------------------------------------------------------------------
# 路线文件
# ---------------------------------------------------------------------------

def get_route_files(routes_dir: str) -> list[str]:
    """扫描 routes 目录，返回 .json 文件名列表（已排序）。"""
    if not os.path.isdir(routes_dir):
        return []
    return sorted(f for f in os.listdir(routes_dir) if f.lower().endswith('.json'))


def load_route_data(routes_dir: str, filename: str) -> dict | None:
    """加载单个路线 JSON 文件。filename 已经过安全校验（basename only）。"""
    filepath = os.path.join(routes_dir, filename)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 圆形选区状态
# ---------------------------------------------------------------------------

def load_circle_state(state_file: str) -> dict | None:
    """从本地 JSON 文件恢复圆形选区状态，失败返回 None。"""
    if not os.path.isfile(state_file):
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_circle_state(state_file: str, cx: float, cy: float, r: float) -> bool:
    """将圆形选区状态写入本地 JSON 文件，返回是否成功。"""
    try:
        data = {'cx': cx, 'cy': cy, 'r': r, 'ts': time.time()}
        with open(state_file, 'w') as f:
            json.dump(data, f)
        print(f"💾 圆形选区已保存: cx={cx:.4f}, cy={cy:.4f}, r={r:.4f}")
        return True
    except Exception as e:
        print(f"[警告] 保存圆形选区失败: {e}")
        return False
