"""
run_web.py - Web 版启动器（薄入口）

用法:
  python run_web.py

gunicorn 生产部署:
  MAP_TRACKER_MODE=sift SOCKETIO_ASYNC_MODE=gevent \
  gunicorn -w 4 --threads 2 \
    -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
    --preload "backend.server:app"
"""

import os
import socket
import sys
from pathlib import Path

from backend import config


def _load_dotenv_defaults(dotenv_path: Path) -> None:
  """加载 .env 中的键值到进程环境（仅填充当前未设置项）。"""
  if not dotenv_path.exists() or not dotenv_path.is_file():
    return

  try:
    for raw in dotenv_path.read_text(encoding='utf-8').splitlines():
      line = raw.strip()
      if not line or line.startswith('#') or '=' not in line:
        continue
      key, value = line.split('=', 1)
      key = key.strip()
      value = value.strip().strip('"').strip("'")
      if key and key not in os.environ:
        os.environ[key] = value
  except Exception as exc:
    print(f"[startup] 读取 .env 失败，已忽略: {exc}")


def _is_port_in_use(port: int, host: str = '127.0.0.1', timeout: float = 0.5) -> bool:
    """检测端口是否已被监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def main() -> int:
    """Web 启动入口：先做端口占用检测，避免重复启动。"""
  project_root = Path(__file__).resolve().parent
  _load_dotenv_defaults(project_root / '.env')

  print(
    "[startup] SOCKETIO_ASYNC_MODE="
    + os.environ.get('SOCKETIO_ASYNC_MODE', '<unset>')
    + ", SOCKETIO_ALLOW_UPGRADES="
    + os.environ.get('SOCKETIO_ALLOW_UPGRADES', '<unset>')
  )

    port = int(config.PORT)
    if _is_port_in_use(port):
        print(f"[startup] 端口 {port} 已被占用，疑似已有实例运行，跳过重复启动。")
        return 0

    # 仅在确认可启动后再导入，避免端口占用时重复初始化 SIFT 资源。
    from backend.server import main as server_main

    server_main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
