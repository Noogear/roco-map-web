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

from backend.server import main

if __name__ == '__main__':
    main()
