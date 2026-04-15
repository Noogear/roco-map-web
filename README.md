# 🗺️ 游戏大地图实时跟点助手 (Game Map Real-time Tracker)

点击云原生开发一键启动！

基于 **计算机视觉 (SIFT)** 的游戏实时悬浮窗地图追踪器。
自动识别屏幕上的游戏小地图，并在超清完整大地图上实时标注玩家的准确位置，支持双层图分离与惯性导航。

## ✨ 核心特性
* 🚀 **传统视觉引擎**：提供极致性能的 SIFT 匹配方案。
* 👁️ **逻辑/显示双图分离**：使用纯净图进行极限特征匹配，在悬浮窗中显示带标记的资源图。
* 🧭 **惯性导航与雷达扫描**：短暂丢失视野时维持原位推测，彻底跟丢后自动启动全局雷达切片扫描。
* 🌊 **弱纹理强化**：集成 CLAHE 算法，强行榨取大海、纯色草地等弱纹理区域的特征点。

## ⚙️ 模式对比与选择

| 模式 | 运行入口 | 原理 | 优点                        | 缺点                         | 适用场景 |
| :--- | :--- | :--- |:--------------------------|:---------------------------| :--- |
| **SIFT 极速版** | `main_sift.py` | 尺度不变特征变换 | CPU 即可流畅运行。               | 面对大面积纯色水域或 UI 遮挡严重时可能短暂跟丢。 | 绝大多数常规游戏跑图 |

## 🛠️ 安装环境

本项目推荐使用 Python 3.9+。

1. 克隆项目到本地并安装依赖：
```bash
git clone https://github.com/你的用户名/你的项目名.git
cd 你的项目名
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 🚀 网络吞吐优化（本版本新增）

已引入以下依赖与架构改造：

- `orjson`：替换默认 JSON 热路径，提升 JSON 编解码吞吐
- `flask-compress`：自动启用 `br/gzip`，降低 HTTP 带宽占用
- `gevent` + `gevent-websocket`：高并发 SocketIO worker

### 推荐启动方式

开发（单进程）：
```bash
python run_web.py
```

高并发（推荐 Linux/容器）：
```bash
set SOCKETIO_ASYNC_MODE=gevent
set COMPRESS_LEVEL=5
set COMPRESS_MIN_SIZE=700
python run_web.py
```

多进程（gunicorn，Linux）：
```bash
SOCKETIO_ASYNC_MODE=gevent gunicorn -w 4 --threads 2 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker --preload "backend.server:app"
```

### 吞吐调优开关

- `SOCKETIO_ASYNC_MODE`：`threading` / `gevent`
- `COMPRESS_LEVEL`：压缩等级（默认 5）
- `COMPRESS_MIN_SIZE`：最小压缩阈值（默认 700B）
- `CV2_NUM_THREADS`：OpenCV 线程数（多进程时建议限制）