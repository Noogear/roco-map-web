# 🗺️ 游戏地图跟点助手（Web）

基于 SIFT 的游戏小地图识别与大地图实时定位工具。

## 项目介绍

本项目用于在网页端实时显示玩家在大地图中的位置，核心流程如下：

1. 捕获并识别游戏小地图（SIFT）
2. 将识别结果映射到完整大地图坐标
3. 通过 Flask + Socket.IO 实时推送给前端展示

主要特点：

- 后端实时识别与状态推送
- 前端多页面（识别台 / 地图页 / 设置页）
- 支持启动前前端自动构建与压缩传输

技术栈：

- 后端：Flask + Socket.IO
- 前端：原生 JS（启动前自动预构建）
- 默认入口：`python run_web.py`

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
npm install
python run_web.py
```

服务启动后访问：`http://127.0.0.1:8686`

## 启动参数总览

> 以下参数统一使用环境变量配置。

| 参数 | 默认值 | 作用 | 典型场景 |
| :-- | :-- | :-- | :-- |
| `SOCKETIO_ASYNC_MODE` | `threading` | Socket.IO 并发模型（`threading` / `gevent`） | 需要更高并发时设为 `gevent` |
| `SOCKETIO_ALLOW_UPGRADES` | `gevent/eventlet` 下默认 `true`，否则 `false` | 是否允许 polling 升级到 websocket | websocket 握手排障 |
| `FRONTEND_PREBUILD` | `auto` | 启动前前端预构建策略：`auto` / `off` / `strict` | 线上建议 `auto`，CI 可用 `strict` |
| `FRONTEND_BUILD_DIR` | `frontend_build/` | 前端构建产物目录 | 自定义部署目录 |
| `FRONTEND_BUILD_TIMEOUT_SEC` | `120` | 前端预构建超时秒数 | 构建机器较慢时调大 |
| `COMPRESS_LEVEL` | `5` | HTTP 压缩级别（br/gzip） | 带宽优先时调高 |
| `COMPRESS_MIN_SIZE` | `700` | 最小压缩阈值（字节） | 小包较多时可调低 |
| `CV2_NUM_THREADS` | `0`（OpenCV 自动） | OpenCV 线程数 | 多进程部署时限制线程争用 |

## 启动方式

### 1) 默认启动（推荐）

```bash
python run_web.py
```

### 2) 关闭前端预构建（排障）

```bash
set FRONTEND_PREBUILD=off
python run_web.py
```

### 3) 严格模式（预构建失败即退出）

```bash
set FRONTEND_PREBUILD=strict
python run_web.py
```

### 4) 高并发示例（Windows 本地）

```bash
set SOCKETIO_ASYNC_MODE=gevent
set COMPRESS_LEVEL=5
set COMPRESS_MIN_SIZE=700
python run_web.py
```

### 5) gunicorn（Linux）

> `gunicorn --preload "backend.server:app"` 不会经过 `run_web.py`，请先手动预构建一次。

```bash
python -m backend.frontend_build
SOCKETIO_ASYNC_MODE=gevent gunicorn -w 4 --threads 2 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker --preload "backend.server:app"
```

## 后端命令

```bash
# 启动 Web 后端（默认入口）
python run_web.py

# 启动 SIFT 模式入口
python run_sift.py

# 仅执行前端预构建（不启动服务）
python -m backend.frontend_build

# 安装后可用脚本入口
map-tracker-build-frontend
map-tracker-web
map-tracker-sift
```

### 后端运行时调参命令（服务端控制台）

启动 `python run_web.py` 后，可在**服务端终端**直接输入以下命令进行热更新：

```text
help
show
set KEY=VALUE [KEY2=VALUE2 ...]
```

示例：

```text
set SEARCH_RADIUS=500 LK_ENABLED=false
set RENDER_EMA_ALPHA=0.4 COMPRESS_MIN_SIZE=900
```

说明：

- `help`：显示命令格式
- `show`：显示当前后端配置快照
- `set ...`：按 `KEY=VALUE` 修改可热更新项
- 多个参数可同一行一起设置
- 无效参数或只读参数会被拒绝（终端会显示 rejected 信息）

## 说明

- 默认端口：`8686`
- 若端口已占用，启动器会跳过重复启动
- 若前端未安装依赖（`node_modules` 缺失），预构建会失败并按策略回退