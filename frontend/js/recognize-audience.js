/* recognize-audience.js — 观众模式（监控室）
 *
 * 职责：
 *  - 维护频道列表（通过 broadcast_list 事件）
 *  - 管理每个订阅频道的画面 tile（canvas + ImageBitmap/createImageBitmap 解码）
 *  - 节流：每 tile 独立维护"正在解码"标志，跳帧不积压
 *  - 多观众互不干扰：每个观众的 canvas 独立，不共享任何全局画布
 *  - 页面切进观众模式时才连接 WS 并拉取频道列表
 */
import * as AppCommon from './common.js';
import TC from './tracker-core.js';

const RecognizeAudience = (function () {
    'use strict';

    /* ── 状态 ── */
    var _ready = false;       // WS 已连接
    var _entered = false;     // 已经 onEnter 过一次

    /* 订阅集合：name → { canvas, ctx, lastRender, decoding, fps, fpsTs, fpsCount } */
    var _tiles = {};

    /* 频道列表数据（最后一次收到的） */
    var _channels = [];

    /* 刷新频道列表的节流 */
    var _listThrottleAt = 0;
    var LIST_THROTTLE_MS = 500;

    /* ── 连接 WS 并绑定事件（幂等） ── */
    function _ensureWS() {
        if (_ready) return Promise.resolve();
        return TC.connectWS().then(function () {
            _ready = true;
            _bindWS();
        });
    }

    function _bindWS() {
        var sock = TC.wsSocket;
        if (!sock) return;

        /* 服务端广播列表更新（展示者上下线、观众数变化） */
        sock.on('broadcast_list', function (data) {
            _channels = (data && Array.isArray(data.rooms)) ? data.rooms : [];
            _scheduleRenderList();
        });

        /* 展示者离线：移除对应 tile */
        sock.on('broadcast_ended', function (data) {
            if (data && data.name) _removeTile(data.name);
            _scheduleList();
        });

        /* 收到帧：仅更新正在订阅的 tile */
        sock.on('broadcast_frame', function (raw) {
            /* raw 可能是 ArrayBuffer / TypedArray / Blob */
            var blob;
            if (raw instanceof Blob) {
                blob = raw;
            } else if (raw instanceof ArrayBuffer) {
                blob = new Blob([raw], { type: 'image/jpeg' });
            } else if (ArrayBuffer.isView(raw)) {
                blob = new Blob([raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)], { type: 'image/jpeg' });
            } else {
                return;
            }
            /* broadcast_frame 到达时服务端已按 name 路由；
               但客户端同一 socket 可能订阅多个频道，
               服务端目前按 viewer sid 定向 emit，
               所以这里把帧分发给所有活跃 tile（同一观众订阅多频道时各 emit 独立） */
            /* 实际上服务端直接 to(vsid) emit，
               所以这里只要更新 "最近一次 emit 对应的 tile" —
               因为服务端不携带 sender name，前端需要依赖订阅关系：
               若只订阅一个频道，直接更新；若多频道，逐个尝试 */
            Object.keys(_tiles).forEach(function (name) {
                _renderTile(name, blob);
            });
        });

        sock.on('broadcast_watch_ack', function (ack) {
            if (!ack.ok) { AppCommon.toast('订阅失败：' + ack.error, 'danger'); }
        });

        sock.on('disconnect', function () { _ready = false; });
    }

    /* ── 调度拉取列表 ── */
    function _scheduleList() {
        var now = Date.now();
        if (now - _listThrottleAt < LIST_THROTTLE_MS) return;
        _listThrottleAt = now;
        if (TC.wsConnected && TC.wsSocket) {
            TC.wsSocket.emit('broadcast_list');
        }
    }

    var _listRenderPending = false;
    function _scheduleRenderList() {
        if (_listRenderPending) return;
        _listRenderPending = true;
        requestAnimationFrame(function () {
            _listRenderPending = false;
            _renderChannelList();
        });
    }

    /* ── 频道列表 DOM ── */
    function _renderChannelList() {
        var el = document.getElementById('channelListEl');
        if (!el) return;

        if (!_channels.length) {
            el.innerHTML = '<div class="rec-channel-empty">暂无展示者在线</div>';
            return;
        }

        el.innerHTML = '';
        _channels.forEach(function (ch) {
            var isWatching = !!_tiles[ch.name];
            var item = document.createElement('div');
            item.className = 'rec-channel-item' + (isWatching ? ' is-watching' : '');
            item.innerHTML =
                '<span class="rec-ch-name">' + _esc(ch.name) + '</span>' +
                '<span class="rec-ch-viewers">👁️ ' + ch.viewers + '</span>' +
                '<span class="rec-ch-action">' + (isWatching ? '取消' : '观看') + '</span>';
            item.addEventListener('click', function () {
                if (isWatching) { _unwatchChannel(ch.name); }
                else { _watchChannel(ch.name); }
            });
            el.appendChild(item);
        });
    }

    /* ── 订阅 / 取消订阅 ── */
    function _watchChannel(name) {
        if (_tiles[name]) return;
        _addTile(name);
        TC.wsSocket.emit('broadcast_watch', { name: name });
        _scheduleRenderList();
    }

    function _unwatchChannel(name) {
        _removeTile(name);
        if (TC.wsConnected && TC.wsSocket) {
            TC.wsSocket.emit('broadcast_unwatch', { name: name });
        }
        _scheduleRenderList();
    }

    /* ── Tile 管理 ── */
    function _addTile(name) {
        if (_tiles[name]) return;

        var grid = document.getElementById('audienceGrid');
        if (!grid) return;

        /* 移除 placeholder */
        var ph = grid.querySelector('.rec-audience-placeholder');
        if (ph) ph.remove();

        var tile = document.createElement('div');
        tile.className = 'rec-viewer-tile';
        tile.dataset.name = name;

        var header = document.createElement('div');
        header.className = 'rec-viewer-header';
        header.innerHTML =
            '<span>' + _esc(name) + '</span>' +
            '<span class="rec-viewer-fps" id="fps_' + _esc(name) + '">-- fps</span>' +
            '<button class="rec-viewer-close" title="关闭">✕</button>';
        header.querySelector('.rec-viewer-close').addEventListener('click', function () {
            _unwatchChannel(name);
        });

        var cvs = document.createElement('canvas');
        cvs.className = 'rec-viewer-canvas';
        cvs.width = 400; cvs.height = 400;

        tile.appendChild(header);
        tile.appendChild(cvs);
        grid.appendChild(tile);

        _tiles[name] = {
            tile: tile,
            canvas: cvs,
            ctx: cvs.getContext('2d'),
            decoding: false,
            lastRender: 0,
            fps: 0, fpsTs: Date.now(), fpsCount: 0,
        };
    }

    function _removeTile(name) {
        var t = _tiles[name];
        if (!t) return;
        t.tile.remove();
        delete _tiles[name];

        /* 若全部取消，恢复 placeholder */
        if (!Object.keys(_tiles).length) {
            var grid = document.getElementById('audienceGrid');
            if (grid) {
                grid.innerHTML =
                    '<div class="rec-audience-placeholder">' +
                    '<div style="font-size:48px;">📺</div>' +
                    '<div>从左侧选择频道开始观看</div>' +
                    '</div>';
            }
        }
    }

    /* ── 渲染帧到 tile ──
       使用 createImageBitmap（硬件解码，比 Image 快，不阻塞主线程）
       节流：若上一帧还在解码则跳过，防止积压 */
    function _renderTile(name, blob) {
        var t = _tiles[name];
        if (!t || t.decoding) return;

        t.decoding = true;
        /* createImageBitmap 是异步的，解码在 GPU/worker 线程 */
        (typeof createImageBitmap === 'function'
            ? createImageBitmap(blob)
            : _fallbackImageDecode(blob)
        ).then(function (bmp) {
            t.decoding = false;
            if (!_tiles[name]) return; // tile 已被关闭
            var ctx = t.ctx;
            if (!ctx) return;
            ctx.drawImage(bmp, 0, 0, t.canvas.width, t.canvas.height);
            if (typeof bmp.close === 'function') bmp.close(); // 释放 GPU 资源

            /* FPS 统计 */
            t.fpsCount++;
            var now = Date.now();
            if (now - t.fpsTs >= 1000) {
                t.fps = t.fpsCount;
                t.fpsCount = 0;
                t.fpsTs = now;
                var fpsEl = document.getElementById('fps_' + name);
                if (fpsEl) fpsEl.textContent = t.fps + ' fps';
            }
        }).catch(function () {
            if (_tiles[name]) _tiles[name].decoding = false;
        });
    }

    /* createImageBitmap 不可用时回退到 Image */
    function _fallbackImageDecode(blob) {
        return new Promise(function (resolve, reject) {
            var url = URL.createObjectURL(blob);
            var img = new Image();
            img.onload = function () { URL.revokeObjectURL(url); resolve(img); };
            img.onerror = function () { URL.revokeObjectURL(url); reject(new Error('decode fail')); };
            img.src = url;
        });
    }

    /* ── HTML 转义（防 XSS：展示名来自网络） ── */
    function _esc(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    /* ── 公开 API ── */
    return {
        /** 切换到观众模式时调用（幂等） */
        onEnter: function () {
            if (_entered) {
                /* 已经进入过，只刷新列表 */
                _scheduleList();
                return;
            }
            _entered = true;
            _ensureWS().then(function () {
                _scheduleList();
            }).catch(function (e) {
                AppCommon.toast('WS 连接失败：' + e.message, 'danger');
            });
        },

        /** 手动刷新频道列表 */
        refresh: function () { _scheduleList(); },
    };
})();

export default RecognizeAudience;
