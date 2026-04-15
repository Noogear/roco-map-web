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
    var _active = false;      // 当前是否处于观众页

    /* 订阅集合：name → { canvas, ctx, lastRender, decoding, fps, fpsTs, fpsCount } */
    var _tiles = {};

    /* 频道列表数据（最后一次收到的） */
    var _channels = [];

    /* 刷新频道列表的节流 */
    var _listThrottleAt = 0;
    var LIST_THROTTLE_MS = 500;

    /* ── 连接 WS 并绑定事件（幂等） ── */
    function _ensureWS() {
        if (TC.wsConnected && TC.wsSocket) {
            _ready = true;
            _bindWS();
            return Promise.resolve();
        }
        return TC.connectWS().then(function () {
            _ready = true;
            _bindWS();
        });
    }

    function _bindWS() {
        var sock = TC.wsSocket;
        if (!sock) return;
        if (sock.__recognizeAudienceBound) return;
        sock.__recognizeAudienceBound = true;

        sock.on('connect', function () {
            _ready = true;
            if (_active) {
                _rewatchAllChannels();
                _scheduleList();
            }
        });

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

        /* 收到帧：服务端显式携带频道名，精确更新对应 tile，避免多频道串台 */
        sock.on('broadcast_frame', function (packet) {
            if (!packet || !packet.name) return;
            var blob = _normalizeFrameBlob(packet.frame);
            if (!blob) return;
            _renderTile(packet.name, blob);
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

    function _rewatchAllChannels() {
        if (!TC.wsConnected || !TC.wsSocket) return;
        Object.keys(_tiles).forEach(function (name) {
            TC.wsSocket.emit('broadcast_watch', { name: name });
        });
    }

    function _normalizeFrameBlob(raw) {
        if (!raw) return null;
        if (raw instanceof Blob) {
            return raw;
        }
        if (raw instanceof ArrayBuffer) {
            return new Blob([raw], { type: 'image/jpeg' });
        }
        if (ArrayBuffer.isView(raw)) {
            return new Blob([raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)], { type: 'image/jpeg' });
        }
        return null;
    }

    function _clearAllTiles(unwatchRemote) {
        var names = Object.keys(_tiles);
        names.forEach(function (name) {
            if (unwatchRemote && TC.wsConnected && TC.wsSocket) {
                TC.wsSocket.emit('broadcast_unwatch', { name: name });
            }
            _removeTile(name);
        });
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
            _active = true;
            _ensureWS().then(function () {
                _scheduleList();
                _rewatchAllChannels();
            }).catch(function (e) {
                AppCommon.toast('WS 连接失败：' + e.message, 'danger');
            });
        },

        /** 离开观众模式时调用：取消订阅并释放 tile DOM/解码负担 */
        onLeave: function () {
            _active = false;
            _clearAllTiles(true);
        },

        /** 手动刷新频道列表 */
        refresh: function () {
            _ensureWS().then(function () {
                _scheduleList();
            }).catch(function (e) {
                AppCommon.toast('WS 连接失败：' + e.message, 'danger');
            });
        },
    };
})();

export default RecognizeAudience;
