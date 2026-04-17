/**
 * tracker-core.js - 地图跟点共用识别核心
 * 供当前识别页 / 地图页 / 配置页共享。
 *
 * 提供:
 *   TC.log(msg)                    日志
 *   TC.init(opts)                   初始化配置
 *   TC.startScreenCapture(o)        启动屏幕捕获
 *   TC.stopScreenCapture()          停止屏幕捕获
 *   TC.captureScreenImg()           圆形区域截图 -> dataURL (JPEG, 文件/HTTP模式)
 *   TC.captureScreenImgBlob()       圆形区域截图 -> Promise<Blob> (PNG, WS二进制模式)
 *   TC.bindFileInput(input, canvas) 绑定文件选择器
 *   TC.loadFile(file, canvas)      手动加载文件 -> Promise<dataURL>
 *   TC.sendAndDisplay(dataURL)      发送到后端 HTTP API 分析
 *   TC.sendViaWS(blob)             通过 Socket.IO 发送二进制帧分析
 *   TC.connectWS()                  建立 Socket.IO 连接
 *   TC.disconnectWS()               断开 Socket.IO
 *   TC.formatStatus(status)         格式化状态文本
 *   TC.updateStatusDOM(status, ids) 更新标准状态栏 DOM
 *   TC.checkEngines()               检测可用引擎
 *   TC.loadTestImages(container)    加载测试图片列表
 *   TC.clamp(v, lo, hi)             数值钳位
 *
 * 状态属性:
 *   TC.mode         'file' | 'screen'
 *   TC.imageData     当前图片 dataURL
 *   TC.isScreenActive  是否正在捕获
 *   TC.selCircle     {cx, cy, r} 归一化圆形选区
 *   TC.wsConnected    Socket.IO 是否已连接
 */
const TrackerCore = (() => {
    'use strict';

    // ==================== 内部状态 ====================
    const S = {
        mode: 'file',
        imageData: null,
        screenStream: null,
        offscreenVid: null,
        videoW: 0,
        videoH: 0,
        selCircle: (function() {
            // 优先读取用户持久化的校准值，否则使用默认值
            var DEF = { cx: (1189 + 62.5) / 1362, cy: (66 + 63.5) / 806, r: Math.max(62.5 / 1362, 63.5 / 806) };
            try {
                var raw = localStorage.getItem('tc_selCircle');
                if (raw) {
                    var saved = JSON.parse(raw);
                    if (saved && typeof saved.cx === 'number' && typeof saved.cy === 'number' && typeof saved.r === 'number') {
                        return saved;
                    }
                }
            } catch (_e) {}
            return DEF;
        })(),
        wsSocket: null,
        wsConnected: false,
        wsTransportName: '',
        wsConnecting: null,    // 连接中的 Promise（防止重复创建 socket）
        wsManualClose: false,  // 标记是否为手动断开（抑制重复日志）
        captureCanvas: null,  // 复用截图 canvas，避免每帧创建导致 GPU 内存泄漏
        lastWasUpdate: false, // logUpdate 状态标记
        nullBlobCount: 0,     // 连续 null blob 计数，用于检测视频流问题
        sessionToken: '',     // 多会话 token
    };

    // ==================== 内部工具 ====================
    /**
     * 构建状态行文本（给 logUpdate 使用）
     * @param {object} st  result.status 对象
     * @param {string|null} kbStr  流量字符串，如 '12.3'，HTTP 模式传 null
     */
    var _fmtResult = function(st, kbStr) {
        var state = st.state || '';
        var icon, tag;
        if (state === 'SCENE_CHANGE') {
            icon = '🔄'; tag = '切场';
        } else if (state === 'GLOBAL_SCAN' && !st.found) {
            icon = '🔍'; tag = '全扫';
        } else if (!st.found) {
            icon = '❌'; tag = '丢失';
        } else if (state === 'INERTIAL') {
            icon = '⚠️'; tag = '惯性';
        } else if (state === 'GLOBAL_SCAN') {
            icon = '🔍'; tag = '全扫';
        } else {
            icon = '✅'; tag = '';
        }
        // 坐标：切场/找到时显示，否则 --
        var pos = (st.found || state === 'SCENE_CHANGE')
            ? '(' + st.position.x + ',' + st.position.y + ')'
            : '--';
        // 品质：有意义时才显示
        var q = (st.match_quality != null && st.match_quality > 0)
            ? ' ' + Math.round(st.match_quality * 100) + '%' : '';
        // 附加标志
        var extras = [];
        if (st.coord_lock)  extras.push('🔒');
        // 拼装
        var line = icon + (tag ? ' [' + tag + ']' : '') + ' ' + pos
                 + ' | ' + st.matches + '匹' + q;
        if (kbStr)        line += ' | ' + kbStr + 'KB';
        if (extras.length) line += ' | ' + extras.join(' ');
        return line;
    };

    var _forEachNode = function(nodes, handler) {
        Array.prototype.forEach.call(nodes || [], handler);
    };

    var _createCustomEvent = function(name, detail) {
        if (typeof window.CustomEvent === 'function') {
            return new CustomEvent(name, { detail: detail });
        }
        var evt = document.createEvent('CustomEvent');
        evt.initCustomEvent(name, false, false, detail);
        return evt;
    };

    var _decodeUtf8 = function(bytes) {
        if (typeof TextDecoder !== 'undefined') {
            try { return new TextDecoder('utf-8').decode(bytes); } catch (e) {}
        }

        var out = '';
        var i = 0;
        while (i < bytes.length) {
            var c = bytes[i++];
            if (c < 128) {
                out += String.fromCharCode(c);
            } else if (c > 191 && c < 224 && i < bytes.length) {
                var c2 = bytes[i++];
                out += String.fromCharCode(((c & 31) << 6) | (c2 & 63));
            } else if (i + 1 < bytes.length) {
                var c3 = bytes[i++];
                var c4 = bytes[i++];
                out += String.fromCharCode(((c & 15) << 12) | ((c3 & 63) << 6) | (c4 & 63));
            }
        }
        return out;
    };

    var _dataURLToBlob = function(dataURL) {
        var parts = (dataURL || '').split(',');
        if (parts.length < 2) return null;

        var header = parts[0];
        var b64 = parts[1];
        var mimeMatch = header.match(/data:([^;]+)/);
        var mime = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
        var bin = atob(b64);
        var len = bin.length;
        var bytes = new Uint8Array(len);
        for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
        return new Blob([bytes], { type: mime });
    };

    var _getDisplayMediaImpl = function() {
        if (navigator.mediaDevices && typeof navigator.mediaDevices.getDisplayMedia === 'function') {
            return function(options) { return navigator.mediaDevices.getDisplayMedia(options); };
        }
        if (typeof navigator.getDisplayMedia === 'function') {
            return function(options) { return navigator.getDisplayMedia(options); };
        }
        return null;
    };

    var _getSocketTransportName = function(sock) {
        try {
            return (sock && sock.io && sock.io.engine && sock.io.engine.transport && sock.io.engine.transport.name) || 'socket.io';
        } catch (e) {
            return 'socket.io';
        }
    };

    var _getCaptureRect = function(vw, vh, sc) {
        var bs = Math.min(vw, vh);
        var cx = sc.cx * vw, cy = sc.cy * vh, r = sc.r * bs;
        var margin = 1.4;
        var sz = Math.max(10, Math.round(r * 2 * margin));
        return {
            sz: sz,
            rx: Math.round(cx - sz / 2),
            ry: Math.round(cy - sz / 2),
        };
    };

    var _drawCaptureToCanvas = function(vid, rect) {
        if (!S.captureCanvas) S.captureCanvas = document.createElement('canvas');
        var c = S.captureCanvas;
        if (c.width !== rect.sz || c.height !== rect.sz) { c.width = rect.sz; c.height = rect.sz; }

        var ctx = c.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
            S.captureCanvas = document.createElement('canvas');
            c = S.captureCanvas;
            c.width = rect.sz; c.height = rect.sz;
            ctx = c.getContext('2d', { willReadFrequently: true });
            if (!ctx) return null;
        }
        ctx.drawImage(vid, rect.rx, rect.ry, rect.sz, rect.sz, 0, 0, rect.sz, rect.sz);
        return c;
    };

    var _bindSocketTransportEvents = function(sock, self) {
        if (!sock || sock.__tcTransportEventsBound) return;
        if (!sock.io || !sock.io.engine || typeof sock.io.engine.on !== 'function') return;

        sock.__tcTransportEventsBound = true;
        sock.io.engine.on('upgrade', function(transport) {
            var name = transport && transport.name ? transport.name : 'websocket';
            S.wsTransportName = name;
            self.log('🚀 Socket.IO 已升级到 ' + name);
        });
    };

    let opts = {
        logEl: null,
        canvasSize: 300,
        onImageLoaded: null,       // function(dataURL, w, h)
        onAnalyzeResult: null,     // function(result)  API 返回结果回调
        onScreenStart: null,       // function()
        onScreenStop: null,        // function()
    };

    // ==================== 公开 API ====================
    return {

        /** 初始化配置 */
        init(options = {}) { Object.assign(opts, options); },

        get mode() { return S.mode; },
        set mode(v) { S.mode = v; },
        get imageData() { return S.imageData; },
        set imageData(v) { S.imageData = v; },
        get selCircle() { return S.selCircle; },
        set selCircle(v) { Object.assign(S.selCircle, v); },
        /** 持久化 selCircle 到 localStorage，保证刷新后仍有效 */
        saveSelCircle() {
            try { localStorage.setItem('tc_selCircle', JSON.stringify({ cx: S.selCircle.cx, cy: S.selCircle.cy, r: S.selCircle.r })); } catch (_e) {}
        },
        /** 重置 selCircle 为默认值并清除持久化 */
        resetSelCircle() {
            S.selCircle.cx = (1189 + 62.5) / 1362;
            S.selCircle.cy = (66 + 63.5) / 806;
            S.selCircle.r  = Math.max(62.5 / 1362, 63.5 / 806);
            S.captureCanvas = null; /* 旧尺寸 canvas 作废 */
            try { localStorage.removeItem('tc_selCircle'); } catch (_e) {}
        },
        get isScreenActive() { return !!S.screenStream; },
        get wsConnected() { return S.wsConnected; },
        get wsSocket() { return S.wsSocket; },
        get wsTransportName() { return S.wsTransportName; },
        get sessionToken() { return S.sessionToken; },
        /** 外部清理复用 canvas（selCircle 改变后旧尺寸失效） */
        set captureCanvas(v) { S.captureCanvas = v; },

        // ========== 日志 ==========
        log(msg) {
            S.lastWasUpdate = false;
            let el = opts.logEl;
            if (typeof el === 'string') el = document.getElementById(el);
            if (!el) return;
            const time = new Date().toLocaleTimeString();
            el.textContent += `[${time}] ${msg}\n`;
            var MAX = parseInt(el.dataset.maxLen || '3000', 10);
            var TRIM = parseInt(el.dataset.trimLen || '2000', 10);
            if (el.textContent.length > MAX) el.textContent = el.textContent.slice(-TRIM);
            el.scrollTop = el.scrollHeight;
        },

        /** 覆盖式日志：高频状态行原地刷新，不堆积 */
        logUpdate(msg) {
            let el = opts.logEl;
            if (typeof el === 'string') el = document.getElementById(el);
            if (!el) return;
            const time = new Date().toLocaleTimeString();
            const line = `[${time}] ${msg}\n`;
            if (S.lastWasUpdate) {
                // 替换最后一行（找倒数第二个换行符）
                var text = el.textContent;
                var end = text.length - 1; // 跳过末尾 \n
                var prev = text.lastIndexOf('\n', end - 1);
                el.textContent = (prev >= 0 ? text.slice(0, prev + 1) : '') + line;
            } else {
                S.lastWasUpdate = true;
                el.textContent += line;
                var MAX = parseInt(el.dataset.maxLen || '3000', 10);
                var TRIM = parseInt(el.dataset.trimLen || '2000', 10);
                if (el.textContent.length > MAX) el.textContent = el.textContent.slice(-TRIM);
            }
            el.scrollTop = el.scrollHeight;
        },

        // ========== 工具 ==========
        clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); },

        getConnectionDiagnostics() {
            var canvas = document.createElement('canvas');
            return {
                httpUpload: typeof fetch === 'function' && typeof FormData !== 'undefined' && typeof JSON !== 'undefined',
                socketIO: typeof io === 'function',
                websocket: typeof WebSocket !== 'undefined',
                binaryFrames: typeof Blob !== 'undefined' && typeof ArrayBuffer !== 'undefined',
                screenCapture: !!_getDisplayMediaImpl(),
                customEvent: typeof window.CustomEvent === 'function',
                textDecoder: typeof TextDecoder !== 'undefined',
                canvasStream: !!canvas.captureStream,
            };
        },

        // ========== 状态格式化 ==========
        formatStatus(status) {
            var text = '--', cls = 'green', label = '';
            var found = !!status.found;
            if (status.mode === 'orb') {
                if (status.state === 'SCENE_CHANGE') {
                    text = '切场'; cls = 'yellow'; label = '场景切换';
                } else if (!found) {
                    text = '丢失'; cls = 'red'; label = '未找到';
                } else if (status.state === 'INERTIAL') {
                    text = '惯性'; cls = 'yellow'; label = '惯性导航';
                } else {
                    text = '正常'; cls = 'green'; label = 'ORB追踪';
                }
            } else {
                if (status.state === 'GLOBAL_SCAN') {
                    text = '扫描'; cls = 'red'; label = '全局扫描';
                } else if (found) {
                    text = '正常'; cls = 'green'; label = '局部追踪';
                } else {
                    text = '丢失'; cls = 'red'; label = '目标丢失';
                }
            }
            return { stateText: text, stateClass: cls, stateLabel: label };
        },

        updateStatusDOM(status, ids) {
            ids = ids || {};
            var fe = document.getElementById(ids.found || 'statusFound');
            var me = document.getElementById(ids.mode || 'statusMode');
            var xe = document.getElementById(ids.x || 'statusX');
            var ye = document.getElementById(ids.y || 'statusY');
            var mche = document.getElementById(ids.matches || 'statusMatches');

            var f = this.formatStatus(status);

            // 状态指示灯
            var dotEl = document.getElementById('statusDot');
            var labelEl = document.getElementById('statusDotLabel');
            if (dotEl) {
                dotEl.className = 'status-dot dot-' + f.stateClass;
                dotEl.title = f.stateText;
            }
            if (labelEl) labelEl.textContent = f.stateLabel;

            if (me && status.mode) {
                me.textContent = status.mode.toUpperCase();
                me.style.color = status.mode === 'orb' ? '#ffa726' : '#00d4ff';
            }
            if (xe && status.position) xe.textContent = status.position.x;
            if (ye && status.position) ye.textContent = status.position.y;
            if (fe) {
                var isFound = !!status.found;
                fe.textContent = isFound ? '\u2705 \u662f' : '\u274c \u5426';
                fe.className = 'status-value ' + (isFound ? 'found-yes' : 'found-no');
            }
            if (mche) mche.textContent = status.matches;

            // 匹配质量显示
            var qe = document.getElementById(ids.quality || 'statusQuality');
            if (qe) {
                var q = status.match_quality || 0;
                qe.textContent = (q * 100).toFixed(0) + '%';
                qe.style.color = q >= 0.7 ? '#4caf50' : q >= 0.4 ? '#ffa726' : '#ff5252';
            }
        },


        // ========== 文件加载 ==========
        bindFileInput(fileInput, previewCanvas) {
            var self = this;
            var cs = opts.canvasSize;
            fileInput.addEventListener('change', function(e) {
                var f = e.target.files[0]; if (!f) return;
                var rd = new FileReader();
                rd.onload = function(ev) {
                    var img = new Image();
                    img.onload = function() {
                        var ctx = previewCanvas.getContext('2d');
                        ctx.clearRect(0, 0, cs, cs);
                        var s = Math.min(cs / img.width, cs / img.height);
                        var w = img.width * s, h = img.height * s;
                        ctx.drawImage(img, (cs - w) / 2, (cs - h) / 2, w, h);
                        S.imageData = previewCanvas.toDataURL('image/png');
                        self.log('\u5df2\u52a0\u8f7d\u56fe\u7247: ' + img.width + 'x' + img.height);
                        if (opts.onImageLoaded) opts.onImageLoaded(S.imageData, img.width, img.height);
                    };
                    img.src = ev.target.result;
                };
                rd.readAsDataURL(f);
            });
        },

        loadFile(file, previewCanvas) {
            var self = this;
            var cs = opts.canvasSize;
            return new Promise(function(resolve) {
                var rd = new FileReader();
                rd.onload = function(ev) {
                    var img = new Image();
                    img.onload = function() {
                        var ctx = previewCanvas.getContext('2d');
                        ctx.clearRect(0, 0, cs, cs);
                        var s = Math.min(cs / img.width, cs / img.height);
                        var w = img.width * s, h = img.height * s;
                        ctx.drawImage(img, (cs - w) / 2, (cs - h) / 2, w, h);
                        S.imageData = previewCanvas.toDataURL('image/png');
                        self.log('\u5df2\u52a0\u8f7d\u56fe\u7247: ' + img.width + 'x' + img.height);
                        if (opts.onImageLoaded) opts.onImageLoaded(S.imageData, img.width, img.height);
                        resolve(S.imageData);
                    };
                    img.src = ev.target.result;
                };
                rd.readAsDataURL(file);
            });
        },


        // ========== 屏幕捕获核心 ==========
        startScreenCapture(o) {
            o = o || {};
            var self = this;
            var getDisplayMedia = _getDisplayMediaImpl();

            if (!getDisplayMedia) {
                self.log('当前浏览器不支持屏幕捕获，请改用 Chrome / Edge / Firefox 新版');
                return Promise.resolve(false);
            }

            return getDisplayMedia({
                video: { cursor: 'always', displaySurface: 'monitor' },
                audio: false,
            }).then(function(strm) {
                S.screenStream = strm;
                S.mode = 'screen';

                var osvId = o.offscreenVidId || 'screenOffscreen';
                var osv = document.getElementById(osvId);
                if (osv) {
                    S.offscreenVid = osv;
                    osv.srcObject = strm;
                    osv.onloadedmetadata = function() {
                        S.videoW = osv.videoWidth;
                        S.videoH = osv.videoHeight;
                        if (o.onMeta) o.onMeta(S.videoW, S.videoH);
                    };
                }

                if (o.previewVideo) {
                    var pv = typeof o.previewVideo === 'string' ? document.getElementById(o.previewVideo) : o.previewVideo;
                    if (pv) pv.srcObject = strm;
                }

                strm.getVideoTracks()[0].onended = function() { self.stopScreenCapture(); };

                if (opts.onScreenStart) opts.onScreenStart();
                self.log('\ud83d\udcbb \u5c4f\u5e55\u6355\u83b7\u5df2\u5f00\u59cb');
                return true;
            }).catch(function(e) {
                self.log('\u5c4f\u5e55\u6355\u83b7\u5931\u8d25/\u53d6\u6d88: ' + e.message);
                return false;
            });
        },

        stopScreenCapture() {
            if (S.screenStream) {
                S.screenStream.getTracks().forEach(function(t) { t.stop(); });
                S.screenStream = null;
            }
            if (S.offscreenVid) {
                S.offscreenVid.srcObject = null;
                S.offscreenVid = null;
            }
            S.videoW = 0; S.videoH = 0;
            S.mode = 'file';

            if (opts.onScreenStop) opts.onScreenStop();
            this.log('\u23cf \u5c4f\u5e55\u6355\u83b7\u5df2\u505c\u6b62');
        },

        /**
         * 从屏幕流截取方形选区图片 (JPEG dataURL，用于 HTTP 模式)
         * 方形截取 + 后端 HoughCircles 圆检测，自动判断小地图是否存在
         * @returns {string|null} dataURL 或 null
         */
        captureScreenImg() {
            var vid = S.offscreenVid;
            if (!vid || !vid.videoWidth) return null;

            var vw = vid.videoWidth || S.videoW;
            var vh = vid.videoHeight || S.videoH;
            if (!vw || !vh) return null;

            var rect = _getCaptureRect(vw, vh, S.selCircle);
            var c = _drawCaptureToCanvas(vid, rect);
            if (!c) return null;
            return c.toDataURL('image/jpeg', 0.80);
        },

        /**
         * 从屏幕流截取方形选区图片 (JPEG Blob，用于 Socket.IO 二进制模式)
         * 方形截取 + 后端 HoughCircles 圆检测，自动判断小地图是否存在
         * @returns {Promise<Blob|null>}
         */
        captureScreenImgBlob() {
            var vid = S.offscreenVid;
            if (!vid || !vid.videoWidth) return Promise.resolve(null);

            var vw = vid.videoWidth || S.videoW;
            var vh = vid.videoHeight || S.videoH;
            if (!vw || !vh) return Promise.resolve(null);

            var rect = _getCaptureRect(vw, vh, S.selCircle);
            var c = _drawCaptureToCanvas(vid, rect);
            if (!c) return Promise.resolve(null);

            return new Promise(function(resolve) {
                if (typeof c.toBlob === 'function') {
                    c.toBlob(function(blob) { resolve(blob); }, 'image/jpeg', 0.82);
                    return;
                }
                try {
                    resolve(_dataURLToBlob(c.toDataURL('image/jpeg', 0.82)));
                } catch (e) {
                    resolve(null);
                }
            });
        },


        // ========== HTTP API 分析 ==========
        /**
         * 截取屏幕流的完整帧（不裁剪 selCircle），返回 Promise<Blob|null>
         * 用于自动定位小地图等需要全画面的场景。
         */
        captureFullFrameBlob(quality) {
            var vid = S.offscreenVid;
            if (!vid || !vid.videoWidth) return Promise.resolve(null);
            var vw = vid.videoWidth, vh = vid.videoHeight;
            var c = document.createElement('canvas');
            c.width = vw; c.height = vh;
            var ctx = c.getContext('2d');
            if (!ctx) return Promise.resolve(null);
            ctx.drawImage(vid, 0, 0, vw, vh);
            return new Promise(function (resolve) {
                c.toBlob(function (blob) { resolve(blob); }, 'image/jpeg', quality || 0.85);
            });
        },
        sendAndDisplay(imageDataURL) {
            var self = this;
            return fetch('/api/upload_minimap', {
                method: 'POST',
                body: JSON.stringify({ image: imageDataURL }),
                headers: { 'Content-Type': 'application/json' }
            }).then(function(resp) {
                if (resp.ok) return resp.json();
                // fallback: FormData（兼容部分反向代理对 JSON body 的限制）
                return fetch(imageDataURL).then(function(r) { return r.blob(); }).then(function(blob) {
                    var fd = new FormData(); fd.append('image', blob, 'minimap.png');
                    return fetch('/api/upload_minimap', { method: 'POST', body: fd }).then(function(r2) { return r2.json(); });
                });
            }).then(function(result) {
                if (result && result.error) {
                    self.logUpdate('⚠️ ' + result.error);
                } else if (result && result.status) {
                    self.logUpdate(_fmtResult(result.status, null));
                }
                if (opts.onAnalyzeResult) opts.onAnalyzeResult(result);
                return result;
            }).catch(function(e) {
                self.log('❌ HTTP 请求失败: ' + e.message);
                throw e;
            });
        },


        // ========== 多会话 Token ==========
        _ensureSessionToken() {
            if (S.sessionToken) return S.sessionToken;
            var stored = null;
            try { stored = localStorage.getItem('rec_session_token'); } catch(e) {}
            if (stored && stored.length >= 8) {
                S.sessionToken = stored;
            } else {
                // crypto.randomUUID() 或 fallback
                if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
                    S.sessionToken = crypto.randomUUID();
                } else {
                    S.sessionToken = 'xxxx-xxxx-xxxx'.replace(/x/g, function() {
                        return Math.floor(Math.random() * 16).toString(16);
                    });
                }
                try { localStorage.setItem('rec_session_token', S.sessionToken); } catch(e) {}
            }
            return S.sessionToken;
        },

        // ========== Socket.IO 二进制通道 ==========
        /**
         * 建立 Socket.IO 连接（使用 socket.io client 库）
         * @returns {Promise<boolean>}
         */
        connectWS() {
            var self = this;
            if (S.wsSocket && S.wsConnected) return Promise.resolve(true);
            if (typeof io !== 'function') {
                var missingErr = new Error('当前页面未加载 Socket.IO 客户端');
                self.log('❌ Socket.IO 客户端未加载，无法建立实时连接');
                return Promise.reject(missingErr);
            }
            // 连接中则复用同一个 Promise，避免并发调用创建多个 socket
            if (S.wsConnecting) return S.wsConnecting;
            S.wsConnecting = new Promise(function(resolve, reject) {
                var settled = false;
                var fallbackTried = false;

                var buildSockOptions = function(pollingOnly) {
                    return {
                        transports: pollingOnly ? ['polling'] : ['polling', 'websocket'],
                        upgrade: !pollingOnly,
                        rememberUpgrade: false,
                        timeout: 5000,
                        forceNew: true,
                    };
                };

                // 接收后端二进制响应: [4字节大端JSON长度][JSON][JPEG图片]
                var _processResultBuf = function(buf, byteLen) {
                    var view = new DataView(buf);
                    if (view.byteLength < 4) return;
                    var jsonLen = view.getUint32(0, false);
                    if (view.byteLength < 4 + jsonLen) return;
                    var jsonBytes = new Uint8Array(buf, 4, jsonLen);
                    var jsonStr = _decodeUtf8(jsonBytes);
                    var status;
                    try { status = JSON.parse(jsonStr); } catch(e) { return; }

                    if (status.error) { self.log('\u274c 解码失败'); return; }

                    var jpegStart = 4 + jsonLen;
                    var jpegBytes = buf.slice(jpegStart);
                    if (jpegBytes.byteLength > 2) {
                        var b64 = _arrayBufferToBase64(jpegBytes);
                        var result = {
                            success: true,
                            image: b64,
                            status: {
                                mode: status.m,
                                state: status.s,
                                position: { x: status.x, y: status.y },
                                found: !!status.f,
                                matches: status.c,
                                match_quality: status.q || 0,
                                arrow_angle: status.a || 0,
                                arrow_stopped: !!status.as,
                                coord_lock: !!status.l,
                                source: status.src || '',
                            }
                        };
                        self.logUpdate(_fmtResult(result.status, (byteLen / 1024).toFixed(1)));
                        if (opts.onAnalyzeResult) opts.onAnalyzeResult(result);
                    }
                };

                // 接收后端 coords 响应（仅坐标，无 JPEG 图片，适合大地图页面）
                var _processCoordsResult = function(buf) {
                    if (!buf || buf.byteLength < 4) return;
                    var view = new DataView(buf);
                    var jsonLen = view.getUint32(0, false);
                    if (buf.byteLength < 4 + jsonLen) return;
                    var jsonBytes = new Uint8Array(buf, 4, jsonLen);
                    var jsonStr = _decodeUtf8(jsonBytes);
                    var status;
                    try { status = JSON.parse(jsonStr); } catch(e) { return; }
                    if (status.error) { self.log('\u274c coords 解码失败'); return; }
                    var result = {
                        success: true,
                        image: null,
                        status: {
                            mode: status.m,
                            state: status.s,
                            position: { x: status.x, y: status.y },
                            found: !!status.f,
                            matches: status.c,
                            match_quality: status.q || 0,
                            arrow_angle: status.a || 0,
                            arrow_stopped: !!status.as,
                            coord_lock: !!status.l,
                            is_teleport: !!status.tp,
                            source: status.src || '',
                        }
                    };
                    self.logUpdate(_fmtResult(result.status, (buf.byteLength / 1024).toFixed(1)));
                    if (opts.onAnalyzeResult) opts.onAnalyzeResult(result);
                };

                var bindResultHandlers = function(sock) {
                    sock.on('result', function(data) {
                        // 标准化为 ArrayBuffer：兼容 ArrayBuffer / TypedArray / Blob
                        if (data instanceof ArrayBuffer) {
                            _processResultBuf(data, data.byteLength);
                        } else if (ArrayBuffer.isView(data)) {
                            var buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                            _processResultBuf(buf, data.byteLength);
                        } else if (typeof Blob !== 'undefined' && data instanceof Blob) {
                            // polling 场景 Socket.IO 某些版本可能传 Blob
                            var blobSize = data.size;
                            data.arrayBuffer().then(function(ab) { _processResultBuf(ab, blobSize); });
                        } else {
                            console.warn('[WS] result: 未知数据类型', typeof data, data);
                        }
                    });

                    sock.on('coords', function(data) {
                        if (data instanceof ArrayBuffer) {
                            _processCoordsResult(data);
                        } else if (ArrayBuffer.isView(data)) {
                            var buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                            _processCoordsResult(buf);
                        } else if (typeof Blob !== 'undefined' && data instanceof Blob) {
                            data.arrayBuffer().then(function(ab) { _processCoordsResult(ab); });
                        }
                    });
                };

                var attemptConnect = function(pollingOnly) {
                    var sock = io(buildSockOptions(pollingOnly));
                    S.wsSocket = sock;

                    sock.on('connect', function() {
                        S.wsConnected = true;
                        S.wsTransportName = _getSocketTransportName(sock);
                        S.wsConnecting = null;
                        _bindSocketTransportEvents(sock, self);

                        try {
                            if (S.wsTransportName === 'websocket') {
                                localStorage.removeItem('tc_ws_polling_only');
                            } else {
                                localStorage.setItem('tc_ws_polling_only', '1');
                            }
                        } catch (_e) {}

                        // 多会话: 报告 token 绑定会话
                        var token = self._ensureSessionToken();
                        sock.emit('session_join', { token: token });
                        self.log('📶 Socket.IO 已连接（' + S.wsTransportName + '）');
                        if (!settled) {
                            settled = true;
                            resolve(true);
                        }
                    });

                    sock.on('disconnect', function(reason) {
                        S.wsConnected = false;
                        S.wsTransportName = '';
                        S.wsConnecting = null;
                        // 手动断开时已在 disconnectWS 记录日志，此处不重复打印
                        if (!S.wsManualClose) {
                            self.log('📵 Socket.IO 已断开' + (reason ? ' (' + reason + ')' : ''));
                        }
                        S.wsManualClose = false;
                    });

                    sock.on('connect_error', function(err) {
                        S.wsConnected = false;
                        S.wsTransportName = '';
                        S.wsConnecting = null;
                        console.error('SIO error:', err);

                        if (!pollingOnly && !fallbackTried) {
                            fallbackTried = true;
                            self.log('⚠️ WebSocket 通道不可用，自动降级为 polling 重连...');
                            try { sock.disconnect(); } catch (_e) {}
                            attemptConnect(true);
                            return;
                        }

                        self.log('❌ Socket.IO 连接失败: ' + err.message + '（已回退 polling）');
                        if (!settled) {
                            settled = true;
                            reject(err);
                        }
                    });

                    bindResultHandlers(sock);
                };

                try {
                    var preferPollingOnly = false;
                    try {
                        preferPollingOnly = localStorage.getItem('tc_ws_polling_only') === '1';
                    } catch (_e) {}
                    attemptConnect(preferPollingOnly);
                } catch(e) {
                    reject(e);
                }
            });
            return S.wsConnecting;
        },

        disconnectWS() {
            S.wsManualClose = true;
            if (S.wsSocket && typeof S.wsSocket.disconnect === 'function') {
                try { S.wsSocket.disconnect(); } catch(e) {}
            }
            S.wsSocket = null;
            S.wsConnected = false;
            S.wsTransportName = '';
            S.wsConnecting = null;
            this.log('Socket.IO 已手动断开');
        },

        /**
         * 通过 Socket.IO emit 发送二进制帧
         * 事件名 'frame' 匹配后端 @socketio.on('frame')
         */
        sendViaWS(blob) {
            if (!S.wsSocket || !S.wsConnected) {
                this.log('Socket.IO 未连接，请先连接！');
                return Promise.reject(new Error('WS not connected'));
            }
            S.wsSocket.emit('frame', blob);
            // 不在此处逐帧打日志，避免每帧触发 DOM 重排（scrollTop）
            return Promise.resolve();
        },

        /**
         * 通过 Socket.IO emit 发送二进制帧（仅返回坐标，无图片，适合大地图页面）
         * 事件名 'frame_coords' 匹配后端 @socketio.on('frame_coords')
         */
        sendCoordsViaWS(blob) {
            if (!S.wsSocket || !S.wsConnected) {
                this.log('Socket.IO 未连接，请先连接！');
                return Promise.reject(new Error('WS not connected'));
            }
            S.wsSocket.emit('frame_coords', blob);
            return Promise.resolve();
        },

        requestLatestJpeg() {
            if (!S.wsSocket || !S.wsConnected) {
                return false;
            }
            S.wsSocket.emit('request_jpeg');
            return true;
        },

        // ========== 测试图片 ==========
        loadTestImages(gridContainerId) {
            var self = this;
            var gid = gridContainerId || 'testGrid';
            return fetch('/api/test_images').then(function(r) { return r.json(); }).then(function(images) {
                var grid = typeof gid === 'string' ? document.getElementById(gid) : gid;
                if (!grid) return images;
                grid.innerHTML = '';
                images.forEach(function(src) {
                    var img = document.createElement('img');
                    img.src = src; img.className = 'test-item'; img.title = src;
                    img.onclick = function() {
                        _forEachNode(grid.querySelectorAll('.test-item'), function(i) { i.classList.remove('selected'); });
                        img.classList.add('selected');
                        var evt = _createCustomEvent('testimage:selected', { src: src, imgEl: img });
                        document.dispatchEvent(evt);
                    };
                    grid.appendChild(img);
                });
                self.log('已加载 ' + images.length + ' 张测试图片');
                return images;
            }).catch(function(e) {
                console.log('无测试图片或加载失败:', e.message);
                return [];
            });
        }

    }; // end of return
})();

export default TrackerCore;

/**
 * ArrayBuffer 转 Base64 (用于 Socket.IO 二进制响应的 JPEG 图片)
 * 分块调用 apply 避免大 buffer 时的字符串拼接性能问题
 */
function _arrayBufferToBase64(buffer) {
    var bytes = new Uint8Array(buffer);
    var chunkSize = 8192;
    var chunks = [];
    for (var i = 0; i < bytes.length; i += chunkSize) {
        chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize)));
    }
    return btoa(chunks.join(''));
}
