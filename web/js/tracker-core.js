/**
 * tracker-core.js - 地图跟点 共用识别核心
 * index.html 和 bigmap.html 共用此模块
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
        selCircle: { cx: (1189 + 62.5) / 1362, cy: (66 + 63.5) / 806, r: Math.max(62.5 / 1362, 63.5 / 806) },
        wsSocket: null,
        wsConnected: false,
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
        get isScreenActive() { return !!S.screenStream; },
        get wsConnected() { return S.wsConnected; },

        // ========== 日志 ==========
        log(msg) {
            let el = opts.logEl;
            if (typeof el === 'string') el = document.getElementById(el);
            if (!el) return;
            const time = new Date().toLocaleTimeString();
            el.textContent += `[${time}] ${msg}\n`;
            var MAX = parseInt(el.dataset.maxLen || '5000', 10);
            var TRIM = parseInt(el.dataset.trimLen || '4000', 10);
            if (el.textContent.length > MAX) el.textContent = el.textContent.slice(-TRIM);
            el.scrollTop = el.scrollHeight;
        },

        // ========== 工具 ==========
        clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); },

        // ========== 状态格式化 ==========
        formatStatus(status) {
            var text = '--', cls = 'green', label = '';
            if (status.mode === 'sift') {
                if (status.state === 'SCENE_CHANGE') {
                    text = '切场'; cls = 'yellow'; label = '场景切换';
                } else if (!status.f) {
                    text = '丢失'; cls = 'red'; label = '未找到';
                } else if (status.state === 'INERTIAL' || status.is_inertial) {
                    text = '惯性'; cls = 'yellow'; label = '惯性导航';
                } else {
                    text = '正常'; cls = 'green'; label = 'SIFT追踪';
                }
            } else {
                if (status.state === 'GLOBAL_SCAN') {
                    text = '扫描'; cls = 'red'; label = '全局扫描';
                } else if (status.f) {
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
                me.style.color = status.mode === 'sift' ? '#ffa726' : '#00d4ff';
            }
            if (xe && status.position) xe.textContent = status.position.x;
            if (ye && status.position) ye.textContent = status.position.y;
            if (fe) {
                fe.textContent = status.f ? '\u2705 \u662f' : '\u274c \u5426';
                fe.className = 'status-value ' + (status.f ? 'found-yes' : 'found-no');
            }
            if (mche) mche.textContent = status.matches;

            // 匹配质量显示
            var qe = document.getElementById(ids.quality || 'statusQuality');
            if (qe) {
                var q = status.match_quality || 0;
                qe.textContent = (q * 100).toFixed(0) + '%';
                qe.style.color = q >= 0.7 ? '#4caf50' : q >= 0.4 ? '#ffa726' : '#ff5252';
            }

            // 混合引擎状态
            var he = document.getElementById(ids.hybrid || 'hybridStatusItem');
            if (he) {
                he.style.display = status.hybrid_busy ? '' : 'none';
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

            return navigator.mediaDevices.getDisplayMedia({
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
         * 从屏幕流截取圆形选区图片 (JPEG dataURL，用于 HTTP 模式)
         * @returns {string|null} dataURL 或 null
         */
        captureScreenImg() {
            var vid = S.offscreenVid;
            if (!vid || !vid.videoWidth) return null;

            var vw = vid.videoWidth || S.videoW;
            var vh = vid.videoHeight || S.videoH;
            if (!vw || !vh) return null;

            var sc = S.selCircle;
            var bs = Math.min(vw, vh);
            var cx = sc.cx * vw, cy = sc.cy * vh, r = sc.r * bs;
            var sz = Math.max(10, Math.round(r * 2));
            var rx = Math.round(cx - r), ry = Math.round(cy - r);

            var c = document.createElement('canvas');
            c.width = sz; c.height = sz;
            var ctx = c.getContext('2d');
            ctx.beginPath(); ctx.arc(sz / 2, sz / 2, sz / 2, 0, Math.PI * 2); ctx.closePath(); ctx.clip();
            ctx.drawImage(vid, rx, ry, sz, sz, 0, 0, sz, sz);
            return c.toDataURL('image/jpeg', 0.80);
        },

        /**
         * 从屏幕流截取圆形选区图片 (JPEG Blob，用于 Socket.IO 二进制模式)
         * 使用 JPEG 代替 PNG，上传体积减少约 90%，识别精度不受影响
         * @returns {Promise<Blob|null>}
         */
        captureScreenImgBlob() {
            var vid = S.offscreenVid;
            if (!vid || !vid.videoWidth) return Promise.resolve(null);

            var vw = vid.videoWidth || S.videoW;
            var vh = vid.videoHeight || S.videoH;
            if (!vw || !vh) return Promise.resolve(null);

            var sc = S.selCircle;
            var bs = Math.min(vw, vh);
            var cx = sc.cx * vw, cy = sc.cy * vh, r = sc.r * bs;
            var sz = Math.max(10, Math.round(r * 2));
            var rx = Math.round(cx - r), ry = Math.round(cy - r);

            var c = document.createElement('canvas');
            c.width = sz; c.height = sz;
            var ctx = c.getContext('2d');
            ctx.beginPath(); ctx.arc(sz / 2, sz / 2, sz / 2, 0, Math.PI * 2); ctx.closePath(); ctx.clip();
            ctx.drawImage(vid, rx, ry, sz, sz, 0, 0, sz, sz);

            return new Promise(function(resolve) {
                c.toBlob(function(blob) { resolve(blob); }, 'image/jpeg', 0.82);
            });
        },


        // ========== HTTP API 分析 ==========
        sendAndDisplay(imageDataURL) {
            var self = this;
            return fetch('/api/upload_minimap', {
                method: 'POST',
                body: JSON.stringify({ image: imageDataURL }),
                headers: { 'Content-Type': 'application/json' }
            }).then(function(resp) {
                if (resp.ok) return resp.json();
                // fallback: FormData
                return fetch(imageDataURL).then(function(r) { return r.blob(); }).then(function(blob) {
                    var fd = new FormData(); fd.append('image', blob, 'minimap.png');
                    return fetch('/api/upload_minimap', { method: 'POST', body: fd }).then(function(r2) { return r2.json(); });
                });
            }).then(function(result) {
                if (result.status) {
                    self.log('\u5206\u6790\u5b8c\u6210 | ' +
                        (result.status.found ? '\u2705 \u627e\u5230\u4f4d\u7f6e' : '\u274c \u672a\u627e\u5230') +
                        ' (' + result.status.matches + ' \u5339\u914d\u70b9)');
                } else {
                    self.log('\u5206\u6790\u5b8c\u6210\uff0c\u4f46\u672a\u8fd4\u56de\u72b6\u6001\u4fe1\u606f');
                }
                if (opts.onAnalyzeResult) opts.onAnalyzeResult(result);
                return result;
            });
        },


        // ========== Socket.IO 二进制通道 ==========
        /**
         * 建立 Socket.IO 连接（使用 socket.io client 库）
         * @returns {Promise<boolean>}
         */
        connectWS() {
            var self = this;
            if (S.wsSocket && S.wsConnected) return Promise.resolve(true);
            return new Promise(function(resolve, reject) {
                try {
                    var sock = io({
                        transports: ['websocket'],
                        forceNew: true,
                    });
                    S.wsSocket = sock;

                    sock.on('connect', function() {
                        S.wsConnected = true;
                        self.log('\ud83d\udcf6 Socket.IO 已连接（二进制模式）');
                        resolve(true);
                    });

                    sock.on('disconnect', function() {
                        S.wsConnected = false;
                        self.log('\ud83d\udcf5 Socket.IO 已断开');
                    });

                    sock.on('connect_error', function(err) {
                        S.wsConnected = false;
                        console.error('SIO error:', err);
                        self.log('\u274c Socket.IO 连接失败: ' + err.message);
                        reject(err);
                    });

                    // 接收后端二进制响应: [4字节大端JSON长度][JSON][JPEG图片]
                    sock.on('result', function(data) {
                        if (!(data instanceof ArrayBuffer)) return;
                        var view = new DataView(data);
                        if (view.byteLength < 4) return;
                        var jsonLen = view.getUint32(0, false);
                        if (view.byteLength < 4 + jsonLen) return;
                        var jsonBytes = new Uint8Array(data, 4, jsonLen);
                        var jsonStr = new TextDecoder().decode(jsonBytes);
                        var status;
                        try { status = JSON.parse(jsonStr); } catch(e) { return; }

                        if (status.error) {
                            self.log('\u274c 解码失败');
                            return;
                        }

                        var jpegStart = 4 + jsonLen;
                        var jpegBytes = data.slice(jpegStart);
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
                                coord_lock: !!status.l,
                                hybrid_busy: !!status.h,
                            }
                        };
                            if (opts.onAnalyzeResult) opts.onAnalyzeResult(result);
                        }
                    });

                } catch(e) {
                    reject(e);
                }
            });
        },

        disconnectWS() {
            if (S.wsSocket && typeof S.wsSocket.disconnect === 'function') {
                try { S.wsSocket.disconnect(); } catch(e) {}
            }
            S.wsSocket = null;
            S.wsConnected = false;
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
            this.log('\u5df2发送二进制帧 (' + Math.round(blob.size / 1024) + ' KB)');
            return Promise.resolve();
        },


        // ========== 引擎检测 ==========
        checkEngines() {
            var self = this;
            return fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'loftr' })
            }).then(function(tr) { return tr.json(); }).then(function(tResult) {
                if (!tResult.success) {
                    self.log('\u26a0\ufe0f SIFT-only 模式：LoFTR 不可用');
                    return { loftr: false };
                }
                // 恢复为 sift
                return fetch('/api/mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: 'sift' })
                }).then(function() { return { loftr: true }; });
            }).catch(function(e) {
                console.log('引擎检测跳过:', e.message);
                return { loftr: false };
            });
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
                        grid.querySelectorAll('.test-item').forEach(function(i) { i.classList.remove('selected'); });
                        img.classList.add('selected');
                        var evt = new CustomEvent('testimage:selected', { detail: { src: src, imgEl: img } });
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

// 兼容短别名
var TC = TrackerCore;

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
