/**
 * tracker-core.js - 地图跟点 共用识别核心
 * index.html 和 bigmap.html 共用此模块
 *
 * 提供:
 *   TC.log(msg)                    日志
 *   TC.init(opts)                   初始化配置
 *   TC.startScreenCapture(o)        启动屏幕捕获
 *   TC.stopScreenCapture()          停止屏幕捕获
 *   TC.captureScreenImg()           圆形区域截图 -> dataURL
 *   TC.bindFileInput(input, canvas) 绑定文件选择器
 *   TC.loadFile(file, canvas)      手动加载文件 -> Promise<dataURL>
 *   TC.sendAndDisplay(dataURL)      发送到后端 HTTP API 分析
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
            var text = '--', cls = 'warn';
            if (status.mode === 'sift') {
                if (status.state === 'INERTIAL') text = '\u60ef\u6027\u5bfc\u822a';
                else if (status.f) { text = 'SIFT \u8ffd\u8e2a'; cls = 'ok'; }
                else { text = 'SIFT \u641c\u7d22'; cls = 'err'; }
            } else {
                if (status.state === 'GLOBAL_SCAN') text = '\u5168\u5c40\u626b\u63cf';
                else if (status.f) { text = '\u90e8\u5206\u8ffd\u8e2a'; cls = 'ok'; }
                else { text = '\u4e22\u5931'; cls = 'err'; }
            }
            return { stateText: text, stateClass: cls };
        },

        updateStatusDOM(status, ids) {
            ids = ids || {};
            var se = document.getElementById(ids.state || 'statusState');
            var fe = document.getElementById(ids.found || 'statusFound');
            var me = document.getElementById(ids.mode || 'statusMode');
            var xe = document.getElementById(ids.x || 'statusX');
            var ye = document.getElementById(ids.y || 'statusY');
            var mche = document.getElementById(ids.matches || 'statusMatches');

            var f = this.formatStatus(status);

            if (me && status.mode) {
                me.textContent = status.mode.toUpperCase();
                me.style.color = status.mode === 'sift' ? '#ffa726' : '#00d4ff';
            }
            if (se) {
                se.textContent = f.stateText;
                var map = { ok: 'state-local', warn: 'state-global', err: 'state-lost' };
                se.className = 'status-value ' + (map[f.stateClass] || '');
            }
            if (xe && status.position) xe.textContent = status.position.x;
            if (ye && status.position) ye.textContent = status.position.y;
            if (fe) {
                fe.textContent = status.f ? '\u2705 \u662f' : '\u274c \u5426';
                fe.className = 'status-value ' + (status.f ? 'found-yes' : 'found-no');
            }
            if (mche) mche.textContent = status.matches;
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
         * 从屏幕流截取圆形选区图片
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


        // ========== HTTP API 分析 ==========
        /**
         * 发送图片到后端分析（和原 index.html 的 sendAndDisplay 一致）
         * @param {string} imageDataURL 
         * @returns {Promise<Object>} {success, image, status}
         */
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


        // ========== 引擎检测 ==========
        checkEngines() {
            var self = this;
            return fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'loftr' })
            }).then(function(tr) { return tr.json(); }).then(function(tResult) {
                if (!tResult.success) {
                    self.log('\u26a0\ufe0f SIFT-only \u6a21\u5f0f\uff1aLoFTR \u4e0d\u53ef\u7528');
                    return { loftr: false };
                }
                // 恢复为 sift
                return fetch('/api/mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: 'sift' })
                }).then(function() { return { loftr: true }; });
            }).catch(function(e) {
                console.log('\u5f15\u64ce\u68c0\u6d4b\u8df3\u8fc7:', e.message);
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
                        // 触发加载
                        var evt = new CustomEvent('testimage:selected', { detail: { src: src, imgEl: img } });
                        document.dispatchEvent(evt);
                    };
                    grid.appendChild(img);
                });
                self.log('\u5df2\u52a0\u8f7d ' + images.length + ' \u5f20\u6d4b\u8bd5\u56fe\u7247');
                return images;
            }).catch(function(e) {
                console.log('\u65e0\u6d4b\u8bd5\u56fe\u7247\u6216\u52a0\u8f7d\u5931\u8d25:', e.message);
                return [];
            });
        }

    }; // end of return
})();

// 兼容短别名
var TC = TrackerCore;
