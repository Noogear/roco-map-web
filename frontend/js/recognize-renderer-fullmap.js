/* recognize-renderer-fullmap.js — Full-map render strategy */
import TC from './tracker-core.js';
import RenderModeJpeg from './recognize-renderer-jpeg.js';

const RenderModeFullmap = (function () {
    'use strict';

    /** 安装 displayMap 生命周期方法到 R */
    function setupOnR(R) {
        R.releaseDisplayMap = function () {
            R.displayMapRequestId += 1;
            R.displayMapLoading = false;
            R.mapLoadProgress = 0;
            if (!R.displayMap) return;
            R.displayMap.onload = null;
            R.displayMap.onerror = null;
            if (R.displayMap.src) R.displayMap.src = '';
            R.displayMap = null;
        };

        R.ensureDisplayMapLoaded = function () {
            if (!R.displayMapUrl) return;
            if (R.displayMapLoading) return;
            if (R.displayMap && R.displayMap.complete && R.displayMap.naturalWidth) return;

            var requestId = ++R.displayMapRequestId;
            R.displayMapLoading = true;
            R.mapLoadProgress = 0.01;

            /* 启动 RAF 确保进度条能被刷新 */
            if (!R.mapRAFActive && typeof R.startMapRAF === 'function') R.startMapRAF();

            var img = new Image();
            var loadDone = false;  // 防止重复回调
            var stallTimer = null;

            /* 超时保护：15 秒未完成则 fallback */
            stallTimer = setTimeout(function () {
                if (loadDone || requestId !== R.displayMapRequestId) return;
                loadDone = true;
                TC.log('⚠️ 全图加载超时（15s），切换到 JPEG 模式');
                _onMapLoadFail(R);
            }, 15000);

            /* 阶段 1：XHR 追踪下载进度（仅用于 UI，不用响应数据） */
            var xhr = new XMLHttpRequest();
            xhr.open('GET', R.displayMapUrl, true);
            xhr.responseType = 'blob';
            xhr.onprogress = function (e) {
                if (requestId !== R.displayMapRequestId) return;
                if (e.lengthComputable && e.total > 0) {
                    R.mapLoadProgress = Math.min(0.92, e.loaded / e.total);
                }
            };
            xhr.onload = function () {
                if (requestId !== R.displayMapRequestId || loadDone) return;
                /* XHR 下载完成，进度推到 95% */
                R.mapLoadProgress = 0.95;
                /* 阶段 2：用 Blob URL 加载 Image（浏览器解码 webp） */
                if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
                    var blobUrl = URL.createObjectURL(xhr.response);
                    img.onload = function () {
                        if (loadDone || requestId !== R.displayMapRequestId) { URL.revokeObjectURL(blobUrl); return; }
                        loadDone = true;
                        clearTimeout(stallTimer);
                        R.displayMap = img;
                        R.displayMapLoading = false;
                        R.mapLoadProgress = 1;
                        URL.revokeObjectURL(blobUrl);
                        if (!R.mapRAFActive && typeof R.renderMapCanvas === 'function') R.renderMapCanvas();
                    };
                    img.onerror = function () {
                        URL.revokeObjectURL(blobUrl);
                        if (loadDone || requestId !== R.displayMapRequestId) return;
                        /* Blob URL 解码失败，尝试直接 URL 加载 */
                        TC.log('⚠️ Blob 解码失败，尝试直接 URL 加载...');
                        R.mapLoadProgress = 0.96;
                        _loadImageDirect(img, R, requestId, stallTimer, function () { loadDone = true; });
                    };
                    img.src = blobUrl;
                } else {
                    /* XHR 状态异常，直接 URL 加载 */
                    _loadImageDirect(img, R, requestId, stallTimer, function () { loadDone = true; });
                }
            };
            xhr.onerror = function () {
                if (requestId !== R.displayMapRequestId || loadDone) return;
                /* XHR 失败，直接 URL 加载 */
                TC.log('⚠️ XHR 下载失败，尝试直接 URL 加载...');
                _loadImageDirect(img, R, requestId, stallTimer, function () { loadDone = true; });
            };
            xhr.send();
        };

        /** 直接 URL 加载兜底（不走 Blob URL） */
        function _loadImageDirect(img, R, requestId, stallTimer, markDone) {
            img.onload = function () {
                if (requestId !== R.displayMapRequestId) return;
                markDone();
                clearTimeout(stallTimer);
                R.displayMap = img;
                R.displayMapLoading = false;
                R.mapLoadProgress = 1;
                if (!R.mapRAFActive && typeof R.renderMapCanvas === 'function') R.renderMapCanvas();
            };
            img.onerror = function () {
                if (requestId !== R.displayMapRequestId) return;
                markDone();
                clearTimeout(stallTimer);
                _onMapLoadFail(R);
            };
            img.src = R.displayMapUrl;
        }

        function _onMapLoadFail(R) {
            R.displayMapLoading = false;
            R.mapLoadProgress = 0;
            R.modeStrategy = RenderModeJpeg;
            R.renderMode = 'jpeg';
            R.updateRenderModeBtn();
            TC.log('⚠️ 全图加载失败，已切换到 JPEG 模式');
        }
    }

    return {
        id: 'fullmap',

        /** 安装生命周期方法（由 RecognizeRenderer.setup 调用一次） */
        setupOnR: setupOnR,

        /** 新位置到达时触发：确保地图已加载 */
        onPositionUpdate: function (R) {
            if (typeof R.ensureDisplayMapLoaded === 'function') R.ensureDisplayMapLoaded();
        },

        /** 全图模式不消费 JPEG 图像帧 */
        onImageResult: function (_src, _R) { /* no-op */ },

        /** 进入全图模式：开始加载地图 */
        onEnter: function (R) {
            if (typeof R.ensureDisplayMapLoaded === 'function') R.ensureDisplayMapLoaded();
        },

        /** 离开全图模式：释放地图资源 */
        onLeave: function (R) {
            if (typeof R.releaseDisplayMap === 'function') R.releaseDisplayMap();
            R.renderPos = null;
        },

        /** 绘制全图模式帧 */
        render: function (ctx, R, ext, drawArrow) {
            var V = 400, half = 200;

            if (ext && typeof R.ensureDisplayMapLoaded === 'function') R.ensureDisplayMapLoaded();

            if (!ext || !R.displayMap || !R.displayMap.complete || !R.displayMap.naturalWidth) {
                R.renderPos = null;
                ctx.fillStyle = '#111'; ctx.fillRect(0, 0, V, V);
                if (R.displayMapLoading) {
                    var bW = Math.round(V * 0.7), bx = (V - bW) / 2, by = half - 20;
                    var prog = Math.max(0.02, Math.min(1, R.mapLoadProgress || 0));
                    ctx.fillStyle = '#333'; ctx.fillRect(bx, by, bW, 8);
                    ctx.fillStyle = '#00d4ff'; ctx.fillRect(bx, by, Math.round(bW * prog), 8);
                    ctx.fillStyle = '#ddd'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText('地图加载中 ' + Math.round(prog * 100) + '%', half, by + 24);
                } else {
                    ctx.fillStyle = '#777'; ctx.font = '15px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText(R.mapState ? '定位中...' : '等待分析...', half, half);
                }
                return;
            }

            var hasTrackPos = !!ext.found || !!ext.isInertial || !!ext.isSceneChange;

            if (!R.renderPos) {
                R.renderPos = { x: ext.x, y: ext.y };
            } else if (hasTrackPos) {
                if (ext.isSceneChange) {
                    R.renderPos = { x: ext.x, y: ext.y };
                } else {
                    R.renderPos.x += 0.4 * (ext.x - R.renderPos.x);
                    R.renderPos.y += 0.4 * (ext.y - R.renderPos.y);
                }
            }

            var pos = {
                x: R.renderPos.x,
                y: R.renderPos.y,
                angle: ext.angle,
                stopped: ext.stopped,
                found: !!ext.found,
                isInertial: ext.isInertial,
                isSceneChange: ext.isSceneChange
            };
            var mW = R.displayMap.naturalWidth, mH = R.displayMap.naturalHeight;
            var sx = Math.max(0, Math.min(Math.round(pos.x - half), mW - V));
            var sy = Math.max(0, Math.min(Math.round(pos.y - half), mH - V));
            var lx = Math.round(pos.x - sx), ly = Math.round(pos.y - sy);
            ctx.drawImage(R.displayMap, sx, sy, V, V, 0, 0, V, V);

            if (hasTrackPos) {
                ctx.fillStyle = pos.isInertial ? '#ffff00' : '#00ff00';
                ctx.beginPath(); ctx.arc(lx, ly + 6, 4, 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.arc(lx, ly + 6, 7, 0, Math.PI * 2); ctx.stroke();
                drawArrow(ctx, lx, ly, pos.angle, pos.stopped);
            } else {
                ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, V, 24);
                ctx.fillStyle = '#ffd27a'; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
                ctx.fillText('未识别到位置，保持上一坐标', 8, 16);
            }

            if (pos.isSceneChange) {
                ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, V, 24);
                ctx.fillStyle = '#ffcc44'; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
                ctx.fillText('这个区域重定位中...', 8, 16);
            }
        }
    };
})();

export default RenderModeFullmap;
