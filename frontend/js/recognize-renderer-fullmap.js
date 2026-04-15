/* recognize-renderer-fullmap.js — Full-map render strategy
 *
 * 加载策略：
 *   页面初始化后立即以浏览器原生 Image 缓存预加载全图。
 *   进入全图模式后直接 drawImage 裁切目标视口，避免 XHR/Blob/createImageBitmap
 *   这条更容易受压缩编码、缓存状态和解码时机影响的不稳定链路。
 *   共享预加载实例独立于渲染模式切换，releaseDisplayMap 只释放当前模式引用，
 *   不会把页面级共享资源一并清掉。
 */
import TC from './tracker-core.js';

var SharedDisplayMap = (function () {
    'use strict';

    var state = {
        url: '',
        img: null,
        ready: false,
        loading: false,
        failed: false,
        progress: 0,
        naturalWidth: 0,
        naturalHeight: 0,
        requestId: 0,
        nextRetryAt: 0
    };

    function _syncToRenderer(R) {
        R.displayMapLoading = state.loading;
        R.mapLoadProgress = state.ready ? 1 : state.progress;
        R.displayMapNaturalWidth = state.naturalWidth;
        R.displayMapNaturalHeight = state.naturalHeight;
        R.displayMapReady = state.ready;
        R.displayMapFailed = state.failed;
        if (state.ready && state.img && state.img.complete && state.img.naturalWidth) {
            R.displayMap = state.img;
        } else {
            R.displayMap = null;
        }
    }

    function _reset(url) {
        if (state.img) {
            state.img.onload = null;
            state.img.onerror = null;
        }
        state.url = url || '';
        state.img = null;
        state.ready = false;
        state.loading = false;
        state.failed = false;
        state.progress = 0;
        state.naturalWidth = 0;
        state.naturalHeight = 0;
        state.nextRetryAt = 0;
    }

    return {
        ensure: function (url, R) {
            if (!url) {
                _syncToRenderer(R);
                return;
            }

            if (state.url && state.url !== url) {
                _reset(url);
            }

            if (!state.url) state.url = url;

            if (state.ready && state.img && state.img.complete && state.img.naturalWidth) {
                _syncToRenderer(R);
                return;
            }

            if (state.loading) {
                _syncToRenderer(R);
                return;
            }

            if (state.failed && Date.now() < state.nextRetryAt) {
                _syncToRenderer(R);
                return;
            }

            state.loading = true;
            state.failed = false;
            state.progress = 0.15;
            state.requestId += 1;

            var requestId = state.requestId;
            var img = new Image();
            img.decoding = 'async';

            img.onload = function () {
                if (requestId !== state.requestId) return;
                state.img = img;
                state.ready = true;
                state.loading = false;
                state.failed = false;
                state.progress = 1;
                state.naturalWidth = img.naturalWidth || 0;
                state.naturalHeight = img.naturalHeight || 0;
                _syncToRenderer(R);
                TC.log('✅ 全图预加载完成: ' + state.naturalWidth + 'x' + state.naturalHeight);
            };

            img.onerror = function () {
                if (requestId !== state.requestId) return;
                state.img = null;
                state.ready = false;
                state.loading = false;
                state.failed = true;
                state.progress = 0;
                state.naturalWidth = 0;
                state.naturalHeight = 0;
                state.nextRetryAt = Date.now() + 3000;
                _syncToRenderer(R);
                TC.log('⚠️ 全图预加载失败');
            };

            state.img = img;
            _syncToRenderer(R);
            img.src = url;
        },

        syncToRenderer: function (R) {
            _syncToRenderer(R);
        }
    };
})();

const RenderModeFullmap = (function () {
    'use strict';

    function setupOnR(R) {
        R.displayMapReady = false;
        R.displayMapFailed = false;

        /* 页面级共享预加载，保留旧方法名兼容调用方。 */
        R.prefetchDisplayMap = function () {
            if (!R.displayMapUrl) return;
            SharedDisplayMap.ensure(R.displayMapUrl, R);
            if (!R.mapRAFActive && typeof R.startMapRAF === 'function') R.startMapRAF();
        };
        R.prefetchDisplayMapBlob = R.prefetchDisplayMap;

        /* ────────────────────────────────────────────────────────────
         * releaseDisplayMap
         * 离开全图模式时只释放当前模式持有的显示引用。
         * 页面级共享预加载资源不在这里销毁，避免模式切换后重新加载。
         * ──────────────────────────────────────────────────────────── */
        R.releaseDisplayMap = function () {
            R.displayMap = null;
            R.displayMapLoading = false;
        };

        /* 兼容旧调用（onPositionUpdate / onEnter 仍会调用）。 */
        R.ensureDisplayMapLoaded = function () {
            SharedDisplayMap.syncToRenderer(R);
            if (!R.displayMapReady && !R.displayMapLoading) {
                R.prefetchDisplayMap();
            }
            if (!R.mapRAFActive && typeof R.startMapRAF === 'function') R.startMapRAF();
        };
    }

    return {
        id: 'fullmap',

        /** 安装生命周期方法（由 RecognizeRenderer.setup 调用一次） */
        setupOnR: setupOnR,

        /** 新位置到达时触发 */
        onPositionUpdate: function (R) {
            if (typeof R.ensureDisplayMapLoaded === 'function') R.ensureDisplayMapLoaded();
        },

        /** 全图模式不消费 JPEG 图像帧 */
        onImageResult: function (_src, _R) { /* no-op */ },

        /** 进入全图模式：同步共享状态并在需要时触发预加载/重试 */
        onEnter: function (R) {
            if (typeof R.ensureDisplayMapLoaded === 'function') R.ensureDisplayMapLoaded();
            if (!R.displayMapReady && !R.displayMapLoading) {
                R.prefetchDisplayMap();
            }
        },

        /** 离开全图模式：释放当前模式引用（共享预载仍保留） */
        onLeave: function (R) {
            if (typeof R.releaseDisplayMap === 'function') R.releaseDisplayMap();
            R.renderPos = null;
        },

        /** 绘制全图模式帧 */
        render: function (ctx, R, ext, drawArrow) {
            var V = 400, half = 200;
            var mW = R.displayMapNaturalWidth, mH = R.displayMapNaturalHeight;

            /* ── 全图未就绪：显示加载态 ── */
            if (!R.displayMapReady || !R.displayMap || !R.displayMap.complete || !R.displayMap.naturalWidth) {
                R.renderPos = null;
                ctx.fillStyle = '#111'; ctx.fillRect(0, 0, V, V);
                if (R.displayMapLoading || R.mapLoadProgress > 0) {
                    var bW = Math.round(V * 0.7), bx = (V - bW) / 2, by = half - 20;
                    var prog = Math.max(0.12, Math.min(0.9, R.mapLoadProgress || 0.5));
                    ctx.fillStyle = '#333'; ctx.fillRect(bx, by, bW, 8);
                    ctx.fillStyle = '#00d4ff'; ctx.fillRect(bx, by, Math.round(bW * prog), 8);
                    ctx.fillStyle = '#ddd'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText('地图加载中 ' + Math.round(prog * 100) + '%', half, by + 24);
                } else if (R.displayMapFailed) {
                    ctx.fillStyle = '#f1b84b'; ctx.font = '14px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText('地图加载失败，请稍后重试', half, half);
                } else {
                    ctx.fillStyle = '#777'; ctx.font = '15px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText(R.mapState ? '定位中...' : '等待分析...', half, half);
                }
                return;
            }

            /* ── 无位置或地图尺寸未就绪 ── */
            if (!ext || !mW || !mH) {
                R.renderPos = null;
                ctx.fillStyle = '#111'; ctx.fillRect(0, 0, V, V);
                ctx.fillStyle = '#777'; ctx.font = '15px sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(R.mapState ? '定位中...' : '等待分析...', half, half);
                return;
            }

            /* ── 平滑位置插值 ── */
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
                x: R.renderPos.x, y: R.renderPos.y,
                angle: ext.angle, stopped: ext.stopped,
                found: !!ext.found, isInertial: ext.isInertial, isSceneChange: ext.isSceneChange
            };

            /* ── 视口起点（地图像素坐标，边界对齐） ── */
            var sx = Math.max(0, Math.min(Math.round(pos.x - half), mW - V));
            var sy = Math.max(0, Math.min(Math.round(pos.y - half), mH - V));

            /* ── 直接使用共享 Image 资源裁切绘制 ── */
            ctx.drawImage(R.displayMap, sx, sy, V, V, 0, 0, V, V);

            /* ── 玩家指示器 ── */
            var lx = Math.round(pos.x - sx), ly = Math.round(pos.y - sy);
            if (hasTrackPos) {
                drawArrow(ctx, lx, ly, pos.angle, pos.stopped, pos.isInertial);
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
