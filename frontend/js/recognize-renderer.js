/* recognize-renderer.js — 共享工具、RAF 循环、PiP、模式切换调度
 *
 * 渲染策略由独立模块导入：
 *   recognize-renderer-fullmap.js  →  RenderModeFullmap
 *   recognize-renderer-jpeg.js     →  RenderModeJpeg
 *
 * R.modeStrategy 指向当前活跃策略，所有渲染/图像处理均委托给它：
 *   .render(ctx, R, ext, drawArrow)    绘制一帧
 *   .onPositionUpdate(R)               新位置坐标到达
 *   .onImageResult(src, R)             收到后端图像数据
 *   .onEnter(R)                        进入该模式
 *   .onLeave(R)                        离开该模式
 */
import * as AppCommon from './common.js';
import TC from './tracker-core.js';
import RenderModeFullmap from './recognize-renderer-fullmap.js';
import RenderModeJpeg from './recognize-renderer-jpeg.js';

export const RecognizeRenderer = {
    setup: function (R) {
        'use strict';

        /* ── Catmull-Rom 插值 ── */
        function catmullRom(p0, p1, p2, p3, t) {
            var t2 = t * t, t3 = t2 * t;
            return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
        }

        function getExtrapolatedPos() {
            if (!R.mapState) return null;
            if (!R.mapState.found) return R.mapState;
            var h = R.mapHistory;
            if (h.length < 2) return R.mapState;
            var prev = h[h.length - 2], curr = h[h.length - 1];
            var interval = curr.ts - prev.ts;
            if (interval < 10) return curr;
            var jdx = curr.x - prev.x, jdy = curr.y - prev.y;
            if (jdx * jdx + jdy * jdy > 400 * 400) return curr;
            var t = Math.min(Math.max((performance.now() - interval - prev.ts) / interval, 0), 1);
            var rx, ry;
            if (h.length >= 4) {
                var p0 = h[h.length - 4], p1 = h[h.length - 3];
                rx = catmullRom(p0.x, p1.x, prev.x, curr.x, t);
                ry = catmullRom(p0.y, p1.y, prev.y, curr.y, t);
            } else if (h.length >= 3) {
                var p0b = h[h.length - 3];
                rx = catmullRom(p0b.x, p0b.x, prev.x, curr.x, t);
                ry = catmullRom(p0b.y, p0b.y, prev.y, curr.y, t);
            } else {
                rx = prev.x + t * jdx; ry = prev.y + t * jdy;
            }
            
            // 角度进行平滑插值 (短路径插值)，避免每秒十帧的生硬跳变
            var prevAngle = prev.angle || 0;
            var currAngle = curr.angle || 0;
            var angleDiff = ((currAngle - prevAngle) % 360 + 540) % 360 - 180;
            var rAngle = (prevAngle + angleDiff * t + 360) % 360;

            return { 
                x: rx, y: ry, 
                angle: rAngle, 
                stopped: curr.stopped, 
                found: curr.found, 
                isInertial: curr.isInertial, 
                isSceneChange: curr.isSceneChange 
            };
        }

        function drawArrow(ctx, cx, cy, angle, stopped, isInertial) {
            var size = 12;
            ctx.save(); ctx.translate(cx, cy); ctx.rotate((angle || 0) * Math.PI / 180);
            
            // 正常状态为天蓝色，惯性模式(无有效位点)为黄色警告
            ctx.fillStyle = isInertial ? 'rgba(255,204,0,0.86)' : 'rgba(48,182,254,0.86)';
            
            if (stopped) {
                ctx.beginPath(); ctx.arc(0, 0, Math.round(size * 0.6), 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
            } else {
                ctx.beginPath(); ctx.moveTo(0, -size); ctx.lineTo(-size * 0.6, size * 0.7); ctx.lineTo(size * 0.6, size * 0.7); ctx.closePath(); ctx.fill();
                ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
            }
            ctx.restore();
        }

        /* ── 安装 displayMap 生命周期（全图模式专用，供策略对象使用） ── */
        RenderModeFullmap.setupOnR(R);

        /* ── 初始化活跃策略 ── */
        R.modeStrategy = R.renderMode === 'jpeg' ? RenderModeJpeg : RenderModeFullmap;
        R.renderFps = Math.max(10, Math.min(60, parseInt(R.renderFps, 10) || 60));
        R.renderIntervalMs = 1000 / R.renderFps;
        R.lastRenderAt = 0;
        R.pipKeepAliveTimer = null;

        R.setRenderFps = function (fps) {
            R.renderFps = Math.max(10, Math.min(60, parseInt(fps, 10) || 60));
            R.renderIntervalMs = 1000 / R.renderFps;
            R.lastRenderAt = 0;
            if (typeof R.updateRenderModeBtn === 'function') R.updateRenderModeBtn();
        };

        /* ── 主渲染：委托给当前策略 ── */
        R.renderMapCanvas = function () {
            var ctx = R.resultCanvas.getContext('2d');
            var ext = getExtrapolatedPos();
            R.modeStrategy.render(ctx, R, ext, drawArrow);
        };

        function stopPiPKeepAlive() {
            if (R.pipKeepAliveTimer) {
                clearTimeout(R.pipKeepAliveTimer);
                R.pipKeepAliveTimer = null;
            }
        }

        function schedulePiPKeepAlive() {
            if (!R.pipRAFActive) {
                stopPiPKeepAlive();
                return;
            }
            if (!document.hidden) {
                stopPiPKeepAlive();
                return;
            }
            if (R.pipKeepAliveTimer) return;

            function tick() {
                R.pipKeepAliveTimer = null;
                if (!R.pipRAFActive || !document.hidden) return;
                try { R.renderMapCanvas(); } catch (e) { console.error(e); }
                schedulePiPKeepAlive();
            }

            R.pipKeepAliveTimer = setTimeout(tick, Math.max(16, Math.round(R.renderIntervalMs)));
        }

        if (!R.pipVisibilityBound) {
            R.pipVisibilityBound = true;
            document.addEventListener('visibilitychange', function () {
                if (R.pipRAFActive && document.hidden) {
                    schedulePiPKeepAlive();
                } else {
                    stopPiPKeepAlive();
                }
            });
        }

        /* ── RAF 循环 ── */
        R.startMapRAF = function () {
            if (R.mapRAFActive) return;
            R.mapRAFActive = true;
            (function loop(now) {
                if (!R.mapRAFActive) return;
                /* Skip rendering when tab is hidden to avoid GPU work piling up */
                if (document.hidden && !R.pipRAFActive) { requestAnimationFrame(loop); return; }
                var ts = typeof now === 'number' ? now : performance.now();
                if (!R.lastRenderAt || ts - R.lastRenderAt >= R.renderIntervalMs - 1) {
                    R.lastRenderAt = ts;
                    try { R.renderMapCanvas(); } catch (e) { console.error(e); }
                    if (R.pipRAFActive) schedulePiPKeepAlive();
                }
                requestAnimationFrame(loop);
            })();
        };

        /* ── PiP ── */
        function stopPiPRefresh() {
            R.pipRAFActive = false;
            stopPiPKeepAlive();
        }

        function stopPiPStream() {
            var pipVideo = document.getElementById('pipVideo');
            var stream = pipVideo && pipVideo.srcObject;
            if (stream && typeof stream.getTracks === 'function') {
                stream.getTracks().forEach(function (track) {
                    try { track.stop(); } catch (_) {}
                });
            }
            if (pipVideo) {
                try { pipVideo.pause(); } catch (_) {}
                pipVideo.srcObject = null;
                pipVideo.removeAttribute('src');
                try { pipVideo.load(); } catch (_) {}
            }
        }

        function isNativePiPActive() {
            var v = document.getElementById('pipVideo');
            return document.pictureInPictureElement === v || v.webkitPresentationMode === 'picture-in-picture';
        }

        R.resetPiPButtonState = function () {
            R.pipBtn.classList.remove('is-active');
            stopPiPRefresh();
        };

        R.cleanupPiP = function () {
            var pipVideo = document.getElementById('pipVideo');
            R.resetPiPButtonState();
            if (pipVideo) {
                try {
                    if (document.pictureInPictureElement === pipVideo && typeof document.exitPictureInPicture === 'function') {
                        document.exitPictureInPicture().catch(function () {});
                    } else if (typeof pipVideo.webkitSetPresentationMode === 'function' && pipVideo.webkitPresentationMode === 'picture-in-picture') {
                        pipVideo.webkitSetPresentationMode('inline');
                    }
                } catch (_) {}
            }
            stopPiPStream();
        };

        R.toggleNativePiP = async function () {
            try {
                if (isNativePiPActive()) {
                    if (document.pictureInPictureElement) await document.exitPictureInPicture();
                    else document.getElementById('pipVideo').webkitSetPresentationMode('inline');
                    R.cleanupPiP(); return;
                }
                var pipVideo = document.getElementById('pipVideo');
                stopPiPStream();
                pipVideo.srcObject = R.resultCanvas.captureStream(Math.max(10, Math.min(60, R.renderFps || 30)));
                if (!R.pipEventsBound) {
                    R.pipEventsBound = true;
                    pipVideo.addEventListener('leavepictureinpicture', function () { R.cleanupPiP(); TC.log('✅ 原生画中画已关闭'); });
                    pipVideo.addEventListener('webkitpresentationmodechanged', function () {
                        if (pipVideo.webkitPresentationMode !== 'picture-in-picture') R.cleanupPiP();
                    });
                }
                await pipVideo.play();
                var support = R.getPiPSupport();
                if (support.standard) await pipVideo.requestPictureInPicture();
                else pipVideo.webkitSetPresentationMode('picture-in-picture');
                R.pipBtn.classList.add('is-active');
                R.pipRAFActive = true;
                schedulePiPKeepAlive();
                TC.log('✅ 原生浏览器画中画已开启');
            } catch (err) {
                R.cleanupPiP();
                AppCommon.toast('画中画启动失败：' + err.message, 'danger');
            }
        };

        /* ── 模式切换按钮 ── */
        R.updateRenderModeBtn = function () {
            var btn = document.getElementById('renderModeBtn');
            var isFullmap = R.modeStrategy.id === 'fullmap';
            btn.textContent = isFullmap ? '🗺️ 全图' : '📸 JPEG';
            btn.classList.toggle('is-fullmap', isFullmap);
            btn.classList.toggle('is-jpeg', !isFullmap);
            btn.title = isFullmap
                ? '当前：全图模式（目标绘制 ' + R.renderFps + 'fps，无边界）\n点击切换到 JPEG 模式（轻量低流量）'
                : '当前：JPEG 模式（目标绘制 ' + R.renderFps + 'fps）\n点击切换到全图模式（无边界）';
        };

        R.toggleRenderMode = function () {
            var next = R.modeStrategy.id === 'fullmap' ? RenderModeJpeg : RenderModeFullmap;
            R.modeStrategy.onLeave(R);
            R.modeStrategy = next;
            R.renderMode = next.id;   /* 保留 renderMode 字段兼容旧引用 */
            AppCommon.updatePref('preferJpegMode', next.id === 'jpeg');
            R.updateRenderModeBtn();
            next.onEnter(R);
        };
    }
};
