/* recognize-renderer-jpeg.js — JPEG render strategy */
import TC from './tracker-core.js';

const RenderModeJpeg = (function () {
    'use strict';

    return {
        id: 'jpeg',

        /** 新位置到达时——JPEG 模式依赖后端推送图像，此处无需主动触发 */
        onPositionUpdate: function (_R) { /* no-op */ },

        /** 收到后端推来的 JPEG 图像：更新帧缓冲、触发淡入 */
        onImageResult: function (src, R) {
            if (src.indexOf('data:') !== 0) src = 'data:image/jpeg;base64,' + src;
            var capX = R.mapState ? R.mapState.x : 0;
            var capY = R.mapState ? R.mapState.y : 0;
            var img = new Image();
            img.onload = function () {
                R.prevJpegImg = R.lastJpegImg; R.prevJpegPos = R.lastJpegPos;
                R.lastJpegImg = img; R.lastJpegPos = { x: capX, y: capY }; R.jpegFadeAlpha = 0;
            };
            img.src = src;
        },

        /** 进入 JPEG 模式：向后端请求最新帧 */
        onEnter: function (_R) {
            if (typeof TC !== 'undefined' && typeof TC.requestLatestJpeg === 'function') {
                TC.requestLatestJpeg();
            }
        },

        /** 离开 JPEG 模式：清空帧缓冲，避免内存驻留 */
        onLeave: function (R) {
            R.lastJpegImg = null; R.lastJpegPos = null;
            R.prevJpegImg = null; R.prevJpegPos = null;
            R.jpegFadeAlpha = 1;
        },

        /** 绘制 JPEG 模式帧（含亚帧 pan 偏移 + 淡入过渡） */
        render: function (ctx, R, ext, drawArrow) {
            var V = 400, half = 200, PAD = 40;

            if (!R.lastJpegImg || !R.lastJpegImg.complete) {
                ctx.fillStyle = '#111'; ctx.fillRect(0, 0, V, V);
                ctx.fillStyle = '#777'; ctx.font = '15px sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(R.mapState ? '定位中...' : '等待分析...', half, half);
                return;
            }

            var dx = 0, dy = 0;
            if (R.lastJpegPos && ext) {
                dx = Math.max(-PAD, Math.min(PAD, Math.round(ext.x - R.lastJpegPos.x)));
                dy = Math.max(-PAD, Math.min(PAD, Math.round(ext.y - R.lastJpegPos.y)));
            }

            if (R.jpegFadeAlpha < 1) R.jpegFadeAlpha = Math.min(1, R.jpegFadeAlpha + 0.25);
            if (R.jpegFadeAlpha < 1 && R.prevJpegImg && R.prevJpegImg.complete) {
                var pdx = 0, pdy = 0;
                if (R.prevJpegPos && ext) {
                    pdx = Math.max(-PAD, Math.min(PAD, Math.round(ext.x - R.prevJpegPos.x)));
                    pdy = Math.max(-PAD, Math.min(PAD, Math.round(ext.y - R.prevJpegPos.y)));
                }
                ctx.save(); ctx.globalAlpha = 1 - R.jpegFadeAlpha;
                ctx.drawImage(R.prevJpegImg, PAD + pdx, PAD + pdy, V, V, 0, 0, V, V); ctx.restore();
                ctx.save(); ctx.globalAlpha = R.jpegFadeAlpha;
                ctx.drawImage(R.lastJpegImg, PAD + dx, PAD + dy, V, V, 0, 0, V, V); ctx.restore();
            } else {
                ctx.drawImage(R.lastJpegImg, PAD + dx, PAD + dy, V, V, 0, 0, V, V);
            }

            if (ext) {
                if (ext.found) {
                    drawArrow(ctx, half, half, ext.angle, ext.stopped, ext.isInertial);
                } else {
                    ctx.fillStyle = '#ff3333';
                    ctx.beginPath(); ctx.arc(half, half, 4, 0, Math.PI * 2); ctx.fill();
                    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
                    ctx.beginPath(); ctx.arc(half, half, 7, 0, Math.PI * 2); ctx.stroke();
                }
                if (ext.isSceneChange) {
                    ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, V, 24);
                    ctx.fillStyle = '#ffcc44'; ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
                    ctx.fillText('这个区域重定位中...', 8, 16);
                }
            }
        }
    };
})();

export default RenderModeJpeg;
