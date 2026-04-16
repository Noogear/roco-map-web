/* recognize.js — Screen capture, broadcast presenter, page tab switching */
import * as AppCommon from './common.js';
import TC from './tracker-core.js';
import { RecognizeRenderer } from './recognize-renderer.js';
import RecognizeAudience from './recognize-audience.js';

AppCommon.buildDock();
AppCommon.initTooltip();

const prefs = AppCommon.loadPrefs();

    /* ── Shared renderer state ── */
    var R = {
        resultCanvas: document.getElementById('resultCanvas'),
        pipBtn: document.getElementById('pipBtn'),
        mapState: null, mapStatePrev: null, mapHistory: [],
        mapRAFActive: false,
        renderFps: prefs.renderFps || 60,
        displayMap: null, mapLoadProgress: 0,
        displayMapLoading: false, displayMapReady: false, displayMapFailed: false,
        displayMapUrl: '/img/map/map_z7.webp',
        displayMapNaturalWidth: 0, displayMapNaturalHeight: 0,
        renderPos: null,
        lastJpegImg: null, lastJpegPos: null, prevJpegImg: null, prevJpegPos: null,
        jpegFadeAlpha: 1,
        renderMode: prefs.preferJpegMode ? 'jpeg' : 'fullmap',
        modeStrategy: null,
        pipRAFActive: false, pipEventsBound: false,
        getPiPSupport: getPiPSupport
    };

    /* ── DOM refs ── */
    var mainScreenBtn    = document.getElementById('mainScreenBtn');
    var mainScreenBtnSub = document.getElementById('mainScreenBtnSub');
    var screenStateChip  = document.getElementById('screenStateChip');
    var watchBadge       = document.getElementById('watchBadge');
    var statusStateText  = document.getElementById('statusStateText');
    var broadcastBtn     = document.getElementById('broadcastBtn');
    var broadcastBtnTitle = document.getElementById('broadcastBtnTitle');
    var broadcastBtnSub = document.getElementById('broadcastBtnSub');
    var statusTransportChip = document.getElementById('statusTransportChip');
    var settingsPanelBtn = document.getElementById('settingsPanelBtn');
    var toolPanelBtn = document.getElementById('toolPanelBtn');
    var settingsPanel = document.getElementById('recognizeSettingsPanel');
    var toolsPanel = document.getElementById('recognizeToolsPanel');
    var recognizeNameInput = document.getElementById('recognizeNameInput');

    /* ── State ── */
    var screenSending = false, useWSMode = true;
    var _sceneChangeMissCount = 0;   /* 连续 SCENE_CHANGE 帧计数，用于触发校准提示 */
    var _sceneChangeWarnAt = 0;      /* 避免重复提示的冷却时间戳 */
    var wsReconnectAt = 0, nullBlobWarnAt = 0, S_nullBlobCount = 0;
    var screenAutoRunning = false, screenAutoTimeoutId = null, screenAutoRafId = null;
    /* Broadcast */
    var bcastName = null;           // null = 未展示，string = 当前展示名
    var bcastViewerCount = 0;
    var bcastCanvas = document.getElementById('bcastCanvas');
    var bcastSending = false;
    var bcastAutoId = null;
    var bcastStarting = false;
    var BCAST_INTERVAL_MS = 100;    // 前端本地节流（10fps），配合服务端节流双保险

    function setBroadcastButtonState(active, name) {
        broadcastBtn.classList.toggle('is-active', !!active);
        if (broadcastBtnTitle) broadcastBtnTitle.textContent = active ? '🔴 停止展示' : '📡 展示';
        if (broadcastBtnSub) broadcastBtnSub.textContent = active
            ? ('当前：' + (name || '未命名频道'))
            : '推流给观众';
    }

    function setFlyoutOpen(panel, trigger, open) {
        if (!panel || !trigger) return;
        panel.classList.toggle('is-open', !!open);
        if (open) positionFlyoutToTrigger(panel, trigger);
        AppCommon.setInteractiveHiddenState(panel, !open);
        trigger.classList.toggle('is-active', !!open);
        trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    function positionFlyoutToTrigger(panel, trigger) {
        if (!panel || !trigger) return;

        var margin = 12;
        var triggerRect = trigger.getBoundingClientRect();

        /* Reset to allow clean measurement */
        panel.style.left = '0px';
        panel.style.top = '-9999px';
        panel.style.right = 'auto';
        panel.style.maxHeight = '';

        var panelWidth = Math.min(Math.max(340, 280), Math.max(280, window.innerWidth - margin * 2));
        panel.style.width = panelWidth + 'px';

        /* Horizontal: right-align to trigger button, clamped to viewport */
        var left = triggerRect.right - panelWidth;
        left = Math.max(margin, Math.min(left, window.innerWidth - panelWidth - margin));

        var spaceBelow = window.innerHeight - triggerRect.bottom - 8 - margin;
        var spaceAbove = triggerRect.top - 8 - margin;
        var shouldOpenUp = spaceBelow < 220 && spaceAbove > spaceBelow;

        var bodyEl = panel.querySelector('.rec-flyout-body');

        if (shouldOpenUp) {
            panel.setAttribute('data-opens-upward', '1');
            var bodyMaxH = Math.max(160, spaceAbove - 72);
            if (bodyEl) bodyEl.style.maxHeight = bodyMaxH + 'px';
            var topUp = Math.max(margin, triggerRect.top - (bodyMaxH + 80) - 8);
            panel.style.top = topUp + 'px';
        } else {
            panel.removeAttribute('data-opens-upward');
            var topDown = triggerRect.bottom + 8;
            var bodyMaxHDown = Math.max(160, spaceBelow - 72);
            if (bodyEl) bodyEl.style.maxHeight = bodyMaxHDown + 'px';
            panel.style.top = topDown + 'px';
        }

        panel.style.left = left + 'px';

        /* Caret: map trigger button center onto the panel's X axis */
        var caretX = triggerRect.left + triggerRect.width / 2 - left;
        caretX = Math.max(24, Math.min(caretX, panelWidth - 24));
        panel.style.setProperty('--caret-x', caretX + 'px');
    }

    function closeTransientPanels() {
        setFlyoutOpen(settingsPanel, settingsPanelBtn, false);
        setFlyoutOpen(toolsPanel, toolPanelBtn, false);
    }

    /* ── Setup renderer ── */
    RecognizeRenderer.setup(R);

    /* ── 后台静默预载全图（页面级共享 Image，不阻塞 UI） ── */
    R.prefetchDisplayMap();

    /* ── KB/s 统计 ── */
    var _kbBytes = 0, _kbAt = Date.now();
    function _trackBytes(n) {
        _kbBytes += n;
        var now = Date.now();
        if (now - _kbAt >= 1000) {
            var kbps = (_kbBytes / 1024 / ((now - _kbAt) / 1000)).toFixed(1);
            var el = document.getElementById('statusKbps');
            if (el) el.textContent = kbps + ' KB/s';
            _kbBytes = 0; _kbAt = now;
        }
    }

    /* ── 强制重置：清空后端+前端所有状态，防止死锁/停滞 ── */
    var _resetInFlight = false;
    async function forceReset() {
        if (_resetInFlight) return;
        _resetInFlight = true;
        try {
            var token = TC._ensureSessionToken();
            await fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: token }),
            });
        } catch (_) { /* 即使后端不可达也继续清前端 */ }
        // 前端状态重置
        R.mapState = null; R.mapStatePrev = null; R.mapHistory = [];
        R.renderPos = null;
        R.lastJpegImg = null; R.lastJpegPos = null;
        R.prevJpegImg = null; R.prevJpegPos = null;
        R.jpegFadeAlpha = 1;
        screenSending = false;
        S_nullBlobCount = 0;
        wsReconnectAt = 0;
        _kbBytes = 0; _kbAt = Date.now();
        // 重置状态 DOM
        statusStateText.textContent = '待机';
        screenStateChip.textContent = TC.isScreenActive ? '🖥️ 实时追踪中' : '📡 待机中';
        if (statusTransportChip) statusTransportChip.textContent = '识别中枢';
        ['statusX', 'statusY', 'statusQuality', 'statusFound', 'statusMatches',
         'statusAngle', 'statusHashIndex', 'statusTransport', 'statusKbps'].forEach(function (id) {
            var el = document.getElementById(id);
            if (el) el.textContent = '--';
        });
        var dotEl = document.getElementById('statusDot');
        if (dotEl) dotEl.className = 'status-dot dot-yellow';
        var dotLabel = document.getElementById('statusDotLabel');
        if (dotLabel) dotLabel.textContent = '等待分析';
        var arrowEl = document.getElementById('statusArrowState');
        if (arrowEl) arrowEl.textContent = '--';
        R.renderMapCanvas();
        TC.log('🔄 已强制重置所有识别状态');
        _resetInFlight = false;
    }

    /* ── TC init ── */
    TC.init({
        onAnalyzeResult: function (result) {
            if (result.status && result.status.position) {
                var st = result.status;
                var hasTrackPos = !!st.found || st.state === 'INERTIAL' || st.state === 'SCENE_CHANGE';
                var prevState = R.mapState;
                var nx = (st.position && typeof st.position.x === 'number') ? st.position.x : (prevState ? prevState.x : 0);
                var ny = (st.position && typeof st.position.y === 'number') ? st.position.y : (prevState ? prevState.y : 0);

                /* 未识别：不要把 (0,0) 写成新轨迹点，避免全图模式跳到左上角 */
                if (!hasTrackPos && prevState) {
                    nx = prevState.x;
                    ny = prevState.y;
                }

                R.mapStatePrev = R.mapState;
                R.mapState = {
                    x: nx, y: ny, angle: st.arrow_angle || 0,
                    stopped: !!st.arrow_stopped, found: !!st.found,
                    isInertial: st.state === 'INERTIAL', isSceneChange: st.state === 'SCENE_CHANGE',
                    ts: performance.now()
                };

                if (hasTrackPos) {
                    R.mapHistory.push(R.mapState);
                    if (R.mapHistory.length > 4) R.mapHistory.shift();
                } else {
                    /* 清空插值历史，避免 NOT_FOUND 期间继续外推形成错误轨迹 */
                    R.mapHistory = [];
                }

                R.modeStrategy.onPositionUpdate(R);
                R.startMapRAF();
            }
            if (result.image) { R.modeStrategy.onImageResult(result.image, R); }
            if (result.status) {
                TC.updateStatusDOM(result.status);
                var fmt = TC.formatStatus(result.status);
                statusStateText.textContent = fmt.stateText;
                screenStateChip.textContent = result.status.found ? '📡 已识别' : '📡 等待识别';
                /* 额外状态格 */
                var angleEl = document.getElementById('statusAngle');
                if (angleEl) angleEl.textContent = result.status.arrow_angle != null
                    ? Math.round(result.status.arrow_angle) + '°' : '--';
                var arrowEl = document.getElementById('statusArrowState');
                if (arrowEl) arrowEl.textContent = result.status.arrow_stopped ? '静止' : '移动中';
                var tpEl = document.getElementById('statusTransport');
                if (tpEl) tpEl.textContent = TC.wsConnected ? (TC.wsTransportName || 'WS') : 'HTTP';
                if (statusTransportChip) statusTransportChip.textContent = (TC.wsConnected ? (TC.wsTransportName || 'WS') : 'HTTP') + ' 通道';
                var hashEl = document.getElementById('statusHashIndex');
                if (hashEl) {
                    var src = result.status.source || '';
                    hashEl.textContent = src.indexOf('HASH') >= 0 ? '命中' : '--';
                }

                /* 连续未检测到小地图时，提示用户校准位置 */
                if (TC.isScreenActive && result.status.state === 'SCENE_CHANGE' && !result.status.found) {
                    _sceneChangeMissCount++;
                    if (_sceneChangeMissCount === 40 && Date.now() >= _sceneChangeWarnAt) {
                        _sceneChangeWarnAt = Date.now() + 30000;
                        TC.log('⚠️ 连续 40 帧未检测到小地图圆形。若游戏非全屏请打开「测试工具」→「自动定位小地图」进行校准，否则无法识别。');
                        AppCommon.toast('未检测到小地图，请点击右上角🧰→自动定位小地图', 'warning');
                    }
                } else {
                    _sceneChangeMissCount = 0;
                }
            }
        },
        onScreenStart: function () { TC.mode = 'screen'; updateScreenButtons(); },
        onScreenStop: function () {
            clearScreenAutoSchedulers();
            screenAutoRunning = false;
            TC.mode = 'file'; updateScreenButtons();
            /* 停止屏幕时也停止展示 */
            if (bcastName) stopBroadcast();
        }
    });

    /* ── Capability guards ── */
    function setButtonAvailability(btns, enabled, title) {
        (btns || []).forEach(function (b) {
            if (!b) return;
            b.disabled = !enabled; b.title = title || '';
            b.style.opacity = enabled ? '' : '0.55';
            b.style.cursor = enabled ? '' : 'not-allowed';
        });
    }

    function getPiPSupport() {
        var c = document.getElementById('pipCanvas'), v = document.getElementById('pipVideo');
        var cs = !!(c && typeof c.captureStream === 'function');
        var std = !!(cs && v && document.pictureInPictureEnabled && typeof v.requestPictureInPicture === 'function');
        var wk = !!(cs && v && typeof v.webkitSupportsPresentationMode === 'function' && v.webkitSupportsPresentationMode('picture-in-picture'));
        return { any: std || wk, standard: std, webkit: wk, canStream: cs };
    }

    function applyCapabilityGuards() {
        var diag = TC.getConnectionDiagnostics(), pip = getPiPSupport();
        setButtonAvailability([mainScreenBtn], diag.screenCapture,
            diag.screenCapture ? '开始屏幕捕获' : '当前浏览器不支持屏幕捕获');
        setButtonAvailability([R.pipBtn], pip.any,
            pip.any ? '打开浏览器画中画' : '当前浏览器不支持原生画中画');
        setButtonAvailability([broadcastBtn], diag.socketIO,
            diag.socketIO ? '展示当前画面给观众' : '当前页面未加载 Socket.IO 客户端');
    }

    /* ── Screen buttons state ── */
    function updateScreenButtons() {
        if (!TC.isScreenActive) {
            mainScreenBtn.classList.remove('is-active');
            mainScreenBtnSub.textContent = '开始追踪';
            screenStateChip.textContent = '📡 待机中';
            return;
        }
        mainScreenBtn.classList.add('is-active');
        mainScreenBtnSub.textContent = screenAutoRunning ? '点击暂停追踪' : '再按停止捕获';
        screenStateChip.textContent = screenAutoRunning ? '🖥️ 实时追踪中' : '🖥️ 已捕获，暂停中';
    }

    /* ── Screen capture ── */
    async function startScreenCapture() {
        return TC.startScreenCapture({ offscreenVidId: 'screenOffscreen' });
    }

    /** 等待屏幕视频流元数据就绪（videoWidth > 0），最多等 3 秒 */
    function waitForVideoReady(timeoutMs) {
        var vid = document.getElementById('screenOffscreen');
        return new Promise(function (resolve) {
            if (vid && vid.videoWidth > 0) { resolve(true); return; }
            var elapsed = 0, poll = 50;
            var timer = setInterval(function () {
                elapsed += poll;
                if ((vid && vid.videoWidth > 0) || elapsed >= (timeoutMs || 3000)) {
                    clearInterval(timer);
                    resolve(!!(vid && vid.videoWidth > 0));
                }
            }, poll);
        });
    }

    async function captureAndSendScreenFrame() {
        if (!TC.isScreenActive || screenSending) return;
        screenSending = true;
        try {
            if (useWSMode && TC.wsConnected) {
                var blob = await TC.captureScreenImgBlob();
                if (blob) {
                    S_nullBlobCount = 0;
                    _trackBytes(blob.size);
                    await TC.sendViaWS(blob);
                } else {
                    S_nullBlobCount++;
                    if (S_nullBlobCount >= 5 && Date.now() >= nullBlobWarnAt) {
                        nullBlobWarnAt = Date.now() + 5000;
                        TC.log('⚠️ 截图连续失败(' + S_nullBlobCount + '帧)，请检查屏幕流是否正常');
                    }
                }
            } else if (useWSMode && !TC.wsConnected) {
                var fd = TC.captureScreenImg();
                if (fd) { TC.imageData = fd; await TC.sendAndDisplay(fd); }
                if (Date.now() >= wsReconnectAt) {
                    wsReconnectAt = Date.now() + 3000;
                    TC.connectWS()
                        .then(function () { bindRecognizeSocketLifecycle(); TC.log('📶 WS 重连成功'); })
                        .catch(function () { TC.log('⚠️ WS 重连失败'); });
                }
            } else {
                var hd = TC.captureScreenImg();
                if (hd) { TC.imageData = hd; await TC.sendAndDisplay(hd); }
            }
        } catch (e) { console.error(e); } finally { screenSending = false; }
    }

    var SCREEN_INTERVAL_MS = Math.round(1000 / (prefs.captureFps || 10));

    /* ── FPS slider ── */
    var fpsRange = document.getElementById('fpsRange');
    var fpsRangeVal = document.getElementById('fpsRangeVal');
    var renderFpsRange = document.getElementById('renderFpsRange');
    var renderFpsRangeVal = document.getElementById('renderFpsRangeVal');
    if (fpsRange) {
        fpsRange.value = prefs.captureFps || 10;
        if (fpsRangeVal) fpsRangeVal.textContent = fpsRange.value;
        fpsRange.addEventListener('input', function () {
            var fps = parseInt(fpsRange.value, 10) || 10;
            if (fpsRangeVal) fpsRangeVal.textContent = fps;
            SCREEN_INTERVAL_MS = Math.round(1000 / fps);
            BCAST_INTERVAL_MS = SCREEN_INTERVAL_MS;
            AppCommon.updatePref('captureFps', fps);
        });
    }
    if (renderFpsRange) {
        renderFpsRange.value = R.renderFps;
        if (renderFpsRangeVal) renderFpsRangeVal.textContent = renderFpsRange.value;
        renderFpsRange.addEventListener('input', function () {
            var fps = parseInt(renderFpsRange.value, 10) || 60;
            if (renderFpsRangeVal) renderFpsRangeVal.textContent = fps;
            if (typeof R.setRenderFps === 'function') R.setRenderFps(fps);
            AppCommon.updatePref('renderFps', fps);
        });
    }
    if (recognizeNameInput) {
        recognizeNameInput.value = prefs.bcastName || '';
        recognizeNameInput.addEventListener('input', function () {
            AppCommon.updatePref('bcastName', recognizeNameInput.value);
        });
    }

    /* ── Sidebar collapse ── */
    var recBody = document.querySelector('.rec-body');
    var panelCollapseBtn = document.getElementById('panelCollapseBtn');
    if (panelCollapseBtn && recBody) {
        AppCommon.bindPersistentToggleState({
            storageKey: 'recognize_panel_collapsed',
            trigger: panelCollapseBtn,
            applyValue: function (collapsed) {
                recBody.classList.toggle('rec-panel-collapsed', collapsed);
                panelCollapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
                panelCollapseBtn.title = collapsed ? '展开侧栏' : '收起侧栏';
            }
        });
    }

    function clearScreenAutoSchedulers() {
        if (screenAutoTimeoutId) {
            clearTimeout(screenAutoTimeoutId);
            screenAutoTimeoutId = null;
        }
        if (screenAutoRafId) {
            cancelAnimationFrame(screenAutoRafId);
            screenAutoRafId = null;
        }
    }

    function runScreenAutoLoop() {
        if (!screenAutoRunning || TC.mode !== 'screen') return;
        if (!document.hidden || R.pipRAFActive) captureAndSendScreenFrame();
        if (document.hidden && R.pipRAFActive) {
            clearScreenAutoSchedulers();
            screenAutoTimeoutId = setTimeout(runScreenAutoLoop, SCREEN_INTERVAL_MS);
            return;
        }
        clearScreenAutoSchedulers();
        screenAutoTimeoutId = setTimeout(function () {
            screenAutoRafId = requestAnimationFrame(function () {
                screenAutoRafId = null;
                runScreenAutoLoop();
            });
        }, SCREEN_INTERVAL_MS);
    }

    async function toggleScreenAutoTrack() {
        screenAutoRunning = !screenAutoRunning;
        if (screenAutoRunning) {
            if (useWSMode && !TC.wsConnected) {
                try {
                    await TC.connectWS();
                    bindRecognizeSocketLifecycle();
                }
                catch (_) { useWSMode = false; TC.log('⚠️ WS 连接失败，回退到 HTTP 模式'); }
            }
            TC.log('📍 屏幕自动追踪已开启');
            runScreenAutoLoop();
        } else {
            clearScreenAutoSchedulers();
            TC.log('⏸ 屏幕自动追踪已暂停');
        }
        updateScreenButtons();
    }

    async function handleMainScreenAction() {
        if (!TC.isScreenActive) {
            var ok = await startScreenCapture();
            if (ok) {
                await forceReset();
                /* 等待视频流就绪后再开始捕获循环，避免前几帧全部空帧 */
                var ready = await waitForVideoReady(3000);
                if (!ready) {
                    TC.log('⚠️ 视频流未就绪，将在就绪后自动开始识别');
                }
                if (!screenAutoRunning) await toggleScreenAutoTrack();
            }
            return;
        }
        if (TC.mode !== 'screen') TC.mode = 'screen';
        if (screenAutoRunning) { await toggleScreenAutoTrack(); } else { TC.stopScreenCapture(); }
    }

    /* ══════════════════════════════════════════
       Broadcast (Presenter) logic
    ══════════════════════════════════════════ */

    /* 截取 resultCanvas → JPEG blob，复用 bcastCanvas */
    function captureBcastBlob() {
        var src = R.resultCanvas;
        if (!src || !src.width) return Promise.resolve(null);
        if (bcastCanvas.width !== src.width || bcastCanvas.height !== src.height) {
            bcastCanvas.width = src.width;
            bcastCanvas.height = src.height;
        }
        var ctx = bcastCanvas.getContext('2d');
        if (!ctx) return Promise.resolve(null);
        ctx.drawImage(src, 0, 0);
        return new Promise(function (resolve) {
            if (typeof bcastCanvas.toBlob === 'function') {
                bcastCanvas.toBlob(resolve, 'image/jpeg', 0.75);
            } else {
                try {
                    var b64 = bcastCanvas.toDataURL('image/jpeg', 0.75);
                    /* Minimal base64 → Blob */
                    var parts = b64.split(',');
                    var bin = atob(parts[1]);
                    var arr = new Uint8Array(bin.length);
                    for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                    resolve(new Blob([arr], { type: 'image/jpeg' }));
                } catch (e) { resolve(null); }
            }
        });
    }

    function runBcastLoop() {
        if (!bcastName) return;
        if (!TC.wsConnected || !TC.wsSocket) {
            bcastAutoId = null;
            return;
        }
        if (bcastViewerCount <= 0) {
            bcastAutoId = null;
            return;
        }
        if (!bcastSending && !document.hidden) {
            bcastSending = true;
            captureBcastBlob().then(function (blob) {
                bcastSending = false;
                if (blob && TC.wsConnected && bcastName && bcastViewerCount > 0) {
                    TC.wsSocket.emit('broadcast_frame', blob);
                }
            }).catch(function () { bcastSending = false; });
        }
        bcastAutoId = setTimeout(runBcastLoop, BCAST_INTERVAL_MS);
    }

    async function startBroadcast(name) {
        if (bcastStarting || bcastName) return;
        bcastStarting = true;
        try {
            if (!TC.wsConnected) {
                await TC.connectWS();
            }
            bindRecognizeSocketLifecycle();

            var ack = await new Promise(function (resolve, reject) {
                var sock = TC.wsSocket;
                if (!sock) { reject(new Error('WS 未就绪')); return; }
                var settled = false;
                var onJoined = function (data) {
                    if (settled) return;
                    settled = true;
                    clearTimeout(timer);
                    sock.off('broadcast_joined', onJoined);
                    resolve(data || { ok: false, error: '空响应' });
                };
                var timer = setTimeout(function () {
                    if (settled) return;
                    settled = true;
                    sock.off('broadcast_joined', onJoined);
                    reject(new Error('等待展示确认超时，请重试'));
                }, 5000);

                sock.on('broadcast_joined', onJoined);
                sock.emit('broadcast_join', { name: name });
            });

            if (!ack.ok) {
                AppCommon.toast('展示失败：' + (ack.error || '未知错误'), 'danger');
                return;
            }

            bcastName = ack.name;
            bcastViewerCount = 0;
            setBroadcastButtonState(true, bcastName);
            watchBadge.textContent = '📡 展示中：' + bcastName;
            watchBadge.classList.add('is-visible');
            AppCommon.toast('已开始展示「' + bcastName + '」', 'success');
            TC.log('📡 展示已开始：' + bcastName);
            runBcastLoop();
        } catch (e) {
            AppCommon.toast('WS 连接失败：' + e.message, 'danger');
        } finally {
            bcastStarting = false;
        }
    }

    function stopBroadcast() {
        if (!bcastName) return;
        if (bcastAutoId) { clearTimeout(bcastAutoId); bcastAutoId = null; }
        bcastSending = false;
        if (TC.wsConnected && TC.wsSocket) {
            TC.wsSocket.emit('broadcast_leave', { name: bcastName });
        }
        TC.log('📡 展示已停止：' + bcastName);
        bcastName = null;
        bcastViewerCount = 0;
        setBroadcastButtonState(false, '');
        watchBadge.classList.remove('is-visible');
    }

    function resetBroadcastLocalState(reason) {
        if (bcastAutoId) { clearTimeout(bcastAutoId); bcastAutoId = null; }
        bcastSending = false;
        bcastStarting = false;
        if (!bcastName) return;
        TC.log('📡 展示已中断' + (reason ? '：' + reason : ''));
        bcastName = null;
        bcastViewerCount = 0;
        setBroadcastButtonState(false, '');
        watchBadge.classList.remove('is-visible');
    }

    function bindRecognizeSocketLifecycle() {
        var sock = TC.wsSocket;
        if (!sock || sock.__recognizeLifecycleBound) return;
        sock.__recognizeLifecycleBound = true;
        sock.on('broadcast_list', function (data) {
            var rooms = (data && Array.isArray(data.rooms)) ? data.rooms : [];
            var prevViewerCount = bcastViewerCount;
            if (!bcastName) {
                bcastViewerCount = 0;
                return;
            }
            var ownRoom = null;
            for (var i = 0; i < rooms.length; i++) {
                if (rooms[i] && rooms[i].name === bcastName) {
                    ownRoom = rooms[i];
                    break;
                }
            }
            bcastViewerCount = ownRoom ? Math.max(0, Number(ownRoom.viewers || 0)) : 0;
            if (bcastViewerCount <= 0 && bcastAutoId) {
                clearTimeout(bcastAutoId);
                bcastAutoId = null;
                bcastSending = false;
            } else if (bcastViewerCount > 0 && prevViewerCount <= 0 && !bcastAutoId) {
                runBcastLoop();
            }
        });
        sock.on('disconnect', function () {
            resetBroadcastLocalState('连接断开');
        });
    }

    broadcastBtn.addEventListener('click', function () {
        if (bcastName) { stopBroadcast(); return; }
        if (!TC.isScreenActive) { AppCommon.toast('请先开始屏幕捕获', 'danger'); return; }
        var name = AppCommon.loadPrefs().bcastName || '';
        name = name.trim();
        if (!name) {
            setFlyoutOpen(settingsPanel, settingsPanelBtn, true);
            if (recognizeNameInput) recognizeNameInput.focus();
            AppCommon.toast('请先在设置中填写展示名称', 'warning');
            return;
        }
        startBroadcast(name);
    });

    /* ══════════════════════════════════════════
       Tab switching
    ══════════════════════════════════════════ */
    var tabTrack    = document.getElementById('tabTrack');
    var tabAudience = document.getElementById('tabAudience');
    var pageTrack    = document.getElementById('pageTrack');
    var pageAudience = document.getElementById('pageAudience');

    function switchTab(tab) {
        var toAudience = tab === 'audience';
        var wasAudience = !pageAudience.classList.contains('rec-page-hidden');
        localStorage.setItem('recognize_active_tab', toAudience ? 'audience' : 'track');
        tabTrack.classList.toggle('is-active', !toAudience);
        tabAudience.classList.toggle('is-active', toAudience);
        tabTrack.setAttribute('aria-pressed', toAudience ? 'false' : 'true');
        tabAudience.setAttribute('aria-pressed', toAudience ? 'true' : 'false');
        pageTrack.classList.toggle('rec-page-hidden', toAudience);
        pageAudience.classList.toggle('rec-page-hidden', !toAudience);
        if (wasAudience && !toAudience && typeof RecognizeAudience !== 'undefined') {
            RecognizeAudience.onLeave();
        }
        if (toAudience && typeof RecognizeAudience !== 'undefined') {
            RecognizeAudience.onEnter();
        }
    }

    function syncRecognizeTab() {
        switchTab(localStorage.getItem('recognize_active_tab') === 'audience' ? 'audience' : 'track');
    }

    function restoreRecognizePageAfterHistoryNavigation() {
        syncRecognizeTab();
        applyCapabilityGuards();
        updateScreenButtons();

        if (settingsPanel && settingsPanel.classList.contains('is-open')) {
            positionFlyoutToTrigger(settingsPanel, settingsPanelBtn);
        }
        if (toolsPanel && toolsPanel.classList.contains('is-open')) {
            positionFlyoutToTrigger(toolsPanel, toolPanelBtn);
        }

        if (R.resultCanvas && (R.resultCanvas.width < 64 || R.resultCanvas.height < 64)) {
            R.resultCanvas.width = 400;
            R.resultCanvas.height = 400;
        }
        if (bcastCanvas && (bcastCanvas.width < 64 || bcastCanvas.height < 64)) {
            bcastCanvas.width = 400;
            bcastCanvas.height = 400;
        }

        if (R.renderMapCanvas) R.renderMapCanvas();
        if (R.mapState && !R.mapRAFActive) R.startMapRAF();

        if (screenAutoRunning && TC.mode === 'screen' && !screenAutoTimeoutId && !screenAutoRafId) runScreenAutoLoop();
        if (bcastName && !bcastAutoId) runBcastLoop();
    }

    tabTrack.addEventListener('click', function () { switchTab('track'); });
    tabAudience.addEventListener('click', function () { switchTab('audience'); });
    AppCommon.bindPageResumeLifecycle({
        rafPasses: 2,
        onResume: function () {
            restoreRecognizePageAfterHistoryNavigation();
            bindRecognizeSocketLifecycle();
        }
    });

    /* 观众模式刷新按钮 */
    var audienceRefreshBtn = document.getElementById('audienceRefreshBtn');
    if (audienceRefreshBtn) {
        audienceRefreshBtn.addEventListener('click', function () {
            if (typeof RecognizeAudience !== 'undefined') RecognizeAudience.refresh();
        });
    }

    /* ── Event bindings ── */
    document.getElementById('renderModeBtn').addEventListener('click', R.toggleRenderMode);
    document.getElementById('mainScreenBtn').addEventListener('click', handleMainScreenAction);
    document.getElementById('pipBtn').addEventListener('click', R.toggleNativePiP);
    document.getElementById('forceResetBtn').addEventListener('click', forceReset);
    if (settingsPanelBtn) {
        AppCommon.setInteractiveHiddenState(settingsPanel, true);
        settingsPanelBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            var next = !settingsPanel.classList.contains('is-open');
            setFlyoutOpen(settingsPanel, settingsPanelBtn, next);
            if (next) setFlyoutOpen(toolsPanel, toolPanelBtn, false);
        });
    }
    if (toolPanelBtn) {
        AppCommon.setInteractiveHiddenState(toolsPanel, true);
        toolPanelBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            var next = !toolsPanel.classList.contains('is-open');
            setFlyoutOpen(toolsPanel, toolPanelBtn, next);
            if (next) setFlyoutOpen(settingsPanel, settingsPanelBtn, false);
        });
    }
    var settingsPanelCloseBtn = document.getElementById('settingsPanelCloseBtn');
    var toolsPanelCloseBtn = document.getElementById('toolsPanelCloseBtn');
    if (settingsPanelCloseBtn) settingsPanelCloseBtn.addEventListener('click', function () { setFlyoutOpen(settingsPanel, settingsPanelBtn, false); });
    if (toolsPanelCloseBtn) toolsPanelCloseBtn.addEventListener('click', function () { setFlyoutOpen(toolsPanel, toolPanelBtn, false); });
    document.addEventListener('click', function (e) {
        var target = e.target;
        var inSettings = settingsPanel && settingsPanel.contains(target);
        var inTools = toolsPanel && toolsPanel.contains(target);
        var onSettingsBtn = settingsPanelBtn && settingsPanelBtn.contains(target);
        var onToolsBtn = toolPanelBtn && toolPanelBtn.contains(target);
        if (!inSettings && !inTools && !onSettingsBtn && !onToolsBtn) closeTransientPanels();
    });
    window.addEventListener('resize', function () {
        if (settingsPanel && settingsPanel.classList.contains('is-open')) {
            positionFlyoutToTrigger(settingsPanel, settingsPanelBtn);
        }
        if (toolsPanel && toolsPanel.classList.contains('is-open')) {
            positionFlyoutToTrigger(toolsPanel, toolPanelBtn);
        }
    });

    /* ── 图片上传测试 ── */
    var uploadFileInput = document.getElementById('uploadFileInput');
    if (uploadFileInput) {
        /**
         * 从游戏全屏截图中裁剪小地图区域
         * 使用与 captureScreenImg 相同的 selCircle 参数
         * @param {HTMLImageElement|HTMLCanvasElement} source
         * @returns {string} dataURL (JPEG)
         */
        function cropMinimapFromImage(source) {
            var vw = source.naturalWidth || source.width;
            var vh = source.naturalHeight || source.height;
            var sc = TC.selCircle;
            var bs = Math.min(vw, vh);
            var cx = sc.cx * vw, cy = sc.cy * vh, r = sc.r * bs;
            var margin = 1.4;
            var sz = Math.max(10, Math.round(r * 2 * margin));
            var rx = Math.max(0, Math.min(Math.round(cx - sz / 2), vw - sz));
            var ry = Math.max(0, Math.min(Math.round(cy - sz / 2), vh - sz));
            var c = document.createElement('canvas');
            c.width = sz; c.height = sz;
            var ctx = c.getContext('2d');
            ctx.drawImage(source, rx, ry, sz, sz, 0, 0, sz, sz);
            return c.toDataURL('image/jpeg', 0.82);
        }

        /** 将整图重编码为 dataURL，作为原图直传失败时的降噪重试 */
        function encodeFullImage(source) {
            var vw = source.naturalWidth || source.width;
            var vh = source.naturalHeight || source.height;
            var c = document.createElement('canvas');
            c.width = vw;
            c.height = vh;
            var ctx = c.getContext('2d');
            ctx.drawImage(source, 0, 0, vw, vh, 0, 0, vw, vh);
            return c.toDataURL('image/jpeg', 0.90);
        }

        async function recognizeByDataUrl(dataUrl) {
            var resp = await fetch('/api/recognize_single', {
                method: 'POST',
                body: JSON.stringify({ image: dataUrl }),
                headers: { 'Content-Type': 'application/json' }
            });
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        }

        async function recognizeByFile(file) {
            var fd = new FormData();
            fd.append('image', file, file.name || 'upload_image');
            var resp = await fetch('/api/recognize_single', {
                method: 'POST',
                body: fd
            });
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        }

        function isGlobalFeatureSource(src) {
            return src.indexOf('SIFT_GLOBAL_') === 0 || src.indexOf('ORB_GLOBAL_') === 0;
        }

        function getStatusScore(st) {
            if (!st) return -1;
            var q = Number(st.match_quality || 0);
            var m = Number(st.matches || 0);
            var src = String(st.source || '');
            var score = q + Math.min(80, m) / 180.0;
            if (isGlobalFeatureSource(src)) score += 0.20;
            else if (src.indexOf('HASH_ECC_') === 0) score += 0.02;
            else if (src.indexOf('HASH_INDEX_') === 0) score -= 0.25;
            return score;
        }

        function isReliableFound(st) {
            if (!(st && st.found)) return false;
            var q = Number(st.match_quality || 0);
            var m = Number(st.matches || 0);
            var src = String(st.source || '');

            // 纯 hash 粗定位：只作为候选，不作为最终命中终止条件
            if (src.indexOf('HASH_INDEX_') === 0) return false;

            // hash+ECC：需要更高质量才能提前接受
            if (src.indexOf('HASH_ECC_') === 0) {
                return q >= 0.35;
            }

            // 全局特征主路径（SIFT/ORB）：放宽但保持可信门槛
            if (isGlobalFeatureSource(src)) {
                return (m >= 8 && q >= 0.20) || (m >= 20 && q >= 0.12) || q >= 0.55;
            }

            return q >= 0.30 || m >= 12;
        }

        /** 加载一个 File 为 Image */
        function loadFileAsImage(file) {
            return new Promise(function (resolve, reject) {
                var reader = new FileReader();
                reader.onload = function () {
                    var img = new Image();
                    img.onload = function () { resolve(img); };
                    img.onerror = function () { reject(new Error('图片解码失败')); };
                    img.src = reader.result;
                };
                reader.onerror = function () { reject(new Error('文件读取失败')); };
                reader.readAsDataURL(file);
            });
        }

        /** 处理单个文件上传 → 裁剪 → 无状态识别 */
        async function processUploadedFile(file, idx, total) {
            try {
                var label = total > 1 ? '[' + (idx + 1) + '/' + total + '] ' : '';
                TC.log('📂 ' + label + '加载图片: ' + file.name);
                var img = await loadFileAsImage(file);
                var attempts = [];
                var result = null;
                var st = null;

                TC.log('🧭 先尝试整图自动定位识别 (' + img.naturalWidth + 'x' + img.naturalHeight + ')');
                result = await recognizeByFile(file);
                st = result && result.status;
                attempts.push({ stage: 'file', result: result });

                if (!(st && isReliableFound(st))) {
                    TC.log('🧪 原图直传未形成高置信命中，尝试整图重编码后再识别');
                    result = await recognizeByDataUrl(encodeFullImage(img));
                    st = result && result.status;
                    attempts.push({ stage: 'full-jpeg', result: result });
                }

                if (!(st && isReliableFound(st))) {
                    TC.log('📐 前两次仍低置信，回退到固定裁剪参数重试');
                    var dataUrl = cropMinimapFromImage(img);
                    result = await recognizeByDataUrl(dataUrl);
                    st = result && result.status;
                    attempts.push({ stage: 'crop-82', result: result });
                }

                // 三段尝试后择优：优先高置信，其次分数最高
                var bestReliable = null;
                var bestAny = null;
                attempts.forEach(function (a) {
                    var s = a.result && a.result.status;
                    if (!(s && s.found)) return;
                    var score = getStatusScore(s);
                    if (!bestAny || score > bestAny.score) bestAny = { score: score, item: a };
                    if (isReliableFound(s) && (!bestReliable || score > bestReliable.score)) {
                        bestReliable = { score: score, item: a };
                    }
                });

                if (bestReliable) {
                    result = bestReliable.item.result;
                    TC.log('🎯 ' + label + '采用最优高置信结果，阶段: ' + bestReliable.item.stage);
                } else if (bestAny) {
                    // 只有低置信候选时，不把粗定位当最终命中，避免“识别到但坐标错”
                    var only = bestAny.item.result.status;
                    TC.log('⚠️ ' + label + '仅得到低置信候选（阶段: ' + bestAny.item.stage + '，来源:' + (only.source || '--') + '），本次按未命中处理');
                    result = {
                        success: false,
                        status: {
                            state: 'LOW_CONFIDENCE',
                            found: false,
                            position: { x: 0, y: 0 },
                            matches: Number(only.matches || 0),
                            match_quality: Number(only.match_quality || 0),
                            mode: 'sift',
                            source: only.source || ''
                        }
                    };
                }

                // 手动更新前端 UI（复用 onAnalyzeResult 的逻辑）
                if (result && result.status) {
                    var st = result.status;
                    var hasTrackPos = !!st.found || st.state === 'INERTIAL' || st.state === 'SCENE_CHANGE';
                    var prevState = R.mapState;

                    R.mapStatePrev = R.mapState;

                    if (!hasTrackPos && !prevState) {
                        /* 上传测试首张即未识别：不制造伪坐标，保持等待态 */
                        R.mapState = null;
                        R.mapHistory = [];
                    } else {
                        var nx = (st.position && typeof st.position.x === 'number') ? st.position.x : (prevState ? prevState.x : 0);
                        var ny = (st.position && typeof st.position.y === 'number') ? st.position.y : (prevState ? prevState.y : 0);
                        if (!hasTrackPos && prevState) {
                            nx = prevState.x;
                            ny = prevState.y;
                        }
                        R.mapState = {
                            x: nx, y: ny, angle: st.arrow_angle || 0,
                            stopped: true, found: !!st.found,
                            isInertial: st.state === 'INERTIAL', isSceneChange: st.state === 'SCENE_CHANGE',
                            ts: performance.now()
                        };

                        if (hasTrackPos) {
                            R.mapHistory.push(R.mapState);
                            if (R.mapHistory.length > 4) R.mapHistory.shift();
                        } else {
                            R.mapHistory = [];
                        }
                    }

                    R.modeStrategy.onPositionUpdate(R);
                    R.startMapRAF();
                    if (result.image) R.modeStrategy.onImageResult(result.image, R);
                    TC.updateStatusDOM(st);
                    if (st.found) {
                        TC.log('✅ ' + label + '识别成功: (' + st.position.x + ', ' + st.position.y + ') 匹配:' + st.matches + ' 质量: ' + Math.round((st.match_quality || 0) * 100) + '% 来源:' + (st.source || '--'));
                    } else {
                        TC.log('❌ ' + label + '未识别到位置, 状态: ' + st.state);
                    }
                } else {
                    TC.log('⚠️ ' + label + '识别失败: ' + (result.error || '未知错误'));
                }
            } catch (err) {
                TC.log('⚠️ 处理 ' + file.name + ' 失败: ' + err.message);
            }
        }

        uploadFileInput.addEventListener('change', async function () {
            var files = Array.from(uploadFileInput.files || []);
            if (!files.length) return;
            uploadFileInput.value = '';  // 允许重复选择相同文件
            TC.mode = 'file';
            TC.log('📂 已选择 ' + files.length + ' 张图片，使用无状态识别模式');

            for (var i = 0; i < files.length; i++) {
                await processUploadedFile(files[i], i, files.length);
                if (files.length > 1 && i < files.length - 1) {
                    await new Promise(function (r) { setTimeout(r, 200); }); // 批量间隔
                }
            }
            TC.log('✅ 所有图片处理完成');
        });
    }

    /* ── 自动定位小地图圆形（从屏幕捕获帧中检测） ── */
    var autoDetectCircleBtn = document.getElementById('autoDetectCircleBtn');
    var autoDetectCircleSub = document.getElementById('autoDetectCircleSub');
    var resetCircleBtn = document.getElementById('resetCircleBtn');
    var circleCalibStatus = document.getElementById('circleCalibStatus');

    function _updateCircleCalibStatus() {
        if (!circleCalibStatus) return;
        var sc = TC.selCircle;
        var DEF_CX = (1189 + 62.5) / 1362, DEF_CY = (66 + 63.5) / 806;
        var isDefault = Math.abs(sc.cx - DEF_CX) < 0.001 && Math.abs(sc.cy - DEF_CY) < 0.001;
        circleCalibStatus.style.display = '';
        circleCalibStatus.textContent = isDefault
            ? '📍 当前：内置默认坐标（cx=' + sc.cx.toFixed(3) + ', cy=' + sc.cy.toFixed(3) + '）'
            : '✅ 已校准：cx=' + sc.cx.toFixed(3) + ', cy=' + sc.cy.toFixed(3) + ', r=' + sc.r.toFixed(3);
        circleCalibStatus.style.color = isDefault ? 'var(--warning,#d28b36)' : 'var(--success,#4f8a4b)';
    }
    _updateCircleCalibStatus();

    if (autoDetectCircleBtn) {
        autoDetectCircleBtn.addEventListener('click', async function () {
            if (!TC.isScreenActive) {
                AppCommon.toast('请先点击"屏幕捕获"开始捕获屏幕', 'warning');
                return;
            }
            autoDetectCircleBtn.disabled = true;
            if (autoDetectCircleSub) autoDetectCircleSub.textContent = '正在截图并检测...';

            try {
                /* 截取全帧（不裁剪 selCircle），用于小地图位置检测 */
                var blob = await TC.captureFullFrameBlob(0.85);
                if (!blob) throw new Error('屏幕流未就绪，请等待画面出现后再试');

                var fd = new FormData();
                fd.append('image', blob, 'fullscreen.jpg');
                var resp = await fetch('/api/detect_minimap_circle', { method: 'POST', body: fd });
                var data = await resp.json();

                if (!data.ok) {
                    AppCommon.toast('🔍 ' + (data.error || '未检测到小地图'), 'danger');
                    if (autoDetectCircleSub) autoDetectCircleSub.textContent = '检测失败，请确保游戏小地图可见';
                    return;
                }

                TC.selCircle = { cx: data.cx, cy: data.cy, r: data.r };
                TC.saveSelCircle();
                TC.captureCanvas = null;  // 作废旧尺寸 canvas

                var pct = Math.round(data.confidence * 100);
                AppCommon.toast('✅ 小地图已校准（置信度 ' + pct + '%）', 'success');
                TC.log('🔍 小地图自动定位成功: cx=' + data.cx.toFixed(3) + ', cy=' + data.cy.toFixed(3) + ', r=' + data.r.toFixed(3) + '（置信度 ' + pct + '%，布局 ' + (data.layout || '--') + '）');
                if (autoDetectCircleSub) autoDetectCircleSub.textContent = '已校准 ✅ 置信度 ' + pct + '%';
                _updateCircleCalibStatus();
            } catch (err) {
                AppCommon.toast('检测异常: ' + err.message, 'danger');
                TC.log('⚠️ 自动定位小地图失败: ' + err.message);
                if (autoDetectCircleSub) autoDetectCircleSub.textContent = '从屏幕捕获中自动识别小地图圆形位置';
            } finally {
                autoDetectCircleBtn.disabled = false;
            }
        });
    }

    if (resetCircleBtn) {
        resetCircleBtn.addEventListener('click', function () {
            TC.resetSelCircle();
            AppCommon.toast('已重置为内置默认坐标', 'success');
            TC.log('↩️ selCircle 已重置为默认值');
            if (autoDetectCircleSub) autoDetectCircleSub.textContent = '从屏幕捕获中自动识别小地图圆形位置';
            _updateCircleCalibStatus();
        });
    }

    /* ── 页面卸载（刷新/关闭）时停止捕获 ── */
    window.addEventListener('beforeunload', function () {
        if (R.cleanupPiP) R.cleanupPiP();
        TC.stopScreenCapture();
    });
    window.addEventListener('pagehide', function (e) {
        /* persisted = bfcache 缓存，不属于真正卸载，不停止捕获 */
        if (e.persisted) return;
        R.mapRAFActive = false;
        R.pipRAFActive = false;
        if (R.cleanupPiP) R.cleanupPiP();
        if (bcastName) stopBroadcast();
        if (typeof RecognizeAudience !== 'undefined') RecognizeAudience.onLeave();
        if (typeof R.releaseDisplayMap === 'function') R.releaseDisplayMap();
    });

    /* ── Boot ── */
    applyCapabilityGuards();
    syncRecognizeTab();
    R.updateRenderModeBtn();
    R.renderMapCanvas();
    updateScreenButtons();
    setBroadcastButtonState(false, '');
    bindRecognizeSocketLifecycle();
    TC.log('系统就绪：识别台已加载');
