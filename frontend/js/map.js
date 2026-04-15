/* map.js — Shared state, viewport, player, events, boot */
import * as AppCommon from './common.js';
import { MapMarkers } from './map-markers.js';
import { MapRoute } from './map-route.js';

AppCommon.buildDock();

const prefs = AppCommon.loadPrefs();

    /* ── Shared application state ── */
    var A = {
        prefs: prefs,
        MC: document.getElementById('mc'),
        MW: document.getElementById('mw'),
        MI: document.getElementById('mi'),
        mapImageUrl: '/img/map/map_z7.webp',
        MARKER_CANVAS: document.getElementById('markerCanvas'),
        markerCtx: document.getElementById('markerCanvas').getContext('2d'),
        PLAYER_DOT: document.getElementById('playerDot'),
        PLAYER_ARROW: document.getElementById('playerArrow'),
        PLAYER_ARROW_POLY: document.getElementById('playerArrowPoly'),
        PLAYER_ARROW_CIRCLE: document.getElementById('playerArrowCircle'),

        mapInfo: null,
        scale: 1, offsetX: 0, offsetY: 0,
        dragging: false, dragStartX: 0, dragStartY: 0, dragOriginX: 0, dragOriginY: 0,
        latestX: null, latestY: null,
        markers: [], markerDetails: {}, markerDetailRequests: Object.create(null),
        categories: {}, markerTypeCounts: {},
        activeTypes: new Set(), selectedMarkerId: null,
        categoryColors: {}, markerIconCache: Object.create(null),
        markersById: Object.create(null), categoryToggleNodes: [],
        renderedMarkerHits: [], markerRenderQueued: false,
        markerDataVersion: null, markerChunkSize: 0,
        totalMarkerCount: 0, totalChunkCount: 0,
        availableMarkerChunks: new Set(),
        loadedMarkerChunks: new Set(), pendingMarkerChunkKeys: new Set(),
        prefetchingMarkerChunkKeys: new Set(), markerLoadGeneration: 0,
        searchIndex: [], searchIndexById: Object.create(null),
        searchIndexLoaded: false, searchIndexLoading: false, searchIndexPromise: null,
        searchIndexWarmPromise: null,
        mapFiltersStorageKey: 'game-map-web:web:map-filters',
        playerLerp: { tx: 0, ty: 0, cx: 0, cy: 0, running: false },
        dragMoved: false, isDragging: false,

        TELEPORT_TYPE: '17310030038',
        routeWaypoints: [], routeResult: null,
        boxSelectEl: null, boxStart: null,
        isLowPowerMode: !!(typeof document !== 'undefined' && document.hidden),
    };

    function getViewportOcclusionInsets() {
        if (!A.MC) return { left: 0, right: 0, top: 0, bottom: 0 };
        if (window.innerWidth <= 860) return { left: 0, right: 0, top: 0, bottom: 0 };

        var leftInset = 0;
        var rightInset = 0;
        var sidebar = document.querySelector('.terra-sidebar');
        var routePanel = document.getElementById('routePanel');

        if (sidebar && !sidebar.classList.contains('is-collapsed') && document.documentElement.getAttribute('data-sidebar-collapsed') !== '1') {
            leftInset = Math.round(sidebar.getBoundingClientRect().width || 0);
        }
        if (routePanel && routePanel.classList.contains('is-open')) {
            rightInset = Math.round(routePanel.getBoundingClientRect().width || 0);
        }

        return { left: leftInset, right: rightInset, top: 0, bottom: 0 };
    }

    function getEffectiveViewportMetrics() {
        var width = A.MC ? A.MC.clientWidth : 0;
        var height = A.MC ? A.MC.clientHeight : 0;
        var inset = getViewportOcclusionInsets();
        return {
            width: width,
            height: height,
            leftInset: inset.left,
            rightInset: inset.right,
            topInset: inset.top,
            bottomInset: inset.bottom,
            availableWidth: Math.max(1, width - inset.left - inset.right),
            availableHeight: Math.max(1, height - inset.top - inset.bottom)
        };
    }
    A.getViewportOcclusionInsets = getViewportOcclusionInsets;

    /* ── Viewport ── */
    function getFitScaleForViewport(viewWidth, viewHeight) {
        if (!A.mapInfo) return 1;
        var metrics = getEffectiveViewportMetrics();
        var width = Math.max(1, Math.min(viewWidth, metrics.availableWidth));
        return Math.min(width / A.mapInfo.map_width, viewHeight / A.mapInfo.map_height);
    }

    function getRecommendedInitialScale(fitScale) {
        var metrics = getEffectiveViewportMetrics();
        var preferredPlayableScale = Math.max(0.30, Math.min(0.38, Math.max(metrics.availableWidth / 5400, metrics.height / 3600)));
        return Math.max(fitScale, preferredPlayableScale);
    }

    function getViewportOffsetBounds(scale) {
        if (!A.mapInfo) return { minX: 0, maxX: 0, minY: 0, maxY: 0 };
        var scaledWidth = A.mapInfo.map_width * scale;
        var scaledHeight = A.mapInfo.map_height * scale;
        var metrics = getEffectiveViewportMetrics();
        var width = metrics.width;
        var height = metrics.height;
        var centeredX = metrics.leftInset + (metrics.availableWidth - scaledWidth) / 2;
        var centeredY = (height - scaledHeight) / 2;
        var edgeSlackX = Math.round(metrics.availableWidth * 0.5);
        var edgeSlackY = Math.round(height * 0.5);
        return {
            minX: scaledWidth <= metrics.availableWidth ? centeredX - edgeSlackX : width - metrics.rightInset - scaledWidth - edgeSlackX,
            maxX: scaledWidth <= metrics.availableWidth ? centeredX + edgeSlackX : metrics.leftInset + edgeSlackX,
            minY: scaledHeight <= height ? centeredY - edgeSlackY : height - scaledHeight - edgeSlackY,
            maxY: scaledHeight <= height ? centeredY + edgeSlackY : edgeSlackY,
        };
    }

    function clampViewportOffset(offsetX, offsetY, scale) {
        var bounds = getViewportOffsetBounds(scale);
        return {
            x: clamp(offsetX, bounds.minX, bounds.maxX),
            y: clamp(offsetY, bounds.minY, bounds.maxY),
        };
    }

    function applyAxisResistance(value, minValue, maxValue) {
        if (value < minValue) return minValue + (value - minValue) * 0.35;
        if (value > maxValue) return maxValue + (value - maxValue) * 0.35;
        return value;
    }

    function applyElasticViewportOffset(offsetX, offsetY, scale) {
        var bounds = getViewportOffsetBounds(scale);
        return {
            x: applyAxisResistance(offsetX, bounds.minX, bounds.maxX),
            y: applyAxisResistance(offsetY, bounds.minY, bounds.maxY),
        };
    }

    var zoomLabelEl = document.getElementById('zoomLabel');

    /* ── Zoom state for render scheduling ── */
    var zoomingTimer = null;
    A.isZooming = false;
    function markZooming() {
        A.isZooming = true;
        if (zoomingTimer) clearTimeout(zoomingTimer);
        zoomingTimer = setTimeout(function () { A.isZooming = false; zoomingTimer = null; }, 120);
    }

    function applyTransform(options) {
        options = options || {};
        if (A.mapInfo && options.clamp !== false) {
            var clamped = clampViewportOffset(A.offsetX, A.offsetY, A.scale);
            A.offsetX = clamped.x;
            A.offsetY = clamped.y;
        }
        A.MW.style.transform = 'translate(' + A.offsetX + 'px,' + A.offsetY + 'px) scale(' + A.scale + ')';
        zoomLabelEl.textContent = Math.round(A.scale * 100) + '%';
        if (A.requestMarkerRender) {
            if (A.isZooming && A.renderMarkersNow) A.renderMarkersNow();
            else A.requestMarkerRender();
        }
        if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync();
        if (A.latestX != null) {
            var sx = A.latestX * A.scale + A.offsetX, sy = A.latestY * A.scale + A.offsetY;
            A.playerLerp.tx = sx; A.playerLerp.ty = sy;
            A.playerLerp.cx = sx; A.playerLerp.cy = sy;
            applyPlayerPos(sx, sy);
        }
    }
    A.applyTransform = applyTransform;
    A.getFitScaleForViewport = getFitScaleForViewport;
    A.getRecommendedInitialScale = getRecommendedInitialScale;
    A.clampViewportOffset = clampViewportOffset;

    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

    function getSidebarAnimationDurationMs() {
        var fallback = 320;
        var shell = document.querySelector('.terra-shell');
        if (!shell || typeof window.getComputedStyle !== 'function') return fallback;
        var value = window.getComputedStyle(shell).getPropertyValue('--sidebar-toggle-duration');
        if (!value) return fallback;
        var raw = String(value).trim();
        if (!raw) return fallback;
        var num = parseFloat(raw);
        if (!isFinite(num) || num <= 0) return fallback;
        if (raw.toLowerCase().endsWith('ms')) return Math.round(num);
        return Math.round(num * 1000);
    }
    A.getSidebarAnimationDurationMs = getSidebarAnimationDurationMs;

    var panAnimId = null;

    function stopViewportAnimation() {
        if (panAnimId !== null) {
            cancelAnimationFrame(panAnimId);
            panAnimId = null;
        }
    }

    function animateViewportTo(targetX, targetY, duration) {
        stopViewportAnimation();
        var startOX = A.offsetX;
        var startOY = A.offsetY;
        var start = null;
        duration = duration || 320;

        if (Math.abs(targetX - startOX) < 0.5 && Math.abs(targetY - startOY) < 0.5) {
            A.offsetX = targetX;
            A.offsetY = targetY;
            applyTransform({ clamp: false });
            return;
        }

        function tick(ts) {
            if (!start) start = ts;
            var t = Math.min(1, (ts - start) / duration);
            var ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            A.offsetX = startOX + (targetX - startOX) * ease;
            A.offsetY = startOY + (targetY - startOY) * ease;
            applyTransform({ clamp: false });
            if (t < 1) {
                panAnimId = requestAnimationFrame(tick);
            } else {
                panAnimId = null;
            }
        }

        panAnimId = requestAnimationFrame(tick);
    }

    function bounceViewportIntoBounds() {
        if (!A.mapInfo) return false;
        var clamped = clampViewportOffset(A.offsetX, A.offsetY, A.scale);
        if (Math.abs(clamped.x - A.offsetX) < 0.5 && Math.abs(clamped.y - A.offsetY) < 0.5) return false;
        animateViewportTo(clamped.x, clamped.y, 220);
        return true;
    }

    function doZoom(factor, px, py) {
        if (px == null) {
            var metrics = getEffectiveViewportMetrics();
            px = metrics.leftInset + metrics.availableWidth / 2;
            py = A.MC.clientHeight / 2;
        }
        stopViewportAnimation();
        markZooming();
        var prevScale = A.scale;
        var next = clamp(A.scale * factor, 0.08, 15);
        A.scale = next;
        var clamped = clampViewportOffset(
            px - (px - A.offsetX) * (next / prevScale),
            py - (py - A.offsetY) * (next / prevScale),
            next
        );
        A.offsetX = clamped.x;
        A.offsetY = clamped.y;
        applyTransform({ clamp: false });
    }

    function resetView() { if (A.fitMap) A.fitMap(); }

    function centerPlayer() {
        if (A.selectedMarkerId) {
            var m = A.markersById[A.selectedMarkerId];
            if (m) { smoothPanTo(m.x, m.y); return; }
        }
        if (A.latestX != null) { smoothPanTo(A.latestX, A.latestY); return; }
        if (A.mapInfo) smoothPanTo(A.mapInfo.map_width / 2, A.mapInfo.map_height / 2);
    }

    /* ── Player ── */
    function applyPlayerPos(sx, sy) {
        A.PLAYER_DOT.style.transform = 'translate(' + sx + 'px,' + sy + 'px) translate(-50%, -50%)';
        A.PLAYER_ARROW.style.transform = 'translate(' + sx + 'px,' + sy + 'px) translate(-50%, -50%)';
    }

    function playerLerpTick() {
        if (A.isLowPowerMode) {
            A.playerLerp.cx = A.playerLerp.tx;
            A.playerLerp.cy = A.playerLerp.ty;
            applyPlayerPos(A.playerLerp.cx, A.playerLerp.cy);
            A.playerLerp.running = false;
            return;
        }
        var lp = A.playerLerp, dx = lp.tx - lp.cx, dy = lp.ty - lp.cy;
        if (Math.abs(dx) < 0.3 && Math.abs(dy) < 0.3) {
            lp.cx = lp.tx; lp.cy = lp.ty; applyPlayerPos(lp.cx, lp.cy); lp.running = false; return;
        }
        lp.cx += dx * 0.25; lp.cy += dy * 0.25;
        if (isNaN(lp.cx) || isNaN(lp.cy)) { lp.running = false; return; }
        applyPlayerPos(lp.cx, lp.cy);
        requestAnimationFrame(playerLerpTick);
    }

    function setPlayer(x, y, animate) {
        A.latestX = x; A.latestY = y;
        if (A._routeCheckNav) A._routeCheckNav(x, y);
        var sx = x * A.scale + A.offsetX, sy = y * A.scale + A.offsetY;
        A.PLAYER_DOT.style.display = '';
        A.PLAYER_ARROW.style.opacity = '0.95';
        A.PLAYER_ARROW_POLY.style.display = 'none';
        A.PLAYER_ARROW_CIRCLE.style.display = '';
        var lp = A.playerLerp;
        if (!animate || A.isZooming || A.isLowPowerMode) {
            lp.tx = sx; lp.ty = sy; lp.cx = sx; lp.cy = sy; lp.running = false; applyPlayerPos(sx, sy);
        } else {
            if (lp.cx === 0 && lp.cy === 0) { lp.cx = sx; lp.cy = sy; }
            lp.tx = sx; lp.ty = sy;
            if (!lp.running) { lp.running = true; requestAnimationFrame(playerLerpTick); }
        }
        document.getElementById('coordX').textContent = Math.round(x);
        document.getElementById('coordY').textContent = Math.round(y);
        document.getElementById('coordState').textContent = '已定位';
    }
    A.setPlayer = setPlayer;

    function placePlayerAt(clientX, clientY) {
        var rect = A.MC.getBoundingClientRect();
        var mapX = (clientX - rect.left - A.offsetX) / A.scale;
        var mapY = (clientY - rect.top - A.offsetY) / A.scale;
        if (mapX < 0 || mapY < 0) return;
        if (A.mapInfo && (mapX > A.mapInfo.map_width || mapY > A.mapInfo.map_height)) return;
        setPlayer(mapX, mapY, true);
        smoothPanTo(mapX, mapY);
    }

    function smoothPanTo(mapX, mapY) {
        var metrics = getEffectiveViewportMetrics();
        var clamped = clampViewportOffset(
            metrics.leftInset + metrics.availableWidth / 2 - mapX * A.scale,
            A.MC.clientHeight / 2 - mapY * A.scale,
            A.scale
        );
        animateViewportTo(clamped.x, clamped.y, 320);
    }
    A.smoothPanTo = smoothPanTo;

    /* ── Map Events ── */
    function bindMapEvents() {
        A.MC.addEventListener('mousedown', function (e) {
            if (e.button !== 0) return;
            stopViewportAnimation();
            markZooming();
            A.dragging = true; A.dragMoved = false; A.MC.classList.add('dragging');
            A.dragStartX = e.clientX; A.dragStartY = e.clientY;
            A.dragOriginX = A.offsetX; A.dragOriginY = A.offsetY;
            document.addEventListener('mousemove', onDragMove);
            document.addEventListener('mouseup', onDragEnd);
            e.preventDefault();
        });

        function onDragMove(e) {
            if (!A.dragging) return;
            var dx = e.clientX - A.dragStartX, dy = e.clientY - A.dragStartY;
            if (!A.dragMoved && Math.abs(dx) + Math.abs(dy) < 4) return;
            A.dragMoved = true; A.isDragging = true;
            var elastic = applyElasticViewportOffset(A.dragOriginX + dx, A.dragOriginY + dy, A.scale);
            A.offsetX = elastic.x; A.offsetY = elastic.y;
            applyTransform({ clamp: false });
        }

        function onDragEnd(e) {
            var wasClick = A.dragging && !A.dragMoved && e;
            A.isDragging = false; A.dragging = false; A.MC.classList.remove('dragging');
            document.removeEventListener('mousemove', onDragMove);
            document.removeEventListener('mouseup', onDragEnd);
            if (A.requestMarkerRender) A.requestMarkerRender();
            if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync(true);
            if (wasClick) {
                var tool = A._currentTool ? A._currentTool() : 'pan';
                var hit = A.hitTestMarkerAt(e.clientX, e.clientY);
                if (tool === 'multiselect' && A.handleMultiSelectHit) {
                    A.handleMultiSelectHit(hit, e.clientX, e.clientY);
                    return;
                }
                if (hit) {
                    if (hit.cluster) A.zoomToCluster(hit);
                    else A.selectMarker(hit.marker.id, false);
                } else {
                    placePlayerAt(e.clientX, e.clientY);
                }
                return;
            }
            bounceViewportIntoBounds();
        }

        A.MC.addEventListener('touchstart', function (e) {
            if (!e.touches[0]) return;
            stopViewportAnimation();
            markZooming();
            A.dragging = true; A.dragMoved = false; A.MC.classList.add('dragging');
            A.dragStartX = e.touches[0].clientX; A.dragStartY = e.touches[0].clientY;
            A.dragOriginX = A.offsetX; A.dragOriginY = A.offsetY;
        }, { passive: true });
        A.MC.addEventListener('touchmove', function (e) {
            if (!A.dragging || !e.touches[0]) return;
            var dx = e.touches[0].clientX - A.dragStartX, dy = e.touches[0].clientY - A.dragStartY;
            if (!A.dragMoved && Math.abs(dx) + Math.abs(dy) < 6) return;
            A.dragMoved = true; A.isDragging = true;
            var elastic = applyElasticViewportOffset(A.dragOriginX + dx, A.dragOriginY + dy, A.scale);
            A.offsetX = elastic.x; A.offsetY = elastic.y;
            applyTransform({ clamp: false });
        }, { passive: true });
        A.MC.addEventListener('touchend', function (e) {
            var touch = e.changedTouches && e.changedTouches[0];
            var wasTap = A.dragging && !A.dragMoved && touch;
            A.isDragging = false;
            A.dragging = false;
            A.MC.classList.remove('dragging');
            if (A.requestMarkerRender) A.requestMarkerRender();
            if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync(true);
            if (wasTap) {
                var tool = A._currentTool ? A._currentTool() : 'pan';
                var hit = A.hitTestMarkerAt(touch.clientX, touch.clientY);
                if (tool === 'multiselect' && A.handleMultiSelectHit) {
                    A.handleMultiSelectHit(hit, touch.clientX, touch.clientY);
                    return;
                }
                if (hit) {
                    if (hit.cluster) A.zoomToCluster(hit);
                    else A.selectMarker(hit.marker.id, false);
                } else {
                    placePlayerAt(touch.clientX, touch.clientY);
                }
            } else {
                bounceViewportIntoBounds();
            }
        }, { passive: true });

        A.MC.addEventListener('touchcancel', function () {
            A.isDragging = false;
            A.dragging = false;
            A.MC.classList.remove('dragging');
            if (A.requestMarkerRender) A.requestMarkerRender();
            if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync(true);
            bounceViewportIntoBounds();
        }, { passive: true });

        /* Coalesce wheel events through RAF to avoid multiple doZoom per frame */
        var pendingWheel = null;
        A.MC.addEventListener('wheel', function (e) {
            e.preventDefault();
            var rect = A.MC.getBoundingClientRect();
            var px = e.clientX - rect.left, py = e.clientY - rect.top;
            if (pendingWheel) {
                /* accumulate: multiply factors so rapid scrolling is merged */
                pendingWheel.factor *= (e.deltaY < 0 ? 1.12 : 0.88);
                pendingWheel.px = px; pendingWheel.py = py;
            } else {
                pendingWheel = { factor: e.deltaY < 0 ? 1.12 : 0.88, px: px, py: py };
                requestAnimationFrame(function () {
                    if (pendingWheel) {
                        markZooming();
                        doZoom(pendingWheel.factor, pendingWheel.px, pendingWheel.py);
                        pendingWheel = null;
                    }
                });
            }
        }, { passive: false });

        window.addEventListener('resize', function () {
            requestAnimationFrame(function () {
                if (!A.mapInfo) return;
                A.ensureMarkerCanvasSize();
                applyTransform();
                if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync(true);
            });
        });
    }

    /* ── Initialize modules ── */
    MapMarkers.setup(A);
    MapRoute.setup(A);

    /* ── Sidebar collapse ── */
    (function () {
        var sidebar = document.querySelector('.terra-sidebar');
        var stageEl = document.querySelector('.terra-stage');
        var panelAnimTimer = null;
        var sidebarUiRefreshTimer = null;
        var btn = document.getElementById('sidebarOpenBtn');
        var closeBtn = document.getElementById('sidebarPanelClose');
        if (!sidebar || !btn) return;

        function markPanelAnimating() {
            if (!stageEl) return;
            var duration = A.getSidebarAnimationDurationMs ? A.getSidebarAnimationDurationMs() : 320;
            stageEl.classList.add('is-panel-animating');
            if (panelAnimTimer) clearTimeout(panelAnimTimer);
            panelAnimTimer = setTimeout(function () {
                stageEl.classList.remove('is-panel-animating');
                panelAnimTimer = null;
            }, duration);
        }

        function applySidebarCollapsed(collapsed) {
            var duration = A.getSidebarAnimationDurationMs ? A.getSidebarAnimationDurationMs() : 320;
            A.uiAnimationHoldUntil = ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) + duration;
            markPanelAnimating();
            sidebar.classList.toggle('is-collapsed', collapsed);
            AppCommon.setInteractiveHiddenState(sidebar, collapsed);
            if (stageEl) stageEl.classList.toggle('has-left-sidebar', !collapsed);
            btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            btn.classList.toggle('is-hidden', !collapsed);
            AppCommon.setInteractiveHiddenState(btn, !collapsed);
            btn.title = collapsed ? '打开资源侧栏' : '资源侧栏已打开';
            document.documentElement.setAttribute('data-sidebar-collapsed', collapsed ? '1' : '0');
            if (sidebarUiRefreshTimer) {
                clearTimeout(sidebarUiRefreshTimer);
                sidebarUiRefreshTimer = null;
            }
            if (!collapsed && A.refreshMapSidebarUi) {
                sidebarUiRefreshTimer = setTimeout(function () {
                    sidebarUiRefreshTimer = null;
                    if (document.documentElement.getAttribute('data-sidebar-collapsed') === '1') return;
                    A.refreshMapSidebarUi();
                }, duration + 16);
            }
            if (A.requestMarkerRender) A.requestMarkerRender();
        }

        AppCommon.bindPersistentToggleState({
            storageKey: 'sidebar_collapsed',
            trigger: btn,
            applyValue: applySidebarCollapsed
        });
        if (closeBtn) {
            closeBtn.addEventListener('click', function () {
                localStorage.setItem('sidebar_collapsed', '1');
                applySidebarCollapsed(true);
            });
        }
    })();

    /* ── Settings popover ── */
    (function () {
        var trigger = document.getElementById('mapSettingsBtn');
        var popover = document.getElementById('mapSettingsPopover');
        var markerSizeInput = document.getElementById('mapPrefMarkerSize');
        var markerSizeVal = document.getElementById('mapPrefMarkerSizeVal');
        var markerOpacityInput = document.getElementById('mapPrefMarkerOpacity');
        var markerOpacityVal = document.getElementById('mapPrefMarkerOpacityVal');
        var sidebarBgInput = document.getElementById('mapPrefSidebarBg');
        var sidebarBgVal = document.getElementById('mapPrefSidebarBgVal');
        var rememberFiltersInput = document.getElementById('mapPrefRememberFilters');
        var autoCenterInput = document.getElementById('mapPrefAutoCenter');
        var resetBtn = document.getElementById('mapPrefResetBtn');
        if (!trigger || !popover || !markerSizeInput || !markerOpacityInput || !sidebarBgInput || !rememberFiltersInput || !autoCenterInput || !resetBtn) return;

        var MAP_PREF_DEFAULTS = {
            markerSize: 10,
            markerOpacityPercent: 88,
            themeSidebarBg: '#fffdf9',
            rememberMapFilters: true,
            mapAutoCenter: true,
        };

        function syncSettingsPopoverValues() {
            A.prefs = AppCommon.loadPrefs();
            markerSizeInput.value = String(Math.max(6, Math.min(28, Number(A.prefs.markerSize || 10))));
            markerSizeVal.textContent = markerSizeInput.value;
            markerOpacityInput.value = String(Math.round(Math.max(0.2, Math.min(1, Number(A.prefs.markerOpacity || 0.88))) * 100));
            markerOpacityVal.textContent = markerOpacityInput.value + '%';
            var nextSidebarBg = String(A.prefs.themeSidebarBg || MAP_PREF_DEFAULTS.themeSidebarBg);
            sidebarBgInput.value = nextSidebarBg;
            if (sidebarBgVal) sidebarBgVal.textContent = nextSidebarBg.toLowerCase();
            rememberFiltersInput.checked = !!A.prefs.rememberMapFilters;
            autoCenterInput.checked = !!A.prefs.mapAutoCenter;
        }

        function applySettingsPopoverOpen(open) {
            popover.classList.toggle('is-open', open);
            AppCommon.setInteractiveHiddenState(popover, !open);
            trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
            if (open) syncSettingsPopoverValues();
        }

        AppCommon.setInteractiveHiddenState(popover, true);

        A.closeMapSettingsPopover = function () {
            if (!popover.classList.contains('is-open')) return false;
            applySettingsPopoverOpen(false);
            return true;
        };

        trigger.addEventListener('click', function () {
            applySettingsPopoverOpen(!popover.classList.contains('is-open'));
        });

        markerSizeInput.addEventListener('input', function () {
            var next = parseInt(markerSizeInput.value, 10) || 10;
            markerSizeVal.textContent = String(next);
            A.prefs.markerSize = next;
            AppCommon.updatePref('markerSize', next);
            if (A.requestMarkerRender) A.requestMarkerRender();
        });

        markerOpacityInput.addEventListener('input', function () {
            var next = Math.max(0.2, Math.min(1, (parseInt(markerOpacityInput.value, 10) || 88) / 100));
            markerOpacityVal.textContent = Math.round(next * 100) + '%';
            A.prefs.markerOpacity = next;
            AppCommon.updatePref('markerOpacity', next);
            if (A.requestMarkerRender) A.requestMarkerRender();
        });

        sidebarBgInput.addEventListener('input', function () {
            var next = String(sidebarBgInput.value || MAP_PREF_DEFAULTS.themeSidebarBg);
            if (sidebarBgVal) sidebarBgVal.textContent = next.toLowerCase();
            A.prefs.themeSidebarBg = next;
            AppCommon.updatePref('themeSidebarBg', next);
            AppCommon.applyTheme();
            if (A.requestMarkerRender) A.requestMarkerRender();
        });

        rememberFiltersInput.addEventListener('change', function () {
            A.prefs.rememberMapFilters = !!rememberFiltersInput.checked;
            AppCommon.updatePref('rememberMapFilters', !!rememberFiltersInput.checked);
        });

        autoCenterInput.addEventListener('change', function () {
            A.prefs.mapAutoCenter = !!autoCenterInput.checked;
            AppCommon.updatePref('mapAutoCenter', !!autoCenterInput.checked);
        });

        resetBtn.addEventListener('click', function () {
            markerSizeInput.value = String(MAP_PREF_DEFAULTS.markerSize);
            markerSizeVal.textContent = String(MAP_PREF_DEFAULTS.markerSize);
            markerOpacityInput.value = String(MAP_PREF_DEFAULTS.markerOpacityPercent);
            markerOpacityVal.textContent = MAP_PREF_DEFAULTS.markerOpacityPercent + '%';
            sidebarBgInput.value = MAP_PREF_DEFAULTS.themeSidebarBg;
            if (sidebarBgVal) sidebarBgVal.textContent = MAP_PREF_DEFAULTS.themeSidebarBg;
            rememberFiltersInput.checked = MAP_PREF_DEFAULTS.rememberMapFilters;
            autoCenterInput.checked = MAP_PREF_DEFAULTS.mapAutoCenter;

            A.prefs.markerSize = MAP_PREF_DEFAULTS.markerSize;
            A.prefs.markerOpacity = MAP_PREF_DEFAULTS.markerOpacityPercent / 100;
            A.prefs.themeSidebarBg = MAP_PREF_DEFAULTS.themeSidebarBg;
            A.prefs.rememberMapFilters = MAP_PREF_DEFAULTS.rememberMapFilters;
            A.prefs.mapAutoCenter = MAP_PREF_DEFAULTS.mapAutoCenter;

            AppCommon.updatePref('markerSize', MAP_PREF_DEFAULTS.markerSize);
            AppCommon.updatePref('markerOpacity', MAP_PREF_DEFAULTS.markerOpacityPercent / 100);
            AppCommon.updatePref('themeSidebarBg', MAP_PREF_DEFAULTS.themeSidebarBg);
            AppCommon.updatePref('rememberMapFilters', MAP_PREF_DEFAULTS.rememberMapFilters);
            AppCommon.updatePref('mapAutoCenter', MAP_PREF_DEFAULTS.mapAutoCenter);
            AppCommon.applyTheme();
            if (A.requestMarkerRender) A.requestMarkerRender();
            AppCommon.toast('地图显示设置已恢复默认', 'success', 1200);
        });

        document.addEventListener('click', function (e) {
            if (!popover.classList.contains('is-open')) return;
            var target = e.target;
            if (trigger.contains(target) || popover.contains(target)) return;
            /* 不干扰右侧路线显示设置弹层 */
            if (target && target.closest && (target.closest('#routeDisplayBtn') || target.closest('#routeDisplayPopup'))) return;
            A.closeMapSettingsPopover();
        });
    })();

    /* ── Button handlers ── */
    document.getElementById('zoomInBtn').addEventListener('click', function () { doZoom(1.4); });
    document.getElementById('zoomOutBtn').addEventListener('click', function () { doZoom(0.6); });
    document.getElementById('resetViewBtn').addEventListener('click', resetView);
    document.getElementById('centerPlayerBtn').addEventListener('click', centerPlayer);

    /* ── Zoom label click-to-edit (range: 10%–1000%) ── */
    (function () {
        var zoomInputEl = document.getElementById('zoomInput');
        var _handled = false;

        function openZoomEdit() {
            _handled = false;
            zoomInputEl.value = Math.round(A.scale * 100);
            zoomLabelEl.style.display = 'none';
            zoomInputEl.classList.add('is-active');
            zoomInputEl.focus();
            zoomInputEl.select();
        }

        function applyZoomEdit() {
            var val = parseFloat(zoomInputEl.value);
            if (isFinite(val) && A.scale > 0) {
                val = Math.max(10, Math.min(1000, Math.round(val)));
                doZoom(val / 100 / A.scale);
            }
            closeZoomEdit();
        }

        function closeZoomEdit() {
            zoomInputEl.classList.remove('is-active');
            zoomLabelEl.style.display = '';
        }

        zoomLabelEl.addEventListener('click', openZoomEdit);

        zoomInputEl.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') { e.preventDefault(); _handled = true; applyZoomEdit(); }
            else if (e.key === 'Escape') { e.preventDefault(); _handled = true; closeZoomEdit(); }
        });

        zoomInputEl.addEventListener('blur', function () {
            if (_handled) { _handled = false; return; }
            applyZoomEdit();
        });
    })();

    document.getElementById('markerDetailCloseBtn').addEventListener('click', function () {
        if (A.getSelectedPoints && A.getSelectedPoints().length > 1) {
            var detailPanel = document.getElementById('markerDetailPanel');
            if (detailPanel) {
                detailPanel.classList.add('is-empty');
                AppCommon.setInteractiveHiddenState(detailPanel, true);
            }
            if (A.syncSelectionInfoDock) A.syncSelectionInfoDock();
            return;
        }
        A.selectedMarkerId = null; A.stopRippleAnim();
        A.renderMarkerDetailPlaceholder(); A.requestMarkerRender();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key !== 'Escape') return;
        if (A.closeMapSettingsPopover && A.closeMapSettingsPopover()) return;
        if (A.selectedMarkerId) {
            A.selectedMarkerId = null; A.stopRippleAnim();
            A.renderMarkerDetailPlaceholder(); A.requestMarkerRender();
        }
    });

    function markMapUiReady() {
        document.body.classList.add('ui-ready');
        document.body.classList.remove('page-preload');
    }

    var mapBootPromise = null;
    function bootMapPage(options) {
        options = options || {};
        if (mapBootPromise && !options.force) return mapBootPromise;
        mapBootPromise = Promise.all([A.initializeMapImage(), A.loadMarkerData(options.forceReloadMarkers)]).then(function () {
            A.renderMarkerDetailPlaceholder();
            if (A.refreshMapSidebarUi) A.refreshMapSidebarUi();
            if (A.requestMarkerRender) A.requestMarkerRender();
            markMapUiReady();
        }).catch(function (err) {
            markMapUiReady();
            throw err;
        });
        return mapBootPromise;
    }

    function restoreMapPageAfterHistoryNavigation() {
        if (!A.MC) return;
        var imageMissing = !A.mapInfo || !A.MI || !A.MI.naturalWidth || !A.MI.getAttribute('src');
        if (imageMissing) {
            mapBootPromise = null;
            bootMapPage();
            return;
        }
        if (A.ensureMarkerCanvasSize) A.ensureMarkerCanvasSize();
        if (A.applyTransform) A.applyTransform();
        if (A.refreshMapSidebarUi) A.refreshMapSidebarUi();
        if (A.requestMarkerRender) A.requestMarkerRender();
        if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync(true);
        markMapUiReady();
    }

    function applyMapPowerMode(lowPower) {
        var next = !!lowPower;
        if (A.isLowPowerMode === next) return;
        A.isLowPowerMode = next;

        if (next) {
            stopViewportAnimation();
            A.playerLerp.running = false;
            if (A.stopRippleAnim) A.stopRippleAnim();
            if (A.stopNavAnim) A.stopNavAnim();
            if (A.cancelMapBackgroundWork) A.cancelMapBackgroundWork();
            if (A.syncRoutePowerMode) A.syncRoutePowerMode(true);
            return;
        }

        if (A.syncMapPowerMode) A.syncMapPowerMode(false);
        if (A.syncRoutePowerMode) A.syncRoutePowerMode(false);
        if (A.latestX != null && A.latestY != null) A.setPlayer(A.latestX, A.latestY, false);
        if (A.requestMarkerRender) A.requestMarkerRender();
        if (A.scheduleVisibleChunkSync) A.scheduleVisibleChunkSync(true);
    }
    A.applyMapPowerMode = applyMapPowerMode;

    /* ── Boot ── */
    window.addEventListener('beforeunload', function () {
        A.playerLerp.running = false;
        A.selectedMarkerId = null;
        A.stopRippleAnim && A.stopRippleAnim();
    });

    AppCommon.bindPageResumeLifecycle({
        rafPasses: 2,
        onResume: function () {
            restoreMapPageAfterHistoryNavigation();
            applyMapPowerMode(false);
        }
    });

    document.addEventListener('visibilitychange', function () {
        applyMapPowerMode(document.hidden);
    });

    bindMapEvents();
    A.attachMarkerSearch();
    A.loadMapFilters();
    bootMapPage();
