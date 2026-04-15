/* map-route.js — Route planning, selection tools, navigation, canvas overlay */
import * as AppCommon from './common.js';
import * as RouteAlgorithms from './route-algorithms.js';

export const MapRoute = {
    setup: function (A) {
        'use strict';

        /* ── DOM refs ── */
        var panel = document.getElementById('routePanel');
        var wpListEl = document.getElementById('routeWpList');
        var wpCountEl = document.getElementById('routeWpCount');
        var wpClearAllBtn = document.getElementById('routeWpClearAll');
        var navHudEl = document.getElementById('routeNavHud');
        var resultArea = document.getElementById('routeResultArea');
        var tcSlider = document.getElementById('teleportCostSlider');
        var tcVal = document.getElementById('teleportCostVal');
        var calcBtn = document.getElementById('routeCalc');
        var algoRangeBtn = document.getElementById('routeAlgoRange');
        var algoNodeBtn = document.getElementById('routeAlgoNode');
        var algoManualBtn = document.getElementById('routeAlgoManual');
        var modeHint = document.getElementById('routeModeHint');
        var navStartBtn = document.getElementById('routeNavStart');
        var resultToggleBtn = document.getElementById('routeResultToggle');
        var autoVisitCheck = document.getElementById('routeAutoVisit');
        var lineColorInput = document.getElementById('routeLineColor');
        var lineWidthSlider = document.getElementById('routeLineWidth');
        var lineWidthValEl = document.getElementById('routeLineWidthVal');
        var dotColorInput = document.getElementById('routeDotColor');
        var arrowColorInput = document.getElementById('routeArrowColor');
        var lineOpacitySlider = document.getElementById('routeLineOpacity');
        var opacityValEl = document.getElementById('routeOpacityVal');
        var displayBtn = document.getElementById('routeDisplayBtn');
        var displayPopup = document.getElementById('routeDisplayPopup');
        var displayResetBtn = document.getElementById('routeDisplayResetBtn');
        var showRadiusCheck = document.getElementById('routeShowRadius');
        var lockCheck = document.getElementById('routeLockCheck');
        var lockToggle = document.getElementById('routeLockToggle');
        var addSelectionBtn = document.getElementById('routeAddSelection');
        var importBtn = document.getElementById('routeImport');
        var exportBtn = document.getElementById('routeExport');
        var serverRouteSelect = document.getElementById('routeServerSelect');
        var serverRouteLoadBtn = document.getElementById('routeServerLoad');
        var serverRouteRefreshBtn = document.getElementById('routeServerRefresh');
        var routeTypeDialog = document.getElementById('routeTypeDialog');
        var routeTypeList = document.getElementById('routeTypeList');
        var routeTypeAllResourceBtn = document.getElementById('routeTypeAllResource');
        var routeTypeAllTeleportBtn = document.getElementById('routeTypeAllTeleport');
        var routeTypeCancelBtn = document.getElementById('routeTypeCancel');
        var routeTypeApplyBtn = document.getElementById('routeTypeApply');

        /* ── Tool dock DOM refs ── */
        var toolBtns = {
            pan: document.getElementById('toolPan'),
            multiselect: document.getElementById('toolMultiSelect'),
            boxselect: document.getElementById('toolBoxSelect'),
        };
        var toolClearBtn = document.getElementById('toolClearSelect');

        /* ── private state ── */
        var routeLocked = false;
        var routeManualMode = false;
        var routeAlgo = 'range'; // range | node | manual
        var routeFixedStartId = null;
        var routeFixedEndId = null;
        var dragSrcIdx = null;
        var routeNavMode = false;
        var visitedIds = new Set();
        var navOrder = [];
        var navCurrentIdx = 0;
        var navAnimId = null;
        var routeLineColor = '#d4a050';
        var routeLineWidth = 3;
        var routeDotColor = '#d4a050';
        var routeArrowColor = '#d4a050';
        var routeLineOpacity = 0.8;
        var showRadius = false;
        var routeResultVisible = false;
        var ROUTE_DISPLAY_DEFAULTS = {
            lineColor: '#d4a050',
            lineWidth: 3,
            dotColor: '#d4a050',
            arrowColor: '#d4a050',
            lineOpacityPercent: 80,
            teleportCost: 500,
            showRadius: false,
        };
        var ROUTE_STATE_STORAGE_KEY = 'game-map-web:web:route-state:v1';
        var routeStateReady = false;
        var rootEl = document.documentElement;
        var suppressInitAnimations = true;
        if (rootEl) rootEl.classList.add('route-init-no-anim');

        function releaseInitAnimationLock() {
            if (!suppressInitAnimations) return;
            suppressInitAnimations = false;
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    if (rootEl) rootEl.classList.remove('route-init-no-anim');
                });
            });
        }

        /* ── Selection state (tool-driven, independent of route) ── */
        var currentTool = 'pan';       // pan | multiselect | boxselect
        var selectedPoints = [];        // ordered list of { id, x, y, name, typeName, isTeleport, isCustom, radius }

        function sanitizeStoredPoint(raw, fallbackIdPrefix, index) {
            if (!raw || typeof raw !== 'object') return null;
            var x = Number(raw.x);
            var y = Number(raw.y);
            if (!isFinite(x) || !isFinite(y)) return null;
            var radius = Number(raw.radius);
            if (!isFinite(radius) || radius <= 0) radius = 30;
            var typeName = String(raw.typeName || (raw.isTeleport ? '传送点' : '资源点'));
            var isTeleport = raw.isTeleport != null ? !!raw.isTeleport : inferTeleportFromTypeName(typeName);
            return {
                id: String(raw.id || (fallbackIdPrefix + '_' + index)),
                x: x,
                y: y,
                radius: radius,
                name: String(raw.name || ('节点 ' + (index + 1))),
                typeName: typeName,
                groupName: String(raw.groupName || (raw.isCustom ? '自定义' : '未分组')),
                isTeleport: isTeleport,
                isCustom: !!raw.isCustom,
                selGroupId: raw.selGroupId ? String(raw.selGroupId) : undefined,
            };
        }

        function saveRouteLocalState() {
            if (!routeStateReady) return;
            try {
                var payload = {
                    routeWaypoints: (A.routeWaypoints || []).map(function (wp, index) {
                        return {
                            id: wp.id || ('wp_' + index),
                            x: wp.x,
                            y: wp.y,
                            radius: wp.radius || 30,
                            name: wp.name || ('路径点 ' + (index + 1)),
                            typeName: wp.typeName || (wp.isTeleport ? '传送点' : '资源点'),
                            isTeleport: !!wp.isTeleport,
                            isCustom: !!wp.isCustom,
                        };
                    }),
                    selectedPoints: selectedPoints.map(function (pt, index) {
                        return {
                            id: pt.id || ('sel_' + index),
                            x: pt.x,
                            y: pt.y,
                            radius: pt.radius || 30,
                            name: pt.name || ('节点 ' + (index + 1)),
                            typeName: pt.typeName || (pt.isTeleport ? '传送点' : '资源点'),
                            groupName: pt.groupName || (pt.isCustom ? '自定义' : '未分组'),
                            isTeleport: !!pt.isTeleport,
                            isCustom: !!pt.isCustom,
                            selGroupId: pt.selGroupId || '',
                        };
                    }),
                    routeAlgo: routeAlgo,
                    routeFixedStartId: routeFixedStartId || null,
                    routeFixedEndId: routeFixedEndId || null,
                    hasRouteResult: !!A.routeResult,
                    routeResult: A.routeResult || null,
                    routeResultVisible: !!routeResultVisible,
                    updatedAt: Date.now(),
                };
                localStorage.setItem(ROUTE_STATE_STORAGE_KEY, JSON.stringify(payload));
            } catch (_err) {}
        }

        function restoreRouteLocalState() {
            try {
                var raw = localStorage.getItem(ROUTE_STATE_STORAGE_KEY);
                if (!raw) return false;
                var parsed = JSON.parse(raw);
                if (!parsed || typeof parsed !== 'object') return false;

                var restoredWaypoints = Array.isArray(parsed.routeWaypoints)
                    ? parsed.routeWaypoints.map(function (wp, idx) {
                        return sanitizeStoredPoint(wp, 'wp', idx);
                    }).filter(Boolean).map(normalizeWaypointType)
                    : [];

                var restoredSelection = Array.isArray(parsed.selectedPoints)
                    ? parsed.selectedPoints.map(function (pt, idx) {
                        return sanitizeStoredPoint(pt, 'sel', idx);
                    }).filter(Boolean)
                    : [];

                if (restoredWaypoints.length) {
                    A.routeWaypoints = restoredWaypoints;
                }
                if (restoredSelection.length) {
                    selectedPoints = restoredSelection;
                }

                if (parsed.routeAlgo === 'range' || parsed.routeAlgo === 'node' || parsed.routeAlgo === 'manual') {
                    setRouteAlgo(parsed.routeAlgo);
                }

                routeFixedStartId = parsed.routeFixedStartId ? String(parsed.routeFixedStartId) : null;
                routeFixedEndId = parsed.routeFixedEndId ? String(parsed.routeFixedEndId) : null;

                if (parsed.hasRouteResult && parsed.routeResult && typeof parsed.routeResult === 'object') {
                    A.routeResult = parsed.routeResult;
                } else {
                    A.routeResult = null;
                }
                routeResultVisible = !!(parsed.hasRouteResult && parsed.routeResultVisible);

                if (parsed.hasRouteResult && !A.routeResult && A.routeWaypoints.length >= 2) {
                    var solved = RouteAlgorithms.solveRouteByAlgorithm({
                        algorithm: routeAlgo,
                        waypoints: A.routeWaypoints,
                        tpCost: getTeleportCost(),
                        fixedStartId: routeFixedStartId,
                        fixedEndId: routeFixedEndId,
                    });
                    A.routeResult = solved.result;
                }

                renderSelectionList();
                renderWaypointList();
                renderRouteResult();
                A.requestMarkerRender();
                return true;
            } catch (_err) {
                return false;
            }
        }

        /* ── display settings ── */
        displayBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            var open = !displayPopup.classList.contains('is-open');
            displayPopup.classList.toggle('is-open', open);
            AppCommon.setInteractiveHiddenState(displayPopup, !open);
        });
        document.addEventListener('click', function (e) {
            var target = e.target;
            if (displayPopup.contains(target) || target === displayBtn) return;
            /* 不干扰左侧地图显示设置浮块 */
            if (target && target.closest && (target.closest('#mapSettingsBtn') || target.closest('#mapSettingsPopover'))) return;
            displayPopup.classList.remove('is-open');
            AppCommon.setInteractiveHiddenState(displayPopup, true);
        });
        AppCommon.setInteractiveHiddenState(displayPopup, true);
        AppCommon.setInteractiveHiddenState(resultArea, true);
        AppCommon.setInteractiveHiddenState(routeTypeDialog, true);

        var savedColor = localStorage.getItem('route_line_color');
        var savedLineWidth = localStorage.getItem('route_line_width');
        var savedDotColor = localStorage.getItem('route_dot_color');
        var savedArrowColor = localStorage.getItem('route_arrow_color');
        var savedOpac = localStorage.getItem('route_line_opacity');
        if (savedColor) { routeLineColor = savedColor; lineColorInput.value = savedColor; }
        if (savedLineWidth) {
            routeLineWidth = Math.max(1, Math.min(8, parseInt(savedLineWidth, 10) || ROUTE_DISPLAY_DEFAULTS.lineWidth));
            if (lineWidthSlider) lineWidthSlider.value = String(routeLineWidth);
        }
        if (savedDotColor) { routeDotColor = savedDotColor; if (dotColorInput) dotColorInput.value = savedDotColor; }
        if (savedArrowColor) { routeArrowColor = savedArrowColor; if (arrowColorInput) arrowColorInput.value = savedArrowColor; }
        if (savedOpac) { routeLineOpacity = parseInt(savedOpac, 10) / 100; lineOpacitySlider.value = savedOpac; opacityValEl.textContent = savedOpac + '%'; }
        if (lineWidthValEl) lineWidthValEl.textContent = String(routeLineWidth) + 'px';

        lineColorInput.addEventListener('input', function () {
            routeLineColor = lineColorInput.value; _rebuildRgbaCache(); localStorage.setItem('route_line_color', routeLineColor); A.requestMarkerRender();
        });
        if (lineWidthSlider) {
            lineWidthSlider.addEventListener('input', function () {
                routeLineWidth = Math.max(1, Math.min(8, parseInt(lineWidthSlider.value, 10) || ROUTE_DISPLAY_DEFAULTS.lineWidth));
                if (lineWidthValEl) lineWidthValEl.textContent = String(routeLineWidth) + 'px';
                localStorage.setItem('route_line_width', String(routeLineWidth));
                A.requestMarkerRender();
            });
        }
        if (dotColorInput) {
            dotColorInput.addEventListener('input', function () {
                routeDotColor = dotColorInput.value;
                localStorage.setItem('route_dot_color', routeDotColor);
                A.requestMarkerRender();
            });
        }
        if (arrowColorInput) {
            arrowColorInput.addEventListener('input', function () {
                routeArrowColor = arrowColorInput.value;
                localStorage.setItem('route_arrow_color', routeArrowColor);
                A.requestMarkerRender();
            });
        }
        lineOpacitySlider.addEventListener('input', function () {
            routeLineOpacity = parseInt(lineOpacitySlider.value, 10) / 100;
            opacityValEl.textContent = lineOpacitySlider.value + '%';
            _rebuildRgbaCache(); localStorage.setItem('route_line_opacity', lineOpacitySlider.value); A.requestMarkerRender();
        });

        function hexToRgb(hex) { return { r: parseInt(hex.slice(1, 3), 16), g: parseInt(hex.slice(3, 5), 16), b: parseInt(hex.slice(5, 7), 16) }; }
        function rgbaFromHex(hex, alpha) {
            var rgb = hexToRgb(hex);
            return 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',' + alpha + ')';
        }
        /* Pre-cached rgba strings — rebuilt only on color/opacity change */
        var _rgbBase = hexToRgb(routeLineColor);
        var _rgbaCache = Object.create(null);
        function _rebuildRgbaCache() {
            _rgbBase = hexToRgb(routeLineColor);
            _rgbaCache = Object.create(null);
        }
        function routeRgba(a) {
            var key = a != null ? a : routeLineOpacity;
            if (_rgbaCache[key]) return _rgbaCache[key];
            var s = 'rgba(' + _rgbBase.r + ',' + _rgbBase.g + ',' + _rgbBase.b + ',' + key + ')';
            _rgbaCache[key] = s;
            return s;
        }

        var savedRadius = localStorage.getItem('route_show_radius');
        if (savedRadius === '1') { showRadius = true; showRadiusCheck.checked = true; }
        showRadiusCheck.addEventListener('change', function () {
            showRadius = showRadiusCheck.checked; localStorage.setItem('route_show_radius', showRadius ? '1' : '0'); A.requestMarkerRender();
        });

        if (displayResetBtn) {
            displayResetBtn.addEventListener('click', function () {
                routeLineColor = ROUTE_DISPLAY_DEFAULTS.lineColor;
                routeLineWidth = ROUTE_DISPLAY_DEFAULTS.lineWidth;
                routeDotColor = ROUTE_DISPLAY_DEFAULTS.dotColor;
                routeArrowColor = ROUTE_DISPLAY_DEFAULTS.arrowColor;
                routeLineOpacity = ROUTE_DISPLAY_DEFAULTS.lineOpacityPercent / 100;
                showRadius = ROUTE_DISPLAY_DEFAULTS.showRadius;

                lineColorInput.value = ROUTE_DISPLAY_DEFAULTS.lineColor;
                if (lineWidthSlider) lineWidthSlider.value = String(ROUTE_DISPLAY_DEFAULTS.lineWidth);
                if (lineWidthValEl) lineWidthValEl.textContent = ROUTE_DISPLAY_DEFAULTS.lineWidth + 'px';
                if (dotColorInput) dotColorInput.value = ROUTE_DISPLAY_DEFAULTS.dotColor;
                if (arrowColorInput) arrowColorInput.value = ROUTE_DISPLAY_DEFAULTS.arrowColor;
                lineOpacitySlider.value = String(ROUTE_DISPLAY_DEFAULTS.lineOpacityPercent);
                opacityValEl.textContent = ROUTE_DISPLAY_DEFAULTS.lineOpacityPercent + '%';
                tcSlider.value = String(ROUTE_DISPLAY_DEFAULTS.teleportCost);
                tcVal.textContent = String(ROUTE_DISPLAY_DEFAULTS.teleportCost);
                showRadiusCheck.checked = ROUTE_DISPLAY_DEFAULTS.showRadius;

                localStorage.setItem('route_line_color', ROUTE_DISPLAY_DEFAULTS.lineColor);
                localStorage.setItem('route_line_width', String(ROUTE_DISPLAY_DEFAULTS.lineWidth));
                localStorage.setItem('route_dot_color', ROUTE_DISPLAY_DEFAULTS.dotColor);
                localStorage.setItem('route_arrow_color', ROUTE_DISPLAY_DEFAULTS.arrowColor);
                localStorage.setItem('route_line_opacity', String(ROUTE_DISPLAY_DEFAULTS.lineOpacityPercent));
                localStorage.setItem('route_teleport_cost', String(ROUTE_DISPLAY_DEFAULTS.teleportCost));
                localStorage.setItem('route_show_radius', '0');

                _rebuildRgbaCache();
                if (A.routeResult && !routeManualMode) {
                    computeSelectedRoute(false);
                }
                A.requestMarkerRender();
                AppCommon.toast('路线显示设置已恢复默认', 'success', 1200);
            });
        }

        /* ── collapsible sections ── */
        function applySectionCollapseState(label, body, collapsed) {
            label.classList.toggle('is-collapsed', collapsed);
            label.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            body.classList.toggle('is-collapsed', collapsed);
            AppCommon.setInteractiveHiddenState(body, collapsed);
        }

        function syncCollapsibleSections() {
            panel.querySelectorAll('.route-section-label[data-collapse]').forEach(function (label) {
                var tid = label.getAttribute('data-collapse');
                var body = document.getElementById(tid);
                if (!body) return;
                applySectionCollapseState(label, body, localStorage.getItem('route_collapse_' + tid) === '1');
            });
        }

        syncCollapsibleSections();

        panel.addEventListener('click', function (e) {
            var label = e.target.closest('.route-section-label[data-collapse]');
            if (!label || !panel.contains(label)) return;
            var tid = label.getAttribute('data-collapse');
            var body = document.getElementById(tid);
            if (!body) return;
            var collapsed = !label.classList.contains('is-collapsed');
            applySectionCollapseState(label, body, collapsed);
            localStorage.setItem('route_collapse_' + tid, collapsed ? '1' : '0');
        });

        window.addEventListener('pageshow', function () {
            syncCollapsibleSections();
        });
        document.addEventListener('visibilitychange', function () {
            if (!document.hidden) syncCollapsibleSections();
        });

        /* ── lock ── */
        lockCheck.addEventListener('change', function () { routeLocked = lockCheck.checked; lockToggle.classList.toggle('is-locked', routeLocked); });

        /* ── teleport cost ── */
        var savedTC = localStorage.getItem('route_teleport_cost');
        if (savedTC) { tcSlider.value = savedTC; tcVal.textContent = savedTC; }
        tcSlider.addEventListener('input', function () { tcVal.textContent = tcSlider.value; localStorage.setItem('route_teleport_cost', tcSlider.value); });
        function getTeleportCost() { return parseInt(tcSlider.value, 10) || 500; }

        /* ── panel toggle ── */
        var dockEl = document.getElementById('mapToolDock');
        var stageEl = document.querySelector('.terra-stage');
        var panelAnimTimer = null;
        function syncDockShift() { dockEl.classList.toggle('dock-shifted', panel.classList.contains('is-open')); }
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
        var routeToggleBtn = document.getElementById('routeToggleBtn');
        var routePanelCloseBtn = document.getElementById('routePanelClose');
        function applyRoutePanelOpen(isOpen) {
            if (!suppressInitAnimations) {
                var duration = A.getSidebarAnimationDurationMs ? A.getSidebarAnimationDurationMs() : 320;
                A.uiAnimationHoldUntil = ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) + duration;
                markPanelAnimating();
            }
            panel.classList.toggle('is-open', isOpen);
            AppCommon.setInteractiveHiddenState(panel, !isOpen);
            if (stageEl) stageEl.classList.toggle('has-right-sidebar', isOpen);
            if (routeToggleBtn) {
                routeToggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                routeToggleBtn.classList.toggle('is-hidden', isOpen);
                AppCommon.setInteractiveHiddenState(routeToggleBtn, isOpen);
            }
            syncDockShift();
            A.requestMarkerRender();
        }
        AppCommon.bindPersistentToggleState({
            storageKey: 'route_panel_open',
            trigger: routeToggleBtn,
            readValue: function () { return localStorage.getItem('route_panel_open') === '1'; },
            writeValue: function (next) { localStorage.setItem('route_panel_open', next ? '1' : '0'); },
            applyValue: applyRoutePanelOpen
        });
        if (routePanelCloseBtn) {
            routePanelCloseBtn.addEventListener('click', function () {
                localStorage.setItem('route_panel_open', '0');
                applyRoutePanelOpen(false);
                if (routeNavMode) stopNavMode();
            });
        }
        releaseInitAnimationLock();

        /* ══════════════════════════════════════════════════════
           §1  SELECTION TOOL DOCK
           ══════════════════════════════════════════════════════ */

        function setTool(name) {
            currentTool = name;
            Object.keys(toolBtns).forEach(function (k) { toolBtns[k].classList.toggle('is-active', k === name); });
            A.MC.style.cursor = (name === 'boxselect' || name === 'multiselect') ? 'crosshair' : '';
            // Exit any lingering box state
            if (name !== 'boxselect' && A.boxSelectEl) { A.boxSelectEl.remove(); A.boxSelectEl = null; A.boxStart = null; }
        }

        Object.keys(toolBtns).forEach(function (key) {
            toolBtns[key].addEventListener('click', function () { setTool(key); });
        });

        /* clear selection — first click arms, second click within 2s confirms */
        var _clearDockArmed = false, _clearDockTimer = null;
        toolClearBtn.addEventListener('click', function () {
            if (selectedPoints.length === 0) return;
            if (_clearDockArmed) {
                clearTimeout(_clearDockTimer); _clearDockArmed = false;
                toolClearBtn.classList.remove('is-armed');
                toolClearBtn.title = '清除选择';
                clearSelection();
                AppCommon.toast('已清除所有选择', 'ok', 1000);
            } else {
                _clearDockArmed = true;
                toolClearBtn.classList.add('is-armed');
                toolClearBtn.title = '再次点击确认清除';
                AppCommon.toast('再次点击确认清除', 'warn', 1800);
                _clearDockTimer = setTimeout(function () { _clearDockArmed = false; toolClearBtn.classList.remove('is-armed'); toolClearBtn.title = '清除选择'; }, 2000);
            }
        });

        function clearSelection() {
            selectedPoints = [];
            renderSelectionList();
            A.requestMarkerRender();
            saveRouteLocalState();
        }
        A.clearSelection = clearSelection;

        var _clearInlineArmed = false;
        var _clearInlineTimer = null;
        var _clearWpArmed = false;
        var _clearWpTimer = null;

        function addToSelection(pt) {
            // Avoid duplicates
            for (var i = 0; i < selectedPoints.length; i++) {
                if (selectedPoints[i].id === pt.id) { AppCommon.toast(pt.name + ' 已在选择中', 'warn', 800); return; }
                if (pt.isCustom && selectedPoints[i].isCustom && selectedPoints[i].x === pt.x && selectedPoints[i].y === pt.y) return;
            }
            selectedPoints.push(pt);
            renderSelectionList();
            A.requestMarkerRender();
            saveRouteLocalState();
        }

        function addManyToSelection(pts) {
            var existIds = new Set();
            selectedPoints.forEach(function (p) { if (p.id) existIds.add(p.id); });
            var added = 0;
            pts.forEach(function (pt) {
                if (pt.id && existIds.has(pt.id)) return;
                if (pt.isCustom) {
                    for (var i = 0; i < selectedPoints.length; i++) {
                        if (selectedPoints[i].isCustom && selectedPoints[i].x === pt.x && selectedPoints[i].y === pt.y) return;
                    }
                }
                selectedPoints.push(pt);
                if (pt.id) existIds.add(pt.id);
                added++;
            });
            if (added > 0) {
                renderSelectionList();
                A.requestMarkerRender();
                saveRouteLocalState();
            }
        }

        function removeFromSelection(index) {
            selectedPoints.splice(index, 1);
            renderSelectionList();
            A.requestMarkerRender();
            saveRouteLocalState();
        }

        /* expose for map.js interaction */
        A.selectedPoints = selectedPoints;
        A.getSelectedPoints = function () { return selectedPoints.slice(); };
        A.getSelectedPointIds = function () { var s = new Set(); selectedPoints.forEach(function (p) { if (p.id) s.add(p.id); }); return s; };

        function getSelectionInfoDockEl() {
            var el = document.getElementById('selectionInfoDock');
            if (el) return el;
            var stage = document.querySelector('.terra-stage');
            if (!stage) return null;
            el = document.createElement('button');
            el.type = 'button';
            el.id = 'selectionInfoDock';
            el.className = 'selection-info-dock';
            el.title = '展开点位信息列表';
            el.addEventListener('click', function () {
                var panel = document.getElementById('markerDetailPanel');
                if (panel) {
                    panel.classList.remove('is-empty');
                    AppCommon.setInteractiveHiddenState(panel, false);
                }
                renderSelectionList();
            });
            stage.appendChild(el);
            AppCommon.setInteractiveHiddenState(el, true);
            return el;
        }

        function syncSelectionInfoDock() {
            var panel = document.getElementById('markerDetailPanel');
            var dock = getSelectionInfoDockEl();
            if (!panel || !dock) return;
            var shouldShow = selectedPoints.length > 0 && panel.classList.contains('is-empty');
            dock.classList.toggle('is-open', shouldShow);
            AppCommon.setInteractiveHiddenState(dock, !shouldShow);
            dock.textContent = '已选 ' + selectedPoints.length + ' 点 · 展开列表';
        }
        A.syncSelectionInfoDock = syncSelectionInfoDock;

        /* ── Unified info panel (single detail / multi list) ── */
        function renderSelectionList() {
            A.selectedPoints = selectedPoints;
            var panel = document.getElementById('markerDetailPanel');
            var title = document.getElementById('markerDetailTitle');
            var chip = document.getElementById('selectedMarkerChip');
            var card = document.getElementById('markerDetailCard');
            if (!panel || !title || !chip || !card) return;
            var keepCollapsed = panel.classList.contains('is-empty');

            if (selectedPoints.length === 0) {
                if (A.renderMarkerDetailPlaceholder) A.renderMarkerDetailPlaceholder();
                syncSelectionInfoDock();
                return;
            }

            if (selectedPoints.length === 1) {
                var only = selectedPoints[0];
                if (!keepCollapsed && !only.isCustom && only.id && A.selectMarker && A.markersById && A.markersById[only.id]) {
                    A.selectMarker(only.id, false);
                    return;
                }

                A.selectedMarkerId = null;
                if (A.stopRippleAnim) A.stopRippleAnim();
                if (!keepCollapsed) {
                    panel.classList.remove('is-empty');
                    AppCommon.setInteractiveHiddenState(panel, false);
                }
                title.textContent = only.name || '节点';
                chip.textContent = (only.typeName || '自定义') + ' / 多选模式';
                card.innerHTML =
                    '<div class="resource-meta">坐标：(' + Math.round(only.x) + ', ' + Math.round(only.y) + ')</div>' +
                    '<p style="margin:6px 0 0;font-size:13px;color:#666;line-height:1.6;">' +
                    (only.isCluster
                        ? ('这是聚合节点（当前聚合数量：' + Number(only.clusterCount || 0) + '）。')
                        : '这是自定义节点，可加入路径并参与路线计算。') +
                    '</p>';
                syncSelectionInfoDock();
                return;
            }

            A.selectedMarkerId = null;
            if (A.stopRippleAnim) A.stopRippleAnim();
            if (!keepCollapsed) {
                panel.classList.remove('is-empty');
                AppCommon.setInteractiveHiddenState(panel, false);
            }
            title.textContent = '已选择 ' + selectedPoints.length + ' 个点';
            chip.textContent = '多选列表（点击可定位）';
            card.innerHTML = '';

            var head = document.createElement('div');
            head.className = 'map-multi-head';
            var headMeta = document.createElement('div');
            headMeta.className = 'resource-meta';
            headMeta.textContent = '已选择 ' + selectedPoints.length + ' 个点';
            var clearBtn = document.createElement('button');
            clearBtn.type = 'button';
            clearBtn.className = 'map-multi-icon-btn map-multi-clear-btn';
            clearBtn.title = '清空选择（再次点击确认）';
            clearBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v9h-2V9zm4 0h2v9h-2V9zM6 9h2v9H6V9z"/></svg><span>清空选择</span>';
            clearBtn.addEventListener('click', function () {
                if (_clearInlineArmed) {
                    clearTimeout(_clearInlineTimer);
                    _clearInlineArmed = false;
                    clearBtn.classList.remove('is-armed');
                    clearSelection();
                    AppCommon.toast('已清除所有选择', 'ok', 1000);
                    return;
                }
                _clearInlineArmed = true;
                clearBtn.classList.add('is-armed');
                AppCommon.toast('再次点击确认清空', 'warn', 1800);
                _clearInlineTimer = setTimeout(function () {
                    _clearInlineArmed = false;
                    clearBtn.classList.remove('is-armed');
                }, 2000);
            });
            head.appendChild(headMeta); head.appendChild(clearBtn);
            card.appendChild(head);

            var list = document.createElement('div');
            list.className = 'map-multi-list';
            selectedPoints.forEach(function (pt, idx) {
                var row = document.createElement('div');
                row.className = 'map-multi-item';

                var seq = document.createElement('span');
                seq.className = 'map-multi-item-seq';
                seq.textContent = String(idx + 1);

                var main = document.createElement('button');
                main.type = 'button';
                main.className = 'map-multi-item-main';
                var titleText = AppCommon.escapeHtml(pt.name || ('(' + Math.round(pt.x) + ', ' + Math.round(pt.y) + ')'));
                var typeTag = AppCommon.escapeHtml(pt.typeName || '资源点');
                var groupTag = AppCommon.escapeHtml(pt.groupName || '未分组');
                var coordText = '(' + Math.round(pt.x) + ', ' + Math.round(pt.y) + ')';
                var showGroupTag = groupTag && groupTag !== typeTag;
                main.innerHTML =
                    '<span class="map-multi-item-title">' + titleText + '</span>' +
                    '<span class="map-multi-item-meta">' +
                    '<span class="map-multi-item-tags">' +
                    '<span class="map-multi-cat">' + typeTag + '</span>' +
                    (showGroupTag ? ('<span class="map-multi-group">' + groupTag + '</span>') : '') +
                    '</span>' +
                    '<span class="map-multi-coords">' + coordText + '</span>' +
                    '</span>';

                var delBtn = document.createElement('button');
                delBtn.type = 'button';
                delBtn.className = 'map-multi-icon-btn';
                delBtn.title = '移除该点';
                delBtn.innerHTML = '<svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M18.3 5.7a1 1 0 0 0-1.4 0L12 10.6 7.1 5.7a1 1 0 1 0-1.4 1.4l4.9 4.9-4.9 4.9a1 1 0 1 0 1.4 1.4l4.9-4.9 4.9 4.9a1 1 0 1 0 1.4-1.4l-4.9-4.9 4.9-4.9a1 1 0 0 0 0-1.4z"/></svg>';
                (function (i) {
                    delBtn.addEventListener('click', function (e) { e.stopPropagation(); removeFromSelection(i); });
                })(idx);

                main.addEventListener('click', function () {
                    A.smoothPanTo(pt.x, pt.y);
                });
                row.appendChild(seq);
                row.appendChild(main);
                row.appendChild(delBtn);
                list.appendChild(row);
            });
            card.appendChild(list);
            syncSelectionInfoDock();
        }

        /* ══════════════════════════════════════════════════════
           §2  MAP CANVAS INTERACTION (tool-driven)
           ══════════════════════════════════════════════════════ */

        /* ── box select ── */
        function startBoxSelect(e) {
            var rect = A.MC.getBoundingClientRect();
            A.boxStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            A.boxSelectEl = document.createElement('div'); A.boxSelectEl.className = 'route-box-select';
            A.MC.appendChild(A.boxSelectEl); return true;
        }
        function moveBoxSelect(e) {
            if (!A.boxSelectEl || !A.boxStart) return;
            var rect = A.MC.getBoundingClientRect(), cx = e.clientX - rect.left, cy = e.clientY - rect.top;
            var x1 = Math.min(A.boxStart.x, cx), y1 = Math.min(A.boxStart.y, cy);
            A.boxSelectEl.style.left = x1 + 'px'; A.boxSelectEl.style.top = y1 + 'px';
            A.boxSelectEl.style.width = Math.abs(cx - A.boxStart.x) + 'px'; A.boxSelectEl.style.height = Math.abs(cy - A.boxStart.y) + 'px';
        }
        function endBoxSelect(e) {
            if (!A.boxSelectEl || !A.boxStart) return;
            var rect = A.MC.getBoundingClientRect(), cx = e.clientX - rect.left, cy = e.clientY - rect.top;
            var x1 = Math.min(A.boxStart.x, cx), y1 = Math.min(A.boxStart.y, cy);
            var x2 = Math.max(A.boxStart.x, cx), y2 = Math.max(A.boxStart.y, cy);

            var toAdd = [];
            A.markers.forEach(function (m) {
                if (!m.isVisible) return;
                var sx = m.x * A.scale + A.offsetX, sy = m.y * A.scale + A.offsetY;
                if (sx >= x1 && sx <= x2 && sy >= y1 && sy <= y2) {
                    var meta = A.categories[m.markType] || {};
                    toAdd.push({ id: m.id, x: m.x, y: m.y, radius: 30,
                        name: (A.markerDetails[m.id] || {}).title || meta.name || '资源点',
                        typeName: meta.name || '资源点',
                        groupName: meta.group || '未分组',
                        isTeleport: m.markType === A.TELEPORT_TYPE, isCustom: false });
                }
            });
            addManyToSelection(toAdd);
            A.boxSelectEl.remove(); A.boxSelectEl = null; A.boxStart = null;
        }

        /* ── custom point in multi-select mode ── */
        var _customPointSeq = 0;
        function placeCustomPointAt(clientX, clientY) {
            var rect = A.MC.getBoundingClientRect();
            var mapX = (clientX - rect.left - A.offsetX) / A.scale;
            var mapY = (clientY - rect.top - A.offsetY) / A.scale;
            _customPointSeq++;
            var name = '节点 ' + _customPointSeq;
            var cp = { id: 'custom_' + Date.now(), x: mapX, y: mapY, name: name, radius: 30, isTeleport: false, isCustom: true, typeName: '自定义', groupName: '自定义' };
            addToSelection(cp);
            AppCommon.toast('已添加 ' + name + ' (' + Math.round(mapX) + ', ' + Math.round(mapY) + ')', 'ok', 1200);
        }

        function handleMultiSelectHit(hit, clientX, clientY) {
            if (hit && hit.cluster && hit.markers && hit.markers.length) {
                var selGroupId = 'selgrp_' + Date.now() + '_' + Math.floor(Math.random() * 100000);
                var clusterPts = hit.markers.map(function (m) {
                    var meta2 = A.categories[m.markType] || {};
                    return {
                        id: m.id,
                        x: m.x,
                        y: m.y,
                        radius: 30,
                        name: (A.markerDetails[m.id] || {}).title || meta2.name || '资源点',
                        typeName: meta2.name || '资源点',
                        groupName: meta2.group || '未分组',
                        isTeleport: m.markType === A.TELEPORT_TYPE || meta2.group === '地点',
                        isCustom: false,
                        selGroupId: selGroupId
                    };
                });
                addManyToSelection(clusterPts);
                AppCommon.toast('已选择该聚合内 ' + clusterPts.length + ' 个点', 'ok', 1200);
                return true;
            }

            if (hit && !hit.cluster && hit.marker) {
                var m = hit.marker, meta = A.categories[m.markType] || {};
                var ptName = (A.markerDetails[m.id] || {}).title || meta.name || '资源点';
                addToSelection({ id: m.id, x: m.x, y: m.y, radius: 30,
                    name: ptName,
                    typeName: meta.name || '资源点',
                    groupName: meta.group || '未分组',
                    isTeleport: m.markType === A.TELEPORT_TYPE || meta.group === '地点', isCustom: false });
                AppCommon.toast('已选中 ' + ptName, 'ok', 800);
                return true;
            }

            placeCustomPointAt(clientX, clientY);
            return true;
        }
        A.handleMultiSelectHit = handleMultiSelectHit;

        /* ── mousedown handler on map (captures before default drag) ── */
        A.MC.addEventListener('mousedown', function (e) {
            if (e.button !== 0) return;

            if (currentTool === 'boxselect') {
                if (startBoxSelect(e)) {
                    e.stopImmediatePropagation(); e.preventDefault();
                    var onMove = function (ev) { moveBoxSelect(ev); };
                    var onUp = function (ev) { endBoxSelect(ev); document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
                    document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
                }
                return;
            }

            if (currentTool === 'multiselect') {
                e.stopImmediatePropagation(); e.preventDefault();
                var hit = A.hitTestMarkerAt(e.clientX, e.clientY);
                handleMultiSelectHit(hit, e.clientX, e.clientY);
                return;
            }

            // pan tool lets the default map.js handler run
        }, true);

        /* Expose tool state so map.js can check whether to drag or not */
        A._currentTool = function () { return currentTool; };

        /* ══════════════════════════════════════════════════════
           §3  ROUTE PANEL – add selection to path, import/export
           ══════════════════════════════════════════════════════ */

        /* ── algorithm toggle ── */
        function setRouteAlgo(nextAlgo) {
            routeAlgo = nextAlgo;
            routeManualMode = (nextAlgo === 'manual');

            if (algoRangeBtn) algoRangeBtn.classList.toggle('is-active', nextAlgo === 'range');
            if (algoNodeBtn) algoNodeBtn.classList.toggle('is-active', nextAlgo === 'node');
            if (algoManualBtn) algoManualBtn.classList.toggle('is-active', nextAlgo === 'manual');

            modeHint.textContent = RouteAlgorithms.getAlgoHint(nextAlgo);
            if (nextAlgo === 'manual') {
                routeFixedStartId = null;
                routeFixedEndId = null;
            }

            A.routeResult = null;
            renderWaypointList();
            renderRouteResult();
            A.requestMarkerRender();
            saveRouteLocalState();
        }

        function computeSelectedRoute(showToast) {
            if (A.routeWaypoints.length < 2) {
                A.routeResult = null;
                renderRouteResult();
                if (showToast) AppCommon.toast('至少需要 2 个点位', 'warn', 1200);
                return;
            }
            if (routeNavMode) stopNavMode();

            var solved = RouteAlgorithms.solveRouteByAlgorithm({
                algorithm: routeAlgo,
                waypoints: A.routeWaypoints,
                tpCost: getTeleportCost(),
                fixedStartId: routeFixedStartId,
                fixedEndId: routeFixedEndId,
            });
            A.routeResult = solved.result;
            renderWaypointList();
            renderRouteResult();
            A.requestMarkerRender();
            saveRouteLocalState();

            if (!showToast) return;
            var toastMeta = RouteAlgorithms.buildAlgorithmToast(solved);
            AppCommon.toast(toastMeta.message, toastMeta.type, toastMeta.duration);
        }

        if (algoRangeBtn) algoRangeBtn.addEventListener('click', function () { setRouteAlgo('range'); });
        if (algoNodeBtn) algoNodeBtn.addEventListener('click', function () { setRouteAlgo('node'); });
        if (algoManualBtn) algoManualBtn.addEventListener('click', function () { setRouteAlgo('manual'); });
        setRouteAlgo('range');
        restoreRouteLocalState();
        routeStateReady = true;
        saveRouteLocalState();

        /* ── navigation ── */
        function startNavMode() {
            if (!A.routeResult || A.routeResult.order.length < 2) { AppCommon.toast('请先计算路线', 'warn', 1500); return; }
            routeNavMode = true; navStartBtn.textContent = '⏹ 停止导航'; navStartBtn.classList.add('is-active');
            visitedIds = new Set(); navCurrentIdx = 0; navOrder = A.routeResult.order.slice();
            renderWaypointList(); A.requestMarkerRender(); startNavAnim();
        }
        function stopNavMode() {
            routeNavMode = false; navStartBtn.textContent = '🧭 开始导航'; navStartBtn.classList.remove('is-active');
            visitedIds = new Set(); navOrder = []; navCurrentIdx = 0; stopNavAnim();
            navHudEl.innerHTML = ''; renderWaypointList(); A.requestMarkerRender();
        }
        navStartBtn.addEventListener('click', function () { if (routeNavMode) stopNavMode(); else startNavMode(); });

        function syncRouteResultVisibility() {
            var hasResult = !!(A.routeResult && A.routeResult.order && A.routeResult.order.length);
            if (!hasResult) routeResultVisible = false;
            if (resultArea) resultArea.classList.toggle('is-open', hasResult && routeResultVisible);
            AppCommon.setInteractiveHiddenState(resultArea, !(hasResult && routeResultVisible));
            if (resultToggleBtn) {
                resultToggleBtn.classList.toggle('is-active', hasResult && routeResultVisible);
                resultToggleBtn.disabled = !hasResult;
                resultToggleBtn.title = hasResult
                    ? (routeResultVisible ? '隐藏计算结果' : '查看计算结果')
                    : '暂无计算结果';
            }
        }

        if (resultToggleBtn) {
            resultToggleBtn.addEventListener('click', function () {
                if (!A.routeResult || !A.routeResult.order || !A.routeResult.order.length) {
                    AppCommon.toast('暂无计算结果，请先点击“计算路线布局”', 'warn', 1400);
                    return;
                }
                routeResultVisible = !routeResultVisible;
                syncRouteResultVisibility();
                saveRouteLocalState();
            });
            document.addEventListener('click', function (e) {
                if (!routeResultVisible) return;
                var target = e.target;
                if (!target) return;
                if (resultToggleBtn.contains(target)) return;
                if (resultArea && resultArea.contains(target)) return;
                routeResultVisible = false;
                syncRouteResultVisibility();
                saveRouteLocalState();
            });
        }

        function startNavAnim() {
            if (A.isLowPowerMode) return;
            if (navAnimId !== null) return;
            var lastRender = 0;
            (function loop(ts) {
                if (!routeNavMode) { navAnimId = null; return; }
                if (A.isLowPowerMode) { navAnimId = null; return; }
                /* throttle to ~30fps to avoid pointless full canvas redraws */
                if (ts - lastRender > 33) { A.requestMarkerRender(); lastRender = ts; }
                navAnimId = requestAnimationFrame(loop);
            })(0);
        }
        function stopNavAnim() { if (navAnimId !== null) { cancelAnimationFrame(navAnimId); navAnimId = null; } }
        A.stopNavAnim = stopNavAnim;
        A.syncRoutePowerMode = function (isLowPower) {
            if (isLowPower) {
                stopNavAnim();
                return;
            }
            if (routeNavMode) startNavAnim();
        };

        function checkNavProgress(px, py) {
            if (!routeNavMode || !navOrder.length) return;
            if (autoVisitCheck && !autoVisitCheck.checked) return;
            var changed = false;
            navOrder.forEach(function (wp) {
                if (visitedIds.has(wp.id)) return;
                var dx = wp.x - px, dy = wp.y - py;
                if (Math.sqrt(dx * dx + dy * dy) <= (wp.radius || 30)) { visitedIds.add(wp.id); changed = true; }
            });
            if (changed) {
                while (navCurrentIdx < navOrder.length && visitedIds.has(navOrder[navCurrentIdx].id)) navCurrentIdx++;
                renderWaypointList(); A.requestMarkerRender();
            }
        }
        A._routeCheckNav = checkNavProgress;

        /* ── waypoint management ── */
        function inferTeleportFromTypeName(typeName) {
            var t = String(typeName || '').toLowerCase();
            return t.indexOf('传送') >= 0 || t.indexOf('teleport') >= 0 || t.indexOf('地点') >= 0;
        }
        function normalizeWaypointType(wp) {
            var next = Object.assign({}, wp);
            if (!next.typeName) next.typeName = next.isTeleport ? '传送点' : '资源点';
            if (next.isTeleport == null) next.isTeleport = inferTeleportFromTypeName(next.typeName);
            if (next.isTeleport && !inferTeleportFromTypeName(next.typeName)) next.typeName = '传送点';
            if (!next.isTeleport && inferTeleportFromTypeName(next.typeName)) next.typeName = '资源点';
            return next;
        }

        function addWaypoints(wps) {
            if (routeLocked) { AppCommon.toast('列表已锁定', 'warn', 1000); return; }
            if (!wps || !wps.length) return;
            var existingIds = new Set();
            A.routeWaypoints.forEach(function (wp) { if (wp.id) existingIds.add(wp.id); });
            var addedCount = 0;
            wps.forEach(function (wp) {
                if (wp.id && existingIds.has(wp.id)) return;
                var duplicateCustom = false;
                if (wp.isCustom) {
                    for (var i = 0; i < A.routeWaypoints.length; i++) {
                        if (A.routeWaypoints[i].isCustom && A.routeWaypoints[i].x === wp.x && A.routeWaypoints[i].y === wp.y) { duplicateCustom = true; break; }
                    }
                }
                if (duplicateCustom) return;
                wp = normalizeWaypointType(wp);
                if (!wp.radius) wp.radius = 30;
                A.routeWaypoints.push(wp);
                if (wp.id) existingIds.add(wp.id);
                addedCount++;
            });
            if (addedCount > 0) {
                A.routeResult = null;
                if (routeNavMode) stopNavMode();
                renderWaypointList(); renderRouteResult(); A.requestMarkerRender();
                saveRouteLocalState();
                AppCommon.toast('已加入 ' + addedCount + ' 个路径点', 'ok', 1200);
            }
        }

        function removeWaypoint(index) {
            if (routeLocked) { AppCommon.toast('列表已锁定', 'warn', 1000); return; }
            var removed = A.routeWaypoints[index];
            if (removed) { if (routeFixedStartId === removed.id) routeFixedStartId = null; if (routeFixedEndId === removed.id) routeFixedEndId = null; }
            A.routeWaypoints.splice(index, 1); A.routeResult = null;
            if (routeNavMode) stopNavMode();
            renderWaypointList(); renderRouteResult(); A.requestMarkerRender();
            saveRouteLocalState();
        }
        function clearWaypoints() {
            if (routeLocked) { AppCommon.toast('列表已锁定', 'warn', 1000); return; }
            A.routeWaypoints = []; A.routeResult = null; routeFixedStartId = null; routeFixedEndId = null;
            if (routeNavMode) stopNavMode();
            renderWaypointList(); renderRouteResult(); A.requestMarkerRender();
            saveRouteLocalState();
        }

        if (wpClearAllBtn) {
            wpClearAllBtn.addEventListener('click', function () {
                if (A.routeWaypoints.length === 0) {
                    AppCommon.toast('路径点列表为空', 'warn', 1000);
                    return;
                }
                if (routeLocked) {
                    AppCommon.toast('列表已锁定', 'warn', 1000);
                    return;
                }
                if (_clearWpArmed) {
                    clearTimeout(_clearWpTimer);
                    _clearWpArmed = false;
                    wpClearAllBtn.classList.remove('is-armed');
                    clearWaypoints();
                    AppCommon.toast('已清空路径点', 'ok', 1000);
                    return;
                }
                _clearWpArmed = true;
                wpClearAllBtn.classList.add('is-armed');
                AppCommon.toast('再次点击确认清空路径点', 'warn', 1800);
                _clearWpTimer = setTimeout(function () {
                    _clearWpArmed = false;
                    wpClearAllBtn.classList.remove('is-armed');
                }, 2000);
            });
        }

        /* ── add selected points to route ── */
        function isTeleportByPoint(p) {
            return p.isTeleport != null ? !!p.isTeleport : inferTeleportFromTypeName(p.typeName);
        }

        function openRouteTypeDialog(points, onApply) {
            if (!routeTypeDialog || !routeTypeList) return;

            // ── Build groups by typeName ──────────────────────────────
            // groups: { [typeName]: { label, count, defaultValue: 'resource'|'teleport' } }
            var groupOrder = [];
            var groups = {};
            points.forEach(function (p) {
                var key = p.typeName || '未知类型';
                if (!groups[key]) {
                    groups[key] = {
                        label: key,
                        count: 0,
                        pts: [],
                        // auto-detect default from first point in group
                        defaultValue: isTeleportByPoint(p) ? 'teleport' : 'resource',
                    };
                    groupOrder.push(key);
                }
                groups[key].count++;
                groups[key].pts.push(p);
            });

            // ── Render one row per group ──────────────────────────────
            routeTypeList.innerHTML = '';
            groupOrder.forEach(function (key) {
                var g = groups[key];
                var item = document.createElement('div');
                item.className = 'route-type-dialog-item';

                // Header row: [▶] TypeName ×N  [select]
                var header = document.createElement('div');
                header.className = 'route-type-dialog-item-header';

                var expandBtn = document.createElement('button');
                expandBtn.type = 'button';
                expandBtn.className = 'route-type-dialog-expand';
                expandBtn.textContent = '▶';

                var nameEl = document.createElement('div');
                nameEl.className = 'route-type-dialog-item-name';
                nameEl.textContent = g.label + (g.count > 1 ? ' ×' + g.count : '');

                var sel = document.createElement('select');
                sel.dataset.typeKey = key;
                sel.innerHTML = '<option value="resource">资源点</option><option value="teleport">传送点</option>';
                sel.value = g.defaultValue;

                // sync blue highlight on init and on change
                function syncTeleportStyle() {
                    item.classList.toggle('is-teleport', sel.value === 'teleport');
                }
                sel.addEventListener('change', syncTeleportStyle);

                header.appendChild(expandBtn);
                header.appendChild(nameEl);
                header.appendChild(sel);

                // Sub-list of individual points in this group
                var ul = document.createElement('ul');
                ul.className = 'route-type-dialog-item-points';
                g.pts.forEach(function (p) {
                    var li = document.createElement('li');
                    var coords = '(' + Math.round(p.x) + ', ' + Math.round(p.y) + ')';
                    if (p.name) {
                        var nameSpan = document.createElement('span');
                        nameSpan.className = 'route-type-dialog-pt-name';
                        nameSpan.textContent = p.name;
                        var coordSpan = document.createElement('span');
                        coordSpan.className = 'route-type-dialog-pt-coord';
                        coordSpan.textContent = coords;
                        li.appendChild(nameSpan);
                        li.appendChild(coordSpan);
                    } else {
                        li.textContent = coords;
                    }
                    ul.appendChild(li);
                });

                expandBtn.addEventListener('click', function () {
                    var isOpen = ul.classList.toggle('is-open');
                    expandBtn.classList.toggle('is-open', isOpen);
                });

                item.appendChild(header);
                item.appendChild(ul);
                routeTypeList.appendChild(item);

                // apply initial color after element is appended
                syncTeleportStyle();
            });

            function getTypeMap() {
                var map = {};
                routeTypeList.querySelectorAll('select').forEach(function (s) {
                    map[s.dataset.typeKey] = s.value;
                });
                return map;
            }
            function setAll(typeValue) {
                routeTypeList.querySelectorAll('select').forEach(function (s) {
                    s.value = typeValue;
                    s.dispatchEvent(new Event('change'));
                });
            }
            function closeDialog() {
                routeTypeDialog.classList.remove('is-open');
                AppCommon.setInteractiveHiddenState(routeTypeDialog, true);
            }
            function applyDialog() {
                var typeMap = getTypeMap();
                var mapped = points.map(function (p) {
                    var key = p.typeName || '未知类型';
                    var sv = typeMap[key] !== undefined ? typeMap[key] : (isTeleportByPoint(p) ? 'teleport' : 'resource');
                    var tp = sv === 'teleport';
                    return {
                        id: p.id, x: p.x, y: p.y, radius: p.radius || 30,
                        name: p.name,
                        typeName: tp ? '传送点' : (p.typeName || '资源点'),
                        isTeleport: tp, isCustom: p.isCustom,
                    };
                });
                closeDialog();
                onApply(mapped);
            }

            if (routeTypeAllResourceBtn) routeTypeAllResourceBtn.onclick = function () { setAll('resource'); };
            if (routeTypeAllTeleportBtn) routeTypeAllTeleportBtn.onclick = function () { setAll('teleport'); };
            if (routeTypeCancelBtn) routeTypeCancelBtn.onclick = function () { closeDialog(); };
            if (routeTypeApplyBtn) routeTypeApplyBtn.onclick = function () { applyDialog(); };
            routeTypeDialog.onclick = function (ev) {
                if (ev.target === routeTypeDialog) closeDialog();
            };

            routeTypeDialog.classList.add('is-open');
            AppCommon.setInteractiveHiddenState(routeTypeDialog, false);
        }

        addSelectionBtn.addEventListener('click', function () {
            var pts = selectedPoints.slice();
            /* 若多选列表为空，但抓手模式有单选标记，则把它补进来 */
            if (pts.length === 0 && A.selectedMarkerId && A.markersById) {
                var m = A.markersById[A.selectedMarkerId];
                if (m) {
                    var meta = A.categories[m.markType] || {};
                    var ptName = (A.markerDetails && A.markerDetails[m.id] || {}).title || meta.name || '资源点';
                    pts = [{ id: m.id, x: m.x, y: m.y, radius: 30,
                        name: ptName,
                        typeName: meta.name || '资源点',
                        groupName: meta.group || '未分组',
                        isTeleport: m.markType === A.TELEPORT_TYPE || meta.group === '地点',
                        isCustom: false }];
                }
            }
            if (pts.length === 0) { AppCommon.toast('当前没有选择点', 'warn', 1200); return; }
            openRouteTypeDialog(pts, function (mapped) {
                addWaypoints(mapped);
            });
        });

        /* ── route storage list/load + import/export ── */
        function mapRoutePoints(data, defaultTypeName) {
            var points = Array.isArray(data) ? data : (data && Array.isArray(data.points) ? data.points : []);
            return points.map(function (p) {
                return {
                    id: p.id || ('import_' + Date.now() + '_' + Math.random()),
                    x: p.x,
                    y: p.y,
                    radius: p.radius || 30,
                    name: p.name || '导入点',
                    typeName: p.typeName || defaultTypeName || '导入',
                    isTeleport: p.isTeleport != null ? !!p.isTeleport : inferTeleportFromTypeName(p.typeName || defaultTypeName),
                    isCustom: !!p.isCustom,
                };
            });
        }

        function rebuildRouteStorageSelect(routes) {
            if (!serverRouteSelect) return;
            serverRouteSelect.innerHTML = '';
            var ph = document.createElement('option');
            ph.value = '';
            ph.textContent = '选择 routes 目录存储路线…';
            serverRouteSelect.appendChild(ph);
            (routes || []).forEach(function (r) {
                var op = document.createElement('option');
                op.value = r.filename;
                op.textContent = (r.name || r.filename) + '（' + (r.point_count || 0) + '点）';
                serverRouteSelect.appendChild(op);
            });
        }

        function fetchRouteStorageList(silent) {
            return AppCommon.fetchJSON('/api/routes').then(function (routes) {
                rebuildRouteStorageSelect(routes);
                if (!silent) AppCommon.toast('已刷新路线存储列表', 'ok', 1000);
                return routes;
            }).catch(function (err) {
                rebuildRouteStorageSelect([]);
                AppCommon.toast('读取路线存储失败：' + err.message, 'warn', 1800);
                return [];
            });
        }

        function loadRouteFromStorage(filename) {
            if (!filename) { AppCommon.toast('请先选择一条路线', 'warn', 1200); return; }
            if (routeLocked) { AppCommon.toast('列表已锁定', 'warn', 1000); return; }
            AppCommon.fetchJSON('/api/routes/' + encodeURIComponent(filename)).then(function (data) {
                var mapped = mapRoutePoints(data, '存储路线');
                if (!mapped.length) { AppCommon.toast('该路线没有可用点位', 'warn', 1400); return; }
                clearWaypoints();
                addWaypoints(mapped);
                AppCommon.toast('已载入存储路线：' + (data.name || filename), 'ok', 1400);
            }).catch(function (err) {
                AppCommon.toast('载入路线失败：' + err.message, 'warn', 1800);
            });
        }

        if (serverRouteRefreshBtn) serverRouteRefreshBtn.addEventListener('click', function () { fetchRouteStorageList(false); });
        if (serverRouteLoadBtn) serverRouteLoadBtn.addEventListener('click', function () { loadRouteFromStorage(serverRouteSelect ? serverRouteSelect.value : ''); });
        if (serverRouteSelect) {
            serverRouteSelect.addEventListener('change', function () {
                if (serverRouteSelect.value) loadRouteFromStorage(serverRouteSelect.value);
            });
        }
        fetchRouteStorageList(true);

        importBtn.addEventListener('click', function () {
            var inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.json';
            inp.addEventListener('change', function () {
                if (!inp.files || !inp.files[0]) return;
                var reader = new FileReader();
                reader.onload = function () {
                    try {
                        var data = JSON.parse(reader.result);
                        var mapped = mapRoutePoints(data, '导入');
                        if (!mapped.length) { AppCommon.toast('无效的路线文件', 'warn', 1500); return; }
                        clearWaypoints();
                        addWaypoints(mapped);
                        AppCommon.toast('路线导入成功', 'ok', 1500);
                    } catch (ex) { AppCommon.toast('文件解析失败: ' + ex.message, 'warn', 2000); }
                };
                reader.readAsText(inp.files[0]);
            });
            inp.click();
        });
        exportBtn.addEventListener('click', function () {
            if (A.routeWaypoints.length === 0) { AppCommon.toast('路径点列表为空', 'warn', 1200); return; }
            var pts = (A.routeResult ? A.routeResult.order : A.routeWaypoints).map(function (wp) {
                return { id: wp.id, x: wp.x, y: wp.y, radius: wp.radius, name: wp.name, typeName: wp.typeName, isTeleport: wp.isTeleport, isCustom: wp.isCustom };
            });
            var blob = new Blob([JSON.stringify(pts, null, 2)], { type: 'application/json' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a'); a.href = url; a.download = 'route_' + new Date().toISOString().slice(0, 10) + '.json';
            document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
            AppCommon.toast('路线已导出', 'ok', 1500);
        });

        /* ── waypoint coordinate popover editor ── */
        var wpCoordPopover = null;
        var wpCoordPopoverWp = null;

        function closeWpCoordPopover() {
            wpCoordPopoverWp = null;
            if (wpCoordPopover) {
                wpCoordPopover.classList.remove('is-open');
                AppCommon.setInteractiveHiddenState(wpCoordPopover, true);
            }
        }

        function ensureWpCoordPopover() {
            if (wpCoordPopover) return wpCoordPopover;
            var pop = document.createElement('div');
            pop.className = 'route-wp-popover';
            pop.innerHTML =
                '<div class="route-wp-popover-title">编辑路径点</div>' +
                '<div class="route-wp-popover-grid">' +
                '<label><span>x</span><input type="number" id="routeWpPopX" step="1"></label>' +
                '<label><span>y</span><input type="number" id="routeWpPopY" step="1"></label>' +
                '<label><span>r</span><input type="number" id="routeWpPopR" step="1" min="1"></label>' +
                '</div>' +
                '<div class="route-wp-popover-actions">' +
                '<button type="button" data-action="cancel">取消</button>' +
                '<button type="button" data-action="save" class="is-primary">保存</button>' +
                '</div>';
            document.body.appendChild(pop);

            function applyEdit() {
                if (!wpCoordPopoverWp) return;
                var xIn = pop.querySelector('#routeWpPopX');
                var yIn = pop.querySelector('#routeWpPopY');
                var rIn = pop.querySelector('#routeWpPopR');
                var nx = parseFloat(xIn.value);
                var ny = parseFloat(yIn.value);
                var nr = parseFloat(rIn.value);

                if (!isFinite(nx)) nx = wpCoordPopoverWp.x;
                if (!isFinite(ny)) ny = wpCoordPopoverWp.y;
                if (!isFinite(nr)) nr = wpCoordPopoverWp.radius || 30;
                nr = Math.max(1, nr);

                var changed = (nx !== wpCoordPopoverWp.x) || (ny !== wpCoordPopoverWp.y) || (nr !== (wpCoordPopoverWp.radius || 30));
                if (changed) {
                    wpCoordPopoverWp.x = nx;
                    wpCoordPopoverWp.y = ny;
                    wpCoordPopoverWp.radius = nr;

                    A.routeResult = null;
                    if (routeNavMode) stopNavMode();
                    if (routeManualMode) drawManualRoute();
                    else {
                        renderWaypointList();
                        renderRouteResult();
                        A.requestMarkerRender();
                    }
                    saveRouteLocalState();
                }
                closeWpCoordPopover();
            }

            pop.addEventListener('click', function (ev) {
                ev.stopPropagation();
                var actionBtn = ev.target.closest('button[data-action]');
                if (!actionBtn) return;
                var action = actionBtn.getAttribute('data-action');
                if (action === 'save') applyEdit();
                else closeWpCoordPopover();
            });

            pop.addEventListener('keydown', function (ev) {
                if (ev.key === 'Enter') {
                    ev.preventDefault();
                    applyEdit();
                } else if (ev.key === 'Escape') {
                    ev.preventDefault();
                    closeWpCoordPopover();
                }
            });

            document.addEventListener('mousedown', function (ev) {
                if (!pop.classList.contains('is-open')) return;
                if (pop.contains(ev.target)) return;
                closeWpCoordPopover();
            });
            document.addEventListener('scroll', function () {
                if (pop.classList.contains('is-open')) closeWpCoordPopover();
            }, true);

            wpCoordPopover = pop;
            AppCommon.setInteractiveHiddenState(wpCoordPopover, true);
            return pop;
        }

        function openWpCoordPopover(anchorEl, wp) {
            if (!anchorEl || !wp || routeNavMode) return;
            var pop = ensureWpCoordPopover();
            wpCoordPopoverWp = wp;

            var xIn = pop.querySelector('#routeWpPopX');
            var yIn = pop.querySelector('#routeWpPopY');
            var rIn = pop.querySelector('#routeWpPopR');
            xIn.value = String(Math.round(wp.x));
            yIn.value = String(Math.round(wp.y));
            rIn.value = String(Math.round(wp.radius || 30));

            var rect = anchorEl.getBoundingClientRect();
            pop.style.left = (rect.left + window.scrollX) + 'px';
            pop.style.top = (rect.bottom + window.scrollY + 8) + 'px';
            pop.classList.add('is-open');
            AppCommon.setInteractiveHiddenState(pop, false);
            xIn.focus();
            xIn.select();
        }

        /* ── render waypoint list ── */
        function renderWaypointList() {
            wpCountEl.textContent = A.routeWaypoints.length;
            wpListEl.innerHTML = '';
            closeWpCoordPopover();

            var pts;
            var usingSolvedOrder = false;
            if (routeNavMode) {
                pts = navOrder.length > 0 ? navOrder : A.routeWaypoints;
            } else if (!routeManualMode && A.routeResult && Array.isArray(A.routeResult.order) && A.routeResult.order.length === A.routeWaypoints.length) {
                pts = A.routeResult.order;
                usingSolvedOrder = true;
            } else {
                pts = A.routeWaypoints;
            }

            /* nav HUD */
            if (routeNavMode && pts.length > 0) {
                var vc = visitedIds.size, tc = pts.length, done = navCurrentIdx >= pts.length, cw = !done ? pts[navCurrentIdx] : null;
                navHudEl.innerHTML = '<div class="route-nav-hud"><div class="nav-target' + (done ? ' is-done' : '') + '">' +
                    (done ? '✅ 路线完成！' : ('▶ ' + AppCommon.escapeHtml(cw.name) + ' <span style="font-weight:400;color:#aaa">(' + Math.round(cw.x) + ', ' + Math.round(cw.y) + ')</span>')) +
                    '</div><div class="route-nav-progress"><span>' + vc + '/' + tc + '</span>' +
                    '<div class="route-nav-bar"><div class="route-nav-bar-fill" style="width:' + (tc ? Math.round(vc / tc * 100) : 0) + '%"></div></div></div></div>';
            } else { navHudEl.innerHTML = ''; }

            pts.forEach(function (wp, idx) {
                var isVis = routeNavMode && visitedIds.has(wp.id);
                var isCur = routeNavMode && idx === navCurrentIdx && !isVis;
                var li = document.createElement('li');
                li.className = 'route-wp-item' + (wp.isTeleport ? ' is-teleport' : '') + (isVis ? ' is-visited' : '') + (isCur ? ' is-current' : '');
                li.dataset.idx = idx;

                /* drag handle — always show for reorder (long press on touchscreens, native drag on desktop) */
                var drag = document.createElement('span'); drag.className = 'route-wp-drag'; drag.textContent = '⠿'; drag.title = '拖拽排序';
                li.appendChild(drag); li.setAttribute('draggable', 'true');

                /* ── desktop drag ── */
                li.addEventListener('dragstart', function (e) { dragSrcIdx = idx; setTimeout(function () { li.classList.add('is-dragging'); }, 0); e.dataTransfer.effectAllowed = 'move'; });
                li.addEventListener('dragend', function () { li.classList.remove('is-dragging'); wpListEl.querySelectorAll('.route-wp-item').forEach(function (el) { el.classList.remove('is-drag-over'); }); });
                li.addEventListener('dragover', function (e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; wpListEl.querySelectorAll('.route-wp-item').forEach(function (el) { el.classList.remove('is-drag-over'); }); li.classList.add('is-drag-over'); });
                li.addEventListener('drop', function (e) {
                    e.preventDefault(); li.classList.remove('is-drag-over');
                    if (dragSrcIdx === null || dragSrcIdx === idx) return;
                    var moved = A.routeWaypoints.splice(dragSrcIdx, 1)[0];
                    var ti = parseInt(li.dataset.idx, 10); if (dragSrcIdx < ti) ti--;
                    A.routeWaypoints.splice(ti, 0, moved); dragSrcIdx = null; A.routeResult = null;
                    renderWaypointList(); if (routeManualMode) drawManualRoute(); A.requestMarkerRender();
                    saveRouteLocalState();
                });

                /* ── touch long-press drag ── */
                (function (liEl, wpIdx) {
                    var longPressTimer = null;
                    var touchDragging = false;
                    var touchClone = null;
                    var touchOverIdx = null;

                    liEl.addEventListener('touchstart', function (ev) {
                        if (ev.touches.length !== 1) return;
                        longPressTimer = setTimeout(function () {
                            touchDragging = true;
                            dragSrcIdx = wpIdx;
                            liEl.classList.add('is-dragging');
                            touchClone = liEl.cloneNode(true);
                            touchClone.style.position = 'fixed';
                            touchClone.style.zIndex = '9999';
                            touchClone.style.opacity = '0.8';
                            touchClone.style.pointerEvents = 'none';
                            touchClone.style.width = liEl.offsetWidth + 'px';
                            document.body.appendChild(touchClone);
                        }, 400);
                    }, { passive: true });

                    liEl.addEventListener('touchmove', function (ev) {
                        if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
                        if (!touchDragging) return;
                        ev.preventDefault();
                        var touch = ev.touches[0];
                        if (touchClone) { touchClone.style.left = touch.clientX + 'px'; touchClone.style.top = (touch.clientY - 20) + 'px'; }
                        var overEl = document.elementFromPoint(touch.clientX, touch.clientY);
                        var overItem = overEl ? overEl.closest('.route-wp-item') : null;
                        wpListEl.querySelectorAll('.route-wp-item').forEach(function (el) { el.classList.remove('is-drag-over'); });
                        if (overItem && overItem.dataset.idx != null) { overItem.classList.add('is-drag-over'); touchOverIdx = parseInt(overItem.dataset.idx, 10); }
                        else touchOverIdx = null;
                    }, { passive: false });

                    liEl.addEventListener('touchend', function () {
                        if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
                        if (touchDragging) {
                            touchDragging = false;
                            liEl.classList.remove('is-dragging');
                            if (touchClone) { document.body.removeChild(touchClone); touchClone = null; }
                            wpListEl.querySelectorAll('.route-wp-item').forEach(function (el) { el.classList.remove('is-drag-over'); });
                            if (dragSrcIdx !== null && touchOverIdx !== null && dragSrcIdx !== touchOverIdx) {
                                var moved = A.routeWaypoints.splice(dragSrcIdx, 1)[0];
                                var ti = touchOverIdx; if (dragSrcIdx < ti) ti--;
                                A.routeWaypoints.splice(ti, 0, moved);
                                A.routeResult = null;
                                renderWaypointList(); if (routeManualMode) drawManualRoute(); A.requestMarkerRender();
                                saveRouteLocalState();
                            }
                            dragSrcIdx = null; touchOverIdx = null;
                        }
                    }, { passive: true });

                    liEl.addEventListener('touchcancel', function () {
                        if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
                        if (touchDragging) {
                            touchDragging = false;
                            liEl.classList.remove('is-dragging');
                            if (touchClone) { document.body.removeChild(touchClone); touchClone = null; }
                            wpListEl.querySelectorAll('.route-wp-item').forEach(function (el) { el.classList.remove('is-drag-over'); });
                        }
                    }, { passive: true });
                })(li, idx);

                var seq = document.createElement('span'); seq.className = 'route-wp-seq'; seq.textContent = isVis ? '✓' : (idx + 1);
                var nameWrap = document.createElement('div'); nameWrap.className = 'route-wp-name-wrap';
                var nameEl = document.createElement('span'); nameEl.className = 'route-wp-name';
                nameEl.textContent = wp.name || ('(' + Math.round(wp.x) + ', ' + Math.round(wp.y) + ')');
                nameWrap.appendChild(nameEl);
                var coordsEl = document.createElement('div'); coordsEl.className = 'route-wp-coords route-wp-coords-editable';
                coordsEl.textContent = '(' + Math.round(wp.x) + ', ' + Math.round(wp.y) + ')  r=' + (wp.radius || 30);
                coordsEl.title = '点击编辑坐标与范围';
                coordsEl.addEventListener('click', function (ev) {
                    ev.stopPropagation();
                    openWpCoordPopover(coordsEl, wp);
                });
                nameWrap.appendChild(coordsEl);

                li.appendChild(seq); li.appendChild(nameWrap);

                if (!routeManualMode && usingSolvedOrder && A.routeResult && idx < pts.length - 1) {
                    var dist = document.createElement('span');
                    var seg = A.routeResult.segments[idx];
                    if (seg) {
                        dist.className = 'route-wp-dist' + (seg.isTp ? ' is-tp' : '');
                        dist.textContent = seg.isTp ? '传送' : Math.round(seg.dist) + 'px';
                        li.appendChild(dist);
                    }
                }

                if (!routeNavMode) {
                    var typeBtn = document.createElement('button');
                    typeBtn.className = 'route-wp-type-toggle' + (wp.isTeleport ? ' is-teleport' : ' is-resource');
                    typeBtn.textContent = wp.isTeleport ? '传' : '资';
                    typeBtn.title = '切换点类型（传送/资源）';
                    (function (ref) { typeBtn.addEventListener('click', function (e) {
                        e.stopPropagation();
                        ref.isTeleport = !ref.isTeleport;
                        ref.typeName = ref.isTeleport ? '传送点' : '资源点';
                        A.routeResult = null;
                        if (routeNavMode) stopNavMode();
                        renderWaypointList();
                        renderRouteResult();
                        A.requestMarkerRender();
                        saveRouteLocalState();
                    }); })(wp);
                    li.appendChild(typeBtn);
                }

                if (!routeManualMode && !routeNavMode) {
                    var bS = document.createElement('button');
                    bS.className = 'route-wp-badge' + (routeFixedStartId === wp.id ? ' is-start' : '');
                    bS.textContent = '起'; bS.title = '设为固定起点';
                    (function (ref) { bS.addEventListener('click', function (e) {
                        e.stopPropagation();
                        if (routeFixedStartId === ref.id) routeFixedStartId = null;
                        else { if (routeFixedEndId === ref.id) routeFixedEndId = null; routeFixedStartId = ref.id; }
                        A.routeResult = null; renderWaypointList(); A.requestMarkerRender(); saveRouteLocalState();
                    }); })(wp);
                    li.appendChild(bS);
                    var bE = document.createElement('button');
                    bE.className = 'route-wp-badge' + (routeFixedEndId === wp.id ? ' is-end' : '');
                    bE.textContent = '终'; bE.title = '设为固定终点';
                    (function (ref) { bE.addEventListener('click', function (e) {
                        e.stopPropagation();
                        if (routeFixedEndId === ref.id) routeFixedEndId = null;
                        else { if (routeFixedStartId === ref.id) routeFixedStartId = null; routeFixedEndId = ref.id; }
                        A.routeResult = null; renderWaypointList(); A.requestMarkerRender(); saveRouteLocalState();
                    }); })(wp);
                    li.appendChild(bE);
                }

                var del = document.createElement('button'); del.className = 'route-wp-del'; del.textContent = '×'; del.title = '移除';
                (function (lp, i) { del.addEventListener('click', function (e) {
                    e.stopPropagation(); var t = lp[i], oi = A.routeWaypoints.indexOf(t);
                    if (oi >= 0) removeWaypoint(oi);
                    else { A.routeWaypoints.splice(i, 1); A.routeResult = null; renderWaypointList(); A.requestMarkerRender(); saveRouteLocalState(); }
                }); })(pts, idx);
                li.appendChild(del);

                li.addEventListener('click', function () {
                    A.smoothPanTo(wp.x, wp.y);
                });
                wpListEl.appendChild(li);
            });
        }

        /* ── manual route ── */
        function drawManualRoute() {
            if (A.routeWaypoints.length < 2) { A.routeResult = null; renderWaypointList(); renderRouteResult(); A.requestMarkerRender(); return; }
            A.routeResult = RouteAlgorithms.solveManualRoute(A.routeWaypoints, getTeleportCost());
            renderWaypointList(); renderRouteResult(); A.requestMarkerRender();
        }

        /* ── calc ── */
        calcBtn.addEventListener('click', function () {
            computeSelectedRoute(true);
        });

        function renderRouteResult() {
            if (!A.routeResult) {
                resultArea.innerHTML = '';
                syncRouteResultVisibility();
                return;
            }
            var r = A.routeResult;
            var ml = RouteAlgorithms.getModeLabel(routeAlgo, r);
            var html = '<div class="route-result-card">' +
                '<div class="rr-row"><span class="rr-label">模式</span><span class="rr-val">' + ml + '</span></div>' +
                '<div class="rr-row"><span class="rr-label">总距离</span><span class="rr-val">' + Math.round(r.totalDist) + ' px</span></div>' +
                '<div class="rr-row"><span class="rr-label">步行距离</span><span class="rr-val">' + Math.round(r.walkDist) + ' px</span></div>' +
                '<div class="rr-row"><span class="rr-label">传送次数</span><span class="rr-val" style="color:#50c8ff">' + r.tpCount + ' 次</span></div>' +
                '<div class="rr-row"><span class="rr-label">路径点数</span><span class="rr-val">' + r.order.length + '</span></div>';
            if (r.isRaro) {
                html += '<div class="rr-divider"></div>' +
                    '<div class="rr-row rr-raro"><span class="rr-label">原始资源点</span><span class="rr-val">' + r.origResourceCount + '</span></div>' +
                    '<div class="rr-row rr-raro"><span class="rr-label">合并后节点</span><span class="rr-val rr-accent">' + r.virtualResourceCount + '</span></div>' +
                    '<div class="rr-row rr-raro"><span class="rr-label">范围合并组</span><span class="rr-val rr-accent">' + r.mergedCount + ' 组</span></div>' +
                    '<div class="rr-row rr-raro"><span class="rr-label">节点节省</span><span class="rr-val rr-highlight">' + r.savingsPercent + '%</span></div>';
            }
            html += '</div>';
            resultArea.innerHTML = html;
            syncRouteResultVisibility();
        }

        /* ══════════════════════════════════════════════════════
           §4  CANVAS OVERLAY — route lines & selection highlight
           ══════════════════════════════════════════════════════ */

        A.drawRouteOverlay = function (ctx, dpr) {
            var pts;
            if (routeNavMode && navOrder.length > 0) pts = navOrder;
            else if (A.routeResult && A.routeResult.order.length >= 2) pts = A.routeResult.order;
            else pts = A.routeWaypoints;
            if (!pts || pts.length === 0) pts = [];

            var scale = A.scale, ox = A.offsetX, oy = A.offsetY;
            ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            /* ── draw selected point highlights ── */
            if (selectedPoints.length > 0) {
                var _sz = Math.max(8, Number(A.prefs.markerSize || 10));
                var _zoomBoost = scale > 1 ? Math.pow(scale, 0.55) : 1;
                var _iconSize = Math.max(10, _sz * 2.8 * _zoomBoost);
                var _ringR = Math.max(_iconSize * 0.56, 12);

                var _clusterSelMode = scale < 0.40;
                if (_clusterSelMode) {
                    /* Stable selected-cluster identity: keep original clicked cluster groups fixed */
                    var cellSz = 220;
                    var sbuckets = Object.create(null);
                    selectedPoints.forEach(function (sp) {
                        if (!sp.isCustom) return;
                        var key = sp.selGroupId || (Math.floor(sp.x / cellSz) + ':' + Math.floor(sp.y / cellSz) + ':' + String(sp.typeName || ''));
                        var b = sbuckets[key];
                        if (!b) b = sbuckets[key] = { points: [], xSum: 0, ySum: 0 };
                        b.points.push(sp);
                        b.xSum += sp.x;
                        b.ySum += sp.y;
                    });

                    ctx.font = '700 11px "Segoe UI",sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    Object.keys(sbuckets).forEach(function (bk) {
                        var b = sbuckets[bk], cnt = b.points.length;
                        var cx = (b.xSum / cnt) * scale + ox;
                        var cy = (b.ySum / cnt) * scale + oy;
                        var cr = Math.max(14, Math.min(34, _ringR * 0.9 + Math.sqrt(cnt) * 2.4));
                        ctx.beginPath();
                        ctx.arc(cx, cy, cr, 0, Math.PI * 2);
                        ctx.lineWidth = 3.0;
                        ctx.strokeStyle = 'rgba(70,200,255,0.98)';
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(cx, cy, cr + 3.8, 0, Math.PI * 2);
                        ctx.lineWidth = 2.0;
                        ctx.strokeStyle = 'rgba(30,120,255,0.62)';
                        ctx.stroke();
                        if (cnt > 1) {
                            var txt = cnt > 99 ? '99+' : String(cnt);
                            ctx.fillStyle = 'rgba(18,52,76,0.96)';
                            ctx.fillText(txt, cx, cy);
                        }
                    });
                } else {
                selectedPoints.forEach(function (sp) {
                    var wx = sp.x * scale + ox, wy = sp.y * scale + oy;
                    if (sp.isCustom) {
                        ctx.beginPath();
                        ctx.arc(wx, wy, Math.max(_ringR * 0.52, 8), 0, Math.PI * 2);
                        ctx.fillStyle = 'rgba(255, 206, 92, 0.92)';
                        ctx.fill();
                        ctx.lineWidth = 2.2;
                        ctx.strokeStyle = 'rgba(68, 40, 10, 0.96)';
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(wx - Math.max(_ringR * 0.85, 10), wy); ctx.lineTo(wx + Math.max(_ringR * 0.85, 10), wy);
                        ctx.moveTo(wx, wy - Math.max(_ringR * 0.85, 10)); ctx.lineTo(wx, wy + Math.max(_ringR * 0.85, 10));
                        ctx.strokeStyle = 'rgba(255, 248, 220, 0.96)';
                        ctx.lineWidth = 2.2;
                        ctx.stroke();
                    } else {
                        ctx.beginPath();
                        ctx.arc(wx, wy, _ringR, 0, Math.PI * 2);
                        ctx.lineWidth = 3.2;
                        ctx.strokeStyle = 'rgba(70,200,255,0.98)';
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(wx, wy, _ringR + 3.8, 0, Math.PI * 2);
                        ctx.lineWidth = 2.0;
                        ctx.strokeStyle = 'rgba(30,120,255,0.62)';
                        ctx.stroke();
                    }
                });
                }
            }

            /* radius circles */
            if ((showRadius || routeNavMode) && pts.length > 0) {
                ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
                var _rcFill = routeRgba(0.07), _rcStroke = routeRgba(0.30);
                pts.forEach(function (wp) {
                    var wx = wp.x * scale + ox, wy = wp.y * scale + oy, r = (wp.radius || 30) * scale;
                    if (r < 2) return;
                    var isV = routeNavMode && visitedIds.has(wp.id);
                    ctx.beginPath(); ctx.arc(wx, wy, r, 0, Math.PI * 2);
                    ctx.fillStyle = isV ? 'rgba(80,200,80,0.10)' : _rcFill;
                    ctx.strokeStyle = isV ? 'rgba(80,200,80,0.45)' : _rcStroke;
                    ctx.fill(); ctx.stroke();
                });
                ctx.setLineDash([]);
            }

            /* RARO: 为合并点绘制被覆盖的各原始资源圆（橙色细虚线） */
            if (A.routeResult && A.routeResult.isRaro && pts.length > 0) {
                var isRaroNav = routeNavMode && navOrder.length > 0;
                var raroLinePts = isRaroNav ? navOrder : (A.routeResult ? A.routeResult.order : []);
                raroLinePts.forEach(function (wp) {
                    if (!wp.isVirtual || !wp.isMultiMerge || !wp.coveredPoints) return;
                    var isV = routeNavMode && visitedIds.has(wp.id);
                    // 被合并的各原始圆：橙黄虚线
                    ctx.setLineDash([3, 4]);
                    ctx.lineWidth = 1.2;
                    wp.coveredPoints.forEach(function (cp) {
                        var cx2 = cp.x * scale + ox, cy2 = cp.y * scale + oy;
                        var cr = (cp.radius || 30) * scale;
                        if (cr < 2) return;
                        ctx.beginPath(); ctx.arc(cx2, cy2, cr, 0, Math.PI * 2);
                        ctx.fillStyle = isV ? 'rgba(80,200,80,0.06)' : 'rgba(255,165,0,0.07)';
                        ctx.strokeStyle = isV ? 'rgba(80,200,80,0.35)' : 'rgba(255,165,0,0.55)';
                        ctx.fill(); ctx.stroke();
                        // 原始圆心小点
                        ctx.setLineDash([]);
                        ctx.beginPath(); ctx.arc(cx2, cy2, 2.5, 0, Math.PI * 2);
                        ctx.fillStyle = isV ? 'rgba(80,200,80,0.60)' : 'rgba(255,165,0,0.70)';
                        ctx.fill();
                        // 从原始圆心到虚拟访问位置的连线
                        var vx = wp.x * scale + ox, vy = wp.y * scale + oy;
                        ctx.setLineDash([2, 3]);
                        ctx.lineWidth = 0.9;
                        ctx.strokeStyle = isV ? 'rgba(80,200,80,0.25)' : 'rgba(255,165,0,0.30)';
                        ctx.beginPath(); ctx.moveTo(cx2, cy2); ctx.lineTo(vx, vy); ctx.stroke();
                        ctx.setLineDash([]);
                    });
                    ctx.setLineDash([]);
                });
            }

            /* path lines */
            var linePts = (routeNavMode && navOrder.length > 0) ? navOrder : (A.routeResult ? A.routeResult.order : []);
            var segs = (A.routeResult && !routeNavMode) ? A.routeResult.segments : null;
            if (linePts.length >= 2) {
                for (var i = 0; i < linePts.length - 1; i++) {
                    var a = linePts[i], b = linePts[i + 1];
                    var ax = a.x * scale + ox, ay = a.y * scale + oy, bx = b.x * scale + ox, by = b.y * scale + oy;
                    var seg = segs ? segs[i] : null, isTp = seg ? seg.isTp : (a.isTeleport && b.isTeleport);
                    var isVisSeg = routeNavMode && visitedIds.has(a.id);
                    ctx.beginPath();
                    if (isTp) { ctx.setLineDash([8, 6]); ctx.strokeStyle = 'rgba(80,200,255,0.75)'; ctx.lineWidth = Math.max(1, routeLineWidth - 0.5); }
                    else if (isVisSeg) { ctx.setLineDash([]); ctx.strokeStyle = 'rgba(80,200,80,0.50)'; ctx.lineWidth = Math.max(1, routeLineWidth - 0.5); }
                    else { ctx.setLineDash([]); ctx.strokeStyle = routeRgba(routeLineOpacity); ctx.lineWidth = routeLineWidth; }
                    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([]);
                    var mx = (ax + bx) / 2, my = (ay + by) / 2, angle = Math.atan2(by - ay, bx - ax);
                    ctx.save(); ctx.translate(mx, my); ctx.rotate(angle);
                    ctx.fillStyle = isTp ? 'rgba(80,200,255,0.85)' : (isVisSeg ? 'rgba(80,200,80,0.70)' : rgbaFromHex(routeArrowColor, Math.min(1, routeLineOpacity + 0.1)));
                    ctx.beginPath(); ctx.moveTo(6, 0); ctx.lineTo(-4, -4); ctx.lineTo(-4, 4); ctx.closePath(); ctx.fill(); ctx.restore();
                }
            }

            /* numbered dots */
            if (pts.length > 0) {
                var now = performance.now();
                var _dotFill = rgbaFromHex(routeDotColor, 0.90);
                ctx.font = '700 10px "Segoe UI",sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                pts.forEach(function (wp, idx) {
                    var wx = wp.x * scale + ox, wy = wp.y * scale + oy;
                    var isV = routeNavMode && visitedIds.has(wp.id);
                    var isCur = routeNavMode && idx === navCurrentIdx && !isV;
                    if (wp.isMultiMerge) {
                        // RARO 合并点：绘制菱形（旋转45°的正方形）
                        var dotR = isCur ? 11 : (isV ? 9 : 9);
                        var pulseFactor = isCur ? (0.55 + 0.45 * Math.sin(now / 380)) : 1;
                        ctx.save();
                        ctx.translate(wx, wy);
                        ctx.rotate(Math.PI / 4);
                        ctx.beginPath();
                        ctx.rect(-dotR * 0.72, -dotR * 0.72, dotR * 1.44, dotR * 1.44);
                        if (isCur) { ctx.fillStyle = 'rgba(255,140,0,' + pulseFactor.toFixed(2) + ')'; }
                        else if (isV) { ctx.fillStyle = 'rgba(72,188,72,0.90)'; }
                        else { ctx.fillStyle = rgbaFromHex(routeDotColor, 0.92); }
                        ctx.fill();
                        ctx.restore();
                        // 外圈描边（高亮区分）
                        ctx.save();
                        ctx.translate(wx, wy);
                        ctx.rotate(Math.PI / 4);
                        ctx.beginPath();
                        ctx.rect(-dotR * 0.72, -dotR * 0.72, dotR * 1.44, dotR * 1.44);
                        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
                        ctx.lineWidth = 1.8;
                        ctx.stroke();
                        ctx.restore();
                        // 数字序号
                        ctx.fillStyle = '#fff';
                        ctx.font = '700 9px "Segoe UI",sans-serif';
                        ctx.fillText(isV ? '✓' : String(idx + 1), wx, wy - 0.5);
                        // 合并数徽章
                        if (!isV && wp.coveredCount > 1) {
                            var badgeX = wx + dotR * 0.7, badgeY = wy - dotR * 0.7;
                            ctx.beginPath(); ctx.arc(badgeX, badgeY, 5.5, 0, Math.PI * 2);
                            ctx.fillStyle = 'rgba(255,220,0,0.95)';
                            ctx.fill();
                            ctx.font = '700 7px "Segoe UI",sans-serif';
                            ctx.fillStyle = '#333';
                            ctx.fillText(String(wp.coveredCount), badgeX, badgeY + 0.5);
                        }
                    } else {
                        ctx.beginPath();
                        if (isCur) { ctx.arc(wx, wy, 10, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,155,15,' + (0.55 + 0.45 * Math.sin(now / 380)).toFixed(2) + ')'; }
                        else if (isV) { ctx.arc(wx, wy, 8, 0, Math.PI * 2); ctx.fillStyle = 'rgba(72,188,72,0.85)'; }
                        else { ctx.arc(wx, wy, 8, 0, Math.PI * 2); ctx.fillStyle = _dotFill; }
                        ctx.fill();
                        ctx.font = '700 10px "Segoe UI",sans-serif';
                        ctx.fillStyle = '#fff'; ctx.fillText(isV ? '✓' : String(idx + 1), wx, wy);
                    }
                });
            }

            /* start/end markers */
            if (!routeNavMode && linePts.length >= 2) {
                ctx.font = 'bold 16px sans-serif'; ctx.fillStyle = routeRgba(1); ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
                var s0 = linePts[0], sx0 = s0.x * scale + ox, sy0 = s0.y * scale + oy;
                ctx.strokeText('★', sx0 - 8, sy0 - 14); ctx.fillText('★', sx0 - 8, sy0 - 14);
                var sN = linePts[linePts.length - 1], sxN = sN.x * scale + ox, syN = sN.y * scale + oy;
                ctx.strokeText('🚩', sxN - 8, syN - 14); ctx.fillText('🚩', sxN - 8, syN - 14);
            }
            ctx.restore();
        };
    }
};
