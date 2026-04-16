/* map-markers.js — Marker rendering, chunk loading, search & filtering */
import * as AppCommon from './common.js';

export const MapMarkers = {
    setup: function (A) {
        'use strict';

        /* ── private state ── */
        var rippleAnimId = null, rippleFrameTick = 0;
        var visibleChunkSyncTimer = null;
        var prefetchIdleHandle = null;
        var searchWarmIdleHandle = null;
        var prefetchGeneration = 0;
        var SEARCH_INDEX_OFFLINE_KEY = 'markers-search-index';
        var CLIENT_PROFILE = buildClientProfile();
        var searchFilterCache = {
            query: null,
            activeSig: null,
            sourceTag: null,
            sourceLen: -1,
            results: null,
        };

        function buildClientProfile() {
            var nav = typeof navigator !== 'undefined' ? navigator : {};
            var connection = nav.connection || nav.mozConnection || nav.webkitConnection || null;
            var effectiveType = String((connection && connection.effectiveType) || '');
            var saveData = !!(connection && connection.saveData);
            var slowNetwork = /(^|slow-)?2g/.test(effectiveType);
            var deviceMemory = Number(nav.deviceMemory || 8);
            var lowMemory = deviceMemory > 0 && deviceMemory <= 4;
            return {
                visibleChunkBuffer: 1,
                prefetchExtraRings: saveData ? 0 : (slowNetwork ? 0 : (lowMemory ? 1 : 2)),
                prefetchBatchSize: saveData ? 0 : (lowMemory ? 4 : 8),
                prefetchDelay: slowNetwork ? 240 : (lowMemory ? 180 : 90),
                eagerSearchWarmup: !saveData && !slowNetwork && !lowMemory,
            };
        }

        function scheduleIdleTask(callback, timeout) {
            if (typeof window.requestIdleCallback === 'function') {
                return { type: 'idle', id: window.requestIdleCallback(callback, { timeout: timeout || 250 }) };
            }
            return {
                type: 'timeout',
                id: window.setTimeout(function () {
                    callback({ didTimeout: true, timeRemaining: function () { return 0; } });
                }, timeout || 120)
            };
        }

        function cancelIdleTask(handle) {
            if (!handle) return;
            if (handle.type === 'idle' && typeof window.cancelIdleCallback === 'function') {
                window.cancelIdleCallback(handle.id);
                return;
            }
            clearTimeout(handle.id);
        }

        function createMarkerApiClient() {
            function markerVersionTag() {
                return A.markerDataVersion || 'static';
            }

            function fetchMarkerJson(path, scope, identity) {
                var versionTag = markerVersionTag();
                var version = encodeURIComponent(versionTag);
                var cacheKey = scope + ':' + versionTag + ':' + identity;
                var sep = path.indexOf('?') >= 0 ? '&' : '?';
                var url = path + sep + 'v=' + version;
                return AppCommon.fetchJSONCached(url, {
                    cacheKey: cacheKey,
                    persistent: true,
                    memory: false,
                    clone: false,
                });
            }

            return {
                fetchChunks: function (keys) {
                    var identity = (keys || []).join('|');
                    return fetchMarkerJson('/api/markers/chunks?keys=' + encodeURIComponent((keys || []).join(',')), 'markers-chunks', identity);
                },
                fetchDetails: function (ids) {
                    var identity = (ids || []).join('|');
                    return fetchMarkerJson('/api/markers/details?ids=' + encodeURIComponent((ids || []).join(',')), 'markers-details', identity);
                },
                fetchSearchIndex: function () {
                    return fetchMarkerJson('/api/markers/search_index', 'markers-search-index', 'all');
                }
            };
        }

        var markerApiClient = createMarkerApiClient();

        function currentMarkerLoadGeneration() {
            return Number(A.markerLoadGeneration || 0);
        }

        function isCurrentMarkerLoadGeneration(generation) {
            return generation === currentMarkerLoadGeneration();
        }

        function startRippleAnim() {
            if (A.isLowPowerMode) return;
            if (rippleAnimId !== null) return;
            (function loop() {
                if (!A.selectedMarkerId) { rippleAnimId = null; return; }
                if (A.isLowPowerMode) { rippleAnimId = null; return; }
                if ((++rippleFrameTick & 1) === 0) requestMarkerRender();
                rippleAnimId = requestAnimationFrame(loop);
            })();
        }
        function stopRippleAnim() {
            if (rippleAnimId !== null) { cancelAnimationFrame(rippleAnimId); rippleAnimId = null; }
        }
        A.startRippleAnim = startRippleAnim;
        A.stopRippleAnim = stopRippleAnim;

        function hashColor(seed) {
            if (A.categoryColors[seed]) return A.categoryColors[seed];
            var hash = 0;
            for (var i = 0; i < seed.length; i++) hash = ((hash << 5) - hash) + seed.charCodeAt(i);
            A.categoryColors[seed] = 'hsl(' + (Math.abs(hash) % 360) + 'deg 72% 58%)';
            return A.categoryColors[seed];
        }

        function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

        function getCurrentSearchText() {
            return (document.getElementById('markerSearchInput').value || '').trim().toLowerCase();
        }

        function getChunkKeyForPoint(x, y) {
            var chunkSize = Math.max(128, Number(A.markerChunkSize || 768));
            return Math.max(0, Math.floor(x / chunkSize)) + ':' + Math.max(0, Math.floor(y / chunkSize));
        }

        function getMarkerMeta(marker) {
            return A.categories[String((marker || {}).markType || '')] || {};
        }

        function getMarkerTitle(marker) {
            if (!marker) return '';
            var detail = A.markerDetails[marker.id] || {};
            return detail.title || marker.title || ('资源点 #' + marker.id);
        }

        function getMarkerDescription(marker) {
            if (!marker) return '';
            var detail = A.markerDetails[marker.id] || {};
            return detail.description || marker.description || '';
        }

        function getMarkerSearchText(marker) {
            if (!marker) return '';
            if (marker.searchText) return marker.searchText;
            var meta = getMarkerMeta(marker);
            return [
                getMarkerTitle(marker),
                getMarkerDescription(marker),
                meta.name || '',
                meta.group || ''
            ].join(' ').toLowerCase();
        }

        function getSearchCollection() {
            return A.searchIndexLoaded && A.searchIndex.length ? A.searchIndex : A.markers;
        }

        function buildActiveTypesSignature() {
            return Array.from(A.activeTypes).sort().join('|');
        }

        function getSelectedMarkerRef() {
            if (!A.selectedMarkerId) return null;
            return A.markersById[A.selectedMarkerId] || A.searchIndexById[A.selectedMarkerId] || null;
        }

        function isSidebarUiVisible() {
            var sidebar = document.querySelector('.terra-sidebar');
            if (!sidebar) return true;
            if (document.documentElement.getAttribute('data-sidebar-collapsed') === '1') return false;
            return !sidebar.classList.contains('is-collapsed');
        }

        function getMarkerRenderPrefs() {
            var sz = Math.max(8, Number(A.prefs.markerSize || 10));
            var opacity = Math.max(0.15, Math.min(1, Number(A.prefs.markerOpacity || 0.88)));
            var zoomBoost = A.scale > 1 ? Math.pow(A.scale, 0.55) : 1;
            var iconSize = Math.max(10, sz * 2.8 * zoomBoost);
            return {
                iconSize: iconSize,
                opacity: opacity,
                hitRadius: Math.max(iconSize * 0.56, 14),
                fallbackRadius: Math.max(iconSize * 0.30, 5)
            };
        }

        function normalizeSearchEntry(item) {
            if (!item) return null;
            var meta = A.categories[String(item.markType || '')] || {};
            var title = String(item.title || meta.name || ('资源点 #' + item.id));
            var description = String(item.description || '');
            return {
                id: String(item.id),
                markType: String(item.markType || ''),
                x: Number(item.x) || 0,
                y: Number(item.y) || 0,
                title: title,
                description: description,
                searchText: [title, description, meta.name || '', meta.group || ''].join(' ').toLowerCase()
            };
        }

        function normalizeMarkerRecord(item) {
            if (!item) return null;
            var markType = String(item.markType || '');
            var meta = A.categories[markType] || {};
            var title = String(item.title || meta.name || ('资源点 #' + item.id));
            var marker = {
                id: String(item.id),
                markType: markType,
                x: Number(item.x) || 0,
                y: Number(item.y) || 0,
                title: title,
                description: '',
                iconUrl: '/img/' + markType + '.png',
                color: hashColor((meta.group || '资源') + ':' + markType),
                isVisible: true,
            };
            marker.searchText = [title, meta.name || '', meta.group || ''].join(' ').toLowerCase();
            marker.iconImage = marker.iconUrl ? getMarkerIcon(marker.iconUrl) : null;
            return marker;
        }

        function addLoadedMarker(item) {
            var marker = normalizeMarkerRecord(item);
            if (!marker || A.markersById[marker.id]) return null;
            A.markersById[marker.id] = marker;
            A.markers.push(marker);
            return marker;
        }

        function updateMarkerLayerChip() {
            var chip = document.getElementById('clusterStateChip');
            if (!chip) return;
            var totalChunks = Number(A.totalChunkCount || 0);
            var loadedChunks = A.loadedMarkerChunks ? A.loadedMarkerChunks.size : 0;
            var prefetchingChunks = A.prefetchingMarkerChunkKeys ? A.prefetchingMarkerChunkKeys.size : 0;
            var pendingChunks = A.pendingMarkerChunkKeys ? Math.max(0, A.pendingMarkerChunkKeys.size - prefetchingChunks) : 0;
            var text = 'Canvas 点位层';
            if (totalChunks > 0) text += ' · ' + loadedChunks + '/' + totalChunks + ' 块';
            if (pendingChunks > 0) text += ' · 加载中 ' + pendingChunks;
            if (prefetchingChunks > 0) text += ' · 预取 ' + prefetchingChunks;
            if (A.searchIndexLoading) text += ' · 搜索索引';
            chip.textContent = text;
        }
        A.updateMarkerLayerChip = updateMarkerLayerChip;

        function scoreSearchMatch(marker, search) {
            var score = 0;
            var title = getMarkerTitle(marker).toLowerCase();
            var meta = getMarkerMeta(marker);
            var catName = String(meta.name || '').toLowerCase();
            var grpName = String(meta.group || '').toLowerCase();
            var searchText = getMarkerSearchText(marker);
            if (title === search) score += 140;
            if (title.indexOf(search) === 0) score += 80;
            if (catName.indexOf(search) === 0) score += 55;
            if (grpName.indexOf(search) === 0) score += 35;
            if (searchText.indexOf(search) >= 0) score += 15;
            return score;
        }

        function zoomToCluster(hit) {
            var ts = clamp(Math.max(A.scale * 1.7, 0.36), 0.08, 15);
            A.scale = ts;
            A.offsetX = A.MC.clientWidth / 2 - hit.mapX * ts;
            A.offsetY = A.MC.clientHeight / 2 - hit.mapY * ts;
            A.applyTransform();
            AppCommon.toast('已放大查看聚合点位', 'success', 1200);
        }
        A.zoomToCluster = zoomToCluster;

        /* ── drawing helpers ── */
        function drawSelectionRipple(ctx, sx, sy, baseR) {
            if (A.isDragging) {
                ctx.beginPath(); ctx.arc(sx, sy, baseR + 2.5, 0, Math.PI * 2);
                ctx.strokeStyle = '#ffc040'; ctx.lineWidth = 2.5; ctx.stroke();
                return;
            }
            var now = performance.now(), period = 1500;
            var prevShadowColor = ctx.shadowColor, prevShadowBlur = ctx.shadowBlur;
            for (var ri = 0; ri < 2; ri++) {
                var t = ((now + ri * period * 0.5) % period) / period;
                ctx.beginPath(); ctx.arc(sx, sy, baseR * (1 + t * 1.1), 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(255,180,40,' + ((1 - t) * 0.55).toFixed(2) + ')';
                ctx.lineWidth = 2; ctx.stroke();
            }
            ctx.beginPath(); ctx.arc(sx, sy, baseR + 2.5, 0, Math.PI * 2);
            ctx.strokeStyle = '#ffc040'; ctx.lineWidth = 3;
            ctx.shadowColor = 'rgba(255,160,0,0.9)'; ctx.shadowBlur = 14;
            ctx.stroke();
            ctx.shadowColor = prevShadowColor; ctx.shadowBlur = prevShadowBlur;
        }

        function drawMarkerGlyph(ctx, marker, sx, sy, rp) {
            if (marker.id === A.selectedMarkerId) drawSelectionRipple(ctx, sx, sy, rp.iconSize * 0.56);
            if (marker.iconImage && marker.iconImage.complete && marker.iconImage.naturalWidth) {
                var prevAlpha = ctx.globalAlpha;
                ctx.globalAlpha = rp.opacity;
                ctx.drawImage(marker.iconImage, sx - rp.iconSize / 2, sy - rp.iconSize / 2, rp.iconSize, rp.iconSize);
                ctx.globalAlpha = prevAlpha;
            } else {
                ctx.beginPath(); ctx.arc(sx, sy, rp.fallbackRadius, 0, Math.PI * 2);
                var prevAlpha2 = ctx.globalAlpha;
                ctx.fillStyle = marker.color || '#d7a95f'; ctx.globalAlpha = rp.opacity; ctx.fill();
                ctx.globalAlpha = 1; ctx.lineWidth = 1.4; ctx.strokeStyle = 'rgba(255,255,255,0.88)'; ctx.stroke();
                ctx.globalAlpha = prevAlpha2;
            }
        }

        function drawClusterIcon(ctx, bucket, sx, sy, rp) {
            var rep = bucket.markers[0], isz = 36, radius = isz / 2;
            if (bucket.selected) drawSelectionRipple(ctx, sx, sy, isz * 0.56);
            if (rep.iconImage && rep.iconImage.complete && rep.iconImage.naturalWidth) {
                ctx.save(); ctx.globalAlpha = rp.opacity;
                ctx.drawImage(rep.iconImage, sx - isz / 2, sy - isz / 2, isz, isz); ctx.restore();
            } else {
                ctx.save(); ctx.beginPath(); ctx.arc(sx, sy, rp.fallbackRadius, 0, Math.PI * 2);
                ctx.fillStyle = rep.color || '#d7a95f'; ctx.globalAlpha = rp.opacity; ctx.fill();
                ctx.globalAlpha = 1; ctx.lineWidth = 1.4; ctx.strokeStyle = 'rgba(255,255,255,0.88)'; ctx.stroke(); ctx.restore();
            }
            var count = bucket.markers.length;
            if (count > 1) {
                var label = count > 99 ? '99+' : String(count);
                var fs = Math.max(8, Math.round(isz * 0.36));
                ctx.save(); ctx.globalAlpha = Math.min(rp.opacity, 0.80);
                ctx.font = '700 ' + fs + 'px "Segoe UI",sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.lineWidth = Math.max(1.5, fs * 0.34); ctx.lineJoin = 'round';
                ctx.strokeStyle = 'rgba(255,255,255,0.92)'; ctx.strokeText(label, sx, sy + isz / 2 + fs * 0.7);
                ctx.fillStyle = '#1a1008'; ctx.fillText(label, sx, sy + isz / 2 + fs * 0.7);
                ctx.restore();
            }
            return radius;
        }

        function getFilteredSearchCollection(search) {
            var source = getSearchCollection();
            var sourceTag = (A.searchIndexLoaded && A.searchIndex.length) ? 'search-index' : 'markers';
            var sourceLen = source.length;
            var activeSig = buildActiveTypesSignature();
            if (searchFilterCache.results &&
                searchFilterCache.query === search &&
                searchFilterCache.activeSig === activeSig &&
                searchFilterCache.sourceTag === sourceTag &&
                searchFilterCache.sourceLen === sourceLen) {
                return searchFilterCache.results;
            }

            var results = source.filter(function (marker) {
                return A.activeTypes.has(String(marker.markType || '')) && (!search || getMarkerSearchText(marker).indexOf(search) >= 0);
            });

            searchFilterCache.query = search;
            searchFilterCache.activeSig = activeSig;
            searchFilterCache.sourceTag = sourceTag;
            searchFilterCache.sourceLen = sourceLen;
            searchFilterCache.results = results;
            return results;
        }

        function updateMarkerCountChipText(search, loadedVisibleCount, totalVisibleCount) {
            var chip = document.getElementById('markerCountChip');
            if (!chip) return '';
            var nextText = '';
            var noChunkReadyYet = !A.loadedMarkerChunks || A.loadedMarkerChunks.size === 0;
            var overallTotal = Math.max(0, Number(A.totalMarkerCount || 0));
            if (!search && noChunkReadyYet && loadedVisibleCount === 0 && totalVisibleCount > 0) {
                try {
                    nextText = localStorage.getItem('map_marker_count_snapshot') || '📍 加载中…';
                } catch (_e) {
                    nextText = '📍 加载中…';
                }
            } else {
                if (search && !A.searchIndexLoaded) {
                    nextText = '📍 筛选中… / 总 ' + (overallTotal || totalVisibleCount);
                } else {
                    nextText = '📍 筛选 ' + totalVisibleCount + ' / 总 ' + (overallTotal || totalVisibleCount);
                }
            }
            chip.textContent = nextText;
            return nextText;
        }

        function updateActiveCategoryChipText() {
            var chip = document.getElementById('activeCategoryChip');
            if (!chip) return '';
            var nextText = A.activeTypes.size + ' / ' + Object.keys(A.categories).length + ' 类';
            chip.textContent = nextText;
            return nextText;
        }

        function persistSidebarStatSnapshots(markerText, categoryText) {
            try {
                if (markerText && markerText.indexOf('📍 筛选 ') === 0) {
                    localStorage.setItem('map_marker_count_snapshot', markerText);
                }
                if (categoryText) localStorage.setItem('map_active_category_snapshot', categoryText);
            } catch (_e) {}
        }

        /* ── canvas ── */
        function ensureMarkerCanvasSize() {
            var dpr = Math.min(window.devicePixelRatio || 1, 2); /* cap at 2 — prevent VRAM explosion on HiDPI */
            var w = Math.max(A.MC.clientWidth, 1), h = Math.max(A.MC.clientHeight, 1);
            var pw = Math.round(w * dpr), ph = Math.round(h * dpr);
            if (A.MARKER_CANVAS.width !== pw || A.MARKER_CANVAS.height !== ph) {
                A.MARKER_CANVAS.width = pw; A.MARKER_CANVAS.height = ph;
                A.MARKER_CANVAS.style.width = w + 'px'; A.MARKER_CANVAS.style.height = h + 'px';
            }
            return { width: w, height: h, dpr: dpr };
        }
        A.ensureMarkerCanvasSize = ensureMarkerCanvasSize;

        function requestMarkerRender() {
            if (A.isLowPowerMode) {
                A.markerRenderQueued = false;
                return;
            }
            var now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            var holdUntil = Number(A.uiAnimationHoldUntil || 0);
            if (holdUntil > now) {
                if (!A.uiAnimationRenderTimer) {
                    A.uiAnimationRenderTimer = window.setTimeout(function () {
                        A.uiAnimationRenderTimer = null;
                        requestMarkerRender();
                    }, Math.max(16, Math.ceil(holdUntil - now) + 8));
                }
                return;
            }
            if (A.markerRenderQueued) return;
            A.markerRenderQueued = true;
            requestAnimationFrame(renderMarkers);
        }
        A.requestMarkerRender = requestMarkerRender;
        A.renderMarkersNow = function () {
            A.markerRenderQueued = false;
            renderMarkers();
        };

        function getMarkerIcon(url) {
            if (!url) return null;
            if (A.markerIconCache[url]) return A.markerIconCache[url];
            var img = new Image(); img.decoding = 'async';
            img.onload = requestMarkerRender; img.onerror = requestMarkerRender;
            img.src = url; A.markerIconCache[url] = img;
            return img;
        }

        function renderMarkers() {
            A.markerRenderQueued = false;
            if (!A.mapInfo || !A.markerCtx) return;
            var cs = ensureMarkerCanvasSize();
            var w = cs.width, h = cs.height, dpr = cs.dpr;
            var rp = getMarkerRenderPrefs();
            var ctx = A.markerCtx;
            var scale = A.scale, ox = A.offsetX, oy = A.offsetY;
            /* Start clustering from 40%, and strengthen progressively while zooming out */
            var clusterMode = scale < 0.40;
            var buckets = Object.create(null);
            var selectedPointIds = A.getSelectedPointIds ? A.getSelectedPointIds() : null;

            A.renderedMarkerHits = [];
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);
            ctx.imageSmoothingEnabled = !A.isDragging && scale > 0.25;
            ctx.imageSmoothingQuality = 'low';
            var clusterScreenCell =
                scale < 0.10 ? 160 :
                scale < 0.14 ? 144 :
                scale < 0.18 ? 128 :
                scale < 0.24 ? 114 :
                scale < 0.32 ? 100 :
                88;
            var cellSz = clusterMode ? (clusterScreenCell / scale) : 0;

            A.markers.forEach(function (marker) {
                if (!marker.isVisible) return;
                var sx = marker.x * scale + ox, sy = marker.y * scale + oy;
                if (sx < -rp.hitRadius || sy < -rp.hitRadius || sx > w + rp.hitRadius || sy > h + rp.hitRadius) return;
                if (clusterMode) {
                    var key = Math.floor(marker.x / cellSz) + ':' + Math.floor(marker.y / cellSz) + ':' + String(marker.markType || '');
                    var bucket = buckets[key];
                    if (!bucket) bucket = buckets[key] = { markers: [], sxSum: 0, sySum: 0, xSum: 0, ySum: 0, color: marker.color, selected: false };
                    bucket.markers.push(marker); bucket.sxSum += sx; bucket.sySum += sy; bucket.xSum += marker.x; bucket.ySum += marker.y;
                    bucket.selected = bucket.selected || marker.id === A.selectedMarkerId || (selectedPointIds && selectedPointIds.has(marker.id));
                } else {
                    drawMarkerGlyph(ctx, marker, sx, sy, rp);
                    A.renderedMarkerHits.push({ type: 'marker', marker: marker, x: sx, y: sy, radius: rp.hitRadius });
                }
            });

            if (clusterMode) {
                Object.keys(buckets).forEach(function (key) {
                    var bucket = buckets[key], count = bucket.markers.length;
                    var ax = bucket.sxSum / count, ay = bucket.sySum / count, mx = bucket.xSum / count, my = bucket.ySum / count;
                    if (count === 1) {
                        drawClusterIcon(ctx, bucket, ax, ay, rp);
                        A.renderedMarkerHits.push({ type: 'marker', marker: bucket.markers[0], x: ax, y: ay, radius: 18 });
                    } else {
                        var radius = drawClusterIcon(ctx, bucket, ax, ay, rp);
                        A.renderedMarkerHits.push({ type: 'cluster', cluster: true, markers: bucket.markers, x: ax, y: ay, radius: radius, mapX: mx, mapY: my });
                    }
                });

                
            }

            /* route/selection overlay — skip during fast drag for perf */
            if (!A.isDragging) {
                var hasSelPts = A.selectedPoints && A.selectedPoints.length > 0;
                if ((A.routeResult && A.routeResult.order.length > 1) || A.routeWaypoints.length > 0 || hasSelPts) {
                    if (A.drawRouteOverlay) A.drawRouteOverlay(ctx, dpr);
                }
            }
        }

        function hitTestMarkerAt(clientX, clientY) {
            if (!A.renderedMarkerHits.length) return null;
            var rect = A.MC.getBoundingClientRect(), px = clientX - rect.left, py = clientY - rect.top;
            for (var i = A.renderedMarkerHits.length - 1; i >= 0; i--) {
                var hit = A.renderedMarkerHits[i], dx = px - hit.x, dy = py - hit.y;
                if (dx * dx + dy * dy <= hit.radius * hit.radius) return hit;
            }
            return null;
        }
        A.hitTestMarkerAt = hitTestMarkerAt;

        /* ── filter persistence ── */
        function saveMapFilters() {
            if (!A.prefs.rememberMapFilters) return;
            var payload = {
                activeTypes: Array.from(A.activeTypes),
                searchText: getCurrentSearchText(),
            };
            localStorage.setItem(A.mapFiltersStorageKey, JSON.stringify(payload));
        }
        A.saveMapFilters = saveMapFilters;

        function loadMapFilters() {
            if (!A.prefs.rememberMapFilters) return;
            try {
                var raw = localStorage.getItem(A.mapFiltersStorageKey);
                if (!raw) return;
                var parsed = JSON.parse(raw);
                var input = document.getElementById('markerSearchInput');
                if (!parsed || typeof parsed !== 'object') return;

                if (Array.isArray(parsed.activeTypes)) {
                    A.activeTypes = new Set(parsed.activeTypes);
                }
                if (input && typeof parsed.searchText === 'string') {
                    input.value = parsed.searchText;
                }
            } catch (_e) {}
        }
        A.loadMapFilters = loadMapFilters;

        function fitMap(options) {
            options = options || {};
            if (!A.mapInfo) return;
            var metrics = A.getViewportOcclusionInsets ? A.getViewportOcclusionInsets() : { left: 0, right: 0 };
            var cw = A.MC.clientWidth, ch = A.MC.clientHeight;
            var visibleWidth = Math.max(1, cw - (metrics.left || 0) - (metrics.right || 0));
            var fitScale = A.getFitScaleForViewport
                ? A.getFitScaleForViewport(visibleWidth, ch)
                : Math.min(cw / A.mapInfo.map_width, ch / A.mapInfo.map_height);
            A.scale = options.mode === 'initial' && A.getRecommendedInitialScale
                ? A.getRecommendedInitialScale(fitScale)
                : fitScale;
            A.offsetX = (metrics.left || 0) + (visibleWidth - A.mapInfo.map_width * A.scale) / 2;
            A.offsetY = (ch - A.mapInfo.map_height * A.scale) / 2;
            if (A.clampViewportOffset) {
                var clamped = A.clampViewportOffset(A.offsetX, A.offsetY, A.scale);
                A.offsetX = clamped.x;
                A.offsetY = clamped.y;
            }
            A.applyTransform({ clamp: false });
        }
        A.fitMap = fitMap;

        /* ── visible chunks ── */
        function getViewportMapBounds() {
            var occlusion = A.getViewportOcclusionInsets ? A.getViewportOcclusionInsets() : { left: 0, right: 0 };
            var scale = Math.max(A.scale || 1, 0.01);
            return {
                minX: clamp((occlusion.left - A.offsetX) / scale, 0, A.mapInfo.map_width),
                minY: clamp((-A.offsetY) / scale, 0, A.mapInfo.map_height),
                maxX: clamp((A.MC.clientWidth - occlusion.right - A.offsetX) / scale, 0, A.mapInfo.map_width),
                maxY: clamp((A.MC.clientHeight - A.offsetY) / scale, 0, A.mapInfo.map_height),
            };
        }

        function computeChunkKeysForBuffer(buffer) {
            if (!A.mapInfo || !A.markerChunkSize) return [];
            var bounds = getViewportMapBounds();
            var chunkSize = A.markerChunkSize;
            var maxCX = Math.max(0, Math.ceil(A.mapInfo.map_width / chunkSize) - 1);
            var maxCY = Math.max(0, Math.ceil(A.mapInfo.map_height / chunkSize) - 1);
            var startCX = clamp(Math.floor(bounds.minX / chunkSize) - buffer, 0, maxCX);
            var startCY = clamp(Math.floor(bounds.minY / chunkSize) - buffer, 0, maxCY);
            var endCX = clamp(Math.floor(bounds.maxX / chunkSize) + buffer, 0, maxCX);
            var endCY = clamp(Math.floor(bounds.maxY / chunkSize) + buffer, 0, maxCY);
            var keys = [];
            for (var cx = startCX; cx <= endCX; cx++) {
                for (var cy = startCY; cy <= endCY; cy++) keys.push(cx + ':' + cy);
            }
            if (A.availableMarkerChunks && A.availableMarkerChunks.size) {
                keys = keys.filter(function (key) { return A.availableMarkerChunks.has(key); });
            }
            return keys;
        }

        function computeVisibleChunkKeys() {
            return computeChunkKeysForBuffer(Math.max(0, Number(CLIENT_PROFILE.visibleChunkBuffer || 1)));
        }

        function sortChunkKeysByDistance(keys) {
            if (!keys.length || !A.markerChunkSize) return keys;
            var bounds = getViewportMapBounds();
            var centerX = (bounds.minX + bounds.maxX) / 2;
            var centerY = (bounds.minY + bounds.maxY) / 2;
            var chunkSize = A.markerChunkSize;
            return keys.slice().sort(function (lhs, rhs) {
                var l = lhs.split(':');
                var r = rhs.split(':');
                var ldx = ((Number(l[0]) + 0.5) * chunkSize) - centerX;
                var ldy = ((Number(l[1]) + 0.5) * chunkSize) - centerY;
                var rdx = ((Number(r[0]) + 0.5) * chunkSize) - centerX;
                var rdy = ((Number(r[1]) + 0.5) * chunkSize) - centerY;
                return (ldx * ldx + ldy * ldy) - (rdx * rdx + rdy * rdy);
            });
        }

        function computeAdjacentPrefetchChunkKeys() {
            var visibleBuffer = Math.max(0, Number(CLIENT_PROFILE.visibleChunkBuffer || 1));
            var extraRings = Math.max(0, Number(CLIENT_PROFILE.prefetchExtraRings || 0));
            if (extraRings <= 0) return [];

            var visibleKeys = computeChunkKeysForBuffer(visibleBuffer);
            var outerKeys = computeChunkKeysForBuffer(visibleBuffer + extraRings);
            var visibleMap = Object.create(null);
            visibleKeys.forEach(function (key) { visibleMap[key] = true; });
            return sortChunkKeysByDistance(outerKeys.filter(function (key) {
                return !visibleMap[key];
            }));
        }

        function cancelScheduledAdjacentChunkPrefetch() {
            if (prefetchIdleHandle) {
                cancelIdleTask(prefetchIdleHandle);
                prefetchIdleHandle = null;
            }
            prefetchGeneration += 1;
        }

        function resetAdjacentChunkPrefetch() {
            cancelScheduledAdjacentChunkPrefetch();
            if (A.prefetchingMarkerChunkKeys && A.prefetchingMarkerChunkKeys.size) {
                A.prefetchingMarkerChunkKeys.clear();
                updateMarkerLayerChip();
            }
        }

        function cancelSearchIndexWarmup() {
            if (!searchWarmIdleHandle) return;
            cancelIdleTask(searchWarmIdleHandle);
            searchWarmIdleHandle = null;
        }

        function shouldScheduleAdjacentChunkPrefetch() {
            if (!A.markerDataVersion || !A.markerChunkSize || !A.mapInfo) return false;
            if (Number(CLIENT_PROFILE.prefetchBatchSize || 0) <= 0 || Number(CLIENT_PROFILE.prefetchExtraRings || 0) <= 0) return false;
            if (window._isUnloading) return false;
            if (typeof document !== 'undefined' && document.hidden) return false;
            if (A.dragging || A.isDragging) return false;
            return true;
        }

        function loadMarkerChunksByKeys(rawKeys, options) {
            options = options || {};
            var generation = options.generation != null ? Number(options.generation) : currentMarkerLoadGeneration();
            if (!A.markerDataVersion || !A.markerChunkSize || !isCurrentMarkerLoadGeneration(generation)) {
                updateMarkerLayerChip();
                return Promise.resolve([]);
            }

            var keys = [];
            var seen = Object.create(null);
            (rawKeys || []).forEach(function (key) {
                key = String(key || '').trim();
                if (!key || seen[key]) return;
                if (A.availableMarkerChunks && A.availableMarkerChunks.size && !A.availableMarkerChunks.has(key)) return;
                seen[key] = true;
                keys.push(key);
            });
            if (!keys.length) {
                updateMarkerLayerChip();
                return Promise.resolve([]);
            }

            var loadedSet = A.loadedMarkerChunks;
            var pendingSet = A.pendingMarkerChunkKeys;
            var prefetchSet = A.prefetchingMarkerChunkKeys;
            var missing = keys.filter(function (key) {
                return !loadedSet.has(key) && !pendingSet.has(key);
            });
            if (!missing.length) {
                updateMarkerLayerChip();
                return Promise.resolve([]);
            }

            missing.forEach(function (key) {
                pendingSet.add(key);
                if (options.prefetch) prefetchSet.add(key);
            });
            updateMarkerLayerChip();

            return markerApiClient.fetchChunks(missing).then(function (payload) {
                if (!isCurrentMarkerLoadGeneration(generation)) return [];
                var chunks = payload.chunks || {};
                missing.forEach(function (key) {
                    (chunks[key] || []).forEach(addLoadedMarker);
                    loadedSet.add(key);
                });
                if (!options.prefetch) {
                    applyMarkerFilters();
                }
                return missing;
            }).catch(function (err) {
                if (!options.silent && isCurrentMarkerLoadGeneration(generation)) {
                    AppCommon.toast('加载点位区块失败：' + err.message, 'danger', 2200);
                }
                return [];
            }).finally(function () {
                missing.forEach(function (key) {
                    pendingSet.delete(key);
                    if (options.prefetch) prefetchSet.delete(key);
                });
                if (pendingSet === A.pendingMarkerChunkKeys) updateMarkerLayerChip();
            });
        }

        function ensureVisibleMarkerChunks(options) {
            options = options || {};
            if (!A.markerDataVersion || !A.markerChunkSize) { updateMarkerLayerChip(); return Promise.resolve([]); }
            var generation = options.generation != null ? Number(options.generation) : currentMarkerLoadGeneration();
            var keys = options.keys || computeVisibleChunkKeys();
            if (!keys.length) {
                updateMarkerLayerChip();
                if (isCurrentMarkerLoadGeneration(generation)) scheduleAdjacentChunkPrefetch();
                return Promise.resolve([]);
            }
            return loadMarkerChunksByKeys(keys, {
                generation: generation,
                silent: options.silent,
            }).then(function (loadedKeys) {
                if (isCurrentMarkerLoadGeneration(generation)) scheduleAdjacentChunkPrefetch();
                return loadedKeys;
            });
        }

        function scheduleVisibleChunkSync(forceNow) {
            if (!A.markerDataVersion) return;
            if (visibleChunkSyncTimer) { clearTimeout(visibleChunkSyncTimer); visibleChunkSyncTimer = null; }
            cancelScheduledAdjacentChunkPrefetch();
            if (A.isLowPowerMode && !forceNow) return;
            if (forceNow) { ensureVisibleMarkerChunks({ generation: currentMarkerLoadGeneration() }); return; }
            visibleChunkSyncTimer = window.setTimeout(function () {
                visibleChunkSyncTimer = null;
                ensureVisibleMarkerChunks();
            }, A.dragging ? 120 : 60);
        }
        A.scheduleVisibleChunkSync = scheduleVisibleChunkSync;

        A.cancelMapBackgroundWork = function () {
            if (visibleChunkSyncTimer) {
                clearTimeout(visibleChunkSyncTimer);
                visibleChunkSyncTimer = null;
            }
            cancelSearchIndexWarmup();
            cancelScheduledAdjacentChunkPrefetch();
            stopRippleAnim();
        };

        A.syncMapPowerMode = function (isLowPower) {
            if (isLowPower) {
                A.cancelMapBackgroundWork();
                return;
            }
            if (A.selectedMarkerId) startRippleAnim();
            scheduleVisibleChunkSync(true);
            scheduleSearchIndexWarmup();
        };

        function ensureMarkerChunkLoadedForMarker(marker) {
            if (!marker) return Promise.resolve([]);
            return ensureVisibleMarkerChunks({ keys: [getChunkKeyForPoint(marker.x, marker.y)] });
        }

        function ensureMarkerDetailsLoaded(ids) {
            var neededIds = (ids || []).map(function (id) { return String(id || '').trim(); }).filter(Boolean);
            if (!neededIds.length || !A.markerDataVersion) return Promise.resolve(A.markerDetails);

            var missingIds = neededIds.filter(function (id) { return !A.markerDetails[id] && !A.markerDetailRequests[id]; });
            if (!missingIds.length) {
                var waiters = neededIds.map(function (id) { return A.markerDetailRequests[id]; }).filter(Boolean);
                return waiters.length ? Promise.all(waiters).then(function () { return A.markerDetails; }) : Promise.resolve(A.markerDetails);
            }

            var generation = currentMarkerLoadGeneration();
            var requestMap = A.markerDetailRequests;
            var requestPromise = markerApiClient.fetchDetails(missingIds).then(function (payload) {
                if (!isCurrentMarkerLoadGeneration(generation)) return A.markerDetails;
                var items = payload.items || {};
                Object.keys(items).forEach(function (id) {
                    var detail = items[id] || {};
                    A.markerDetails[id] = {
                        title: String(detail.title || ((A.markersById[id] || {}).title) || ((A.searchIndexById[id] || {}).title) || ('资源点 #' + id)),
                        description: String(detail.description || ''),
                    };
                });
                return A.markerDetails;
            }).catch(function (err) {
                AppCommon.toast('加载点位详情失败：' + err.message, 'warning', 2200);
                return A.markerDetails;
            }).finally(function () {
                missingIds.forEach(function (id) {
                    if (requestMap[id] === requestPromise) delete requestMap[id];
                });
            });

            missingIds.forEach(function (id) { requestMap[id] = requestPromise; });
            return requestPromise;
        }

        function hydrateSearchIndex(items) {
            A.searchIndex = (items || []).map(normalizeSearchEntry).filter(Boolean);
            A.searchIndexById = Object.create(null);
            A.searchIndex.forEach(function (item) {
                A.searchIndexById[item.id] = item;
                if (!A.markerDetails[item.id]) {
                    A.markerDetails[item.id] = {
                        title: item.title,
                        description: item.description || '',
                    };
                }
            });
            A.searchIndexLoaded = true;
            return A.searchIndex;
        }

        function onSearchIndexReady() {
            updateMarkerLayerChip();
            if (getCurrentSearchText()) applyMarkerFilters();
        }

        function warmSearchIndexFromOffline() {
            if (A.searchIndexLoaded) return Promise.resolve(A.searchIndex);
            if (A.searchIndexWarmPromise) return A.searchIndexWarmPromise;
            if (!A.markerDataVersion || !AppCommon.readOfflineJSON) return Promise.resolve([]);

            var generation = currentMarkerLoadGeneration();
            A.searchIndexWarmPromise = AppCommon.readOfflineJSON(SEARCH_INDEX_OFFLINE_KEY).then(function (snapshot) {
                if (!isCurrentMarkerLoadGeneration(generation)) return [];
                if (!snapshot || snapshot.version !== A.markerDataVersion || !Array.isArray(snapshot.items)) return [];
                hydrateSearchIndex(snapshot.items);
                onSearchIndexReady();
                return A.searchIndex;
            }).catch(function () {
                return [];
            }).finally(function () {
                if (isCurrentMarkerLoadGeneration(generation)) A.searchIndexWarmPromise = null;
            });
            return A.searchIndexWarmPromise;
        }

        function ensureSearchIndexLoaded(options) {
            options = options || {};
            if (A.searchIndexLoaded) return Promise.resolve(A.searchIndex);
            if (A.searchIndexPromise) return A.searchIndexPromise;
            if (!A.markerDataVersion) return Promise.resolve([]);

            var generation = currentMarkerLoadGeneration();
            A.searchIndexPromise = Promise.resolve(warmSearchIndexFromOffline()).then(function () {
                if (A.searchIndexLoaded || !isCurrentMarkerLoadGeneration(generation)) return A.searchIndex;

                A.searchIndexLoading = true;
                updateMarkerLayerChip();
                return markerApiClient.fetchSearchIndex().then(function (payload) {
                    if (!isCurrentMarkerLoadGeneration(generation)) return [];
                    var items = Array.isArray(payload.items) ? payload.items : [];
                    hydrateSearchIndex(items);
                    if (AppCommon.writeOfflineJSON) {
                        AppCommon.writeOfflineJSON(SEARCH_INDEX_OFFLINE_KEY, {
                            version: A.markerDataVersion,
                            items: items,
                            updatedAt: Date.now(),
                        }).catch(function () {});
                    }
                    return A.searchIndex;
                }).catch(function (err) {
                    if (!options.background) AppCommon.toast('搜索索引加载失败：' + err.message, 'warning', 2200);
                    return [];
                });
            }).finally(function () {
                if (!isCurrentMarkerLoadGeneration(generation)) return;
                A.searchIndexLoading = false;
                A.searchIndexPromise = null;
                onSearchIndexReady();
            });
            return A.searchIndexPromise;
        }

        function prefetchAdjacentChunkBatch(runGeneration) {
            if (runGeneration !== prefetchGeneration || !shouldScheduleAdjacentChunkPrefetch()) return;
            if (A.pendingMarkerChunkKeys.size > 12) {
                scheduleAdjacentChunkPrefetch();
                return;
            }

            var batchSize = Math.max(1, Number(CLIENT_PROFILE.prefetchBatchSize || 4));
            var batch = computeAdjacentPrefetchChunkKeys().slice(0, batchSize);
            if (!batch.length) {
                updateMarkerLayerChip();
                return;
            }

            loadMarkerChunksByKeys(batch, {
                prefetch: true,
                silent: true,
                generation: currentMarkerLoadGeneration(),
            }).finally(function () {
                if (runGeneration === prefetchGeneration) scheduleAdjacentChunkPrefetch();
            });
        }

        function scheduleAdjacentChunkPrefetch() {
            cancelScheduledAdjacentChunkPrefetch();
            if (!shouldScheduleAdjacentChunkPrefetch()) return;

            var runGeneration = ++prefetchGeneration;
            prefetchIdleHandle = scheduleIdleTask(function () {
                prefetchIdleHandle = null;
                prefetchAdjacentChunkBatch(runGeneration);
            }, CLIENT_PROFILE.prefetchDelay || 120);
        }

        function scheduleSearchIndexWarmup() {
            cancelSearchIndexWarmup();
            if (!A.markerDataVersion || A.searchIndexLoaded || A.searchIndexPromise) return;
            if (!CLIENT_PROFILE.eagerSearchWarmup) return;
            if (window._isUnloading) return;
            if (typeof document !== 'undefined' && document.hidden) return;

            searchWarmIdleHandle = scheduleIdleTask(function () {
                searchWarmIdleHandle = null;
                ensureSearchIndexLoaded({ background: true });
            }, 500);
        }

        /* ── category groups ── */
        function buildCategoryGroups() {
            var grouped = {};
            Object.keys(A.categories).forEach(function (id) {
                var meta = A.categories[id], group = meta.group || '未分组';
                if (!grouped[group]) grouped[group] = [];
                grouped[group].push({ id: id, name: meta.name || id, group: group });
            });

            var groupOrder = Object.keys(grouped).sort(function (a, b) { return a.localeCompare(b, 'zh-CN'); });
            A.categoryToggleNodes = [];

            var container = document.getElementById('categoryGroups');
            container.innerHTML = '';
            groupOrder.forEach(function (groupName) {
                var wrap = document.createElement('details');
                wrap.className = 'category-folder'; wrap.open = true;
                var title = document.createElement('summary');
                title.className = 'category-folder-summary';
                title.innerHTML = '<span class="category-folder-title"><span>📂</span><span>' + AppCommon.escapeHtml(groupName) + '</span></span><span class="category-folder-meta">' + grouped[groupName].length + ' 类</span>';
                wrap.appendChild(title);
                var chipList = document.createElement('div');
                chipList.className = 'category-chip-list';
                grouped[groupName].forEach(function (item) {
                    var b = document.createElement('button');
                    b.type = 'button'; b.className = 'category-toggle'; b.dataset.type = item.id;
                    b.innerHTML = '<img class="category-icon" src="/img/' + item.id + '.png" onerror="this.style.display=\'none\'" alt=""><span>' + AppCommon.escapeHtml(item.name) + '</span><small class="category-count">' + (A.markerTypeCounts[item.id] || 0) + '</small>';
                    b.addEventListener('click', function () {
                        if (A.activeTypes.has(item.id)) A.activeTypes.delete(item.id); else A.activeTypes.add(item.id);
                        saveMapFilters(); applyMarkerFilters();
                    });
                    chipList.appendChild(b);
                    A.categoryToggleNodes.push(b);
                });
                wrap.appendChild(chipList);
                container.appendChild(wrap);
            });
        }

        /* ── details panel ── */
        function renderMarkerDetailPlaceholder() {
            if (A.getSelectedPoints && A.getSelectedPoints().length > 1) return;
            stopRippleAnim();
            var panel = document.getElementById('markerDetailPanel');
            panel.classList.add('is-empty');
            AppCommon.setInteractiveHiddenState(panel, true);
            document.getElementById('selectedMarkerChip').textContent = '未选中';
            document.getElementById('markerDetailCard').innerHTML = '<p style="margin:0;font-size:13px;color:#999;">在地图上点击资源点查看详情</p>';
        }
        A.renderMarkerDetailPlaceholder = renderMarkerDetailPlaceholder;

        function renderSelectedMarkerDetail(marker, loading) {
            if (A.getSelectedPoints && A.getSelectedPoints().length > 1) return;
            if (!marker) { renderMarkerDetailPlaceholder(); return; }
            var detail = A.markerDetails[marker.id] || {};
            var meta = getMarkerMeta(marker);
            var panel = document.getElementById('markerDetailPanel');
            panel.classList.remove('is-empty');
            AppCommon.setInteractiveHiddenState(panel, false);
            document.getElementById('markerDetailTitle').textContent = detail.title || getMarkerTitle(marker);
            document.getElementById('selectedMarkerChip').textContent = (meta.group || '未分组') + ' / ' + (meta.name || marker.markType);
            var desc = detail.description || getMarkerDescription(marker);
            document.getElementById('markerDetailCard').innerHTML =
                '<div class="resource-meta">坐标：(' + Math.round(marker.x) + ', ' + Math.round(marker.y) + ')</div>' +
                (loading ? '<div class="resource-meta" style="margin-top:6px;">⏳ 正在加载详情…</div>' : '') +
                '<p style="margin:6px 0 0;font-size:13px;color:#666;line-height:1.6;">' + AppCommon.escapeHtml(desc || (loading ? '正在加载详情…' : '暂无描述')) + '</p>';
        }

        function selectMarker(markerId, centerOnMarker, markerRef) {
            var marker = A.markersById[markerId] || markerRef || A.searchIndexById[markerId] || null;
            if (!marker) return Promise.resolve();

            A.selectedMarkerId = markerId;
            startRippleAnim();
            if (centerOnMarker) {
                A.offsetX = A.MC.clientWidth / 2 - marker.x * A.scale;
                A.offsetY = A.MC.clientHeight / 2 - marker.y * A.scale;
                A.applyTransform();
            } else {
                requestMarkerRender();
            }
            renderSelectedMarkerDetail(marker, !A.markerDetails[markerId]);

            return Promise.all([
                ensureMarkerChunkLoadedForMarker(marker),
                ensureMarkerDetailsLoaded([markerId])
            ]).then(function () {
                if (A.selectedMarkerId === markerId) {
                    renderSelectedMarkerDetail(A.markersById[markerId] || marker, false);
                    requestMarkerRender();
                }
            });
        }
        A.selectMarker = selectMarker;

        function applyMarkerFilters() {
            var search = getCurrentSearchText();
            var loadedVisibleCount = 0;

            A.markers.forEach(function (marker) {
                var categoryVisible = A.activeTypes.has(marker.markType);
                var searchVisible = !search || getMarkerSearchText(marker).indexOf(search) >= 0;
                marker.isVisible = categoryVisible && searchVisible;
                if (marker.isVisible) loadedVisibleCount++;
            });

            var selected = getSelectedMarkerRef();
            var selectedStillVisible = !selected || (A.activeTypes.has(String(selected.markType || '')) && (!search || getMarkerSearchText(selected).indexOf(search) >= 0));
            if (!selectedStillVisible && A.selectedMarkerId) A.selectedMarkerId = null;

            var totalVisibleCount;
            if (!search) {
                totalVisibleCount = Array.from(A.activeTypes).reduce(function (sum, typeId) {
                    return sum + Number(A.markerTypeCounts[typeId] || 0);
                }, 0);
            } else if (A.searchIndexLoaded) {
                totalVisibleCount = getFilteredSearchCollection(search).length;
            } else {
                totalVisibleCount = loadedVisibleCount;
            }

            if (isSidebarUiVisible()) {
                A.categoryToggleNodes.forEach(function (node) {
                    node.classList.toggle('is-active', A.activeTypes.has(node.dataset.type));
                });
                var markerChipText = updateMarkerCountChipText(search, loadedVisibleCount, totalVisibleCount);
                var activeChipText = updateActiveCategoryChipText();
                persistSidebarStatSnapshots(markerChipText, activeChipText);
            }

            if (!A.selectedMarkerId) renderMarkerDetailPlaceholder();
            else renderSelectedMarkerDetail(getSelectedMarkerRef(), !A.markerDetails[A.selectedMarkerId]);
            requestMarkerRender();
        }
        A.applyMarkerFilters = applyMarkerFilters;

        function refreshSidebarUi() {
            if (!isSidebarUiVisible()) return;
            var search = getCurrentSearchText();
            var loadedVisibleCount = 0;
            A.markers.forEach(function (marker) {
                if (marker.isVisible) loadedVisibleCount++;
            });
            var totalVisibleCount;
            if (!search) {
                totalVisibleCount = Array.from(A.activeTypes).reduce(function (sum, typeId) {
                    return sum + Number(A.markerTypeCounts[typeId] || 0);
                }, 0);
            } else if (A.searchIndexLoaded) {
                totalVisibleCount = getFilteredSearchCollection(search).length;
            } else {
                totalVisibleCount = loadedVisibleCount;
            }

            A.categoryToggleNodes.forEach(function (node) {
                node.classList.toggle('is-active', A.activeTypes.has(node.dataset.type));
            });
            var markerChipText = updateMarkerCountChipText(search, loadedVisibleCount, totalVisibleCount);
            var activeChipText = updateActiveCategoryChipText();
            persistSidebarStatSnapshots(markerChipText, activeChipText);
        }
        A.refreshMapSidebarUi = refreshSidebarUi;

        function attachMarkerSearch() {
            var input = document.getElementById('markerSearchInput');
            var suggest = document.getElementById('markerSearchSuggest');

            function closeSuggest() {
                suggest.classList.remove('is-open');
                AppCommon.setInteractiveHiddenState(suggest, true);
                suggest.innerHTML = '';
            }

            function openSuggest(search) {
                if (!search) { closeSuggest(); return; }
                var matches = getFilteredSearchCollection(search).slice().sort(function (a, b) {
                    return scoreSearchMatch(b, search) - scoreSearchMatch(a, search);
                }).slice(0, 8);

                if (!matches.length) { closeSuggest(); return; }

                var frag = document.createDocumentFragment();
                matches.forEach(function (marker) {
                    var meta = getMarkerMeta(marker);
                    var item = document.createElement('div');
                    item.className = 'search-suggest-item';
                    item.innerHTML =
                        '<img class="search-suggest-icon" src="/img/' + marker.markType + '.png" onerror="this.style.display=\'none\'" alt="">' +
                        '<span class="search-suggest-name">' + AppCommon.escapeHtml(getMarkerTitle(marker)) + '</span>' +
                        '<span class="search-suggest-cat">' + AppCommon.escapeHtml(meta.name || marker.markType) + '</span>';
                    item.addEventListener('mousedown', function (e) {
                        e.preventDefault();
                        input.value = getMarkerTitle(marker);
                        closeSuggest();
                        applyMarkerFilters();
                        selectMarker(marker.id, true, marker);
                    });
                    frag.appendChild(item);
                });
                suggest.innerHTML = '';
                suggest.appendChild(frag);
                suggest.classList.add('is-open');
                AppCommon.setInteractiveHiddenState(suggest, false);
                suggest.onmousedown = function (e) { e.preventDefault(); };
            }

            AppCommon.setInteractiveHiddenState(suggest, true);

            input.addEventListener('input', AppCommon.debounce(function () {
                var search = getCurrentSearchText();
                if (search) {
                    ensureSearchIndexLoaded().then(function () {
                        if (getCurrentSearchText() === search) {
                            openSuggest(search);
                            saveMapFilters();
                            applyMarkerFilters();
                        }
                    });
                }
                openSuggest(search);
                saveMapFilters();
                applyMarkerFilters();
            }, 120));

            var suggestBlurTimer = null;
            input.addEventListener('focus', function () {
                if (suggestBlurTimer) { clearTimeout(suggestBlurTimer); suggestBlurTimer = null; }
                var search = getCurrentSearchText();
                if (search) {
                    ensureSearchIndexLoaded({ background: true });
                    openSuggest(search);
                }
            });
            input.addEventListener('blur', function () {
                if (suggestBlurTimer) clearTimeout(suggestBlurTimer);
                suggestBlurTimer = setTimeout(function () { closeSuggest(); suggestBlurTimer = null; }, 180);
            });

            document.getElementById('showAllBtn').addEventListener('click', function () {
                A.activeTypes = new Set(Object.keys(A.categories)); saveMapFilters(); applyMarkerFilters();
            });
            document.getElementById('hideAllBtn').addEventListener('click', function () {
                A.activeTypes = new Set(); saveMapFilters(); applyMarkerFilters();
            });
            document.getElementById('refreshMarkersBtn').addEventListener('click', function () {
                loadMarkerData(true);
            });
        }
        A.attachMarkerSearch = attachMarkerSearch;

        function loadMarkerData(forceReload) {
            var manifestUrl = '/api/markers/manifest' + (forceReload ? '?refresh=1' : '');
            var nextGeneration = currentMarkerLoadGeneration() + 1;
            A.markerLoadGeneration = nextGeneration;
            cancelSearchIndexWarmup();
            resetAdjacentChunkPrefetch();
            return AppCommon.fetchJSON(manifestUrl).then(function (payload) {
                A.categories = payload.categories || {};
                A.markerTypeCounts = payload.markerTypeCounts || {};
                A.totalMarkerCount = Number(payload.totalMarkers || 0);
                A.totalChunkCount = Number(payload.totalChunkCount || 0);
                A.markerDataVersion = payload.version || 'static';
                A.markerChunkSize = Math.max(128, Number(payload.chunkSize || 768));
                A.availableMarkerChunks = new Set(payload.populatedChunkKeys || []);

                A.markers = [];
                A.markerDetails = {};
                A.markerDetailRequests = Object.create(null);
                A.markersById = Object.create(null);
                A.loadedMarkerChunks = new Set();
                A.pendingMarkerChunkKeys = new Set();
                A.prefetchingMarkerChunkKeys = new Set();
                A.searchIndex = [];
                A.searchIndexById = Object.create(null);
                A.searchIndexLoaded = false;
                A.searchIndexLoading = false;
                A.searchIndexPromise = null;
                A.searchIndexWarmPromise = null;
                A.selectedMarkerId = null;
                stopRippleAnim();

                var validTypes = Object.keys(A.categories);
                if (!A.activeTypes.size) {
                    A.activeTypes = new Set(validTypes);
                } else {
                    A.activeTypes = new Set(Array.from(A.activeTypes).filter(function (id) {
                        return Object.prototype.hasOwnProperty.call(A.categories, id);
                    }));
                    if (!A.activeTypes.size && validTypes.length) A.activeTypes = new Set(validTypes);
                }

                buildCategoryGroups();
                updateMarkerLayerChip();
                return ensureVisibleMarkerChunks({ generation: nextGeneration });
            }).then(function () {
                renderMarkerDetailPlaceholder();
                applyMarkerFilters();
                scheduleSearchIndexWarmup();
            }).catch(function (err) {
                AppCommon.toast('加载资源点失败：' + err.message, 'danger');
            });
        }
        A.loadMarkerData = loadMarkerData;

        function initializeMapImage() {
            return new Promise(function (resolve, reject) {
                function finalizeImageReady() {
                    A.mapInfo = {
                        map_width: A.MI.naturalWidth,
                        map_height: A.MI.naturalHeight,
                    };
                    ensureMarkerCanvasSize();
                    fitMap({ mode: 'initial' });
                    scheduleVisibleChunkSync(true);
                    resolve();
                }

                A.MI.decoding = 'async';
                A.MI.onload = finalizeImageReady;
                A.MI.onerror = function () { reject(new Error('大地图加载失败')); };
                A.MI.src = A.mapImageUrl;
                if (A.MI.complete && A.MI.naturalWidth) finalizeImageReady();
            });
        }
        A.initializeMapImage = initializeMapImage;

        if (typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', function () {
                if (document.hidden) {
                    if (A.cancelMapBackgroundWork) A.cancelMapBackgroundWork();
                    return;
                }
                if (A.syncMapPowerMode) A.syncMapPowerMode(false);
            });
        }

        updateMarkerLayerChip();
    }
};
