/* common.js — Shared utilities, ES module */
'use strict';

const PREFS_KEY = 'game-map-web:web:prefs';
const DOCK_POS_KEY = 'game-map-web:web:dock-position';
const JSON_RESPONSE_CACHE = 'game-map-web:web:json-cache:v1';
const OFFLINE_DB_NAME = 'game-map-web:web:offline:v1';
const OFFLINE_DB_STORE = 'json';
const JSON_MEMORY_CACHE = new Map();
const JSON_INFLIGHT_CACHE = new Map();
const JSON_MEMORY_MAX_ENTRIES = 48;
const JSON_MEMORY_MAX_BYTES = 2 * 1024 * 1024;
let jsonMemoryBytes = 0;
let offlineDbPromise = null;
const DEFAULT_PREFS = {
        logMaxLen: 6000,
        logTrimLen: 3600,
        markerSize: 10,
        markerOpacity: 0.88,
        rememberMapFilters: true,
        autoOpenAdvancedTools: false,
        preferJpegMode: false,
        mapAutoCenter: true,
        captureFps: 10,
        themeAccent: '#9a6948',
        themeBgImage: '',
        playerDotColor: '#00ff7f',
        playerArrowColor: '#30b6fe',
        routeLineColor: '#d4a050',
        routeLineOpacity: 0.8,
        themeText: '#4b3528',
        themeTextSoft: '#7b6555',
        themeTextFaint: '#a18b7b',
        themePanelStrong: '#fffdf9',
        themeMapBg: '#1b1512',
        themeSidebarBg: '#fffdf9',
        themeToolbarBg: '#261c16',
        themeHudBg: '#261c16',
        themeDetailBg: '#fffcf6',
        themeCssVarsJson: '',
    };

const FRONTEND_PREFS_META = [
        { key: 'logMaxLen', label: '日志最大长度', group: '识别页', type: 'int', min: 1000, max: 20000, description: '识别页和地图页日志面板的最大保留字符数。' },
        { key: 'logTrimLen', label: '日志裁剪长度', group: '识别页', type: 'int', min: 500, max: 12000, description: '日志超长后保留的尾部字符数。' },
        { key: 'autoOpenAdvancedTools', label: '默认展开高级工具', group: '识别页', type: 'bool', description: '识别页打开时是否自动展开高级工具区。' },
        { key: 'preferJpegMode', label: '识别页默认 JPEG 模式', group: '识别页', type: 'bool', description: '开启后识别页默认使用轻量 JPEG 视图。' },
        { key: 'captureFps', label: '屏幕捕获帧率', group: '识别页', type: 'int', min: 2, max: 60, description: '屏幕捕获发送帧率（fps）。降低可减轻 CPU 和带宽压力。' },
    { key: 'playerArrowColor', label: '玩家箭头颜色', group: '识别页', type: 'color', description: '识别页中的玩家方向箭头颜色。' },
        { key: 'markerSize', label: '资源点大小', group: '地图页', type: 'int', min: 6, max: 28, description: '大地图页资源点的默认显示大小。' },
        { key: 'markerOpacity', label: '资源点透明度', group: '地图页', type: 'float', min: 0.2, max: 1, step: 0.05, description: '大地图资源点图层的不透明度。' },
        { key: 'rememberMapFilters', label: '记住地图筛选', group: '地图页', type: 'bool', description: '刷新或重开地图页时保留上次资源筛选状态。' },
        { key: 'mapAutoCenter', label: '玩家自动保持可见', group: '地图页', type: 'bool', description: '收到坐标更新时自动确保玩家点位留在地图视口内。' },
        { key: 'routeLineColor', label: '路线颜色', group: '地图页', type: 'color', description: '路线规划线段的颜色。' },
        { key: 'routeLineOpacity', label: '路线透明度', group: '地图页', type: 'float', min: 0.1, max: 1, step: 0.05, description: '路线规划线段的不透明度。' },
    { key: 'playerDotColor', label: '玩家定位点颜色', group: '地图页', type: 'color', description: '大地图上玩家位置圆点的颜色。' },
        { key: 'themeAccent', label: '主题强调色', group: '外观主题', type: 'color', description: '影响按钮、标签、高亮等主要强调色，修改后全站生效。' },
        { key: 'themeBgImage', label: '自定义背景图', group: '外观主题', type: 'string', description: '填入图片 URL，留空使用默认渐变背景。仅非地图页生效。' },
    ];

    function loadPrefs() {
        try {
            var raw = localStorage.getItem(PREFS_KEY);
            if (!raw) return Object.assign({}, DEFAULT_PREFS);
            return Object.assign({}, DEFAULT_PREFS, JSON.parse(raw));
        } catch (_err) {
            return Object.assign({}, DEFAULT_PREFS);
        }
    }

    function savePrefs(nextPrefs) {
        var merged = Object.assign({}, DEFAULT_PREFS, nextPrefs || {});
        localStorage.setItem(PREFS_KEY, JSON.stringify(merged));
        return merged;
    }

    function updatePref(key, value) {
        var prefs = loadPrefs();
        prefs[key] = value;
        return savePrefs(prefs);
    }

    function applyNavState() {
        var current = document.body.getAttribute('data-page');
        var nodes = document.querySelectorAll('[data-nav-page]');
        Array.prototype.forEach.call(nodes, function (node) {
            node.classList.toggle('is-active', node.getAttribute('data-nav-page') === current);
        });
    }

    /* ── Dock position persistence ── */
    function loadDockPosition() {
        try { var raw = localStorage.getItem(DOCK_POS_KEY); return raw ? JSON.parse(raw) : null; } catch (_e) { return null; }
    }
    function saveDockPosition(pos) { localStorage.setItem(DOCK_POS_KEY, JSON.stringify(pos)); return pos; }
    function clampDockPosition(pos, dock) {
        var margin = 12, w = dock.offsetWidth || 88, h = dock.offsetHeight || 240;
        return {
            x: Math.min(Math.max(margin, pos.x), Math.max(margin, window.innerWidth - w - margin)),
            y: Math.min(Math.max(margin, pos.y), Math.max(margin, window.innerHeight - h - margin))
        };
    }
    function applyDockPosition(dock, pos) {
        var next = clampDockPosition(pos, dock);
        dock.style.left = next.x + 'px'; dock.style.top = next.y + 'px';
        dock.style.right = 'auto'; dock.style.bottom = 'auto'; dock.style.transform = 'none';
        return next;
    }
    function getDefaultDockPosition(dock) {
        return clampDockPosition({ x: 20, y: Math.max(20, Math.round((window.innerHeight - (dock.offsetHeight || 280)) / 2)) }, dock);
    }

    function initFloatingDock() {
        var dock = document.querySelector('.app-dock');
        if (!dock || dock.dataset.dragReady === '1') return;
        dock.dataset.dragReady = '1';

        if (!dock.querySelector('.app-dock-grip')) {
            var grip = document.createElement('button');
            grip.type = 'button'; grip.className = 'app-dock-grip';
            grip.setAttribute('aria-label', '拖动页面导航'); grip.title = '拖动页面导航';
            grip.textContent = '⋮⋮';
            dock.insertBefore(grip, dock.firstChild);
        }

        var handle = dock.querySelector('.app-dock-grip');
        var dragging = null;

        function persistCurrentPosition() { saveDockPosition({ x: dock.offsetLeft, y: dock.offsetTop }); }

        function onMove(event) {
            if (!dragging) return;
            var next = applyDockPosition(dock, { x: event.clientX - dragging.offsetX, y: event.clientY - dragging.offsetY });
            dragging.last = next;
        }
        function onUp() {
            if (!dragging) return;
            dock.classList.remove('is-dragging');
            if (dragging.last) saveDockPosition(dragging.last);
            dragging = null;
            document.removeEventListener('pointermove', onMove);
            document.removeEventListener('pointerup', onUp);
        }

        handle.addEventListener('pointerdown', function (event) {
            event.preventDefault();
            dragging = { offsetX: event.clientX - dock.offsetLeft, offsetY: event.clientY - dock.offsetTop, last: { x: dock.offsetLeft, y: dock.offsetTop } };
            dock.classList.add('is-dragging');
            document.addEventListener('pointermove', onMove);
            document.addEventListener('pointerup', onUp);
        });
        handle.addEventListener('dblclick', function () {
            var reset = applyDockPosition(dock, getDefaultDockPosition(dock));
            saveDockPosition(reset);
        });

        requestAnimationFrame(function () {
            var saved = loadDockPosition();
            applyDockPosition(dock, saved || getDefaultDockPosition(dock));
        });
        window.addEventListener('resize', debounce(function () {
            persistCurrentPosition();
            applyDockPosition(dock, loadDockPosition() || getDefaultDockPosition(dock));
        }, 60));
    }

    /* ── buildDock: inject nav dock HTML into the page ── */
    function buildDock() {
        var pages = [
            { href: '/recognize', page: 'recognize', icon: '🎮', label: '识别台' },
            { href: '/map', page: 'map', icon: '🗺️', label: '大地图' },
            { href: '/settings', page: 'settings', icon: '⚙️', label: '配置' }
        ];
        var nav = document.createElement('nav');
        nav.className = 'app-dock';
        nav.setAttribute('aria-label', '页面切换');
        pages.forEach(function (p) {
            var a = document.createElement('a');
            a.className = 'app-dock-btn'; a.href = p.href;
            a.setAttribute('data-nav-page', p.page);
            a.innerHTML = '<span>' + p.icon + '</span><small>' + p.label + '</small>';
            nav.appendChild(a);
        });
        document.body.appendChild(nav);
        applyNavState();
        initFloatingDock();
        applyTheme();
        requestAnimationFrame(function () {
            if (document.body.getAttribute('data-delay-ready') === '1') return;
            document.body.classList.add('ui-ready');
            document.body.classList.remove('page-preload');
        });
    }

    /* ── Toast ── */
    function ensureToastHost() {
        var host = document.getElementById('appToastStack');
        if (!host) { host = document.createElement('div'); host.id = 'appToastStack'; host.className = 'app-toast-stack'; document.body.appendChild(host); }
        return host;
    }
    function toast(message, tone, timeout) {
        var host = ensureToastHost();
        var item = document.createElement('div');
        item.className = 'app-toast';
        if (tone) item.setAttribute('data-tone', tone);
        item.textContent = message;
        host.appendChild(item);
        window.setTimeout(function () {
            item.style.opacity = '0'; item.style.transform = 'translateY(8px)';
            window.setTimeout(function () { if (item.parentNode) item.parentNode.removeChild(item); }, 180);
        }, timeout || 2800);
    }

    function debounce(fn, delay) {
        var timer = null;
        return function () {
            var args = arguments;
            var ctx = this;
            if (timer) clearTimeout(timer);
            timer = setTimeout(function () { fn.apply(ctx, args); }, delay);
        };
    }

    function fetchJSON(url, opts) {
        return fetch(url, opts).then(function (resp) {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        });
    }

    function cloneJSONValue(value) {
        if (value == null) return value;
        if (typeof structuredClone === 'function') return structuredClone(value);
        return JSON.parse(JSON.stringify(value));
    }

    function estimateJSONBytes(value) {
        try {
            return JSON.stringify(value).length;
        } catch (_err) {
            return Infinity;
        }
    }

    function getMemoryCacheValue(cacheKey) {
        if (!JSON_MEMORY_CACHE.has(cacheKey)) return undefined;
        var entry = JSON_MEMORY_CACHE.get(cacheKey);
        JSON_MEMORY_CACHE.delete(cacheKey);
        JSON_MEMORY_CACHE.set(cacheKey, entry);
        return entry.payload;
    }

    function setMemoryCacheValue(cacheKey, payload) {
        var approxBytes = estimateJSONBytes(payload);
        if (!isFinite(approxBytes) || approxBytes > JSON_MEMORY_MAX_BYTES) return;

        if (JSON_MEMORY_CACHE.has(cacheKey)) {
            jsonMemoryBytes -= JSON_MEMORY_CACHE.get(cacheKey).bytes;
            JSON_MEMORY_CACHE.delete(cacheKey);
        }

        while (JSON_MEMORY_CACHE.size >= JSON_MEMORY_MAX_ENTRIES || (jsonMemoryBytes + approxBytes > JSON_MEMORY_MAX_BYTES && JSON_MEMORY_CACHE.size)) {
            var oldestKey = JSON_MEMORY_CACHE.keys().next().value;
            var oldestEntry = JSON_MEMORY_CACHE.get(oldestKey);
            jsonMemoryBytes -= oldestEntry.bytes;
            JSON_MEMORY_CACHE.delete(oldestKey);
        }

        JSON_MEMORY_CACHE.set(cacheKey, { payload: payload, bytes: approxBytes });
        jsonMemoryBytes += approxBytes;
    }

    function getOfflineDb() {
        if (offlineDbPromise) return offlineDbPromise;
        if (typeof window === 'undefined' || !('indexedDB' in window)) return Promise.resolve(null);

        offlineDbPromise = new Promise(function (resolve, reject) {
            var req = window.indexedDB.open(OFFLINE_DB_NAME, 1);
            req.onupgradeneeded = function (event) {
                var db = event.target.result;
                if (!db.objectStoreNames.contains(OFFLINE_DB_STORE)) {
                    db.createObjectStore(OFFLINE_DB_STORE, { keyPath: 'key' });
                }
            };
            req.onsuccess = function () { resolve(req.result); };
            req.onerror = function () { reject(req.error || new Error('IndexedDB open failed')); };
        }).catch(function () {
            offlineDbPromise = null;
            return null;
        });
        return offlineDbPromise;
    }

    function readOfflineJSON(key) {
        return getOfflineDb().then(function (db) {
            if (!db) return null;
            return new Promise(function (resolve, reject) {
                var tx = db.transaction(OFFLINE_DB_STORE, 'readonly');
                var req = tx.objectStore(OFFLINE_DB_STORE).get(key);
                req.onsuccess = function () {
                    resolve(req.result ? req.result.value : null);
                };
                req.onerror = function () {
                    reject(req.error || new Error('IndexedDB read failed'));
                };
            });
        });
    }

    function writeOfflineJSON(key, value) {
        return getOfflineDb().then(function (db) {
            if (!db) return value;
            return new Promise(function (resolve, reject) {
                var tx = db.transaction(OFFLINE_DB_STORE, 'readwrite');
                tx.objectStore(OFFLINE_DB_STORE).put({
                    key: key,
                    value: value,
                    updatedAt: Date.now(),
                });
                tx.oncomplete = function () { resolve(value); };
                tx.onerror = function () {
                    reject(tx.error || new Error('IndexedDB write failed'));
                };
            });
        });
    }

    function fetchJSONCached(url, options) {
        options = options || {};
        var fetchOptions = options.fetchOptions || {};
        if (options.method && !fetchOptions.method) {
            fetchOptions = Object.assign({}, fetchOptions, { method: options.method });
        }

        var cacheKey = options.cacheKey || url;
        var useMemory = options.memory !== false;
        var cloneResponse = options.clone !== false;
        var method = String(fetchOptions.method || 'GET').toUpperCase();
        var usePersistentCache = !!options.persistent && method === 'GET' && typeof window !== 'undefined' && 'caches' in window;

        var memoryValue = useMemory ? getMemoryCacheValue(cacheKey) : undefined;
        if (memoryValue !== undefined) {
            return Promise.resolve(cloneResponse ? cloneJSONValue(memoryValue) : memoryValue);
        }
        if (JSON_INFLIGHT_CACHE.has(cacheKey)) {
            return JSON_INFLIGHT_CACHE.get(cacheKey).then(function (payload) {
                return cloneResponse ? cloneJSONValue(payload) : payload;
            });
        }

        var task = (async function () {
            if (usePersistentCache) {
                /* Cache API requires http(s) URLs; wrap bare keys into a synthetic URL */
                var cacheReq = cacheKey;
                if (!/^https?:\/\//.test(cacheKey)) cacheReq = new Request('https://cache.invalid/' + encodeURIComponent(cacheKey));
                var cache = await window.caches.open(JSON_RESPONSE_CACHE);
                var cached = await cache.match(cacheReq);
                if (cached) {
                    var cachedPayload = await cached.json();
                    if (useMemory) setMemoryCacheValue(cacheKey, cachedPayload);
                    return cachedPayload;
                }
            }

            var resp = await fetch(url, fetchOptions);
            if (!resp.ok) throw new Error('HTTP ' + resp.status);

            if (usePersistentCache) {
                var cacheReq2 = cacheKey;
                if (!/^https?:\/\//.test(cacheKey)) cacheReq2 = new Request('https://cache.invalid/' + encodeURIComponent(cacheKey));
                var responseCache = await window.caches.open(JSON_RESPONSE_CACHE);
                await responseCache.put(cacheReq2, resp.clone());
            }

            var payload = await resp.json();
            if (useMemory) setMemoryCacheValue(cacheKey, payload);
            return payload;
        })();

        JSON_INFLIGHT_CACHE.set(cacheKey, task);
        return task.then(function (payload) {
            return cloneResponse ? cloneJSONValue(payload) : payload;
        }).finally(function () {
            JSON_INFLIGHT_CACHE.delete(cacheKey);
        });
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    function formatNumber(val, digits) {
        return Number(val).toFixed(digits != null ? digits : 1);
    }

    function applyLogPrefs(logEl) {
        if (!logEl) return;
        var prefs = loadPrefs();
        logEl.dataset.maxLen = prefs.logMaxLen || 6000;
        logEl.dataset.trimLen = prefs.logTrimLen || 3600;
    }

    /* ── Theme application ── */
    function adjustColor(hex, amount) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        if (amount >= 0) {
            r = Math.round(r + (255 - r) * amount / 100);
            g = Math.round(g + (255 - g) * amount / 100);
            b = Math.round(b + (255 - b) * amount / 100);
        } else {
            var f = 1 + amount / 100;
            r = Math.round(r * f); g = Math.round(g * f); b = Math.round(b * f);
        }
        r = Math.min(255, Math.max(0, r)); g = Math.min(255, Math.max(0, g)); b = Math.min(255, Math.max(0, b));
        return '#' + r.toString(16).padStart(2, '0') + g.toString(16).padStart(2, '0') + b.toString(16).padStart(2, '0');
    }

    function applyTheme() {
        var prefs = loadPrefs();
        var root = document.documentElement;
        var page = document.body.getAttribute('data-page');

        function setColorVar(prefKey, cssVar, fallback) {
            var val = prefs[prefKey];
            if (val && val !== fallback) root.style.setProperty(cssVar, val);
            else root.style.removeProperty(cssVar);
        }

        if (prefs.themeAccent && prefs.themeAccent !== '#9a6948') {
            root.style.setProperty('--accent', prefs.themeAccent);
            root.style.setProperty('--accent-deep', adjustColor(prefs.themeAccent, -20));
            root.style.setProperty('--accent-soft', adjustColor(prefs.themeAccent, 40));
        } else {
            root.style.removeProperty('--accent');
            root.style.removeProperty('--accent-deep');
            root.style.removeProperty('--accent-soft');
        }

        setColorVar('themeText', '--text', '#4b3528');
        setColorVar('themeTextSoft', '--text-soft', '#7b6555');
        setColorVar('themeTextFaint', '--text-faint', '#a18b7b');
        setColorVar('themePanelStrong', '--panel-strong', '#fffdf9');
        setColorVar('themeMapBg', '--map-bg', '#1b1512');
        setColorVar('themeSidebarBg', '--map-sidebar-bg', '#fffdf9');
        setColorVar('themeToolbarBg', '--map-toolbar-bg', '#261c16');
        setColorVar('themeHudBg', '--map-hud-bg', '#261c16');
        setColorVar('themeDetailBg', '--map-detail-bg', '#fffcf6');

        if (page !== 'map') {
            var shell = document.querySelector('.app-page-shell');
            if (shell && prefs.themeBgImage) {
                shell.style.backgroundImage = 'url(' + JSON.stringify(prefs.themeBgImage) + ')';
                shell.style.backgroundSize = 'cover';
                shell.style.backgroundPosition = 'center';
            } else if (shell) {
                shell.style.backgroundImage = '';
            }
        }
        if (prefs.playerDotColor && prefs.playerDotColor !== '#00ff7f') {
            root.style.setProperty('--player-dot-color', prefs.playerDotColor);
        } else {
            root.style.removeProperty('--player-dot-color');
        }
        if (prefs.playerArrowColor && prefs.playerArrowColor !== '#30b6fe') {
            root.style.setProperty('--player-arrow-color', prefs.playerArrowColor);
        } else {
            root.style.removeProperty('--player-arrow-color');
        }

        root.style.removeProperty('--_theme-json-applied');
        if (prefs.themeCssVarsJson) {
            try {
                var vars = JSON.parse(prefs.themeCssVarsJson);
                if (vars && typeof vars === 'object') {
                    Object.keys(vars).forEach(function (k) {
                        if (k && k.indexOf('--') === 0 && typeof vars[k] === 'string') root.style.setProperty(k, vars[k]);
                    });
                }
            } catch (_err) {
                /* ignore malformed JSON */
            }
        }
    }

    function syncPrefsToScatteredKeys(prefs) {
        if (prefs.routeLineColor) localStorage.setItem('route_line_color', prefs.routeLineColor);
        if (prefs.routeLineOpacity != null) localStorage.setItem('route_line_opacity', String(Math.round(prefs.routeLineOpacity * 100)));
    }

    var INTERACTIVE_HIDDEN_FOCUSABLE_SELECTOR = 'a[href],area[href],button,input,select,textarea,iframe,summary,[tabindex],[contenteditable=""],[contenteditable="true"]';

    function collectFocusableNodes(root) {
        if (!root || root.nodeType !== 1) return [];
        var nodes = [];
        if (root.matches && root.matches(INTERACTIVE_HIDDEN_FOCUSABLE_SELECTOR)) nodes.push(root);
        return nodes.concat(Array.prototype.slice.call(root.querySelectorAll(INTERACTIVE_HIDDEN_FOCUSABLE_SELECTOR)));
    }

    function setFallbackFocusableState(element, hidden) {
        collectFocusableNodes(element).forEach(function (node) {
            if (hidden) {
                if (node.hasAttribute('data-app-hidden-focus-managed')) return;
                node.setAttribute('data-app-hidden-prev-tabindex', node.hasAttribute('tabindex') ? node.getAttribute('tabindex') : '');
                node.setAttribute('tabindex', '-1');
                node.setAttribute('data-app-hidden-focus-managed', '1');
                return;
            }
            if (!node.hasAttribute('data-app-hidden-focus-managed')) return;
            var prev = node.getAttribute('data-app-hidden-prev-tabindex');
            if (prev === '') node.removeAttribute('tabindex');
            else node.setAttribute('tabindex', prev);
            node.removeAttribute('data-app-hidden-prev-tabindex');
            node.removeAttribute('data-app-hidden-focus-managed');
        });
    }

    function setInteractiveHiddenState(element, hidden, options) {
        options = options || {};
        if (!element) return hidden;

        var nextHidden = !!hidden;
        var activeEl = typeof document !== 'undefined' ? document.activeElement : null;

        if (options.className) element.classList.toggle(options.className, nextHidden);
        if (options.shownClassName) element.classList.toggle(options.shownClassName, !nextHidden);
        if (options.useHiddenAttribute) element.hidden = nextHidden;
        if (options.setDisabled && 'disabled' in element) element.disabled = nextHidden;

        if (options.syncAriaHidden !== false) {
            element.setAttribute('aria-hidden', nextHidden ? 'true' : 'false');
        }

        if (!element.hasAttribute('data-app-hidden-prev-pointer-events')) {
            element.setAttribute('data-app-hidden-prev-pointer-events', element.style.pointerEvents || '');
        }
        element.style.pointerEvents = nextHidden ? 'none' : element.getAttribute('data-app-hidden-prev-pointer-events');
        if (!nextHidden) {
            element.removeAttribute('data-app-hidden-prev-pointer-events');
        }

        if ('inert' in element) {
            element.inert = nextHidden;
        } else {
            setFallbackFocusableState(element, nextHidden);
        }

        if (nextHidden && activeEl && element.contains(activeEl) && typeof activeEl.blur === 'function') {
            activeEl.blur();
        }

        return nextHidden;
    }

    function bindPersistentToggleState(options) {
        options = options || {};
        var storageKey = options.storageKey;
        var trigger = options.trigger;
        var readValue = options.readValue || function () { return localStorage.getItem(storageKey) === '1'; };
        var writeValue = options.writeValue || function (next) { localStorage.setItem(storageKey, next ? '1' : '0'); };
        var applyValue = options.applyValue || function () {};

        function sync() {
            var current = !!readValue();
            applyValue(current);
            return current;
        }

        function set(next) {
            var normalized = !!next;
            writeValue(normalized);
            applyValue(normalized);
            return normalized;
        }

        if (options.syncOnInit !== false) sync();

        if (trigger) {
            trigger.addEventListener('click', function (event) {
                if (typeof options.onBeforeToggle === 'function') options.onBeforeToggle(event);
                if (event.defaultPrevented) return;
                set(!readValue());
            });
        }

        if (options.syncOnRestore !== false) {
            window.addEventListener('pageshow', sync);
            document.addEventListener('visibilitychange', function () {
                if (!document.hidden) sync();
            });
        }

        return {
            sync: sync,
            set: set,
            get: function () { return !!readValue(); }
        };
    }

    /* ── Page resume lifecycle (BFCache / history back) ── */
    function bindPageResumeLifecycle(options) {
        options = options || {};
        var onResume = typeof options.onResume === 'function' ? options.onResume : function () {};
        var rafPasses = Math.max(0, Number(options.rafPasses || 0));

        function runResume(reason, event) {
            var passes = rafPasses;
            function invoke() {
                if (passes <= 0) {
                    onResume({ reason: reason, event: event, persisted: !!(event && event.persisted) });
                    return;
                }
                passes -= 1;
                requestAnimationFrame(invoke);
            }
            invoke();
        }

        if (options.onPageShow !== false) {
            window.addEventListener('pageshow', function (event) {
                window._isUnloading = false;
                runResume('pageshow', event);
            });
        }

        if (options.onVisibility !== false) {
            document.addEventListener('visibilitychange', function (event) {
                if (document.hidden) return;
                window._isUnloading = false;
                runResume('visibilitychange', event);
            });
        }

        if (options.runOnInit) runResume('init', null);

        return {
            runNow: function () {
                runResume('manual', null);
            }
        };
    }

    function releaseHeavyPageResources() {
        try {
            window._isUnloading = true;
            var canvases = document.getElementsByTagName('canvas');
            for (var i = 0; i < canvases.length; i++) {
                canvases[i].width = 1;
                canvases[i].height = 1;
            }
            var videos = document.getElementsByTagName('video');
            for (var j = 0; j < videos.length; j++) {
                if (videos[j].srcObject && videos[j].srcObject.getTracks) {
                    var tracks = videos[j].srcObject.getTracks();
                    for (var t = 0; t < tracks.length; t++) { tracks[t].stop(); }
                    videos[j].srcObject = null;
                }
                videos[j].removeAttribute('src');
                videos[j].load();
            }
            var mi = document.getElementById('mi');
            if (mi) mi.removeAttribute('src'); // 针对大地图图片专门释放
        } catch (e) { /* ignore */ }
    }

    /* ── 预防切换黑屏死机：仅在真正离开页面时释放大内存占用，避免破坏 BFCache 返回恢复 ── */
    window.addEventListener('pagehide', function (event) {
        if (event && event.persisted) {
            window._isUnloading = false;
            return;
        }
        releaseHeavyPageResources();
    });

    window.addEventListener('beforeunload', function () {
        window._isUnloading = true;
    });

    window.addEventListener('pageshow', function () {
        window._isUnloading = false;
    });

export {
    DEFAULT_PREFS, FRONTEND_PREFS_META,
    loadPrefs, savePrefs, updatePref, applyNavState,
    initFloatingDock, buildDock, toast, debounce,
    fetchJSON, fetchJSONCached, readOfflineJSON, writeOfflineJSON,
    escapeHtml, formatNumber, applyLogPrefs,
    setInteractiveHiddenState,
    applyTheme, syncPrefsToScatteredKeys, bindPersistentToggleState,
    bindPageResumeLifecycle,
};
