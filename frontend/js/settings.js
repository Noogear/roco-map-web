/* settings.js — Personal settings: frontend prefs, theme, backend tuning, import/export */
import * as AppCommon from './common.js';

AppCommon.buildDock();

var PREFS_META = AppCommon.FRONTEND_PREFS_META;
var SECTION_ORDER = ['外观主题', '地图页', '识别页'];
var SECTION_ICONS = { '外观主题': '🎨', '地图页': '🗺️', '识别页': '🎮' };
var SECTION_DESCRIPTIONS = {
    '外观主题': '调整主题、配色与基础界面观感。',
    '地图页': '控制地图展示、路线样式与浏览体验。',
    '识别页': '管理识别页面相关的交互与显示偏好。',
    'backend': '后端识别引擎的热更新参数，保存后立即生效。',
    'data': '导入、导出或重置当前浏览器中的本地设置。'
};
var BACKEND_STATE = null;
var THEME_VAR_GROUPS = [
    {
        key: 'global',
        title: '全局基础',
        description: '控制全站基础底色、文字和边框。',
        items: [
            { cssVar: '--bg', label: '页面背景', defaultValue: '#f7f1e7' },
            { cssVar: '--bg-soft', label: '浅背景', defaultValue: '#fffaf2' },
            { cssVar: '--panel', label: '普通面板', defaultValue: '#fffbf4' },
            { cssVar: '--line', label: '边框/分割线', defaultValue: '#714e5529' },
            { cssVar: '--panel-strong', label: '强调面板', defaultValue: '#fffdf9' },
            { cssVar: '--text', label: '主文字', defaultValue: '#4b3528' },
            { cssVar: '--text-soft', label: '次级文字', defaultValue: '#7b6555' },
            { cssVar: '--text-faint', label: '弱化文字', defaultValue: '#a18b7b' }
        ]
    },
    {
        key: 'accent',
        title: '品牌/状态色',
        description: '按钮、高亮、成功/警告/错误色。',
        items: [
            { cssVar: '--accent-deep', label: '深强调色', defaultValue: '#7b4f33' },
            { cssVar: '--accent-soft', label: '浅强调色', defaultValue: '#ead7c3' },
            { cssVar: '--accent-gold', label: '金色强调', defaultValue: '#d7a95f' },
            { cssVar: '--success', label: '成功色', defaultValue: '#4f8a4b' },
            { cssVar: '--warning', label: '警告色', defaultValue: '#d28b36' },
            { cssVar: '--danger', label: '危险色', defaultValue: '#c35f5f' }
        ]
    },
    {
        key: 'map',
        title: '地图专用',
        description: '地图页舞台、侧栏、工具栏、HUD 与路线规划栏。',
        items: [
            { cssVar: '--map-bg', label: '地图背景', defaultValue: '#1b1512' },
            { cssVar: '--map-sidebar-bg', label: '地图侧栏', defaultValue: '#fffdf9' },
            { cssVar: '--map-toolbar-bg', label: '地图工具栏', defaultValue: '#261c16' },
            { cssVar: '--map-hud-bg', label: '地图 HUD', defaultValue: '#261c16' },
            { cssVar: '--map-detail-bg', label: '地图信息面板', defaultValue: '#fffcf6' },
            { cssVar: '--map-route-panel-bg', label: '路线栏背景', defaultValue: '#fffdf9' },
            { cssVar: '--map-route-panel-border', label: '路线栏边框', defaultValue: 'rgba(113,78,55,0.08)' },
            { cssVar: '--map-route-section-bg', label: '路线分组背景', defaultValue: '#ffffff' },
            { cssVar: '--map-route-item-bg', label: '路径项背景', defaultValue: 'rgba(255,255,255,0.72)' },
            { cssVar: '--map-route-item-hover-bg', label: '路径项悬停背景', defaultValue: '#ffffff' },
            { cssVar: '--map-route-accent-bg', label: '路线强调底色', defaultValue: 'rgba(154,105,72,0.14)' },
            { cssVar: '--map-route-nav-btn-start', label: '导航按钮起始色', defaultValue: '#aa7756' },
            { cssVar: '--map-route-nav-btn-end', label: '导航按钮结束色', defaultValue: '#8a5c3d' }
        ]
    }
];

function parseThemeVars(raw) {
    if (!raw) return {};
    try {
        var parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (_err) {
        return {};
    }
}

function stableStringifyThemeVars(obj) {
    var clean = {};
    Object.keys(obj || {}).sort().forEach(function (key) {
        if (obj[key]) clean[key] = obj[key];
    });
    return Object.keys(clean).length ? JSON.stringify(clean, null, 2) : '';
}

function saveThemeVars(nextVars) {
    savePref('themeCssVarsJson', stableStringifyThemeVars(nextVars));
}

function getSectionStorageKey() { return 'stg_active_section'; }

function getActiveSectionKey(fallback) {
    return localStorage.getItem(getSectionStorageKey()) || fallback;
}

function setActiveSectionKey(key) {
    localStorage.setItem(getSectionStorageKey(), key);
}

/* ── Build all sections ── */
function buildSettings() {
    var body = document.getElementById('settingsBody');
    body.innerHTML = '';
    body.className = 'settings-layout';

    var sidebar = document.createElement('aside');
    sidebar.className = 'settings-sidebar';
    var sidebarInner = document.createElement('div');
    sidebarInner.className = 'settings-sidebar-card';
    sidebarInner.innerHTML = '<div class="settings-sidebar-head"><div><div class="settings-sidebar-kicker">设置目录</div>' +
        '<h2>配置分组</h2><p class="app-note">从左侧选择分类，右侧展开对应的可修改项。</p></div></div>';
    var nav = document.createElement('nav');
    nav.className = 'settings-nav';
    nav.setAttribute('aria-label', '设置分组');
    sidebarInner.appendChild(nav);
    sidebar.appendChild(sidebarInner);

    var detail = document.createElement('div');
    detail.className = 'settings-detail';

    var grouped = {};
    var sections = [];
    PREFS_META.forEach(function (m) { if (!grouped[m.group]) grouped[m.group] = []; grouped[m.group].push(m); });
    if (!grouped['外观主题']) grouped['外观主题'] = [];
    grouped['外观主题'].push({
        key: 'themePaletteGlobal',
        label: '高级主题色板',
        group: '外观主题',
        type: 'theme-palette',
        editorGroupKeys: ['global', 'accent'],
        description: '管理全局基础色、强调色和状态色。'
    });
    if (!grouped['地图页']) grouped['地图页'] = [];
    grouped['地图页'].push({
        key: 'themePaletteMap',
        label: '地图专用色板',
        group: '地图页',
        type: 'theme-palette',
        editorGroupKeys: ['map'],
        description: '管理地图页舞台、侧栏、工具栏、HUD、信息卡片与路线规划栏配色。'
    });

    SECTION_ORDER.forEach(function (gn) {
        if (grouped[gn]) sections.push(buildFrontendSection(gn, SECTION_ICONS[gn] || '⚙️', grouped[gn]));
    });
    sections.push(buildBackendSection());
    sections.push(buildDataSection());

    sections.forEach(function (section) {
        nav.appendChild(buildSectionNavButton(section));
        detail.appendChild(section.element);
    });

    body.appendChild(sidebar);
    body.appendChild(detail);

    if (sections.length) {
        var activeKey = getActiveSectionKey(sections[0].key);
        if (!sections.some(function (section) { return section.key === activeKey; })) activeKey = sections[0].key;
        setActiveSection(activeKey);
    }

    loadBackendConfig();
}

/* ── Frontend section ── */
function buildFrontendSection(name, icon, metas) {
    var prefs = AppCommon.loadPrefs();
    var section = document.createElement('section');
    section.className = 'settings-section settings-panel';
    section.dataset.sectionKey = name;

    var toggle = document.createElement('div');
    toggle.className = 'settings-section-toggle';
    toggle.innerHTML = '<span class="settings-section-icon">' + icon + '</span><span class="settings-section-title-wrap"><span>' + AppCommon.escapeHtml(name) +
        '</span><small>' + AppCommon.escapeHtml(SECTION_DESCRIPTIONS[name] || '管理该分类中的页面偏好。') + '</small></span>' +
        '<span class="settings-section-count">' + metas.length + ' 项</span>';

    var body = document.createElement('div');
    body.className = 'settings-section-body';
    var grid = document.createElement('div');
    grid.className = 'settings-items';
    metas.forEach(function (meta) { grid.appendChild(buildItem(meta, prefs[meta.key])); });
    body.appendChild(grid);
    section.appendChild(toggle);
    section.appendChild(body);

    return {
        key: name,
        icon: icon,
        title: name,
        description: SECTION_DESCRIPTIONS[name] || '管理该分类中的页面偏好。',
        countText: metas.length + ' 项',
        element: section
    };
}

function buildSectionNavButton(section) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'settings-nav-item';
    button.dataset.sectionKey = section.key;
    button.innerHTML = '<span class="settings-nav-item-icon">' + section.icon + '</span>' +
        '<span class="settings-nav-item-main"><strong>' + AppCommon.escapeHtml(section.title) + '</strong>' +
        '<small>' + AppCommon.escapeHtml(section.description) + '</small></span>' +
        '<span class="settings-nav-item-count">' + AppCommon.escapeHtml(section.countText || '') + '</span>';
    button.addEventListener('click', function () { setActiveSection(section.key); });
    return button;
}

function setActiveSection(key) {
    setActiveSectionKey(key);
    document.querySelectorAll('.settings-nav-item').forEach(function (item) {
        var active = item.dataset.sectionKey === key;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    document.querySelectorAll('.settings-panel').forEach(function (panel) {
        panel.classList.toggle('is-active', panel.dataset.sectionKey === key);
    });
}

/* ── Build a single setting item ── */
function buildItem(meta, value) {
    var isWide = meta.type === 'string' || meta.type === 'theme-palette';
    var item = document.createElement('div');
    item.className = 'settings-item' + (isWide ? ' is-wide' : '');

    var info = document.createElement('div');
    info.className = 'settings-item-info';
    info.innerHTML = '<div class="settings-item-label">' + AppCommon.escapeHtml(meta.label) +
        '</div><div class="settings-item-desc">' + AppCommon.escapeHtml(meta.description) + '</div>';

    var ctrl = document.createElement('div');
    ctrl.className = 'settings-item-control';
    ctrl.appendChild(createControl(meta, value, false));

    item.appendChild(info);
    item.appendChild(ctrl);
    return item;
}

/* ── Create control element ── */
function createControl(meta, value, isBackend) {
    var type = meta.type || 'string';
    var keyAttr = isBackend ? 'data-backend-key' : 'data-key';

    if (!isBackend && meta.type === 'theme-palette') {
        return createThemePaletteEditor(AppCommon.loadPrefs().themeCssVarsJson, meta.editorGroupKeys || []);
    }

    if (type === 'bool') {
        var lbl = document.createElement('label'); lbl.className = 'settings-toggle';
        var cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = !!value;
        cb.setAttribute(keyAttr, meta.key);
        if (!isBackend) cb.addEventListener('change', function () { savePref(meta.key, cb.checked); });
        var track = document.createElement('span'); track.className = 'settings-toggle-track';
        lbl.appendChild(cb); lbl.appendChild(track); return lbl;
    }

    if (type === 'color') {
        var ci = document.createElement('input'); ci.type = 'color'; ci.className = 'settings-color-input';
        ci.value = value || '#000000'; ci.setAttribute(keyAttr, meta.key);
        if (!isBackend) ci.addEventListener('input', function () { savePref(meta.key, ci.value); });
        return ci;
    }

    if (type === 'int' || type === 'float') {
        var wrap = document.createElement('div'); wrap.className = 'settings-range-wrap';
        var rng = document.createElement('input'); rng.type = 'range';
        rng.min = meta.min != null ? meta.min : 0; rng.max = meta.max != null ? meta.max : 100;
        rng.step = meta.step != null ? meta.step : (type === 'int' ? 1 : 0.01);
        rng.value = value != null ? value : rng.min;
        rng.setAttribute(keyAttr, meta.key);
        var valEl = document.createElement('span'); valEl.className = 'settings-range-val';
        valEl.textContent = type === 'float' ? Number(rng.value).toFixed(2) : rng.value;
        rng.addEventListener('input', function () {
            var v = type === 'int' ? parseInt(rng.value, 10) : parseFloat(rng.value);
            valEl.textContent = type === 'float' ? v.toFixed(2) : String(v);
            if (!isBackend) savePref(meta.key, v);
        });
        wrap.appendChild(rng); wrap.appendChild(valEl); return wrap;
    }

    /* string */
    var si = document.createElement('input'); si.type = 'text'; si.className = 'app-field';
    si.value = value != null ? String(value) : ''; si.placeholder = meta.description || '';
    si.setAttribute(keyAttr, meta.key);
    if (!isBackend) {
        var dt = null;
        si.addEventListener('input', function () {
            clearTimeout(dt);
            dt = setTimeout(function () { savePref(meta.key, si.value); }, 400);
        });
    }
    return si;
}

function createThemePaletteEditor(value, allowedGroupKeys) {
    var vars = parseThemeVars(value);
    var wrap = document.createElement('div');
    wrap.className = 'theme-editor';

    var summary = document.createElement('div');
    summary.className = 'theme-editor-summary';
    summary.innerHTML = '<div><strong>可视化主题色板</strong><div class="app-note">这里管理剩余的高级 CSS 变量覆盖；已在上方提供的颜色项不会在这里重复出现。</div></div>';

    var resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'app-btn-ghost';
    resetBtn.textContent = '重置高级变量';
    resetBtn.addEventListener('click', function () {
        saveThemeVars({});
        buildSettings();
        AppCommon.toast('高级主题变量已重置', 'success');
    });
    summary.appendChild(resetBtn);
    wrap.appendChild(summary);

    var groups = document.createElement('div');
    groups.className = 'theme-editor-groups';

    THEME_VAR_GROUPS.forEach(function (group) {
        if (allowedGroupKeys && allowedGroupKeys.length && allowedGroupKeys.indexOf(group.key) < 0) return;
        var card = document.createElement('section');
        card.className = 'theme-editor-group';
        card.innerHTML = '<div class="theme-editor-group-head"><h3>' + AppCommon.escapeHtml(group.title) + '</h3><p>' + AppCommon.escapeHtml(group.description) + '</p></div>';

        var grid = document.createElement('div');
        grid.className = 'theme-editor-grid';

        group.items.forEach(function (item) {
            var row = document.createElement('div');
            row.className = 'theme-editor-item';

            var text = document.createElement('div');
            text.className = 'theme-editor-item-text';
            text.innerHTML = '<strong>' + AppCommon.escapeHtml(item.label) + '</strong>';

            var controls = document.createElement('div');
            controls.className = 'theme-editor-item-controls';

            var color = document.createElement('input');
            color.type = 'color';
            color.className = 'settings-color-input';
            color.value = normalizeColorForInput(vars[item.cssVar] || item.defaultValue);

            var code = document.createElement('input');
            code.type = 'text';
            code.className = 'theme-color-code';
            code.value = vars[item.cssVar] || item.defaultValue;

            var clear = document.createElement('button');
            clear.type = 'button';
            clear.className = 'theme-editor-item-reset';
            clear.textContent = '默认';

            function commit(nextValue) {
                var current = parseThemeVars(AppCommon.loadPrefs().themeCssVarsJson);
                if (!nextValue || normalizeCssColorText(nextValue) === normalizeCssColorText(item.defaultValue)) delete current[item.cssVar];
                else current[item.cssVar] = nextValue;
                saveThemeVars(current);
            }

            color.addEventListener('input', function () {
                code.value = color.value;
                commit(color.value);
            });

            code.addEventListener('change', function () {
                var normalized = normalizeCssColorText(code.value) || item.defaultValue;
                code.value = normalized;
                color.value = normalizeColorForInput(normalized);
                commit(normalized);
            });

            clear.addEventListener('click', function (event) {
                event.preventDefault();
                code.value = item.defaultValue;
                color.value = normalizeColorForInput(item.defaultValue);
                commit(item.defaultValue);
            });

            controls.appendChild(color);
            controls.appendChild(code);
            controls.appendChild(clear);
            row.appendChild(text);
            row.appendChild(controls);
            grid.appendChild(row);
        });

        card.appendChild(grid);
        groups.appendChild(card);
    });

    wrap.appendChild(groups);

    return wrap;
}

function normalizeColorForInput(value) {
    var normalized = normalizeCssColorText(value);
    return /^#[0-9a-f]{6}$/i.test(normalized) ? normalized : '#000000';
}

function normalizeCssColorText(value) {
    if (!value) return '';
    var text = String(value).trim();
    if (/^#[0-9a-f]{3}$/i.test(text)) {
        return '#' + text.slice(1).split('').map(function (ch) { return ch + ch; }).join('');
    }
    if (/^#[0-9a-f]{6}$/i.test(text)) return text.toLowerCase();
    return text;
}

/* ── Save a frontend pref (auto-save on change) ── */
function savePref(key, value) {
    AppCommon.updatePref(key, value);
    AppCommon.syncPrefsToScatteredKeys(AppCommon.loadPrefs());
    AppCommon.applyTheme();
}

/* ── Backend config section (editable items only) ── */
function buildBackendSection() {
    var section = document.createElement('section');
    section.className = 'settings-section settings-panel'; section.id = 'backendSection';
    section.dataset.sectionKey = 'backend';

    var toggle = document.createElement('div');
    toggle.className = 'settings-section-toggle';
    toggle.innerHTML = '<span class="settings-section-icon">🔧</span><span class="settings-section-title-wrap"><span>识别参数微调</span>' +
        '<small>' + AppCommon.escapeHtml(SECTION_DESCRIPTIONS.backend) + '</small></span>' +
        '<span class="settings-section-count" id="backendCount">加载中</span>';

    var body = document.createElement('div');
    body.className = 'settings-section-body';
    body.innerHTML = '<div class="settings-backend-note">以下为后端识别引擎的可热更新参数，修改后立即生效。注意：调整会影响所有连接用户。</div>' +
        '<div id="backendGrid" class="settings-items"></div>' +
        '<div style="margin-top:14px;display:flex;gap:8px;">' +
        '<button class="app-btn-soft" type="button" id="saveBackendBtn">💾 保存修改</button>' +
        '<button class="app-btn-ghost" type="button" id="refreshBackendBtn">↻ 重新读取</button></div>';

    section.appendChild(toggle); section.appendChild(body);
    return {
        key: 'backend',
        icon: '🔧',
        title: '识别参数微调',
        description: SECTION_DESCRIPTIONS.backend,
        countText: '加载中',
        element: section
    };
}

function loadBackendConfig() {
    AppCommon.fetchJSON('/api/config').then(function (p) {
        BACKEND_STATE = p; renderBackendConfig();
    }).catch(function (e) {
        var g = document.getElementById('backendGrid');
        if (g) g.innerHTML = '<div class="empty-note">读取配置失败：' + AppCommon.escapeHtml(e.message) + '</div>';
        updateSectionCount('backend', '读取失败');
    });
}

function updateSectionCount(sectionKey, text) {
    var countEl = document.querySelector('.settings-panel[data-section-key="' + sectionKey + '"] .settings-section-count');
    if (countEl) countEl.textContent = text;
    var navCountEl = document.querySelector('.settings-nav-item[data-section-key="' + sectionKey + '"] .settings-nav-item-count');
    if (navCountEl) navCountEl.textContent = text;
}

function renderBackendConfig() {
    if (!BACKEND_STATE) return;
    var grid = document.getElementById('backendGrid');
    if (!grid) return;
    grid.innerHTML = '';
    var count = 0;
    BACKEND_STATE.editableKeys.forEach(function (key) {
        var meta = BACKEND_STATE.meta[key], value = BACKEND_STATE.values[key];
        if (!meta) return;
        count++;
        var item = document.createElement('div'); item.className = 'settings-item';
        var info = document.createElement('div'); info.className = 'settings-item-info';
        info.innerHTML = '<div class="settings-item-label">' + AppCommon.escapeHtml(meta.label) +
            '</div><div class="settings-item-desc">' + AppCommon.escapeHtml(meta.description || meta.key) + '</div>';
        var ctrl = document.createElement('div'); ctrl.className = 'settings-item-control';
        ctrl.appendChild(createControl(meta, value, true));
        item.appendChild(info); item.appendChild(ctrl); grid.appendChild(item);
    });
    updateSectionCount('backend', count + ' 项');
}

function saveBackendConfig() {
    if (!BACKEND_STATE) return;
    var updates = {};
    BACKEND_STATE.editableKeys.forEach(function (key) {
        var meta = BACKEND_STATE.meta[key];
        var el = document.querySelector('[data-backend-key="' + key + '"]');
        if (!el) return;
        var nv;
        if (meta.type === 'bool') nv = !!el.checked;
        else if (meta.type === 'int') nv = parseInt(el.value, 10);
        else if (meta.type === 'float') nv = parseFloat(el.value);
        else nv = el.value;
        if (String(nv) !== String(BACKEND_STATE.values[key])) updates[key] = nv;
    });
    if (!Object.keys(updates).length) { AppCommon.toast('没有检测到变更', 'warning'); return; }
    fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ updates: updates }) })
        .then(function (r) { return r.json().then(function (p) { return { ok: r.ok, payload: p }; }); })
        .then(function (res) {
            if (res.payload.values) { BACKEND_STATE = res.payload; renderBackendConfig(); }
            if (res.ok) AppCommon.toast('后端配置已应用', 'success');
            else AppCommon.toast(res.payload.error || '保存失败', 'danger', 4000);
        }).catch(function (e) { AppCommon.toast('保存失败：' + e.message, 'danger', 4000); });
}

/* ── Data management section ── */
function buildDataSection() {
    var section = document.createElement('section');
    section.className = 'settings-section settings-panel';
    section.dataset.sectionKey = 'data';

    var toggle = document.createElement('div');
    toggle.className = 'settings-section-toggle';
    toggle.innerHTML = '<span class="settings-section-icon">💾</span><span class="settings-section-title-wrap"><span>数据管理</span>' +
        '<small>' + AppCommon.escapeHtml(SECTION_DESCRIPTIONS.data) + '</small></span>' +
        '<span class="settings-section-count">3 项</span>';

    var body = document.createElement('div');
    body.className = 'settings-section-body';
    body.innerHTML = '<div class="settings-data-actions">' +
        '<button class="app-btn-soft" type="button" id="exportAllBtn">📤 导出全部设置</button>' +
        '<button class="app-btn-soft" type="button" id="importAllBtn">📥 导入设置</button>' +
        '<button class="app-btn-danger" type="button" id="resetAllBtn">🔄 恢复默认</button></div>' +
        '<div class="settings-item-desc" style="margin-top:10px;">导出为 JSON 文件，可在其他浏览器导入恢复。重置将清除所有本地前端偏好。</div>';
    section.appendChild(toggle); section.appendChild(body);
    return {
        key: 'data',
        icon: '💾',
        title: '数据管理',
        description: SECTION_DESCRIPTIONS.data,
        countText: '3 项',
        element: section
    };
}

/* ── Import / Export ── */
var SCATTERED_KEYS = ['route_line_color', 'route_line_opacity', 'route_show_radius', 'route_teleport_cost'];

function exportSettings() {
    var data = { version: 1, exported: new Date().toISOString(), prefs: AppCommon.loadPrefs(), scattered: {} };
    SCATTERED_KEYS.forEach(function (k) { var v = localStorage.getItem(k); if (v != null) data.scattered[k] = v; });
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'game-map-settings-' + new Date().toISOString().slice(0, 10) + '.json';
    a.click(); URL.revokeObjectURL(a.href);
    AppCommon.toast('设置已导出', 'success');
}

function importSettings(file) {
    var reader = new FileReader();
    reader.onload = function (e) {
        try {
            var data = JSON.parse(e.target.result);
            if (!data.prefs || typeof data.prefs !== 'object') throw new Error('无效的设置文件');
            AppCommon.savePrefs(data.prefs);
            if (data.scattered) {
                Object.keys(data.scattered).forEach(function (k) { if (data.scattered[k] != null) localStorage.setItem(k, data.scattered[k]); });
            }
            AppCommon.applyTheme();
            buildSettings();
            AppCommon.toast('设置已导入并生效', 'success');
        } catch (err) { AppCommon.toast('导入失败：' + err.message, 'danger', 4000); }
    };
    reader.readAsText(file);
}

var _resetArmed = false, _resetTimer = null;
function resetAll() {
    if (_resetArmed) {
        clearTimeout(_resetTimer); _resetArmed = false;
        localStorage.removeItem('game-map-web:web:prefs');
        SCATTERED_KEYS.forEach(function (k) { localStorage.removeItem(k); });
        AppCommon.applyTheme();
        buildSettings();
        AppCommon.toast('已恢复默认设置', 'success');
    } else {
        _resetArmed = true;
        AppCommon.toast('再次点击确认重置', 'warning', 2000);
        _resetTimer = setTimeout(function () { _resetArmed = false; }, 2000);
    }
}

/* ── Event bindings ── */
document.getElementById('exportBtn').addEventListener('click', exportSettings);
document.getElementById('importBtn').addEventListener('click', function () { document.getElementById('importFile').click(); });
document.getElementById('importFile').addEventListener('change', function (e) {
    if (e.target.files[0]) importSettings(e.target.files[0]);
    e.target.value = '';
});

/* Delegate clicks for dynamically created buttons */
document.addEventListener('click', function (e) {
    var id = e.target.id;
    if (id === 'exportAllBtn') exportSettings();
    else if (id === 'importAllBtn') document.getElementById('importFile').click();
    else if (id === 'resetAllBtn') resetAll();
    else if (id === 'saveBackendBtn') saveBackendConfig();
    else if (id === 'refreshBackendBtn') loadBackendConfig();
});

/* ── Boot ── */
buildSettings();
requestAnimationFrame(function () {
    document.body.classList.add('ui-ready');
    document.body.classList.remove('page-preload');
});
