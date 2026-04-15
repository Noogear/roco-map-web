/* route-algorithms.js — 路线算法统一入口（范围优化 / 节点优化 / 手动排序） */
'use strict';

import * as TSPSolver from './tsp-solver.js';
import * as RangeOptimizer from './range-optimizer.js';

const ALGO_HINTS = {
    range: '范围优化：重叠采集圈合并为单一站点，传送点不参与合并',
    node: '节点优化：TSP最短路径，可指定固定起点/终点',
    manual: '手动排序：长按拖拽列表项调整顺序，首项为起点，末项为终点',
};

const MODE_LABELS = {
    range: '🔮 范围优化',
    node: '⚡ 节点优化',
    manual: '手动排序',
};

function getAlgoHint(algo) {
    return ALGO_HINTS[algo] || ALGO_HINTS.node;
}

function getModeLabel(algorithm, result) {
    if (algorithm === 'manual') return MODE_LABELS.manual;
    if (result && result.isRaro) return MODE_LABELS.range;
    return MODE_LABELS.node;
}

function solveManualRoute(waypoints, tpCost) {
    if (!Array.isArray(waypoints) || waypoints.length < 2) {
        return { order: Array.isArray(waypoints) ? waypoints.slice() : [], segments: [], totalDist: 0, walkDist: 0, tpCount: 0 };
    }
    var pts = waypoints;
    var segments = [];
    var walkDist = 0;
    var tpCount = 0;

    for (var i = 0; i < pts.length - 1; i++) {
        var a = pts[i], b = pts[i + 1];
        var isTp = a.isTeleport && b.isTeleport;
        var d = isTp ? tpCost : TSPSolver.euclidean(a, b);
        segments.push({ dist: d, isTp: isTp });
        if (isTp) tpCount++; else walkDist += d;
    }

    var total = segments.reduce(function (s, seg) { return s + seg.dist; }, 0);
    return { order: pts.slice(), segments: segments, totalDist: total, walkDist: walkDist, tpCount: tpCount };
}

function solveRouteByAlgorithm(options) {
    var opts = options || {};
    var algorithm = opts.algorithm || 'node';
    var waypoints = Array.isArray(opts.waypoints) ? opts.waypoints : [];
    var tpCost = Number(opts.tpCost || 0);
    var fixedStartId = opts.fixedStartId || null;
    var fixedEndId = opts.fixedEndId || null;

    var t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    var result;

    if (algorithm === 'manual') {
        result = solveManualRoute(waypoints, tpCost);
    } else if (algorithm === 'range') {
        result = RangeOptimizer.solveRangeOptimizedTSP(waypoints, tpCost, fixedStartId, fixedEndId);
    } else {
        result = TSPSolver.solveTSP(waypoints, tpCost, fixedStartId, fixedEndId);
    }

    var t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    return {
        algorithm: algorithm,
        elapsedMs: Math.max(0, Math.round(t1 - t0)),
        result: result,
    };
}

function buildAlgorithmToast(solved) {
    var s = solved || {};
    var algorithm = s.algorithm || 'node';
    var elapsedMs = Number(s.elapsedMs || 0);
    var result = s.result || {};

    if (algorithm === 'manual') {
        return { message: '已按手动顺序计算路线', type: 'ok', duration: 1200 };
    }
    if (algorithm === 'range') {
        var msg = '范围优化完成（' + elapsedMs + 'ms）';
        if (result && result.savingsPercent > 0) msg += '，节省 ' + result.savingsPercent + '% 路径点';
        return { message: msg, type: 'ok', duration: 2200 };
    }
    return { message: '节点优化完成（' + elapsedMs + 'ms）', type: 'ok', duration: 1200 };
}

export { getAlgoHint, getModeLabel, solveManualRoute, solveRouteByAlgorithm, buildAlgorithmToast };
