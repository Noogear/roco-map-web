/* route-core.js — 路线计算共享内核（TSP 基础工具） */
'use strict';

function euclidean(a, b) {
    var dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function buildDistMatrix(points, tpCost) {
    var n = points.length, D = new Array(n);
    for (var i = 0; i < n; i++) {
        D[i] = new Array(n);
        for (var j = 0; j < n; j++) {
            D[i][j] = (i === j) ? 0 : ((points[i].isTeleport && points[j].isTeleport) ? tpCost : euclidean(points[i], points[j]));
        }
    }
    return D;
}

function findStartIndex(points) {
    var resources = points.filter(function (p) { return !p.isTeleport; });
    var tpIdxs = [];
    points.forEach(function (p, i) { if (p.isTeleport) tpIdxs.push(i); });
    if (!tpIdxs.length) return 0;
    if (!resources.length) return tpIdxs[0];
    var cx = 0, cy = 0;
    resources.forEach(function (p) { cx += p.x; cy += p.y; });
    cx /= resources.length; cy /= resources.length;
    var bestIdx = tpIdxs[0], bestDist = Infinity;
    tpIdxs.forEach(function (i) {
        var d = euclidean(points[i], { x: cx, y: cy });
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    });
    return bestIdx;
}

function greedyTSP(D, startIdx, fixedEndIdx) {
    var n = D.length, visited = new Array(n);
    for (var i = 0; i < n; i++) visited[i] = false;
    if (fixedEndIdx != null) visited[fixedEndIdx] = true;
    var order = [startIdx]; visited[startIdx] = true;
    for (var step = 1; step < n - (fixedEndIdx != null ? 1 : 0); step++) {
        var last = order[order.length - 1], bestJ = -1, bestDv = Infinity;
        for (var j = 0; j < n; j++) {
            if (!visited[j] && D[last][j] < bestDv) { bestDv = D[last][j]; bestJ = j; }
        }
        if (bestJ === -1) break;
        order.push(bestJ); visited[bestJ] = true;
    }
    if (fixedEndIdx != null) order.push(fixedEndIdx);
    return order;
}

function totalCost(order, D) {
    var s = 0;
    for (var i = 0; i < order.length - 1; i++) s += D[order[i]][order[i + 1]];
    return s;
}

function twoOpt(order, D, maxIter, fixBoundaries) {
    var improved = true, iter = 0, jMax = fixBoundaries ? order.length - 2 : order.length - 1;
    // Protect against massive freeze if order is too large (e.g. 500+ nodes)
    if (order.length > 200) maxIter = Math.min(maxIter, 20);
    if (order.length > 500) maxIter = Math.min(maxIter, 2);

    while (improved && iter < maxIter) {
        improved = false; iter++;
        for (var i = 1; i < jMax; i++) {
            for (var j = i + 1; j <= jMax; j++) {
                var nj = (j + 1 < order.length) ? order[j + 1] : -1;
                var oldD = D[order[i - 1]][order[i]] + (nj >= 0 ? D[order[j]][nj] : 0);
                var newD = D[order[i - 1]][order[j]] + (nj >= 0 ? D[order[i]][nj] : 0);
                if (newD < oldD - 0.01) {
                    var l = i, r = j;
                    while (l < r) { var t = order[l]; order[l] = order[r]; order[r] = t; l++; r--; }
                    improved = true;
                }
            }
        }
    }
    return order;
}

function pruneTeleports(order, points, D) {
    var changed = true;
    while (changed) {
        changed = false;
        for (var i = 1; i < order.length - 1; i++) {
            if (!points[order[i]].isTeleport) continue;
            if (D[order[i - 1]][order[i + 1]] <= D[order[i - 1]][order[i]] + D[order[i]][order[i + 1]] + 0.01) {
                order.splice(i, 1); changed = true; break;
            }
        }
    }
    return order;
}

export { euclidean, buildDistMatrix, findStartIndex, greedyTSP, totalCost, twoOpt, pruneTeleports };