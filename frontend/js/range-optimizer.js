/* range-optimizer.js — 范围感知路线优化器（RARO）
 *
 * 核心思想：
 *   每个资源点都有采集范围 r，玩家只需进入范围内即可采集。
 *   当多个圆的范围有公共交集时，站在公共交集内可一次性采集所有点。
 *   本模块将资源点进行范围聚合，输出更少的"虚拟拜访点"，
 *   再将这些虚拟点与传送点一起交给 TSP 求最短路径。
 *
 * 算法：RARO-GC（贪心聚类 + Chebyshev中心）
 *   1. 构建范围重叠邻接矩阵
 *   2. 按邻居度从高到低贪心扩展聚类
 *   3. 每次扩展用交替投影法验证公共交集是否非空
 *   4. 每个聚类的访问位置 = 多圆公共区域的最深可行点（近似Chebyshev中心）
 *   5. TSP 作用于：虚拟拜访点 + 传送点（传送点不参与范围合并）
 *
 * 导出：
 *   solveRangeOptimizedTSP(waypoints, tpCost, fixedStartId, fixedEndId)
 *   buildRangeOptimizedWaypoints(waypoints)  // 仅聚类，不含TSP
 */

'use strict';

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   §1  几何工具函数
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

function euclidean(a, b) {
    var dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * 将点 P=(px,py) 投影到以 (cx,cy) 为圆心、半径 r 的圆盘内部（边界含）。
 * 如果点已在盘内，直接返回原点；否则投影到盘边界。
 */
function projectOntoDisk(px, py, cx, cy, r) {
    var dx = px - cx, dy = py - cy;
    var d2 = dx * dx + dy * dy;
    if (d2 <= r * r) return { x: px, y: py };
    var d = Math.sqrt(d2);
    return { x: cx + dx * r / d, y: cy + dy * r / d };
}

/**
 * 检查点 (px,py) 是否在所有圆内（含 tol 容差）。
 */
function isInAllDisks(px, py, circles, tol) {
    tol = tol || 0;
    for (var i = 0; i < circles.length; i++) {
        var c = circles[i];
        var dx = px - c.x, dy = py - c.y;
        if (dx * dx + dy * dy > (c.radius + tol) * (c.radius + tol)) return false;
    }
    return true;
}

/**
 * 求 N 个圆公共交集的近似 Chebyshev 中心（最深可行点）。
 *
 * 方法：带重启的交替投影法（AP）。
 *   - 首先从加权重心出发，轮询投影到每个违反约束的圆上。
 *   - 若收敛到可行点，再做一轮"向内推进"以求更深的中心。
 *   - 如果多次重启后仍无法满足所有约束，返回 null（交集为空）。
 *
 * @param {Array} circles  [{x, y, radius}, ...]
 * @param {number} [tol=1.5]  容差（像素），允许浮点误差和地图精度误差
 * @returns {{x,y}|null}
 */
function findChebyshevCenter(circles, tol) {
    tol = tol === undefined ? 1.5 : tol;
    var n = circles.length;
    if (n === 0) return null;
    if (n === 1) return { x: circles[0].x, y: circles[0].y };

    // 快速配对可行性检查：任意一对不交叉则整体无解（必要条件）
    for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
            var d = euclidean(circles[i], circles[j]);
            if (d > circles[i].radius + circles[j].radius + tol) return null;
        }
    }

    // 候选起始点：各圆圆心、加权重心、最小覆盖圆圆心（用圆心简化）
    var starts = [];

    // 加权重心（权重=半径）
    var wx = 0, wy = 0, wsum = 0;
    for (var i = 0; i < n; i++) { wx += circles[i].x * circles[i].radius; wy += circles[i].y * circles[i].radius; wsum += circles[i].radius; }
    starts.push({ x: wx / wsum, y: wy / wsum });

    // 每个圆心也作为起点（适合极端不均匀情形）
    for (var i = 0; i < n; i++) starts.push({ x: circles[i].x, y: circles[i].y });

    // 对每个起始点运行交替投影
    var MAX_ITER = 120;
    var bestX = null, bestY = null, bestSlack = -Infinity;

    for (var si = 0; si < starts.length; si++) {
        var px = starts[si].x, py = starts[si].y;

        for (var iter = 0; iter < MAX_ITER; iter++) {
            var prevX = px, prevY = py;

            // 轮询投影：先投影最违反的约束
            for (var ci = 0; ci < n; ci++) {
                var c = circles[ci];
                var dx = px - c.x, dy = py - c.y;
                var d2 = dx * dx + dy * dy;
                if (d2 > c.radius * c.radius) {
                    var d = Math.sqrt(d2);
                    px = c.x + dx * c.radius / d;
                    py = c.y + dy * c.radius / d;
                }
            }

            var moved = Math.sqrt((px - prevX) * (px - prevX) + (py - prevY) * (py - prevY));
            if (moved < 0.005) break;
        }

        // 计算当前点的最小松弛量（最深度）
        var minSlack = Infinity;
        for (var ci = 0; ci < n; ci++) {
            var c = circles[ci];
            var d = euclidean({ x: px, y: py }, c);
            var slack = c.radius - d;
            if (slack < minSlack) minSlack = slack;
        }

        if (minSlack > bestSlack) {
            bestSlack = minSlack;
            bestX = px; bestY = py;
        }
    }

    // 可行性判断
    if (bestSlack < -tol) return null;
    return { x: bestX, y: bestY };
}


/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   §2  核心算法：贪心聚类（RARO-GC）
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * 将一组资源点（非传送点）聚合成虚拟拜访点。
 *
 * 返回：{ clusters: [{visitPos, points, coveredIds}], singles: [] }
 *   其中 singles 是无法与任何点合并的孤立点（仍需单独访问）。
 */
function buildClusters(resources) {
    var n = resources.length;
    if (n === 0) return { clusters: [] };

    // 构建两两重叠邻接矩阵
    var adj = [];
    for (var i = 0; i < n; i++) {
        adj.push(new Array(n).fill(false));
        for (var j = 0; j < i; j++) {
            var d = euclidean(resources[i], resources[j]);
            var overlap = d < resources[i].radius + resources[j].radius;
            adj[i][j] = adj[j][i] = overlap;
        }
    }

    // 每个节点的邻居度
    var degrees = [];
    for (var i = 0; i < n; i++) {
        var deg = 0;
        for (var j = 0; j < n; j++) if (adj[i][j]) deg++;
        degrees.push(deg);
    }

    // 按度排序（降序），相同度按半径降序，使聚类从"最互连"的点开始
    var order = [];
    for (var i = 0; i < n; i++) order.push(i);
    order.sort(function (a, b) {
        if (degrees[b] !== degrees[a]) return degrees[b] - degrees[a];
        return resources[b].radius - resources[a].radius;
    });

    var assigned = new Array(n).fill(false);
    var clusters = [];

    for (var oi = 0; oi < order.length; oi++) {
        var startIdx = order[oi];
        if (assigned[startIdx]) continue;

        assigned[startIdx] = true;
        var clusterIdxs = [startIdx];

        // 候选集：与当前所有聚类成员都两两重叠的未分配点
        // 每轮扩展后重新过滤候选集
        var expandAgain = true;
        while (expandAgain) {
            expandAgain = false;

            // 找当前可合并候选（必须与所有当前聚类成员两两重叠）
            var candidates = [];
            for (var j = 0; j < n; j++) {
                if (assigned[j]) continue;
                var pairwiseOk = true;
                for (var k = 0; k < clusterIdxs.length; k++) {
                    if (!adj[j][clusterIdxs[k]]) { pairwiseOk = false; break; }
                }
                if (pairwiseOk) {
                    var distToCluster = euclidean(resources[j], resources[clusterIdxs[0]]);
                    candidates.push({ idx: j, distToStart: distToCluster });
                }
            }

            // 按到聚类重心的距离排序（近的优先，更可能进入公共交集）
            if (candidates.length > 0) {
                var gcx = 0, gcy = 0;
                clusterIdxs.forEach(function (i) { gcx += resources[i].x; gcy += resources[i].y; });
                gcx /= clusterIdxs.length; gcy /= clusterIdxs.length;

                candidates.sort(function (a, b) {
                    var da = euclidean(resources[a.idx], { x: gcx, y: gcy });
                    var db = euclidean(resources[b.idx], { x: gcx, y: gcy });
                    return da - db;
                });
            }

            // 依次尝试加入候选
            for (var ci = 0; ci < candidates.length; ci++) {
                var candIdx = candidates[ci].idx;
                // 对扩展后的聚类做 Chebyshev 可行性检查
                var testGroup = clusterIdxs.concat([candIdx]).map(function (i) { return resources[i]; });
                var center = findChebyshevCenter(testGroup);
                if (center !== null) {
                    clusterIdxs.push(candIdx);
                    assigned[candIdx] = true;
                    expandAgain = true; // 成功扩展后再扫一遍
                    break; // 本轮先加一个，再重新排序候选
                }
            }
        }

        // 计算最终聚类的访问位置
        var clusterCircles = clusterIdxs.map(function (i) { return resources[i]; });
        var visitPos = findChebyshevCenter(clusterCircles);
        if (!visitPos) {
            // Fallback：使用邻居最多的点的位置
            visitPos = { x: resources[startIdx].x, y: resources[startIdx].y };
        }

        clusters.push({
            visitPos: visitPos,
            points: clusterCircles,
            coveredIds: clusterCircles.map(function (p) { return p.id; }),
            merged: clusterIdxs.length > 1,
        });
    }

    return { clusters: clusters };
}


/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   §3  构建虚拟路径点
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * 主函数：将路径点（含传送点）转换为范围优化后的虚拟路径点集合。
 *
 * 传送点（isTeleport=true）不参与范围合并，直接传入 TSP。
 *
 * @param {Array} waypoints  原始路径点 [{id, x, y, radius, isTeleport, ...}]
 * @returns {{
 *   virtualWaypoints: Array,   // 供 TSP 使用的路径点（已减少）
 *   clusters: Array,           // 各聚类信息（供可视化）
 *   origResourceCount: number,
 *   virtualResourceCount: number,
 *   savingsPercent: number,    // 节点节省百分比
 * }}
 */
function buildRangeOptimizedWaypoints(waypoints) {
    var resources = waypoints.filter(function (w) { return !w.isTeleport; });
    var teleports = waypoints.filter(function (w) { return w.isTeleport; });

    if (resources.length === 0) {
        return {
            virtualWaypoints: teleports.slice(),
            clusters: [],
            origResourceCount: 0,
            virtualResourceCount: 0,
            savingsPercent: 0,
        };
    }

    var result = buildClusters(resources);
    var clusters = result.clusters;

    // 构建虚拟路径点
    var virtualWaypoints = [];
    var wpSeq = 0;

    clusters.forEach(function (cl) {
        wpSeq++;
        var isMulti = cl.points.length > 1;
        var firstPt = cl.points[0];

        // 虚拟点的导航半径：使用聚类中最小圆半径的一半（确保抵达虚拟位置就算"到达"）
        // 对单点则保留原始半径语义（兼容导航检测）
        var navRadius = isMulti
            ? Math.min.apply(null, cl.points.map(function (p) { return p.radius || 30; })) * 0.5
            : (firstPt.radius || 30);

        var names = cl.points.map(function (p) { return p.name || p.id; });
        var vwp = {
            id: 'raro_vwp_' + wpSeq,
            x: cl.visitPos.x,
            y: cl.visitPos.y,
            radius: navRadius,
            isTeleport: false,
            isCustom: false,
            isVirtual: true,
            isMultiMerge: isMulti,
            coveredPoints: cl.points,
            coveredIds: cl.coveredIds,
            coveredCount: cl.points.length,
            name: isMulti
                ? ('合并×' + cl.points.length + ': ' + names.slice(0, 3).join('、') + (names.length > 3 ? '…' : ''))
                : (firstPt.name || firstPt.id || '资源点'),
            typeName: isMulti ? '范围合并点' : (firstPt.typeName || '资源点'),
            groupName: firstPt.groupName || '',
            // 保留原始点信息用于导出
            _origFirstPt: firstPt,
        };
        virtualWaypoints.push(vwp);
    });

    // 传送点原样保留
    teleports.forEach(function (tp) { virtualWaypoints.push(tp); });

    var origN = resources.length;
    var virtN = clusters.length;

    return {
        virtualWaypoints: virtualWaypoints,
        clusters: clusters,
        origResourceCount: origN,
        virtualResourceCount: virtN,
        savingsPercent: origN > 0 ? Math.round((origN - virtN) / origN * 100) : 0,
    };
}


/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   §4  TSP 工具（内联，避免循环依赖）
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

function buildDistMatrix(points, tpCost) {
    var n = points.length;
    var D = new Array(n);
    for (var i = 0; i < n; i++) {
        D[i] = new Array(n);
        for (var j = 0; j < n; j++) {
            if (i === j) { D[i][j] = 0; continue; }
            D[i][j] = (points[i].isTeleport && points[j].isTeleport)
                ? tpCost
                : euclidean(points[i], points[j]);
        }
    }
    return D;
}

function findStartIndex(points) {
    var tpIdxs = [];
    var resIdxs = [];
    points.forEach(function (p, i) {
        if (p.isTeleport) tpIdxs.push(i); else resIdxs.push(i);
    });
    if (!tpIdxs.length) return 0;
    if (!resIdxs.length) return tpIdxs[0];
    var cx = 0, cy = 0;
    resIdxs.forEach(function (i) { cx += points[i].x; cy += points[i].y; });
    cx /= resIdxs.length; cy /= resIdxs.length;
    var best = tpIdxs[0], bestD = Infinity;
    tpIdxs.forEach(function (i) {
        var d = euclidean(points[i], { x: cx, y: cy });
        if (d < bestD) { bestD = d; best = i; }
    });
    return best;
}

function greedyTSP(D, startIdx, fixedEndIdx) {
    var n = D.length;
    var visited = new Array(n).fill(false);
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
    var improved = true, iter = 0;
    var jMax = fixBoundaries ? order.length - 2 : order.length - 1;
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


/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   §5  完整流水线：RARO 聚类 + TSP
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/**
 * 范围优化 TSP（主入口）。
 *
 * 1. 用 RARO-GC 将资源点聚合成虚拟拜访点
 * 2. 对 [虚拟点 + 传送点] 运行贪心TSP + 2-opt
 * 3. 返回与标准 solveTSP 兼容的结果对象（含 RARO 额外字段）
 *
 * @param {Array}  waypoints       原始路径点数组
 * @param {number} tpCost          传送权重（像素等效距离）
 * @param {string} [fixedStartId]  固定起点 id
 * @param {string} [fixedEndId]    固定终点 id
 * @returns {object}
 */
function solveRangeOptimizedTSP(waypoints, tpCost, fixedStartId, fixedEndId) {
    if (!waypoints || waypoints.length < 2) {
        var trivial = waypoints || [];
        return {
            order: trivial, segments: [], totalDist: 0, walkDist: 0, tpCount: 0,
            isRaro: true, clusters: [], origResourceCount: 0, virtualResourceCount: 0,
            savingsPercent: 0, mergedCount: 0,
        };
    }

    // ── 步骤 1：RARO 聚类 ──
    var raroResult = buildRangeOptimizedWaypoints(waypoints);
    var vwps = raroResult.virtualWaypoints;

    if (vwps.length < 2) {
        // 极端情况：全部被合并到一个点
        return {
            order: vwps, segments: [], totalDist: 0, walkDist: 0, tpCount: 0,
            isRaro: true,
            clusters: raroResult.clusters,
            origResourceCount: raroResult.origResourceCount,
            virtualResourceCount: raroResult.virtualResourceCount,
            savingsPercent: raroResult.savingsPercent,
            mergedCount: raroResult.clusters.filter(function (c) { return c.merged; }).length,
        };
    }

    // ── 步骤 2：TSP ──
    var D = buildDistMatrix(vwps, tpCost);

    var startIdx = -1, endIdx = -1;
    // 固定起/终点映射（虚拟点可能合并了原始起/终点）
    vwps.forEach(function (vwp, i) {
        if (fixedStartId) {
            if (vwp.id === fixedStartId || (vwp.coveredIds && vwp.coveredIds.indexOf(fixedStartId) >= 0))
                startIdx = i;
        }
        if (fixedEndId) {
            if (vwp.id === fixedEndId || (vwp.coveredIds && vwp.coveredIds.indexOf(fixedEndId) >= 0))
                endIdx = i;
        }
    });
    if (startIdx === -1) startIdx = findStartIndex(vwps);
    if (endIdx === startIdx) endIdx = -1;

    var fixEnd = endIdx !== -1;
    var order = greedyTSP(D, startIdx, fixEnd ? endIdx : null);
    order = twoOpt(order, D, 1000, fixEnd);
    order = pruneTeleports(order, vwps, D);

    var orderedPts = order.map(function (i) { return vwps[i]; });
    var segments = [], walkDist = 0, tpCount = 0;
    for (var i = 0; i < order.length - 1; i++) {
        var a = vwps[order[i]], b = vwps[order[i + 1]];
        var isTp = a.isTeleport && b.isTeleport;
        var segDist = D[order[i]][order[i + 1]];
        segments.push({ dist: segDist, isTp: isTp });
        if (isTp) tpCount++; else walkDist += segDist;
    }

    var mergedClusters = raroResult.clusters.filter(function (c) { return c.merged; });

    return {
        order: orderedPts,
        segments: segments,
        totalDist: totalCost(order, D),
        walkDist: walkDist,
        tpCount: tpCount,
        // RARO 专有字段
        isRaro: true,
        clusters: raroResult.clusters,
        origResourceCount: raroResult.origResourceCount,
        virtualResourceCount: raroResult.virtualResourceCount,
        savingsPercent: raroResult.savingsPercent,
        mergedCount: mergedClusters.length,
    };
}


/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   §6  导出
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

export { solveRangeOptimizedTSP, buildRangeOptimizedWaypoints, findChebyshevCenter };
