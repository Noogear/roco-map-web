/* tsp-solver.js — Pure TSP algorithm (greedy + 2-opt), zero dependencies */
'use strict';

import {
    euclidean,
    buildDistMatrix,
    findStartIndex,
    greedyTSP,
    totalCost,
    twoOpt,
    pruneTeleports
} from './route-core.js';

function solveTSP(waypoints, tpCost, fixedStartId, fixedEndId) {
    var D = buildDistMatrix(waypoints, tpCost);
    var startIdx = -1, endIdx = -1;
    waypoints.forEach(function (wp, i) {
        if (fixedStartId && wp.id === fixedStartId) startIdx = i;
        if (fixedEndId && wp.id === fixedEndId) endIdx = i;
    });
    if (startIdx === -1) startIdx = findStartIndex(waypoints);
    if (endIdx === startIdx) endIdx = -1;
    var fixEnd = endIdx !== -1;
    var order = greedyTSP(D, startIdx, fixEnd ? endIdx : null);
    order = twoOpt(order, D, 1000, fixEnd);
    order = pruneTeleports(order, waypoints, D);
    var orderedPts = order.map(function (i) { return waypoints[i]; });
    var segments = [], walkDist = 0, tpCount = 0;
    for (var i = 0; i < order.length - 1; i++) {
        var a = waypoints[order[i]], b = waypoints[order[i + 1]];
        var isTp = a.isTeleport && b.isTeleport;
        segments.push({ dist: D[order[i]][order[i + 1]], isTp: isTp });
        if (isTp) tpCount++; else walkDist += D[order[i]][order[i + 1]];
    }
    return { order: orderedPts, segments: segments, totalDist: totalCost(order, D), walkDist: walkDist, tpCount: tpCount };
}

export { euclidean, solveTSP, twoOpt };
