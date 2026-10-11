#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Make a driving route for record_cameras.py: a closed loop in the right-hand lane.

    python tools/route.py scene/ routes/east_loop.csv

The loop is given below as rough waypoints, a couple of metres from where they should be. Each
is pulled onto the middle of the pavement, measured across the road on a raster of
speedway_road.obj, then moved into the right-hand lane where the road is narrow enough to have
lanes. Where it is wider, on the skid pad, the waypoint is used as given.
"""

import math
import os
import sys

import numpy as np

# Anticlockwise round the east end: east along the south road, north up the east side, west
# along the top, down the connector onto the skid pad and across it, back onto the south road.
EAST_LOOP = [
    (118, 9), (140, 9), (165, 8), (200, 5), (225, 4), (242, 7), (254, 14), (261, 26), (264, 45),
    (265, 70), (265, 100), (265, 130), (264, 152), (258, 170), (246, 182), (228, 187), (205, 187),
    (190, 188), (178, 188), (172, 183), (172, 175), (172.5, 165), (172, 153), (167, 137), (150, 120), (130, 103),
    (110, 85), (95, 70), (84, 56), (81, 44), (84, 32), (91, 21), (102, 13),
]

PIXEL = 0.2         # raster resolution, metres
SPACING = 2.0       # distance between route points, metres
LANE_ROADS = 12.0   # roads narrower than this have two lanes
LANE_OFFSET = 1.8   # how far right of the middle to drive, at most a quarter of the road
JUNCTION = 30.0     # a wide stretch shorter than this, metres, is a junction the lane carries across


def pavement(scene_dir, bounds):
    """Boolean raster of the pavement inside bounds = (x0, y0, x1, y1), PIXEL per cell."""
    verts, faces = [], []
    with open(os.path.join(scene_dir, "speedway_road.obj")) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:3]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    v, f = np.array(verts), np.array(faces)
    x0, y0, x1, y1 = bounds
    w, h = int((x1 - x0) / PIXEL), int((y1 - y0) / PIXEL)
    grid = np.zeros((h, w), bool)
    for a, b, c in v[f]:
        lo = np.floor((np.minimum(np.minimum(a, b), c) - (x0, y0)) / PIXEL).astype(int)
        hi = np.ceil((np.maximum(np.maximum(a, b), c) - (x0, y0)) / PIXEL).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, (w, h))
        if (hi <= lo).any():
            continue
        px, py = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        qx, qy = x0 + (px + 0.5) * PIXEL, y0 + (py + 0.5) * PIXEL
        # Barycentric inside test, either winding.
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (qx - c[0]) + (c[0] - b[0]) * (qy - c[1])) / d
        l2 = ((c[1] - a[1]) * (qx - c[0]) + (a[0] - c[0]) * (qy - c[1])) / d
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        grid[py[inside], px[inside]] = True
    return grid


def resample(points, spacing, closed=True):
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(3, int(s[-1] / spacing))
    t = np.linspace(0, s[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def smooth(p, passes):
    for _ in range(passes):
        p = 0.25 * np.roll(p, 1, axis=0) + 0.5 * p + 0.25 * np.roll(p, -1, axis=0)
    return p


def tangents(p):
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    scene_dir, out = sys.argv[1], sys.argv[2]
    coarse = np.array(EAST_LOOP, float)
    bounds = (coarse[:, 0].min() - 40, coarse[:, 1].min() - 40, coarse[:, 0].max() + 40, coarse[:, 1].max() + 40)
    grid = pavement(scene_dir, bounds)
    h, w = grid.shape

    def on_road(q):
        i, j = int((q[1] - bounds[1]) / PIXEL), int((q[0] - bounds[0]) / PIXEL)
        return 0 <= i < h and 0 <= j < w and grid[i, j]

    def reach(q, d, limit=15.0):
        """How far from q the pavement goes in direction d."""
        r = 0.0
        while r < limit and on_road(q + r * d):
            r += PIXEL / 2
        return r

    p = smooth(resample(coarse, SPACING), 4)
    t = tangents(p)
    normal = np.column_stack([-t[:, 1], t[:, 0]])  # to the left
    target = np.full_like(p, np.nan)
    for k, (q, n) in enumerate(zip(p, normal)):
        if not on_road(q):
            raise SystemExit(f"waypoint near ({q[0]:.1f}, {q[1]:.1f}) is off the pavement")
        left, right = reach(q, n), reach(q, -n)
        width = left + right
        if width < LANE_ROADS:
            middle = q + 0.5 * (left - right) * n
            target[k] = middle - min(LANE_OFFSET, width / 4) * n
    # A junction widens the road for a few metres. Carry the lane straight across it instead of
    # jumping to the waypoint and back. Longer wide stretches, the skid pad, keep the waypoints.
    wide = np.isnan(target[:, 0])
    n = len(p)
    k = 0
    while k < n:
        if not wide[k]:
            k += 1
            continue
        end = k
        while end < k + n and wide[end % n]:
            end += 1
        if end - k <= JUNCTION / SPACING and not wide[(k - 1) % n] and not wide[end % n]:
            a, b = target[(k - 1) % n], target[end % n]
            for j in range(k, end):
                f = (j - k + 1) / (end - k + 1)
                target[j % n] = (1 - f) * a + f * b
        else:
            for j in range(k, end):
                target[j % n] = p[j % n]
        k = end
    p = smooth(target, 6)

    for q in p:
        if not on_road(q):
            raise SystemExit(f"the route leaves the pavement near ({q[0]:.1f}, {q[1]:.1f})")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("# Closed loop, right-hand lane. x, y in the scene frame, metres. Made by tools/route.py\n")
        f.write("x,y\n")
        for x, y in p:
            f.write(f"{x:.2f},{y:.2f}\n")
    length = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T).sum()
    print(f"{out}: {len(p)} points, {length:.0f} m round")


if __name__ == "__main__":
    main()
