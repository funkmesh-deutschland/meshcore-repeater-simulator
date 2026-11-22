"""XYZ-Kacheln (Web-Mercator) aus den UTM-Rastern rendern – pixelgenau reprojiziert."""
from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image
from pyproj import Transformer

_T = Transformer.from_crs(3857, 25832, always_xy=True)
R = 6378137.0
EMPTY = None


def _empty_png():
    global EMPTY
    if EMPTY is None:
        b = io.BytesIO()
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(b, "PNG")
        EMPTY = b.getvalue()
    return EMPTY


def tile_utm(z, x, y, size=256):
    n = 2 ** z
    span = 2 * math.pi * R / n
    x0 = -math.pi * R + x * span
    y0 = math.pi * R - y * span
    px = x0 + (np.arange(size) + 0.5) * span / size
    py = y0 - (np.arange(size) + 0.5) * span / size
    mx, my = np.meshgrid(px, py)
    ux, uy = _T.transform(mx, my)
    return ux, uy


def _lut(stops, n=256):
    """stops: Liste (pos 0..1, (r,g,b,a))."""
    lut = np.zeros((n, 4), np.uint8)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], float)
    xs = np.linspace(0, 1, n)
    for ch in range(4):
        lut[:, ch] = np.interp(xs, pos, cols[:, ch]).round().astype(np.uint8)
    return lut


# Farbskalen (Funkmesh-Palette: Petrol #0f766e, Türkis #14b8a6, Orange #fc790d)
LUT_RSSI = _lut([(0.0, (68, 1, 84, 150)), (0.25, (59, 82, 139, 175)), (0.5, (33, 145, 140, 190)),
                 (0.75, (94, 201, 98, 200)), (1.0, (253, 231, 37, 210))])
LUT_PROB = _lut([(0.0, (220, 38, 38, 0)), (0.5, (220, 38, 38, 0)), (0.501, (220, 38, 38, 150)),
                 (0.7, (252, 121, 13, 170)), (0.85, (250, 204, 21, 180)), (0.9, (20, 184, 166, 185)),
                 (1.0, (15, 118, 110, 200))])
LUT_LOAD = _lut([(0.0, (20, 184, 166, 120)), (0.33, (250, 204, 21, 170)), (0.66, (252, 121, 13, 190)),
                 (1.0, (185, 28, 28, 210))])
LUT_BLD = _lut([(0.0, (180, 170, 200, 200)), (1.0, (60, 30, 110, 230))])
LUT_VEG = _lut([(0.0, (170, 220, 140, 120)), (1.0, (20, 90, 40, 200))])

MARGIN_BINS = [(-1e9, -10, (0, 0, 0, 0)), (-10, 0, (220, 38, 38, 140)), (0, 5, (252, 121, 13, 170)),
               (5, 10, (250, 204, 21, 180)), (10, 20, (94, 234, 212, 185)), (20, 1e9, (15, 118, 110, 200))]
COUNT_COLS = np.array([(0, 0, 0, 0), (252, 121, 13, 170), (250, 204, 21, 180), (20, 184, 166, 185),
                       (15, 118, 110, 200), (8, 70, 66, 215)], np.uint8)
CAT = np.array([(15, 118, 110), (252, 121, 13), (37, 99, 235), (219, 39, 119), (101, 163, 13),
                (124, 58, 237), (234, 179, 8), (8, 145, 178), (190, 18, 60), (71, 85, 105),
                (16, 185, 129), (249, 115, 22), (99, 102, 241), (236, 72, 153), (132, 204, 22)], np.uint8)


class TileRenderer:
    def __init__(self, grid, cov):
        self.grid = grid
        self.cov = cov
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.load_arr = None

    def _sample(self, arr, ux, uy, res, x0, y1):
        r = np.floor((y1 - uy) / res).astype(np.int64)
        c = np.floor((ux - x0) / res).astype(np.int64)
        ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
        rr = np.where(ok, r, 0)
        cc = np.where(ok, c, 0)
        v = arr[rr, cc]  # Fancy-Indexing liest nur die benötigten Seiten der memmap
        return np.asarray(v), ok

    def render(self, layer, z, x, y, version):
        key = (layer, version, z, x, y)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        png = self._render(layer, z, x, y)
        with self.lock:
            self.cache[key] = png
            if len(self.cache) > 3000:
                self.cache.popitem(last=False)
        return png

    def _render(self, layer, z, x, y):
        m = self.grid.meta
        ux, uy = tile_utm(z, x, y)
        if ux.max() < m.x0 or ux.min() > m.x1 or uy.max() < m.y0 or uy.min() > m.y1:
            return _empty_png()
        rgba = np.zeros(ux.shape + (4,), np.uint8)
        if layer in ("terrain", "buildings", "vegetation"):
            src = {"terrain": self.grid.terrain, "buildings": self.grid.bld, "vegetation": self.grid.veg}[layer]
            v, ok = self._sample(src, ux, uy, m.res, m.x0, m.y1)
            if layer == "terrain":
                v = v.astype(np.float32)
                px = (ux[0, 1] - ux[0, 0])
                gy, gx = np.gradient(v, px, px)
                slope = np.pi / 2 - np.arctan(np.hypot(gx, gy) * 3)
                asp = np.arctan2(-gx, gy)
                az, alt = np.radians(315), np.radians(45)
                hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - asp)
                g = np.clip(hs * 255, 0, 255).astype(np.uint8)
                lo, hi = 40.0, 190.0
                t = np.clip((v - lo) / (hi - lo), 0, 1)
                rgba[..., 0] = (g * (0.75 + 0.25 * t)).astype(np.uint8)
                rgba[..., 1] = (g * (0.85 + 0.1 * t)).astype(np.uint8)
                rgba[..., 2] = (g * (0.8 - 0.1 * t)).astype(np.uint8)
                rgba[..., 3] = np.where(ok, 170, 0)
            else:
                h = v.astype(np.float32) / 10.0
                lut = LUT_BLD if layer == "buildings" else LUT_VEG
                idx = np.clip(h / (25.0 if layer == "buildings" else 30.0) * 255, 0, 255).astype(int)
                rgba[:] = lut[idx]
                rgba[~(ok & (h > 0))] = 0
        else:
            meta = self.cov.result_meta
            if not meta:
                return _empty_png()
            res = m.res * meta["k"]
            name = {"rssi": "rssi", "margin": "margin", "prob": "prob", "count": "count", "best": "best",
                    "load": "load"}.get(layer)
            arr = self.load_arr if layer == "load" else self.cov.array(name)
            if arr is None:
                return _empty_png()
            v, ok = self._sample(arr, ux, uy, res, m.x0, m.y1)
            if layer == "rssi":
                ok &= v != -32768
                t = np.clip((v / 10.0 + 135.0) / 60.0, 0, 1)
                rgba[:] = LUT_RSSI[(t * 255).astype(int)]
                rgba[~ok] = 0
            elif layer == "margin":
                ok &= v != -32768
                mv = v / 10.0
                for lo, hi, col in MARGIN_BINS:
                    sel = ok & (mv >= lo) & (mv < hi)
                    rgba[sel] = col
            elif layer == "prob":
                ok &= v != 255
                rgba[:] = LUT_PROB[np.clip(v.astype(int) * 255 // 100, 0, 255)]
                rgba[~ok] = 0
            elif layer == "count":
                rgba[:] = COUNT_COLS[np.clip(v.astype(int), 0, len(COUNT_COLS) - 1)]
                rgba[~ok] = 0
            elif layer == "best":
                ok &= v > 0
                col = CAT[(v.astype(int) - 1) % len(CAT)]
                rgba[..., :3] = col
                rgba[..., 3] = 170
                rgba[~ok] = 0
            elif layer == "load":
                ok &= v != 255
                rgba[:] = LUT_LOAD[np.clip(v.astype(int) * 255 // 30, 0, 255)]
                rgba[~ok | (v == 0)] = 0
        b = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(b, "PNG", compress_level=3)
        return b.getvalue()
