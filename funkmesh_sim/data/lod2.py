"""CityGML-LoD2 (AdV-Profil, z. B. NRW) einlesen und Dachflächen rastern.

Jede ``bldg:RoofSurface`` ist ein ebenes 3D-Polygon. Wir bestimmen die Ebene über die
Newell-Normale und füllen das Polygon (inkl. Innenringen, even-odd) per Scanline auf
Pixelmittelpunkte. Pro Pixel wird die maximale Dachhöhe [m ü. NHN] gespeichert – so bleiben
First und Traufe erhalten, Straßen zwischen den Häusern bleiben frei.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from lxml import etree
from numba import njit

NS_GML = "http://www.opengis.net/gml"
ROOF_TAGS = (
    "{http://www.opengis.net/citygml/building/1.0}RoofSurface",
    "{http://www.opengis.net/citygml/building/2.0}RoofSurface",
)


def _parse_poslist(text: str) -> np.ndarray:
    v = np.array(text.split(), dtype=np.float64)
    return v.reshape(-1, 3)


def read_roof_polygons(path: Path):
    """Liefert (coords[N,3], ring_off[R+1], poly_ring_off[P+1]) aller Dachpolygone."""
    coords: list[np.ndarray] = []
    ring_off = [0]
    poly_ring_off = [0]
    n = 0
    ctx = etree.iterparse(str(path), events=("end",), tag=ROOF_TAGS, huge_tree=True)
    for _ev, elem in ctx:
        for poly in elem.iter(f"{{{NS_GML}}}Polygon"):
            rings = []
            for tag in ("exterior", "interior"):
                for ring in poly.iter(f"{{{NS_GML}}}{tag}"):
                    pl = ring.find(f".//{{{NS_GML}}}posList")
                    if pl is not None and pl.text:
                        pts = _parse_poslist(pl.text)
                    else:
                        ps = ring.findall(f".//{{{NS_GML}}}pos")
                        if not ps:
                            continue
                        pts = np.array([[float(t) for t in p.text.split()] for p in ps])
                    if len(pts) >= 3:
                        rings.append(pts)
            if not rings:
                continue
            for pts in rings:
                coords.append(pts)
                n += len(pts)
                ring_off.append(n)
            poly_ring_off.append(len(ring_off) - 1)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    if not coords:
        return np.zeros((0, 3)), np.zeros(1, np.int64), np.zeros(1, np.int64)
    return (
        np.concatenate(coords),
        np.asarray(ring_off, np.int64),
        np.asarray(poly_ring_off, np.int64),
    )


@njit(cache=True)
def rasterize_roofs(coords, ring_off, poly_ring_off, x0, y1, res, nrows, ncols, out):
    """Füllt ``out`` (float32, NaN = kein Dach) mit max. Dachhöhe je Pixel."""
    npoly = poly_ring_off.shape[0] - 1
    xs_buf = np.empty(4096, np.float64)
    for p in range(npoly):
        r0 = poly_ring_off[p]
        r1 = poly_ring_off[p + 1]
        a = ring_off[r0]
        b = ring_off[r0 + 1]
        # Newell-Normale des Außenrings
        nx = 0.0
        ny = 0.0
        nz = 0.0
        cx = 0.0
        cy = 0.0
        cz = 0.0
        m = b - a
        for i in range(a, b):
            j = i + 1 if i + 1 < b else a
            xi = coords[i, 0]
            yi = coords[i, 1]
            zi = coords[i, 2]
            xj = coords[j, 0]
            yj = coords[j, 1]
            zj = coords[j, 2]
            nx += (yi - yj) * (zi + zj)
            ny += (zi - zj) * (xi + xj)
            nz += (xi - xj) * (yi + yj)
            cx += xi
            cy += yi
            cz += zi
        cx /= m
        cy /= m
        cz /= m
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        if nn <= 0.0:
            continue
        if abs(nz) / nn < 0.05:  # nahezu senkrecht -> keine Dachfläche
            continue
        # Bounding Box in Pixeln
        miny = 1e30
        maxy = -1e30
        minx = 1e30
        maxx = -1e30
        zmax_poly = -1e30
        zmin_poly = 1e30
        for i in range(ring_off[r0], ring_off[r1]):
            if coords[i, 1] < miny:
                miny = coords[i, 1]
            if coords[i, 1] > maxy:
                maxy = coords[i, 1]
            if coords[i, 0] < minx:
                minx = coords[i, 0]
            if coords[i, 0] > maxx:
                maxx = coords[i, 0]
            if coords[i, 2] > zmax_poly:
                zmax_poly = coords[i, 2]
            if coords[i, 2] < zmin_poly:
                zmin_poly = coords[i, 2]
        row_a = int(np.floor((y1 - maxy) / res - 0.5))
        row_b = int(np.ceil((y1 - miny) / res - 0.5))
        if row_a < 0:
            row_a = 0
        if row_b > nrows - 1:
            row_b = nrows - 1
        for row in range(row_a, row_b + 1):
            yc = y1 - (row + 0.5) * res
            k = 0
            for r in range(r0, r1):
                ra = ring_off[r]
                rb = ring_off[r + 1]
                for i in range(ra, rb):
                    j = i + 1 if i + 1 < rb else ra
                    ya = coords[i, 1]
                    yb = coords[j, 1]
                    if (ya <= yc < yb) or (yb <= yc < ya):
                        t = (yc - ya) / (yb - ya)
                        if k < xs_buf.shape[0]:
                            xs_buf[k] = coords[i, 0] + t * (coords[j, 0] - coords[i, 0])
                            k += 1
            if k < 2:
                continue
            xs = np.sort(xs_buf[:k])
            for q in range(0, k - 1, 2):
                ca = int(np.ceil((xs[q] - x0) / res - 0.5))
                cb = int(np.floor((xs[q + 1] - x0) / res - 0.5))
                if ca < 0:
                    ca = 0
                if cb > ncols - 1:
                    cb = ncols - 1
                for col in range(ca, cb + 1):
                    xc = x0 + (col + 0.5) * res
                    z = cz - (nx * (xc - cx) + ny * (yc - cy)) / nz
                    # Ebene nicht über die Polygonhöhen hinaus extrapolieren
                    if z > zmax_poly:
                        z = zmax_poly
                    if z < zmin_poly:
                        z = zmin_poly
                    cur = out[row, col]
                    if not (cur >= z):  # NaN oder kleiner
                        out[row, col] = z
    return out


def roof_raster(path: Path, x0: float, y1: float, res: float, nrows: int, ncols: int) -> np.ndarray:
    coords, ring_off, poly_ring_off = read_roof_polygons(path)
    out = np.full((nrows, ncols), np.nan, np.float32)
    if len(coords):
        rasterize_roofs(coords, ring_off, poly_ring_off, x0, y1, res, nrows, ncols, out)
    return out
