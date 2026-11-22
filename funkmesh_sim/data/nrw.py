"""Geodaten NRW (OpenGeodata.NRW, Lizenz dl-de/zero-2.0) laden und ins Gebietsraster schreiben.

Quellen (je 1-km-Kachel, ETRS89/UTM32, Höhen DHHN2016):

* DGM1  – Digitales Geländemodell 1 m (Laserscanning)
* DOM1  – Digitales Oberflächenmodell 1 m (Laserscanning, enthält Bewuchs)
* LoD2  – 3D-Gebäudemodelle (CityGML) mit Dachformen

Verarbeitung je Kachel (1 m -> Zielauflösung, Standard 5 m):

* Gelände  = Mittelwert DGM1
* Gebäude  = max. LoD2-Dachhöhe - Gelände, wenn >= 50 % der Zelle bebaut
* Bewuchs  = 75%-Perzentil von (DOM1 - DGM1) über den Bewuchs-Pixeln (> 2 m, nicht Gebäude),
             wenn >= 30 % der Zelle bewachsen
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage

from .. import config

log = logging.getLogger(__name__)

BASE = "https://www.opengeodata.nrw.de/produkte/geobasis"
PRODUCTS = {
    "dgm": f"{BASE}/hm/dgm1_tiff/dgm1_tiff",
    "dom": f"{BASE}/hm/dom1_tiff/dom1_tiff",
    "lod2": f"{BASE}/3dg/lod2_gml/lod2_gml",
}
PAD = 50  # m Rand für Gebäude, die über die Kachelgrenze ragen
UA = {"User-Agent": "funkmesh-repeater-simulator/0.1 (+https://funkmesh.de)"}


def _index(product: str, cache: Path) -> dict[tuple[int, int], str]:
    f = cache / f"index_{product}.json"
    if not f.exists() or time.time() - f.stat().st_mtime > 30 * 86400:
        r = requests.get(f"{PRODUCTS[product]}/index.json", headers=UA, timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
    files = json.loads(f.read_text())["datasets"][0]["files"]
    out = {}
    for x in files:
        parts = x["name"].split("_")
        try:
            e, n = int(parts[2]), int(parts[3])
        except (IndexError, ValueError):
            continue
        out[(e, n)] = x["name"]
    return out


def region_tiles(boundary_geojson: Path, buffer_m: float):
    """1-km-Kacheln (E, N in km), die die gepufferte Regionsgrenze schneiden."""
    from shapely.geometry import box, shape
    from shapely import transform as shp_transform
    from pyproj import Transformer

    gj = json.loads(Path(boundary_geojson).read_text())
    geom = shape(gj["features"][0]["geometry"])
    t = Transformer.from_crs(4326, 25832, always_xy=True)
    g = shp_transform(geom, lambda xy: np.column_stack(t.transform(xy[:, 0], xy[:, 1])))
    gb = g.buffer(buffer_m)
    minx, miny, maxx, maxy = gb.bounds
    tiles = []
    for e in range(int(minx // 1000), int(maxx // 1000) + 1):
        for n in range(int(miny // 1000), int(maxy // 1000) + 1):
            if gb.intersects(box(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)):
                tiles.append((e, n))
    return tiles, g


def _download(url: str, dest: Path, session: requests.Session) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with session.get(url, headers=UA, timeout=120, stream=True) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            os.replace(tmp, dest)
            return dest
        except FileNotFoundError:
            raise
        except Exception as exc:  # Netzwerkfehler -> erneut versuchen
            log.warning("Download fehlgeschlagen (%s), Versuch %d: %s", url, attempt + 1, exc)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Download dauerhaft fehlgeschlagen: {url}")


def _read_tif(path: Path | None, e: int, n: int) -> np.ndarray | None:
    if path is None:
        return None
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float32)
        tr = ds.transform
        if abs(tr.c - e * 1000) > 0.01 or abs(tr.f - (n + 1) * 1000) > 0.01 or a.shape != (1000, 1000):
            raise ValueError(f"Unerwartete Kachelgeometrie {path}: {tr} {a.shape}")
        nd = ds.nodata
    if nd is not None:
        a[a == nd] = np.nan
    a[a < -500] = np.nan
    return a


def process_tile(e: int, n: int, paths: dict, res: int) -> dict:
    """Läuft im Worker-Prozess. Gibt aggregierte Arrays in Zielauflösung zurück."""
    from .lod2 import roof_raster

    k = res
    m = 1000 // k
    dgm = _read_tif(paths.get("dgm"), e, n)
    dom = _read_tif(paths.get("dom"), e, n)
    if dgm is None:
        return {"e": e, "n": n, "ok": False}
    terrain = np.nanmean(dgm.reshape(m, k, m, k), axis=(1, 3)).astype(np.float32)

    # Gebäude mit Rand rastern (Dachhöhe absolut, 1 m)
    side = 1000 + 2 * PAD
    if paths.get("lod2"):
        roof = roof_raster(paths["lod2"], e * 1000 - PAD, (n + 1) * 1000 + PAD, 1.0, side, side)
    else:
        roof = np.full((side, side), np.nan, np.float32)
    mp = side // k
    rb = roof.reshape(mp, k, mp, k)
    covered = np.isfinite(rb)
    bcount = covered.sum(axis=(1, 3)).astype(np.uint8)
    with np.errstate(all="ignore"):
        roofmax = np.where(covered.any(axis=(1, 3)), np.nanmax(np.where(covered, rb, -np.inf), axis=(1, 3)), np.nan)
    roofmax = roofmax.astype(np.float32)

    veg = np.zeros((m, m), np.uint16)
    if dom is not None:
        inner = np.isfinite(roof[PAD:PAD + 1000, PAD:PAD + 1000])
        bmask = ndimage.binary_dilation(inner, iterations=1)
        ndom = dom - dgm
        vpx = (ndom > 2.0) & ~bmask & np.isfinite(ndom)
        vb = vpx.reshape(m, k, m, k)
        frac = vb.mean(axis=(1, 3))
        hb = np.where(vb, ndom.reshape(m, k, m, k), np.nan)
        hb = hb.transpose(0, 2, 1, 3).reshape(m, m, k * k)
        with np.errstate(all="ignore"), _warnings_off():
            h75 = np.nanpercentile(hb, 75, axis=2)
        h75 = np.where((frac >= 0.3) & np.isfinite(h75), h75, 0.0)
        veg = np.clip(np.round(h75 * 10), 0, 65000).astype(np.uint16)
    return {"e": e, "n": n, "ok": True, "terrain": terrain, "veg": veg, "roofmax": roofmax, "bcount": bcount}


class _warnings_off:
    def __enter__(self):
        import warnings

        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        self._c.__exit__(*a)


def nrw_tiles(bbox_utm, raw: Path) -> list[tuple[int, int]]:
    """1-km-Kacheln mit DGM1 NRW, die das UTM-Rechteck (minx, miny, maxx, maxy) schneiden."""
    minx, miny, maxx, maxy = bbox_utm
    idx = _index("dgm", raw)
    return [(e, n) for e in range(int(minx // 1000), int(maxx // 1000) + 1)
            for n in range(int(miny // 1000), int(maxy // 1000) + 1) if (e, n) in idx]


def process_tiles(rdir: Path, raw: Path, tiles, meta, keep_raw=False, workers=None, downloads=6,
                  progress=None, stop_event: threading.Event | None = None) -> bool:
    """Lädt die NRW-Kacheln und schreibt Gelände/Bewuchs sowie Dach-Zwischenraster in das
    Gebietsraster (fortsetzbar). Rückgabe: alle Kacheln fertig."""
    res = int(meta.res)
    if 1000 % res or PAD % res:
        raise ValueError("Auflösung muss 1000 und 50 teilen (5, 10, 25 m)")
    x0, y1, nx, ny = meta.x0, meta.y1, meta.nx, meta.ny
    idx = {p: _index(p, raw) for p in PRODUCTS}
    terrain = np.load(rdir / "terrain.npy", mmap_mode="r+")
    veg = np.load(rdir / "veg.npy", mmap_mode="r+")

    def mm(name, dtype, fill):
        p = rdir / f"{name}.npy"
        if p.exists():
            return np.load(p, mmap_mode="r+")
        a = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=(ny, nx))
        for i in range(0, ny, 2000):
            a[i:i + 2000] = fill
        return a

    roofmax = mm("_roofmax", np.float32, np.nan)
    bcount = mm("_bcount", np.uint8, 0)

    done_file = rdir / "_done_tiles.json"
    done = set(map(tuple, json.loads(done_file.read_text()))) if done_file.exists() else set()
    tiles = [tuple(t) for t in tiles]
    todo = [t for t in tiles if t not in done]
    total = len(tiles)
    if progress:
        progress(len(done), total, "NRW-Kacheln")

    workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=downloads, pool_maxsize=downloads * 3)
    session.mount("https://", adapter)

    def fetch(tile):
        e, n = tile
        paths = {}
        for p in PRODUCTS:
            name = idx[p].get((e, n))
            if name is None:
                continue
            paths[p] = _download(f"{PRODUCTS[p]}/{name}", raw / name, session)
        return tile, paths

    def merge(r):
        e, n = r["e"], r["n"]
        if r["ok"]:
            m = 1000 // res
            row = int(round((y1 - (n + 1) * 1000) / res))
            col = int(round((e * 1000 - x0) / res))
            terrain[row:row + m, col:col + m] = r["terrain"]
            veg[row:row + m, col:col + m] = r["veg"]
            p = PAD // res
            rs, cs = row - p, col - p
            r0, c0 = max(rs, 0), max(cs, 0)
            r1, c1 = min(rs + r["roofmax"].shape[0], ny), min(cs + r["roofmax"].shape[1], nx)
            sub_rm = r["roofmax"][r0 - rs:r1 - rs, c0 - cs:c1 - cs]
            sub_bc = r["bcount"][r0 - rs:r1 - rs, c0 - cs:c1 - cs]
            cur = roofmax[r0:r1, c0:c1]
            roofmax[r0:r1, c0:c1] = np.fmax(cur, sub_rm)
            bc = bcount[r0:r1, c0:c1].astype(np.uint16) + sub_bc
            bcount[r0:r1, c0:c1] = np.minimum(bc, 255).astype(np.uint8)

    from concurrent.futures import wait, FIRST_COMPLETED

    max_inflight = workers * 2
    with ThreadPoolExecutor(downloads) as dl, ProcessPoolExecutor(workers) as pp:
        pending_dl = set()
        pending_pp = {}
        it = iter(todo)
        exhausted = False
        while True:
            if stop_event is not None and stop_event.is_set():
                for f in pending_dl:
                    f.cancel()
                break
            while not exhausted and len(pending_dl) + len(pending_pp) < max_inflight + downloads:
                t = next(it, None)
                if t is None:
                    exhausted = True
                    break
                pending_dl.add(dl.submit(fetch, t))
            if not pending_dl and not pending_pp:
                break
            fin, _ = wait(list(pending_dl) + list(pending_pp), return_when=FIRST_COMPLETED, timeout=600)
            for f in fin:
                if f in pending_dl:
                    pending_dl.discard(f)
                    tile, paths = f.result()
                    pending_pp[pp.submit(process_tile, tile[0], tile[1], paths, res)] = (tile, paths)
                else:
                    tile, paths = pending_pp.pop(f)
                    merge(f.result())
                    done.add(tile)
                    if not keep_raw:
                        for pth in paths.values():
                            try:
                                Path(pth).unlink()
                            except OSError:
                                pass
                    if len(done) % 20 == 0 or len(done) == total:
                        for a in (terrain, veg, roofmax, bcount):
                            a.flush()
                        done_file.write_text(json.dumps(sorted(done)))
                    if progress:
                        progress(len(done), total, f"NRW-Kachel {tile[0]}/{tile[1]}")
    for a in (terrain, veg, roofmax, bcount):
        a.flush()
    done_file.write_text(json.dumps(sorted(done)))
    return len(done) >= total


def tile_cover(meta, tiles):
    """Funktion (r0, c0, h, w) -> bool-Maske der Zellen, die in einer NRW-Kachel liegen."""
    tset = {tuple(t) for t in tiles}
    if not tset:
        return None
    es = [t[0] for t in tset]
    ns = [t[1] for t in tset]
    e0, n0 = min(es), min(ns)
    km = np.zeros((max(ns) - n0 + 1, max(es) - e0 + 1), bool)
    for e, n in tset:
        km[n - n0, e - e0] = True

    def fn(r0, c0, h, w):
        xs = meta.x0 + (np.arange(c0, c0 + w) + 0.5) * meta.res
        ys = meta.y1 - (np.arange(r0, r0 + h) + 0.5) * meta.res
        ei = np.floor(xs / 1000).astype(int) - e0
        ni = np.floor(ys / 1000).astype(int) - n0
        okx = (ei >= 0) & (ei < km.shape[1])
        oky = (ni >= 0) & (ni < km.shape[0])
        sub = km[np.clip(ni, 0, km.shape[0] - 1)][:, np.clip(ei, 0, km.shape[1] - 1)]
        return sub & oky[:, None] & okx[None, :]

    return fn


def finalize_nrw(rdir: Path, meta, tiles) -> None:
    """Gebäudehöhe aus Dachmaximum ableiten – nur in Zellen der NRW-Kacheln."""
    res = meta.res
    ny, nx = meta.ny, meta.nx
    if not (rdir / "_roofmax.npy").exists():
        return
    terrain = np.load(rdir / "terrain.npy", mmap_mode="r+")
    roofmax = np.load(rdir / "_roofmax.npy", mmap_mode="r")
    bcount = np.load(rdir / "_bcount.npy", mmap_mode="r")
    veg = np.load(rdir / "veg.npy", mmap_mode="r+")
    bld = np.load(rdir / "bld.npy", mmap_mode="r+")
    cover = tile_cover(meta, tiles)
    half = res * res / 2.0
    for i in range(0, ny, 1000):
        h_ = min(1000, ny - i)
        cov = cover(i, 0, h_, nx)
        t = np.array(terrain[i:i + h_])
        h = np.array(roofmax[i:i + h_]) - t
        isb = (np.array(bcount[i:i + h_]) >= half) & np.isfinite(h) & (h > 0.5)
        b = np.where(isb, np.clip(np.round(h * 10), 5, 65000), 0).astype(np.uint16)
        bld[i:i + h_] = np.where(cov, b, bld[i:i + h_])
        v = np.array(veg[i:i + h_])
        v[isb & cov] = 0
        veg[i:i + h_] = v
    for a in (terrain, bld, veg):
        a.flush()
    del roofmax, bcount
    for f in ("_roofmax.npy", "_bcount.npy"):
        try:
            (rdir / f).unlink()
        except OSError:
            pass
