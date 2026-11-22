"""Gebiete: frei gewählte Ausschnitte Deutschlands (bis hin zu ganz Deutschland) mit eigenem Raster.

Ein Gebiet besteht aus ``data/regions/<name>/`` (Raster + ``meta.json`` + ``boundary.geojson``),
einem Projekt ``data/projects/<name>.json`` (Repeater, Parameter), Ergebnissen
``data/results/<name>/`` und dem Repeater-Cache ``data/cache/<name>/``.

Datenquellen je nach Lage und Rasterweite:

* Raster <= 10 m und Kacheln in NRW: DGM1/DOM1/LoD2 NRW (1 m, beste Qualität)
* sonst: Copernicus GLO-30 + ESA WorldCover (+ OSM-Gebäude bei <= 10 m und <= 4000 km²),
  siehe :mod:`funkmesh_sim.data.generic`
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .grid import GridMeta, write_meta

log = logging.getLogger(__name__)

# Deutschland (lon/lat) inkl. kleinem Rand
GERMANY_BBOX = (5.85, 47.26, 15.06, 55.07)
RESOLUTIONS = (5, 10, 25, 50, 100)
NRW_MAX_TILES = 12000      # ~ 12 000 km² NRW in 1 m – darüber gröberes Raster verwenden
SEC_PER_NRW_TILE = 1.2     # gemessen (Coesfeld, 8 Prozesse)
SEC_PER_BLOCK = 6.0        # Copernicus/WorldCover je 1024² Block (Netz abhängig)


def slug(title: str) -> str:
    t = title.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return (t or "gebiet")[:40]


def nrw_raw() -> Path:
    p = config.data_dir() / "raw" / "_nrw"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class AreaSpec:
    title: str
    bbox: tuple            # (west, south, east, north) in Grad
    res: int = 10
    buffer_m: float = 5000.0
    use_nrw: bool = True
    use_osm: bool = True
    default_building_m: float = 9.0
    boundary: dict | None = None   # GeoJSON (FeatureCollection) oder None = Rechteck
    name: str = ""

    def __post_init__(self):
        self.bbox = tuple(float(v) for v in self.bbox)
        if not self.name:
            self.name = slug(self.title)
        if int(self.res) not in RESOLUTIONS:
            raise ValueError(f"Rasterweite muss eine von {RESOLUTIONS} m sein")
        self.res = int(self.res)


def utm_bounds(bbox, buffer_m=0.0):
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 25832, always_xy=True)
    w, s, e, n = bbox
    lo = np.concatenate([np.linspace(w, e, 50), np.full(50, e), np.linspace(e, w, 50), np.full(50, w)])
    la = np.concatenate([np.full(50, s), np.linspace(s, n, 50), np.full(50, n), np.linspace(n, s, 50)])
    x, y = t.transform(lo, la)
    return x.min() - buffer_m, y.min() - buffer_m, x.max() + buffer_m, y.max() + buffer_m


def plan(spec: AreaSpec) -> dict:
    """Rastergeometrie, Datenquellen und Aufwand – ohne etwas herunterzuladen (bis auf den NRW-Index)."""
    from .data import nrw, generic

    minx, miny, maxx, maxy = utm_bounds(spec.bbox, spec.buffer_m)
    res = spec.res
    tiles = []
    if spec.use_nrw and res <= 10:
        # NRW nur, wenn das Gebiet NRW überhaupt berührt (grob: 5.8–9.5 °E, 50.3–52.6 °N)
        w, s, e, n = spec.bbox
        if e > 5.8 and w < 9.5 and n > 50.3 and s < 52.6:
            try:
                tiles = nrw.nrw_tiles((minx, miny, maxx, maxy), nrw_raw())
            except Exception as exc:  # noqa: BLE001
                log.warning("NRW-Index nicht verfügbar: %s", exc)
                tiles = []
    use_nrw = 0 < len(tiles) <= NRW_MAX_TILES
    if not use_nrw:
        tiles = []
    if use_nrw:
        pad = nrw.PAD
        x0 = math.floor(minx / 1000) * 1000.0 - pad
        y1 = math.ceil(maxy / 1000) * 1000.0 + pad
        x1 = math.ceil(maxx / 1000) * 1000.0 + pad
        y0 = math.floor(miny / 1000) * 1000.0 - pad
    else:
        a = max(50, res)
        x0 = float(math.floor(minx / a) * a)
        y1 = float(math.ceil(maxy / a) * a)
        x1 = float(math.ceil(maxx / a) * a)
        y0 = float(math.floor(miny / a) * a)
    nx = int(round((x1 - x0) / res))
    ny = int(round((y1 - y0) / res))
    km2 = (x1 - x0) * (y1 - y0) / 1e6
    use_osm = spec.use_osm and res <= 10 and km2 <= generic.OSM_MAX_KM2
    cells = nx * ny
    nblocks = math.ceil(nx / generic.BLOCK) * math.ceil(ny / generic.BLOCK)
    nrw_km2 = len(tiles)
    gen_frac = max(0.0, 1.0 - nrw_km2 / max(km2, 1e-9))
    t_prep = len(tiles) * SEC_PER_NRW_TILE + nblocks * gen_frac * SEC_PER_BLOCK * (2.0 if use_osm else 1.0)
    if use_osm:
        t_prep += km2 * gen_frac / 100 * 4.0  # Overpass ~4 s je 100 km²
    dl_mb = len(tiles) * 11.0 + km2 * gen_frac * 0.006
    return {
        "name": spec.name, "title": spec.title, "res": res, "x0": x0, "y1": y1, "nx": nx, "ny": ny,
        "km2": round(km2), "cells": cells, "disk_gb": round(cells * 8 / 1e9, 2),
        "nrw_tiles": len(tiles), "use_nrw": use_nrw, "use_osm": use_osm,
        "prep_s": round(t_prep), "download_mb": round(dl_mb),
        "tiles": tiles,
    }


def estimate_compute(bbox, res_out: float, radius_km: float, backend: str) -> dict:
    """Grobe Rechenzeit für alle bekannten Repeater im Gebiet (map.meshcore.io-Cache)."""
    n = None
    try:
        from .meshcore_import import repeaters_in_bounds

        n = len(repeaters_in_bounds(bbox, 30))
    except Exception:  # noqa: BLE001
        pass
    cells = (2 * radius_km * 1000 / res_out) ** 2
    per = cells * (1.1e-6 if backend == "cuda" else 1.5e-5) + 1.5
    return {"repeaters_known": n, "sec_per_repeater": round(per, 1),
            "total_s": None if n is None else round(per * n)}


# --------------------------------------------------------------- Liste ----

def area_dir(name: str) -> Path:
    return config.data_dir() / "regions" / name


def list_areas() -> list[dict]:
    root = config.data_dir() / "regions"
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        mf = d / "meta.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
        except ValueError:
            continue
        info = m.get("info") or {}
        size = sum(f.stat().st_size for f in d.glob("*.npy"))
        out.append({"name": d.name, "title": info.get("title") or d.name.capitalize(), "res": m["res"],
                    "complete": bool(m.get("complete")), "bbox": info.get("bbox") or _bbox_from_meta(m),
                    "km2": round(m["nx"] * m["ny"] * m["res"] ** 2 / 1e6), "disk_gb": round(size / 1e9, 2),
                    "sources": m.get("sources", {}), "created": info.get("created")})
    return out


def _bbox_from_meta(m):
    from pyproj import Transformer

    t = Transformer.from_crs(25832, 4326, always_xy=True)
    x0, y1 = m["x0"], m["y1"]
    x1, y0 = x0 + m["nx"] * m["res"], y1 - m["ny"] * m["res"]
    lo, la = t.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return [min(lo), min(la), max(lo), max(la)]


def boundary(name: str) -> dict:
    """Grenze des Gebiets (GeoJSON). Fällt auf das Rechteck des Rasters zurück."""
    f = area_dir(name) / "boundary.geojson"
    if f.exists():
        return json.loads(f.read_text())
    f2 = config.REGIONS_DIR / f"{name}.geojson"
    if f2.exists():
        return json.loads(f2.read_text())
    m = json.loads((area_dir(name) / "meta.json").read_text())
    w, s, e, n = (m.get("info") or {}).get("bbox") or _bbox_from_meta(m)
    return rect_geojson((w, s, e, n))


def rect_geojson(b):
    w, s, e, n = b
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}


def delete_area(name: str) -> None:
    for p in (area_dir(name), config.data_dir() / "raw" / name, config.data_dir() / "cache" / name,
              config.data_dir() / "results" / name):
        if p.exists():
            shutil.rmtree(p)
    pf = config.projects_dir() / f"{name}.json"
    if pf.exists():
        pf.unlink()


# ------------------------------------------------------------ Aufbau ----

def prepare(spec: AreaSpec, progress=None, stop: threading.Event | None = None, workers=None) -> bool:
    """Baut das Gebietsraster auf (fortsetzbar). Rückgabe: vollständig."""
    from .data import nrw, generic

    pl = plan(spec)
    rdir = area_dir(spec.name)
    rdir.mkdir(parents=True, exist_ok=True)
    raw = config.data_dir() / "raw" / spec.name
    raw.mkdir(parents=True, exist_ok=True)
    ny, nx, res = pl["ny"], pl["nx"], spec.res
    from . import memguard

    memguard.require_disk(rdir, ny * nx * 8 * 1.6, "Gebietsraster")
    srcs = {}
    if pl["use_nrw"]:
        srcs.update({"terrain_nrw": "DGM1 NRW (Land NRW, dl-de/zero-2.0)",
                     "buildings_nrw": "3D-Gebäudemodell LoD2 NRW (Land NRW, dl-de/zero-2.0)",
                     "vegetation_nrw": "DOM1 - DGM1 NRW (Land NRW, dl-de/zero-2.0)"})
    gen_needed = (not pl["use_nrw"]) or pl["nrw_tiles"] < pl["km2"]
    if gen_needed:
        srcs.update({"terrain": generic.SOURCES["terrain"], "vegetation": generic.SOURCES["vegetation"],
                     "buildings": generic.SOURCES["buildings_osm" if pl["use_osm"] else "buildings_clutter"]})
    if pl["use_nrw"] and not gen_needed:
        srcs.update({"terrain": srcs["terrain_nrw"], "buildings": srcs["buildings_nrw"], "vegetation": srcs["vegetation_nrw"]})
    srcs["buffer_m"] = spec.buffer_m
    meta = GridMeta(name=spec.name, res=float(res), x0=pl["x0"], y1=pl["y1"], nx=nx, ny=ny, sources=srcs,
                    complete=False, info={"title": spec.title, "bbox": list(spec.bbox), "use_nrw": pl["use_nrw"],
                                          "use_osm": pl["use_osm"], "default_building_m": spec.default_building_m,
                                          "created": time.strftime("%Y-%m-%d %H:%M")})
    mp = rdir / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if (old["res"], old["x0"], old["y1"], old["nx"], old["ny"]) != (res, pl["x0"], pl["y1"], nx, ny):
            raise RuntimeError(f"Gebiet '{spec.name}' existiert bereits mit anderer Geometrie – erst löschen.")
        if old.get("complete"):
            return True
    write_meta(rdir, meta)
    (rdir / "boundary.geojson").write_text(json.dumps(spec.boundary or rect_geojson(spec.bbox)))

    def mm(name, dtype, fill):
        p = rdir / f"{name}.npy"
        if p.exists():
            return np.load(p, mmap_mode="r+")
        a = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=(ny, nx))
        for i in range(0, ny, 2000):
            a[i:i + 2000] = fill
        a.flush()
        return a

    terrain = mm("terrain", np.float32, np.nan)
    bld = mm("bld", np.uint16, 0)
    veg = mm("veg", np.uint16, 0)
    del terrain, bld, veg

    phases = (["nrw"] if pl["use_nrw"] else []) + (["osm"] if gen_needed and pl["use_osm"] else []) + \
        (["generic"] if gen_needed else []) + ["final"]

    def phase_prog(ph):
        i = phases.index(ph)

        def f(done, total, msg):
            if progress:
                progress((i + done / max(total, 1)) / len(phases), msg)
        return f

    if pl["use_nrw"]:
        ok = nrw.process_tiles(rdir, nrw_raw_dir(raw), pl["tiles"], meta, workers=workers,
                               progress=phase_prog("nrw"), stop_event=stop)
        if not ok:
            return False
        nrw.finalize_nrw(rdir, meta, pl["tiles"])
    if gen_needed:
        bindex = None
        if pl["use_osm"]:
            blds = generic.fetch_osm_buildings(_ll_of(meta), raw / "osm", spec.default_building_m, phase_prog("osm"), stop)
            if stop is not None and stop.is_set():
                return False
            bindex = generic.BuildingIndex(blds)
            del blds
        arrays = tuple(np.load(rdir / f"{n}.npy", mmap_mode="r+") for n in ("terrain", "bld", "veg"))
        cover = nrw.tile_cover(meta, pl["tiles"]) if pl["use_nrw"] else None
        cover = _with_outside(cover, meta, spec)
        ok = generic.run_generic(arrays, meta, cover, pl["use_osm"], bindex, phase_prog("generic"), stop,
                                 rdir / "_done_blocks.json")
        del arrays
        if not ok:
            return False
    phase_prog("final")(0, 1, "Abschluss …")
    _finalize(rdir, meta)
    meta.complete = True
    write_meta(rdir, meta)
    for f in ("_done_tiles.json", "_done_blocks.json"):
        (rdir / f).unlink(missing_ok=True)
    if progress:
        progress(1.0, "fertig")
    return True


def _with_outside(cover, meta, spec):
    """Blöcke weit außerhalb der Gebietsgrenze (> Puffer, mind. 20 km) überspringen – spart bei
    „ganz Deutschland“ fast die Hälfte (Nachbarländer, Meer)."""
    if not spec.boundary:
        return cover
    from pyproj import Transformer
    from shapely import transform as shp_transform
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    t = Transformer.from_crs(4326, 25832, always_xy=True)
    g = unary_union([shape(f["geometry"]) for f in spec.boundary["features"]])
    g = shp_transform(g, lambda xy: np.column_stack(t.transform(xy[:, 0], xy[:, 1])))
    g = g.simplify(200).buffer(max(spec.buffer_m, 20000.0))

    def fn(r0, c0, h, w):
        x0 = meta.x0 + c0 * meta.res
        y1 = meta.y1 - r0 * meta.res
        if not g.intersects(box(x0, y1 - h * meta.res, x0 + w * meta.res, y1)):
            return np.ones((h, w), bool)
        return cover(r0, c0, h, w) if cover else None

    return fn


def nrw_raw_dir(raw: Path) -> Path:
    """Kachel-Downloads im Gebiets-Rohordner, Index-Dateien gemeinsam für alle Gebiete."""
    for f in nrw_raw().glob("index_*.json"):
        dst = raw / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    return raw


def _ll_of(meta):
    return _bbox_from_meta({"x0": meta.x0, "y1": meta.y1, "nx": meta.nx, "ny": meta.ny, "res": meta.res})


def _finalize(rdir: Path, meta) -> None:
    """Restliche Lücken im Gelände (Meer, Ausland ohne Daten) mit dem Median füllen."""
    terrain = np.load(rdir / "terrain.npy", mmap_mode="r+")
    sample = np.asarray(terrain[::max(1, meta.ny // 300), ::max(1, meta.nx // 300)])
    fill = float(np.nanmedian(sample)) if np.isfinite(sample).any() else 0.0
    for i in range(0, meta.ny, 1000):
        t = np.array(terrain[i:i + 1000])
        nan = ~np.isfinite(t)
        if nan.any():
            t[nan] = fill
            terrain[i:i + 1000] = t
    terrain.flush()


# ----------------------------------------------------------- Auftrag ----

class AreaJob:
    """Hintergrundauftrag „Gebiet vorbereiten“ (eins zur Zeit)."""

    def __init__(self):
        self.state = "idle"
        self.progress = 0.0
        self.message = ""
        self.name = None
        self.title = None
        self.stop = threading.Event()
        self.thread = None
        self.started = 0.0
        self.finished = 0.0
        self.error = None

    def to_dict(self):
        return {"state": self.state, "progress": round(self.progress, 4), "message": self.message,
                "name": self.name, "title": self.title, "error": self.error,
                "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else 0}

    def start(self, spec: AreaSpec, on_done=None):
        if self.state == "running":
            raise RuntimeError("Es wird bereits ein Gebiet vorbereitet")
        self.__init__()
        self.state, self.name, self.title, self.started = "running", spec.name, spec.title, time.time()

        def prog(f, msg):
            self.progress, self.message = f, msg

        def run():
            try:
                ok = prepare(spec, prog, self.stop)
                self.state = "done" if ok else "canceled"
                self.message = "fertig" if ok else "angehalten – „Vorbereiten“ setzt fort"
                if ok and on_done:
                    on_done(spec)
            except Exception as exc:  # noqa: BLE001
                log.exception("Gebiet vorbereiten")
                self.state, self.error, self.message = "error", str(exc), f"Fehler: {exc}"
            finally:
                self.finished = time.time()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


def spec_from_meta(name: str) -> AreaSpec:
    """Spezifikation eines (unvollständigen) Gebiets zum Fortsetzen."""
    m = json.loads((area_dir(name) / "meta.json").read_text())
    info = m.get("info") or {}
    bnd = area_dir(name) / "boundary.geojson"
    return AreaSpec(title=info.get("title", name), bbox=tuple(info["bbox"]), res=int(m["res"]),
                    buffer_m=float(m.get("sources", {}).get("buffer_m", 5000)), use_nrw=info.get("use_nrw", True),
                    use_osm=info.get("use_osm", True), default_building_m=info.get("default_building_m", 9.0),
                    boundary=json.loads(bnd.read_text()) if bnd.exists() else None, name=name)


# -------------------------------------------------------- Ortssuche ----

def search_places(q: str, limit: int = 6, country: bool = False) -> list[dict]:
    """Kreise/Städte/Länder in Deutschland über Nominatim (© OpenStreetMap, ODbL)."""
    import requests

    r = requests.get("https://nominatim.openstreetmap.org/search", timeout=30,
                     headers={"User-Agent": "funkmesh-repeater-simulator/0.2 (+https://funkmesh.de)"},
                     params={"q": q, "format": "jsonv2", "countrycodes": "de", "polygon_geojson": 1,
                             "polygon_threshold": 0.0015, "limit": limit, "featureType": "country" if country else "settlement"})
    r.raise_for_status()
    out = []
    for x in r.json():
        if x.get("geojson", {}).get("type") not in ("Polygon", "MultiPolygon"):
            continue
        s, n, w, e = (float(v) for v in x["boundingbox"])
        out.append({"name": x.get("name"), "display": x.get("display_name"), "type": x.get("addresstype") or x.get("type"),
                    "bbox": [w, s, e, n],
                    "boundary": {"type": "FeatureCollection", "features": [
                        {"type": "Feature", "properties": {"name": x.get("name")}, "geometry": x["geojson"]}]}})
    return out
