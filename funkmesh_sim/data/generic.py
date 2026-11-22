"""Deutschlandweite Geodaten (außerhalb NRW bzw. für große Gebiete).

Quellen (alle frei nutzbar, werden blockweise per HTTP-Range aus Cloud-Optimized-GeoTIFFs gelesen,
es wird nichts komplett heruntergeladen):

* **Copernicus DEM GLO-30** (30 m, © DLR e.V. 2010-2014 und © Airbus Defence and Space GmbH
  2014-2018, bereitgestellt unter COPERNICUS durch EU und ESA). Ein *Oberflächen*modell
  (X-Band-Radar): enthält Wald und Bebauung teilweise.
* **ESA WorldCover 2021 v200** (10 m Landbedeckung, © ESA WorldCover project / Contains modified
  Copernicus Sentinel data (2021), CC BY 4.0) – Klassen 10 = Baumbestand, 50 = bebaut.
* **OpenStreetMap-Gebäude** (© OpenStreetMap-Mitwirkende, ODbL) über die Overpass-API – nur für
  Gebiete bis ``OSM_MAX_KM2`` bei feinem Raster.

Ableitung je Block (mit 2 km Überlappung):

* Gelände: GLO-30 dort, wo frei (kein Wald, nicht bebaut); unter Wald/Bebauung aus der freien
  Umgebung mehrskalig interpoliert (normalisierte Faltung) und höchstens so hoch wie die Oberfläche;
  danach morphologische Öffnung (150 m) gegen Hecken/Feldgehölze/Radar-Rauschen und 20 m Glättung.

Abgleich mit DGM1/DOM1/LoD2 NRW (Dülmen, 10 m, gleicher Repeater, 4,5 km Radius): Pfadverlust
im Mittel -2.5 dB (30-m-Kirchturm) bis +3.8 dB (10-m-Dach), Streuung ~5 dB je Zelle.
* Bewuchs: Waldzellen (Baumanteil >= 30 %), Höhe 0.6 · (Oberfläche - Gelände) + 9.2 m (an NRW-
  Laserscanning kalibriert, X-Band dringt in die Krone ein), begrenzt auf 6..35 m.
* Gebäude: OSM-Grundrisse (Höhe aus ``height`` / ``building:levels`` × 3 m + 1,5 m Dach, sonst
  Vorgabe), Zelle = Gebäude bei >= 50 % Überdeckung. Ohne OSM (große Gebiete) wird Bebauung als
  „Clutter“ nach ITU-R P.1812 behandelt: Bauflächen mit 8 m Höhe im Bewuchskanal (Beugung über die
  Bebauung, keine Gebäudeeintrittsdämpfung).
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage

log = logging.getLogger(__name__)

GLO_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
WC_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{lat:02d}{ew}{lon:03d}_Map.tif"
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "funkmesh-repeater-simulator/0.2 (+https://funkmesh.de)"}

OSM_MAX_KM2 = 4000.0      # darüber keine Einzelgebäude (Overpass-Last, Rasterweite)
CLUTTER_H_M = 8.0         # Bebauung als Clutter (große Gebiete)
TREE_FRAC = 0.30
CANOPY_A, CANOPY_B = 0.60, 9.2    # siehe process_block
OPEN_M, SMOOTH_M = 150.0, 20.0     # Geländebereinigung, siehe process_block
BUILT_FRAC = 0.50
BLOCK = 1024              # Zielzellen je Block (Kante)
PAD_M = 2000.0            # Überlappung für die Geländeinterpolation

SOURCES = {
    "terrain": "Copernicus DEM GLO-30 (© DLR/Airbus, COPERNICUS/EU/ESA), unter Wald/Bebauung interpoliert",
    "vegetation": "ESA WorldCover 2021 (CC BY 4.0) + Höhe aus GLO-30",
    "buildings_osm": "OpenStreetMap-Gebäude (© OpenStreetMap-Mitwirkende, ODbL)",
    "buildings_clutter": "ESA WorldCover 2021 Bauflächen als Clutter (8 m, ITU-R P.1812)",
}

_env_lock = threading.Lock()


def _gdal_env():
    import rasterio

    return rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                        GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE",
                        VSI_CACHE_SIZE=64 << 20, GDAL_CACHEMAX=256, GDAL_HTTP_USERAGENT=UA["User-Agent"])


def _tile_url(kind: str, lat: int, lon: int) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    tpl = GLO_URL if kind == "glo" else WC_URL
    return tpl.format(ns=ns, ew=ew, lat=abs(lat), lon=abs(lon))


def _tiles_for(kind: str, w, s, e, n):
    step = 1 if kind == "glo" else 3
    out = []
    for la in range(int(math.floor(s / step) * step), int(math.floor(n / step) * step) + 1, step):
        for lo in range(int(math.floor(w / step) * step), int(math.floor(e / step) * step) + 1, step):
            out.append((la, lo, step))
    return out


_missing: set = set()


def _reproject_source(kind, dst, dst_transform, ll_bounds, src_res_m, resampling):
    """Liest alle Quellkacheln, die ``ll_bounds`` schneiden, mit passender Übersichtsstufe und
    projiziert sie in ``dst`` (EPSG:25832). Fehlende Kacheln (Meer, Ausland) bleiben nodata."""
    import rasterio
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    w, s, e, n = ll_bounds
    for la, lo, step in _tiles_for(kind, w, s, e, n):
        url = _tile_url(kind, la, lo)
        if url in _missing:
            continue
        tb = (lo, la, lo + step, la + step)
        iw, is_, ie, in_ = max(w, tb[0]), max(s, tb[1]), min(e, tb[2]), min(n, tb[3])
        if iw >= ie or is_ >= in_:
            continue
        try:
            with _gdal_env(), rasterio.open("/vsicurl/" + url) as ds:
                win = from_bounds(iw, is_, ie, in_, ds.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
                if win.width < 1 or win.height < 1:
                    continue
                # Quellpixelgröße in m (Breite ~ Kosinus der Breite)
                px_m = abs(ds.transform.e) * 111_320.0
                dec = max(1, int(src_res_m / px_m))
                oh, ow = max(1, int(win.height // dec)), max(1, int(win.width // dec))
                rs = Resampling.average if kind == "glo" else Resampling.nearest
                a = ds.read(1, window=win, out_shape=(oh, ow), resampling=rs)
                st = ds.window_transform(win) * rasterio.Affine.scale(win.width / ow, win.height / oh)
                if kind == "glo":
                    a = a.astype(np.float32)
                    a[(a < -100) | (a > 9000)] = np.nan
                    nd = np.nan
                else:
                    nd = 0
            tmp = np.full(dst.shape, nd, dst.dtype)
            reproject(a, tmp, src_transform=st, src_crs="EPSG:4326", dst_transform=dst_transform,
                      dst_crs="EPSG:25832", resampling=resampling, src_nodata=nd, dst_nodata=nd)
            if kind == "glo":
                m = np.isfinite(tmp)
            else:
                m = tmp != 0
            dst[m] = tmp[m]
        except rasterio.errors.RasterioIOError as exc:
            if "404" in str(exc) or "does not exist" in str(exc) or "not recognized" in str(exc):
                _missing.add(url)  # z. B. Kachel über offenem Meer
                continue
            raise


def _fill_terrain(dsm, ground, res):
    """Gelände unter Wald/Bebauung aus freien Zellen interpolieren (mehrskalige normalisierte Faltung)."""
    ok = ground & np.isfinite(dsm)
    terr = np.where(ok, dsm, np.nan).astype(np.float32)
    hole = ~ok
    if not hole.any():
        return terr
    v0 = np.where(ok, dsm, 0).astype(np.float32)
    w0 = ok.astype(np.float32)
    for sig_m in (40, 120, 360, 1000, 2500):
        sig = sig_m / res
        if sig < 0.6:
            continue
        w = ndimage.gaussian_filter(w0, sig, mode="nearest")
        v = ndimage.gaussian_filter(v0, sig, mode="nearest")
        est = np.where(w > 0.02, v / np.maximum(w, 1e-6), np.nan)
        sel = hole & np.isfinite(est)
        terr[sel] = est[sel]
        hole &= ~sel
        if not hole.any():
            break
    fin = np.isfinite(dsm)
    terr = np.where(fin, np.fmin(terr, dsm), terr)          # Gelände nie über der Oberfläche
    return terr


# ------------------------------------------------------------------ OSM --

def _parse_height(tags: dict, default_h: float) -> float:
    for k in ("height", "building:height"):
        v = tags.get(k)
        if v:
            m = re.match(r"\s*([0-9]+(?:[.,][0-9]+)?)", v)
            if m:
                h = float(m.group(1).replace(",", "."))
                if "ft" in v:
                    h *= 0.3048
                if 2 <= h <= 400:
                    return h
    lv = tags.get("building:levels")
    if lv:
        m = re.match(r"\s*([0-9]+(?:[.,][0-9]+)?)", lv)
        if m:
            n = float(m.group(1).replace(",", "."))
            if 0 < n < 120:
                rl = 0.0
                rm = re.match(r"\s*([0-9]+)", tags.get("roof:levels", "") or "")
                if rm:
                    rl = min(float(rm.group(1)), 5)
                return n * 3.0 + rl * 2.5 + 1.5
    b = tags.get("building", "")
    if b in ("garage", "garages", "carport", "shed", "hut", "roof", "kiosk"):
        return 3.0
    return default_h


def fetch_osm_buildings(bbox_ll, cache_dir: Path, default_h: float, progress=None, stop=None):
    """Gebäudegrundrisse als Liste (Höhe, Polygon in UTM32) – in ~10×10-km-Kacheln, gecacht."""
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 25832, always_xy=True)
    w, s, e, n = bbox_ll
    dlat, dlon = 0.09, 0.14
    tiles = []
    la = s
    while la < n:
        lo = w
        while lo < e:
            tiles.append((round(la, 4), round(lo, 4), round(min(la + dlat, n), 4), round(min(lo + dlon, e), 4)))
            lo += dlon
        la += dlat
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    sess = requests.Session()
    for i, (s_, w_, n_, e_) in enumerate(tiles):
        if stop is not None and stop.is_set():
            break
        f = cache_dir / f"osm_{s_}_{w_}_{n_}_{e_}.json"
        if not f.exists():
            q = (f"[out:json][timeout:180];(way[\"building\"]({s_},{w_},{n_},{e_});"
                 f"relation[\"building\"][\"type\"=\"multipolygon\"]({s_},{w_},{n_},{e_}););out tags geom;")
            for attempt in range(6):
                try:
                    r = sess.post(OVERPASS, data={"data": q}, headers=UA, timeout=240)
                    if r.status_code in (429, 504, 502, 503):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    f.write_bytes(r.content)
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning("Overpass %s (Versuch %d): %s", f.name, attempt + 1, exc)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("Overpass-API dauerhaft nicht erreichbar")
            time.sleep(1.0)
        data = json.loads(f.read_text())
        for el in data.get("elements", []):
            tags = el.get("tags") or {}
            h = _parse_height(tags, default_h)
            rings = []
            if el["type"] == "way" and el.get("geometry"):
                rings.append(el["geometry"])
            elif el["type"] == "relation":
                for mb in el.get("members", []):
                    if mb.get("role") == "outer" and mb.get("geometry"):
                        rings.append(mb["geometry"])
            for g in rings:
                if len(g) < 4:
                    continue
                lon = np.array([p["lon"] for p in g])
                lat = np.array([p["lat"] for p in g])
                x, y = t.transform(lon, lat)
                out.append((h, np.column_stack([x, y])))
        if progress:
            progress(i + 1, len(tiles), f"OSM-Gebäude {i + 1}/{len(tiles)}")
    # Duplikate (Gebäude über Kachelgrenzen) sind unschädlich: gleiche Höhe, gleiche Fläche
    return out


class BuildingIndex:
    """Schneller Bounding-Box-Filter für die Rasterung je Block."""

    def __init__(self, blds):
        self.h = np.array([b[0] for b in blds], np.float32)
        self.polys = [b[1] for b in blds]
        if blds:
            self.bb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in self.polys])
        else:
            self.bb = np.zeros((0, 4))

    def query(self, x0, y0, x1, y1):
        if not len(self.bb):
            return []
        sel = (self.bb[:, 2] >= x0) & (self.bb[:, 0] <= x1) & (self.bb[:, 3] >= y0) & (self.bb[:, 1] <= y1)
        idx = np.nonzero(sel)[0]
        idx = idx[np.argsort(self.h[idx], kind="stable")]  # höchste zuletzt -> gewinnt
        return [(self.polys[i], float(self.h[i])) for i in idx]


def _raster_buildings(index: BuildingIndex, x0, y1, res, ny, nx):
    """Gebäudehöhe [m] je Zelle (>= 50 % überdeckt), Subraster 2 m (bzw. res/2)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    sub = max(1, int(round(res / 2.0)))
    sres = res / sub
    shapes = index.query(x0, y1 - ny * res, x0 + nx * res, y1)
    if not shapes:
        return np.zeros((ny, nx), np.float32)
    geoms = [({"type": "Polygon", "coordinates": [p.tolist()]}, h) for p, h in shapes]
    geoms = [(g, int(round(h * 10))) for g, h in geoms]
    hr = rasterize(geoms, out_shape=(ny * sub, nx * sub), transform=from_origin(x0, y1, sres, sres),
                   fill=0, dtype="uint16", all_touched=False)
    hb = hr.reshape(ny, sub, nx, sub)
    cnt = np.count_nonzero(hb, axis=(1, 3))
    hmax = hb.max(axis=(1, 3))
    del hr, hb
    return np.where(cnt * 2 >= sub * sub, hmax / 10.0, 0.0).astype(np.float32)


# ------------------------------------------------------------- Block ----

def process_block(r0, c0, h, w, meta, use_osm, bindex, skip_mask=None):
    """Berechnet Gelände/Gebäude/Bewuchs für Zielzellen [r0:r0+h, c0:c0+w]."""
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from rasterio.enums import Resampling

    res = meta.res
    pad = int(math.ceil(PAD_M / res))
    R0, C0 = r0 - pad, c0 - pad
    H, W = h + 2 * pad, w + 2 * pad
    x0 = meta.x0 + C0 * res
    y1 = meta.y1 - R0 * res
    tr = from_origin(x0, y1, res, res)
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    xs = np.array([x0, x0 + W * res, x0, x0 + W * res])
    ys = np.array([y1, y1, y1 - H * res, y1 - H * res])
    lo, la = to_ll.transform(xs, ys)
    llb = (lo.min() - 0.01, la.min() - 0.01, lo.max() + 0.01, la.max() + 0.01)

    dsm = np.full((H, W), np.nan, np.float32)
    rs = Resampling.bilinear if res < 30 else Resampling.average
    _reproject_source("glo", dsm, tr, llb, min(res, 30.0), rs)

    # Landbedeckung im Subraster ~10 m -> Anteile je Zielzelle
    s = max(1, int(round(res / 10.0)))
    wc = np.zeros((H * s, W * s), np.uint8)
    _reproject_source("wc", wc, from_origin(x0, y1, res / s, res / s), llb, res / s, Resampling.nearest)
    wcb = wc.reshape(H, s, W, s)
    tree = (wcb == 10).mean(axis=(1, 3))
    built = (wcb == 50).mean(axis=(1, 3))
    known = (wcb != 0).mean(axis=(1, 3)) > 0.5
    del wc, wcb

    bld = np.zeros((H, W), np.float32)
    if use_osm and bindex is not None:
        bld = _raster_buildings(bindex, x0, y1, res, H, W)
    ground = known & (tree < 0.25) & (built < 0.25) & (bld == 0)
    ground |= ~known & np.isfinite(dsm)          # ohne Landbedeckung (Ausland/Meer): DSM nehmen
    terr = _fill_terrain(dsm, ground, res)
    fin = np.isfinite(terr)
    if not fin.all() and np.isfinite(dsm).any():
        terr = np.where(fin, terr, dsm - np.where(tree >= TREE_FRAC, 15.0, np.where(built >= BUILT_FRAC, 6.0, 0.0)))
    # Hecken, Baumreihen, Feldgehölze (nicht als Wald klassifiziert) und Radar-Rauschen stecken als
    # schmale Erhebungen im Oberflächenmodell und wirken für niedrige Antennen wie Beugungskanten.
    # Morphologische Öffnung (150 m) + Glättung (20 m) entfernt sie; geprüft gegen DGM1 NRW:
    # Geländefehler 0.6 ± 2.4 m -> 0.0 ± 1.9 m, Pfadverlust-Abweichung niedriger Repeater ~6 dB kleiner.
    if np.isfinite(terr).all():
        k = max(1, int(round(OPEN_M / res)))
        if k >= 3:
            terr = ndimage.grey_opening(terr, size=(k | 1, k | 1), mode="nearest")
        sg = SMOOTH_M / res
        if sg >= 0.5:
            terr = ndimage.gaussian_filter(terr, sg, mode="nearest")
    # GLO-30 (X-Band) dringt in die Krone ein: Oberfläche - Gelände unterschätzt die Baumhöhe.
    # Korrektur kalibriert an DOM1/DGM1 NRW (Münsterland, 608 000 Waldzellen à 10 m, Mittel über
    # die 5-m-Zellen): Kronenhöhe ≈ 0.6 · (DSM - Gelände) + 9.2 m (ohne Korrektur ~ 40 % zu niedrig).
    d = np.clip(np.nan_to_num(dsm - terr, nan=8.0), 0.0, 40.0)
    vh = np.where(tree >= TREE_FRAC, np.clip(CANOPY_A * d + CANOPY_B, 6.0, 35.0), 0.0)
    if not use_osm:
        vh = np.maximum(vh, np.where(built >= BUILT_FRAC, CLUTTER_H_M, 0.0))
    sl = (slice(pad, pad + h), slice(pad, pad + w))
    T = terr[sl].astype(np.float32)
    B = np.clip(np.round(bld[sl] * 10), 0, 65000).astype(np.uint16)
    V = np.where(B > 0, 0, np.clip(np.round(vh[sl] * 10), 0, 65000)).astype(np.uint16)
    return T, B, V


def run_generic(arrays, meta, cover_mask_fn, use_osm, bindex, progress=None, stop=None, done_file=None):
    """Füllt die Rasterarrays blockweise (fortsetzbar). ``cover_mask_fn(r0,c0,h,w)`` liefert die
    Zellen, die bereits von einer besseren Quelle (NRW) abgedeckt sind (dort nichts schreiben)."""
    terrain, bld, veg = arrays
    blocks = [(r, c) for r in range(0, meta.ny, BLOCK) for c in range(0, meta.nx, BLOCK)]
    done = set(map(tuple, json.loads(done_file.read_text()))) if done_file and done_file.exists() else set()
    for i, (r0, c0) in enumerate(blocks):
        if stop is not None and stop.is_set():
            break
        if (r0, c0) in done:
            continue
        h, w = min(BLOCK, meta.ny - r0), min(BLOCK, meta.nx - c0)
        cov = cover_mask_fn(r0, c0, h, w) if cover_mask_fn else None
        if cov is not None and cov.all():
            done.add((r0, c0))
            continue
        T, B, V = process_block(r0, c0, h, w, meta, use_osm, bindex)
        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        if cov is not None and cov.any():
            terrain[sl] = np.where(cov, terrain[sl], T)
            bld[sl] = np.where(cov, bld[sl], B)
            veg[sl] = np.where(cov, veg[sl], V)
        else:
            terrain[sl], bld[sl], veg[sl] = T, B, V
        done.add((r0, c0))
        if done_file and (len(done) % 8 == 0):
            for a in arrays:
                a.flush()
            done_file.write_text(json.dumps(sorted(done)))
        if progress:
            progress(len(done), len(blocks), f"Block {len(done)}/{len(blocks)} (Copernicus/WorldCover)")
    for a in arrays:
        a.flush()
    if done_file:
        done_file.write_text(json.dumps(sorted(done)))
    return len(done) == len(blocks)
