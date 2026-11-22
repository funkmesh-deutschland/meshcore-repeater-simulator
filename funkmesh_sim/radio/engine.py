"""Berechnung der Pfadverlust-Karte je Repeater auf GPU (CUDA) oder CPU.

Ablauf je Repeater:
1. Fenster (Radius + Rand) aus dem Regionsraster ausschneiden (RAM/VRAM-geprüft)
2. Kernel zeilenblockweise starten (Fortschritt + Abbruch möglich)
3. optional Straßenschlucht-Nachlauf (:mod:`funkmesh_sim.radio.canyon`)
4. Ergebnis als komprimierte uint16-Datei cachen (0.02 dB Auflösung)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import config, memguard
from ..grid import RegionGrid
from . import kernel as K
from .lora import antenna_elements, bel_p2109

log = logging.getLogger(__name__)
C0 = 299792458.0


@dataclass
class Environment:
    """Physikalische Modellparameter (alle mit Quelle, siehe README)."""
    freq_mhz: float = 869.618
    k_factor: float = 4.0 / 3.0          # effektiver Erdradiusfaktor (Standardatmosphäre)
    rx_height_m: float = 1.5             # Empfänger über Grund (getragenes Gerät)
    veg_gamma_db_m: float = 0.25         # ITU-R P.833 Abb. 2, ~900 MHz, belaubt
    wall_db: float = 10.0                # je Außenwand, ITU-R P.2040 Mauerwerk ~30 cm
    interior_db_m: float = 0.6           # Innenräume (Zwischenwände, Möbel)
    building_type: str = "traditional"   # ITU-R P.2109: traditional | thermally_efficient
    building_max_db: float = 60.0        # Obergrenze Durchdringung
    diffraction: str = "delta_bullington"  # oder "deygout"
    ground_eps_r: float = 22.0           # ITU-R P.527 mittlerer Boden
    ground_sigma: float = 0.003
    step_factor: float = 1.0             # Profilschritt = step_factor * Rasterweite
    canyon: bool = True                  # Straßenschlucht-/Eckenausbreitung (ITU-R P.1411)
    canyon_max_m: float = 400.0

    @property
    def lam(self) -> float:
        return C0 / (self.freq_mhz * 1e6)

    @property
    def veg_am(self) -> float:
        # ITU-R P.833 Gl. (2): Am = A1 * f^alpha, A1 = 0.18, alpha = 0.752 (f in MHz)
        return 0.18 * self.freq_mhz ** 0.752

    def bel_db(self) -> float:
        return bel_p2109(self.freq_mhz / 1e3, 0.0, self.building_type)


@dataclass
class Repeater:
    id: str
    name: str
    lat: float
    lon: float
    height_m: float = 10.0       # Antenne über Grund
    tx_dbm: float = 22.0
    gain_dbi: float = 3.0
    cable_db: float = 1.0
    tilt_deg: float = 0.0
    enabled: bool = True
    source: str = "manuell"
    public_key: str | None = None

    def geometry_key(self, env: Environment, out_res: float, radius_m: float) -> str:
        g = {k: getattr(self, k) for k in ("lat", "lon", "height_m", "gain_dbi", "tilt_deg")}
        e = asdict(env)
        s = json.dumps([g, e, out_res, radius_m, K.__doc__[:40], 3], sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:16]


@dataclass
class LossResult:
    """Pfadverlust [dB] im Ausgaberaster; ``r0, c0`` = Lage im globalen Ausgaberaster."""
    loss: np.ndarray
    r0: int
    c0: int
    out_res: float
    tx_rc: tuple
    meta: dict = field(default_factory=dict)


_backend_lock = threading.Lock()
_backend: str | None = None


def backend() -> str:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = "cpu"
            if not config.DISABLE_GPU:
                try:
                    from numba import cuda

                    if cuda.is_available():
                        cuda.get_current_device()
                        _backend = "cuda"
                except Exception as exc:  # pragma: no cover
                    log.warning("CUDA nicht verfügbar (%s) – CPU wird verwendet", exc)
        return _backend


def to_utm(lat: float, lon: float):
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 25832, always_xy=True)
    return t.transform(lon, lat)


class Canceled(Exception):
    pass


def _params(env: Environment, res: float, out_res: float, radius: float, rep: Repeater,
            tx_x: float, tx_y: float, gtx: float, tx_in: bool) -> np.ndarray:
    P = np.zeros(K.P_N, np.float64)
    P[K.P_RES] = res
    P[K.P_STEP] = res * env.step_factor
    P[K.P_TXX] = tx_x
    P[K.P_TXY] = tx_y
    P[K.P_ZTX] = gtx + rep.height_m
    P[K.P_GTX] = gtx
    P[K.P_TXIN] = 1.0 if tx_in else 0.0
    P[K.P_HRX] = env.rx_height_m
    P[K.P_LAM] = env.lam
    P[K.P_AE] = 6371e3 * env.k_factor
    P[K.P_FGHZ] = env.freq_mhz / 1e3
    P[K.P_VGAM] = env.veg_gamma_db_m
    P[K.P_VAM] = env.veg_am
    P[K.P_WALL] = env.wall_db
    P[K.P_INT] = env.interior_db_m
    P[K.P_BEL] = env.bel_db()
    P[K.P_ORES] = out_res
    P[K.P_OX0] = 0.5 * out_res
    P[K.P_OY0] = 0.5 * out_res
    P[K.P_MAXD] = radius
    P[K.P_GANT] = rep.gain_dbi
    P[K.P_NEL] = antenna_elements(rep.gain_dbi)
    P[K.P_TILT] = rep.tilt_deg
    P[K.P_DMODE] = 0.0 if env.diffraction == "delta_bullington" else 1.0
    P[K.P_EPSR] = env.ground_eps_r
    P[K.P_SIGMA] = env.ground_sigma
    P[K.P_BLDMAX] = env.building_max_db
    return P


def _window(grid: RegionGrid, x: float, y: float, radius: float, k: int):
    """Datenfenster (Zeilen/Spalten im Datenraster), an das Ausgaberaster (Faktor k) ausgerichtet."""
    m = grid.meta
    res = m.res
    margin = 2 * res * k
    c_a = int(math.floor((x - radius - margin - m.x0) / res / k)) * k
    c_b = int(math.ceil((x + radius + margin - m.x0) / res / k)) * k
    r_a = int(math.floor((m.y1 - (y + radius + margin)) / res / k)) * k
    r_b = int(math.ceil((m.y1 - (y - radius - margin)) / res / k)) * k
    nxo = (m.nx // k) * k
    nyo = (m.ny // k) * k
    return max(r_a, 0), min(r_b, nyo), max(c_a, 0), min(c_b, nxo)


def compute_repeater(grid: RegionGrid, rep: Repeater, env: Environment, out_res: float,
                     radius_m: float, progress=None, cancel: threading.Event | None = None) -> LossResult:
    m = grid.meta
    res = m.res
    k = int(round(out_res / res))
    if k < 1 or abs(k * res - out_res) > 1e-6:
        raise ValueError(f"Ausgabeauflösung muss Vielfaches von {res} m sein")
    x, y = to_utm(rep.lat, rep.lon)
    if not (m.x0 < x < m.x1 and m.y0 < y < m.y1):
        raise ValueError(f"Repeater {rep.name} liegt außerhalb der Geodaten")
    r_a, r_b, c_a, c_b = _window(grid, x, y, radius_m, k)
    wy, wx = r_b - r_a, c_b - c_a
    oy, ox = wy // k, wx // k
    nbytes_in = wy * wx * (4 + 2 + 2)
    nbytes_out = oy * ox * 4
    memguard.require_ram(nbytes_in + 3 * nbytes_out, f"Fenster {rep.name}")

    T = np.ascontiguousarray(grid.terrain[r_a:r_b, c_a:c_b])
    B = np.ascontiguousarray(grid.bld[r_a:r_b, c_a:c_b])
    V = np.ascontiguousarray(grid.veg[r_a:r_b, c_a:c_b])
    wl = m.x0 + c_a * res
    wt = m.y1 - r_a * res
    tx_x, tx_y = x - wl, wt - y
    gtx = float(K._bil(T, tx_x, tx_y, res))
    hb_tx = float(K._near(B, tx_x, tx_y, res)) * 0.1
    tx_in = hb_tx > 0 and rep.height_m < hb_tx + 0.3
    P = _params(env, res, out_res, radius_m, rep, tx_x, tx_y, gtx, tx_in)
    out = np.full((oy, ox), np.nan, np.float32)

    be = backend()
    t0 = time.time()
    if be == "cuda":
        from numba import cuda

        memguard.require_vram(nbytes_in + nbytes_out + (1 << 24), f"GPU-Fenster {rep.name}")
        fn = K.build("cuda")["coverage_kernel"]
        dT, dB, dV = cuda.to_device(T), cuda.to_device(B), cuda.to_device(V)
        dP = cuda.to_device(P.astype(np.float32))
        dO = cuda.to_device(out)
        tpb = (32, 4)
        rows_per = 64
        for r0 in range(0, oy, rows_per):
            if cancel is not None and cancel.is_set():
                raise Canceled()
            nr = min(rows_per, oy - r0)
            blocks = ((ox + tpb[0] - 1) // tpb[0], (nr + tpb[1] - 1) // tpb[1])
            fn[blocks, tpb](dT, dB, dV, dP, dO, r0)
            cuda.synchronize()
            if progress:
                progress((r0 + nr) / oy)
        out = dO.copy_to_host()
        del dT, dB, dV, dP, dO
    else:
        fn = K.build("cpu")["coverage_cpu"]
        rows_per = 16
        for r0 in range(0, oy, rows_per):
            if cancel is not None and cancel.is_set():
                raise Canceled()
            fn(T, B, V, P, out, r0, min(rows_per, oy - r0))
            if progress:
                progress(min(r0 + rows_per, oy) / oy)
    t_kernel = time.time() - t0

    t1 = time.time()
    # ITU-R P.1411: Straßenschlucht-Modell nur, wenn die Senderantenne unterhalb der
    # umgebenden Dachkanten liegt (sonst: Über-Dach-Ausbreitung = Profil-Kernel).
    rr = int(150 / res)
    tr, tc = int(tx_y / res), int(tx_x / res)
    nb = B[max(tr - rr, 0):tr + rr + 1, max(tc - rr, 0):tc + rr + 1]
    roof_level = float(np.percentile(nb[nb > 0], 90)) * 0.1 if (nb > 0).sum() > 20 else 0.0
    below_roof = rep.height_m <= roof_level + 1.0
    if env.canyon and below_roof:
        from .canyon import canyon_pass

        nan_mask = ~np.isfinite(out)
        out = canyon_pass(out, B[k // 2::k, k // 2::k][:oy, :ox], (tx_y / out_res - 0.5, tx_x / out_res - 0.5),
                          out_res, env, cancel=cancel)
        out[nan_mask] = np.nan
    t_canyon = time.time() - t1
    out = local_mean(out, k_cells=3)
    log.info("%s: %dx%d Zellen, Kernel %.1fs, Schlucht %.1fs (%s)", rep.name, oy, ox, t_kernel, t_canyon, be)
    return LossResult(out, r_a // k, c_a // k, out_res, (tx_y / out_res, tx_x / out_res),
                      {"t_kernel": t_kernel, "t_canyon": t_canyon, "backend": be, "ground_m": gtx,
                       "building_m": hb_tx, "indoor": bool(tx_in),
                       "roof_level_m": roof_level, "canyon": bool(env.canyon and below_roof)})


def local_mean(loss: np.ndarray, k_cells: int = 3) -> np.ndarray:
    """Lokaler Mittelwert (Leistungsmittel) über k x k Zellen. Entspricht dem 'local mean' der
    Messpraxis (schnelles Fading und Abtast-Aliasing schmaler Sichtkorridore werden gemittelt)."""
    from scipy import ndimage

    if k_cells <= 1:
        return loss
    ok = np.isfinite(loss)
    p = np.where(ok, 10.0 ** (-0.1 * np.where(ok, loss, 0.0)), 0.0).astype(np.float64)
    num = ndimage.uniform_filter(p, k_cells, mode="nearest")
    den = ndimage.uniform_filter(ok.astype(np.float64), k_cells, mode="nearest")
    with np.errstate(all="ignore"):
        m = -10.0 * np.log10(num / den)
    return np.where(ok, m, np.nan).astype(np.float32)


def compute_points(grid: RegionGrid, rep: Repeater, env: Environment, xs, ys, rx_abs=None) -> np.ndarray:
    """Pfadverlust zu einzelnen Punkten (UTM). ``rx_abs``: feste Empfangsantennenhöhe ü. NHN
    (Repeater-zu-Repeater-Strecken). Läuft auf der CPU, Fenster = Bounding-Box."""
    m = grid.meta
    res = m.res
    x, y = to_utm(rep.lat, rep.lon)
    xs = np.atleast_1d(np.asarray(xs, np.float64))
    ys = np.atleast_1d(np.asarray(ys, np.float64))
    out = np.full(xs.shape, np.nan)
    for i in range(xs.size):
        c_a = max(int((min(x, xs[i]) - m.x0) / res) - 4, 0)
        c_b = min(int((max(x, xs[i]) - m.x0) / res) + 5, m.nx)
        r_a = max(int((m.y1 - max(y, ys[i])) / res) - 4, 0)
        r_b = min(int((m.y1 - min(y, ys[i])) / res) + 5, m.ny)
        if c_b - c_a < 3 or r_b - r_a < 3:
            continue
        memguard.require_ram((c_b - c_a) * (r_b - r_a) * 8, "Punkt-Fenster")
        T = np.ascontiguousarray(grid.terrain[r_a:r_b, c_a:c_b])
        B = np.ascontiguousarray(grid.bld[r_a:r_b, c_a:c_b])
        V = np.ascontiguousarray(grid.veg[r_a:r_b, c_a:c_b])
        wl = m.x0 + c_a * res
        wt = m.y1 - r_a * res
        tx_x, tx_y = x - wl, wt - y
        gtx = float(K._bil(T, tx_x, tx_y, res))
        hb_tx = float(K._near(B, tx_x, tx_y, res)) * 0.1
        tx_in = hb_tx > 0 and rep.height_m < hb_tx + 0.3
        P = _params(env, res, res, 1e9, rep, tx_x, tx_y, gtx, tx_in)
        if rx_abs is not None:
            P[K.P_FIXRX] = 1.0
            P[K.P_ZRXABS] = float(np.atleast_1d(rx_abs)[i])
        fn = K.build("cpu")["points_cpu"]
        o = np.zeros(1)
        fn(T, B, V, P, np.array([xs[i] - wl]), np.array([wt - ys[i]]), o)
        out[i] = o[0]
    return out


# ---------------------------------------------------------------- Cache ----

def cache_dir(region: str) -> Path:
    p = config.data_dir() / "cache" / region
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_result(path: Path, res: LossResult) -> None:
    """Speichert als unkomprimiertes uint16-.npy (memory-mappable, 0.02 dB) + JSON."""
    q = np.where(np.isfinite(res.loss), np.clip(np.round(res.loss * 50.0), 0, 65534), 65535).astype(np.uint16)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, q)
    meta = dict(res.meta, r0=res.r0, c0=res.c0, out_res=res.out_res, tx_rc=list(res.tx_rc))
    path.with_suffix(".json").write_text(json.dumps(meta))
    tmp.replace(path.with_suffix(".npy"))


class CachedLoss:
    """Memory-mapped Pfadverlust eines Repeaters."""

    def __init__(self, path: Path):
        self.q = np.load(path.with_suffix(".npy"), mmap_mode="r")
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.r0 = int(self.meta["r0"])
        self.c0 = int(self.meta["c0"])
        self.shape = self.q.shape

    def rows(self, a: int, b: int) -> np.ndarray:
        q = np.asarray(self.q[a:b])
        return np.where(q == 65535, np.nan, q.astype(np.float32) / 50.0).astype(np.float32)

    def at(self, r: int, c: int) -> float:
        rr, cc = r - self.r0, c - self.c0
        if 0 <= rr < self.shape[0] and 0 <= cc < self.shape[1]:
            v = int(self.q[rr, cc])
            return float("nan") if v == 65535 else v / 50.0
        return float("nan")


def cached(path: Path) -> bool:
    return path.with_suffix(".npy").exists() and path.with_suffix(".json").exists()


def _strip(grid: RegionGrid, xa, ya, xb, yb):
    """Profilstreifen Punkt a -> b im Rasterabstand: (terrain, bld, veg, step) oder None."""
    from scipy import ndimage

    m = grid.meta
    d = math.hypot(xb - xa, yb - ya)
    n = max(int(d / m.res) + 1, 2)
    t = np.arange(n + 1) / n
    xs, ys = xa + (xb - xa) * t, ya + (yb - ya) * t
    r, c = m.to_rc(xs, ys)
    if r.min() < 0 or c.min() < 0 or r.max() > m.ny - 1 or c.max() > m.nx - 1:
        return None
    ri = np.clip(np.round(r).astype(int), 0, m.ny - 1)
    ci = np.clip(np.round(c).astype(int), 0, m.nx - 1)
    terr = np.empty(n + 1, np.float32)
    for k in range(0, n + 1, 256):
        rr, cc = r[k:k + 256], c[k:k + 256]
        a0, a1 = int(rr.min()), min(int(rr.max()) + 2, m.ny)
        b0, b1 = int(cc.min()), min(int(cc.max()) + 2, m.nx)
        blk = np.asarray(grid.terrain[a0:a1, b0:b1], np.float32)
        terr[k:k + 256] = ndimage.map_coordinates(blk, [rr - a0, cc - b0], order=1, mode="nearest")
    return terr, np.asarray(grid.bld[ri, ci]), np.asarray(grid.veg[ri, ci]), d / n


class StripBatch:
    """Profilstreifen vieler Verbindungen Repeater -> Punkt, einmal extrahiert und danach mit
    beliebigen Modellparametern in einem parallelen Aufruf auswertbar (Kalibrier-Fit)."""

    def __init__(self, grid: RegionGrid, items):
        """items: Liste (repeater, x, y, z_abs)."""
        self.items = items
        T, B, V, offs, lens, xs, meta = [], [], [], [], [], [], []
        off = 0
        for rep, xb, yb, zb in items:
            xa, ya = to_utm(rep.lat, rep.lon)
            st = _strip(grid, xa, ya, xb, yb)
            if st is None:
                meta.append(None)
                offs.append(off)
                lens.append(0)
                xs.append(0.0)
                continue
            terr, bld, veg, step = st
            T.append(terr)
            B.append(bld)
            V.append(veg)
            offs.append(off)
            lens.append(len(terr))
            off += len(terr)
            xs.append((len(terr) - 0.5) * step)
            meta.append((step, float(terr[0]), float(bld[0]) * 0.1, zb))
        cat = (lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(1, dt))
        self.T = np.repeat(cat(T, np.float32)[None, :], 3, 0)
        self.B = np.repeat(cat(B, np.uint16)[None, :], 3, 0)
        self.V = np.repeat(cat(V, np.uint16)[None, :], 3, 0)
        self.offs = np.array(offs, np.int64)
        self.lens = np.array(lens, np.int64)
        self.xs = np.array(xs, np.float64)
        self.meta = meta
        memguard.require_ram(self.T.nbytes * 3, "Kalibrier-Profile")

    def evaluate(self, env: Environment) -> np.ndarray:
        n = len(self.items)
        Pall = np.zeros((n, K.P_N))
        ys = np.zeros(n)
        valid = np.zeros(n, bool)
        for i, ((rep, _x, _y, _z), mt) in enumerate(zip(self.items, self.meta)):
            if mt is None:
                continue
            step, gtx, hb_tx, zb = mt
            tx_in = hb_tx > 0 and rep.height_m < hb_tx + 0.3
            P = _params(env, step, step, 1e9, rep, 0.5 * step, 1.5 * step, gtx, tx_in)
            P[K.P_FIXRX] = 1.0
            P[K.P_ZRXABS] = zb
            Pall[i] = P
            ys[i] = 1.5 * step
            valid[i] = True
        out = np.full(n, np.nan)
        if valid.any():
            idx = np.nonzero(valid)[0]
            o = np.zeros(idx.size)
            K.build("cpu")["batch_cpu"](self.T, self.B, self.V, self.offs[idx], self.lens[idx],
                                        np.ascontiguousarray(Pall[idx]), self.xs[idx], ys[idx], o)
            out[idx] = o
        return out


def link_loss(grid: RegionGrid, rep: Repeater, env: Environment, xb: float, yb: float, zb_abs: float) -> float:
    """Pfadverlust Repeater -> Punkt (feste Empfangshöhe ü. NHN) über einen Profilstreifen
    entlang der Verbindungslinie – identische Physik wie die Karte, kaum Speicherbedarf."""
    return float(StripBatch(grid, [(rep, xb, yb, zb_abs)]).evaluate(env)[0])
