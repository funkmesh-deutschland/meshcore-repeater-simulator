"""Berechnungsauftrag: Pfadverlust je Repeater (gecacht) -> kombinierte Abdeckungsraster + Links.

Ergebnisraster (memory-mapped, ``data/results/<projekt>/``), Auflösung = Ausgaberaster:

* ``rssi``   int16  [0.1 dBm] bester Downlink-Empfangspegel am Endgerät
* ``margin`` int16  [0.1 dB]  beste Linkreserve min(Downlink, Uplink) - Empfindlichkeit
* ``count``  uint8           Anzahl Repeater mit Ortswahrscheinlichkeit >= Ziel (Redundanz)
* ``best``   uint16          Index+1 des besten Repeaters (0 = keiner)
* ``prob``   uint8  [%]      Wahrscheinlichkeit, mindestens einen Repeater zu erreichen
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import traceback
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy.special import ndtr

from . import config
from .grid import RegionGrid
from .project import Project
from .radio import engine as E
from .radio import kernel as K
from .radio.lora import antenna_elements, sensitivity_dbm

log = logging.getLogger(__name__)
NODATA16 = -32768


def results_dir(project: Project) -> Path:
    p = config.data_dir() / "results" / project.name
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path(project: Project, rep, env, out_res, radius_m) -> Path:
    return E.cache_dir(project.region) / f"{rep.geometry_key(env, out_res, radius_m)}"


class Job:
    def __init__(self):
        self.state = "idle"          # idle | running | done | error | canceled
        self.progress = 0.0
        self.message = ""
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = 0.0
        self.finished = 0.0
        self.log: list[str] = []

    def to_dict(self):
        return {"state": self.state, "progress": round(self.progress, 4), "message": self.message,
                "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else 0,
                "log": self.log[-30:]}


class CoverageService:
    def __init__(self, grid: RegionGrid, store):
        self.grid = grid
        self.store = store
        self.job = Job()
        self.lock = threading.Lock()
        self.version = 0
        self._res_meta = None
        self._arrays = {}
        self._load_results()

    # ----------------------------------------------------------- Ergebnisse --
    def _load_results(self):
        rd = results_dir(self.store.project)
        mp = rd / "meta.json"
        self._arrays = {}
        if mp.exists():
            self._res_meta = json.loads(mp.read_text())
            for n in ("rssi", "margin", "count", "best", "prob"):
                f = rd / f"{n}.npy"
                if f.exists():
                    self._arrays[n] = np.load(f, mmap_mode="r")
            self.version = self._res_meta.get("version", 0)
        else:
            self._res_meta = None

    @property
    def result_meta(self):
        return self._res_meta

    def array(self, name):
        return self._arrays.get(name)

    # ----------------------------------------------------------- Auftrag ----
    def start(self, only_missing: bool = False):
        with self.lock:
            if self.job.state == "running":
                raise RuntimeError("Es läuft bereits eine Berechnung")
            self.job = Job()
            self.job.state = "running"
            self.job.started = time.time()
            self.job.thread = threading.Thread(target=self._run, daemon=True)
            self.job.thread.start()

    def cancel(self):
        self.job.cancel.set()

    def _msg(self, m):
        self.job.message = m
        self.job.log.append(time.strftime("%H:%M:%S ") + m)
        log.info(m)

    def _run(self):
        job = self.job
        try:
            p = self.store.project
            env = p.environment()
            out_res = float(p.compute.out_res)
            radius = float(p.compute.radius_km) * 1000.0
            m = self.grid.meta
            reps, outside = [], 0
            for r in p.repeaters:
                if not r.enabled:
                    continue
                x, y = E.to_utm(r.lat, r.lon)
                if m.x0 < x < m.x1 and m.y0 < y < m.y1:
                    reps.append(r)
                else:
                    outside += 1
            if not reps:
                raise RuntimeError("Keine aktiven Repeater im Gebiet")
            if outside:
                self._msg(f"{outside} Repeater außerhalb des Gebiets – übersprungen")
            n = len(reps)
            self._msg(f"Berechnung: {n} Repeater, Raster {out_res:g} m, Radius {radius / 1000:g} km, Backend {E.backend()}")
            paths = []
            for i, rep in enumerate(reps):
                if job.cancel.is_set():
                    raise E.Canceled()
                cp = cache_path(p, rep, env, out_res, radius)
                paths.append(cp)
                if E.cached(cp):
                    self._msg(f"[{i + 1}/{n}] {rep.name}: aus Cache")
                    job.progress = (i + 1) / (n + 1)
                    continue
                self._msg(f"[{i + 1}/{n}] {rep.name}: berechne …")

                def prog(f, i=i):
                    job.progress = (i + 0.9 * f) / (n + 1)

                res = E.compute_repeater(self.grid, rep, env, out_res, radius, progress=prog, cancel=job.cancel)
                E.save_result(cp, res)
                self._msg(f"[{i + 1}/{n}] {rep.name}: fertig ({res.meta['t_kernel'] + res.meta['t_canyon']:.0f} s)")
            self._msg("Kombiniere Abdeckung …")
            self._combine(p, reps, paths, out_res, radius)
            job.progress = 0.97
            self._msg("Berechne Repeater-Verbindungen …")
            self.links(recompute=True)
            job.progress = 1.0
            job.state = "done"
            self._msg(f"Fertig in {time.time() - job.started:.0f} s")
        except E.Canceled:
            job.state = "canceled"
            self._msg("Abgebrochen")
        except Exception as exc:  # noqa: BLE001
            job.state = "error"
            self._msg(f"Fehler: {exc}")
            log.error(traceback.format_exc())
        finally:
            job.finished = time.time()

    # ----------------------------------------------------------- Kombination --
    def budget(self, p: Project):
        sens = sensitivity_dbm(p.radio.sf, p.radio.bw_khz, p.radio.nf_db)
        return sens

    def _combine(self, p: Project, reps, paths, out_res, radius):
        m = self.grid.meta
        k = int(round(out_res / m.res))
        ny, nx = m.ny // k, m.nx // k
        rd = results_dir(p)
        self._arrays = {}
        sens = self.budget(p)
        tgt_z = NormalDist().inv_cdf(p.stats.target_prob)
        sig_out = p.stats.sigma_outdoor_db
        sig_in = math.hypot(p.stats.sigma_outdoor_db, p.stats.sigma_indoor_db)
        node = p.node

        def mm(name, dtype, fill):
            a = np.lib.format.open_memmap(rd / f"_{name}.npy", mode="w+", dtype=dtype, shape=(ny, nx))
            for i in range(0, ny, 2000):
                a[i:i + 2000] = fill
            return a

        rssi = mm("rssi", np.int16, NODATA16)
        marg = mm("margin", np.int16, NODATA16)
        cnt = mm("count", np.uint8, 0)
        best = mm("best", np.uint16, 0)
        lq = mm("lq", np.float32, 0.0)  # sum log(1 - p)
        bstride = self.grid.bld[k // 2::k, k // 2::k]
        for idx, (rep, cp) in enumerate(zip(reps, paths)):
            cl = E.CachedLoss(cp)
            h, w = cl.shape
            r0, c0 = cl.r0, cl.c0
            h = min(h, ny - r0)
            w = min(w, nx - c0)
            for a in range(0, h, 512):
                b = min(a + 512, h)
                L = cl.rows(a, b)[:, :w] + p.stats.calib_offset_db
                ok = np.isfinite(L)
                if not ok.any():
                    continue
                gr = slice(r0 + a, r0 + b)
                gc = slice(c0, c0 + w)
                indoor = np.asarray(bstride[gr, gc]) > 0
                down = rep.tx_dbm - rep.cable_db + node.gain_dbi - node.body_loss_db - L
                up = node.tx_dbm + node.gain_dbi - node.body_loss_db - rep.cable_db - L
                mg = np.minimum(down, up) - sens
                sig = np.where(indoor, sig_in, sig_out)
                pr = ndtr(np.where(ok, mg, -99.0) / sig)
                # Ergebnisse aktualisieren
                cur_r = np.asarray(rssi[gr, gc])
                d16 = np.where(ok, np.clip(np.round(down * 10), -32000, 32000), NODATA16).astype(np.int16)
                rssi[gr, gc] = np.maximum(cur_r, d16)
                cur_m = np.asarray(marg[gr, gc])
                m16 = np.where(ok, np.clip(np.round(mg * 10), -32000, 32000), NODATA16).astype(np.int16)
                better = m16 > cur_m
                marg[gr, gc] = np.where(better, m16, cur_m)
                cur_b = np.asarray(best[gr, gc])
                best[gr, gc] = np.where(better, idx + 1, cur_b).astype(np.uint16)
                cnt[gr, gc] = np.minimum(np.asarray(cnt[gr, gc]).astype(np.int16) + (ok & (mg >= tgt_z * sig)), 255).astype(np.uint8)
                lq[gr, gc] = np.asarray(lq[gr, gc]) + np.log1p(-np.clip(pr, 0, 1 - 1e-7)).astype(np.float32)
            del cl
        prob = mm("prob", np.uint8, 255)
        for i in range(0, ny, 1000):
            q = np.asarray(lq[i:i + 1000])
            has = np.asarray(marg[i:i + 1000]) != NODATA16
            prob[i:i + 1000] = np.where(has, np.round(100 * (1 - np.exp(q))), 255).astype(np.uint8)
        for a in (rssi, marg, cnt, best, prob):
            a.flush()
        del rssi, marg, cnt, best, prob, lq
        for n in ("rssi", "margin", "count", "best", "prob"):
            (rd / f"_{n}.npy").replace(rd / f"{n}.npy")
        (rd / "_lq.npy").unlink(missing_ok=True)
        self.version = int(time.time())
        meta = {"version": self.version, "out_res": out_res, "k": k, "ny": ny, "nx": nx,
                "radius_m": radius, "repeaters": [r.id for r in reps], "names": [r.name for r in reps],
                "cache": [str(c) for c in paths], "sens_dbm": sens, "target_prob": p.stats.target_prob}
        (rd / "meta.json").write_text(json.dumps(meta))
        self._load_results()

    # ----------------------------------------------------------- Links -----
    def _link_reps(self, p):
        m = self.grid.meta
        out = []
        for r in p.repeaters:
            if r.enabled:
                x, y = E.to_utm(r.lat, r.lon)
                if m.x0 < x < m.x1 and m.y0 < y < m.y1:
                    out.append((r, (x, y)))
        return out

    def links(self, recompute: bool = False, max_dist_m: float = 40000.0):
        """Repeater-zu-Repeater-Strecken (beide Antennen auf Montagehöhe). Fehlende Strecken
        werden gesammelt und in Paketen über Profilstreifen gerechnet (auch für Tausende Repeater)."""
        p = self.store.project
        rd = results_dir(p)
        f = rd / "links.json"
        if f.exists() and not recompute:
            return json.loads(f.read_text())
        env = p.environment()
        rl = self._link_reps(p)
        sens = self.budget(p)
        out = []
        if len(rl) >= 2:
            from scipy.spatial import cKDTree

            reps = [r for r, _ in rl]
            utm = np.array([xy for _, xy in rl])
            pairs = sorted(cKDTree(utm).query_pairs(max_dist_m))
            cache_f = E.cache_dir(p.region) / "links_cache.json"
            lc = json.loads(cache_f.read_text()) if cache_f.exists() else {}
            keys = {}
            missing = []
            for i, j in pairs:
                a, b = reps[i], reps[j]
                key = "|".join(sorted([a.geometry_key(env, 0, 0), b.geometry_key(env, 0, 0)]))
                keys[(i, j)] = key
                if key not in lc:
                    missing.append((i, j, key))
            for k0 in range(0, len(missing), 1000):
                if self.job.cancel.is_set():
                    raise E.Canceled()
                chunk = missing[k0:k0 + 1000]
                vals = self._link_losses([(reps[i], reps[j], tuple(utm[j])) for i, j, _ in chunk], env)
                for (_i, _j, key), v in zip(chunk, vals):
                    lc[key] = v
                if len(missing) > 1000:
                    self._msg(f"Verbindungen {min(k0 + 1000, len(missing))}/{len(missing)}")
            for (i, j), key in keys.items():
                a, b = reps[i], reps[j]
                lb = lc.get(key)  # Grundübertragungsdämpfung ohne Antennengewinne
                if lb is None:
                    continue
                d = float(np.hypot(*(utm[i] - utm[j])))
                ga = a.gain_dbi - a.cable_db
                gb = b.gain_dbi - b.cable_db
                lb = lb + p.stats.calib_offset_db
                ab = a.tx_dbm + ga + gb - lb
                ba = b.tx_dbm + ga + gb - lb
                if max(ab, ba) < sens - 20:   # weit unter der Empfindlichkeit: nicht speichern
                    continue
                out.append({"a": a.id, "b": b.id, "dist_m": round(d), "loss_db": round(lb, 1),
                            "rssi_ab": round(ab, 1), "rssi_ba": round(ba, 1),
                            "margin": round(min(ab, ba) - sens, 1)})
            cache_f.write_text(json.dumps(lc))
        f.write_text(json.dumps(out))
        return out

    def _link_losses(self, items, env):
        """Grundübertragungsdämpfung a->b mit realen Höhen (Antennendiagramme berücksichtigt,
        Spitzengewinne herausgerechnet -> Gewinne werden im Budget addiert). items: (a, b, xy_b)."""
        g = self.grid
        batch_items, meta = [], []
        for a, b, xy_b in items:
            ib = g.sample_point(*xy_b)
            xa, ya = E.to_utm(a.lat, a.lon)
            ia = g.sample_point(xa, ya)
            if not ib.get("inside") or not ia.get("inside"):
                meta.append(None)
                continue
            zb = ib["terrain"] + b.height_m
            za = ia["terrain"] + a.height_m
            d = math.dist((xa, ya), xy_b)
            th = math.atan2(zb - za, max(d, 1.0))
            gb_pat = K._pattern_db(-th, b.gain_dbi, antenna_elements(b.gain_dbi), b.tilt_deg) - b.gain_dbi
            meta.append((len(batch_items), a.gain_dbi - gb_pat))
            batch_items.append((a, xy_b[0], xy_b[1], zb))
        vals = E.StripBatch(g, batch_items).evaluate(env) if batch_items else []
        out = []
        for mt in meta:
            if mt is None:
                out.append(None)
                continue
            v = float(vals[mt[0]])
            # Kernel: l_eff = Lb - (gain_a + ga_pat)  ->  Rückgabe Lb - ga_pat - gb_pat
            out.append(v + mt[1] if np.isfinite(v) else None)
        return out

    # ----------------------------------------------------------- Punkt ----
    def point(self, x, y):
        g = self.grid
        out = {"x": x, "y": y, **g.sample_point(x, y)}
        meta = self._res_meta
        if not meta or not self._arrays:
            return out
        k = meta["k"]
        r = int((g.meta.y1 - y) / (g.meta.res * k))
        c = int((x - g.meta.x0) / (g.meta.res * k))
        if not (0 <= r < meta["ny"] and 0 <= c < meta["nx"]):
            return out
        rs = int(self._arrays["rssi"][r, c])
        if rs == NODATA16:
            return out
        out["rssi"] = rs / 10
        out["margin"] = int(self._arrays["margin"][r, c]) / 10
        out["count"] = int(self._arrays["count"][r, c])
        b = int(self._arrays["best"][r, c])
        out["best"] = meta["names"][b - 1] if b else None
        out["prob"] = int(self._arrays["prob"][r, c])
        p = self.store.project
        per = []
        reps = {r_.id: r_ for r_ in p.repeaters}
        sens = meta["sens_dbm"]
        for rid, cp in zip(meta["repeaters"], meta["cache"]):
            rep = reps.get(rid)
            if rep is None or not E.cached(Path(cp)):
                continue
            L = E.CachedLoss(Path(cp)).at(r, c) + p.stats.calib_offset_db
            if not np.isfinite(L):
                continue
            n = p.node
            down = rep.tx_dbm - rep.cable_db + n.gain_dbi - n.body_loss_db - L
            up = n.tx_dbm + n.gain_dbi - n.body_loss_db - rep.cable_db - L
            per.append({"name": rep.name, "loss": round(L, 1), "down": round(down, 1), "up": round(up, 1),
                        "margin": round(min(down, up) - sens, 1)})
        per.sort(key=lambda d: -d["margin"])
        out["repeaters"] = per[:12]
        return out
