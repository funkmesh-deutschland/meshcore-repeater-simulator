"""Web-Server: REST-API + Kacheln + statische Oberfläche (läuft lokal, Linux & Windows).

Mehrere Gebiete (Ausschnitte Deutschlands) können vorbereitet werden; eines ist aktiv. Jedes
Gebiet hat sein eigenes Projekt (Repeater, Parameter), eigene Ergebnisse und Kalibrierung.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pyproj import Transformer

from .. import __version__, areas as A, config
from ..coverage import CoverageService, results_dir
from ..grid import RegionGrid
from ..project import Project, ProjectStore
from ..radio import engine as E
from ..radio.lora import airtime_s, sensitivity_dbm
from .tiles import TileRenderer

log = logging.getLogger(__name__)
_to_ll = Transformer.from_crs(25832, 4326, always_xy=True)


def _migrate_legacy() -> None:
    """Version 0.1 hatte ein Projekt ``standard`` für die Region ``coesfeld``."""
    pd = config.projects_dir()
    old = pd / "standard.json"
    if old.exists():
        try:
            d = json.loads(old.read_text())
        except ValueError:
            return
        reg = d.get("region", "coesfeld")
        new = pd / f"{reg}.json"
        if not new.exists() and RegionGrid.exists(reg):
            d["name"] = reg
            new.write_text(json.dumps(d, indent=1, ensure_ascii=False))
            ord_ = config.data_dir() / "results" / "standard"
            nrd = config.data_dir() / "results" / reg
            if ord_.exists() and not nrd.exists():
                ord_.rename(nrd)
                mf = nrd / "meta.json"
                if mf.exists():  # Cache-Pfade bleiben gültig (Cache liegt je Region)
                    pass
            old.rename(pd / "standard.json.bak")


class Ctx:
    """Aktives Gebiet mit Projekt, Berechnungsdienst und Kachel-Renderer."""

    def __init__(self):
        self.lock = threading.RLock()
        self.name = None
        self.grid = None
        self.store = None
        self.cov = None
        self.tiles = None
        self.load_state = {"result": None, "running": False, "error": None}
        self.calib = {"report": None}
        self.bounds = None
        self.boundary = None
        self.area_job = A.AreaJob()

    def require(self):
        if self.cov is None:
            raise HTTPException(409, "Kein Gebiet aktiv – zuerst ein Gebiet anlegen oder auswählen")
        return self

    def activate(self, name: str, seed_from: Project | None = None):
        if not RegionGrid.exists(name):
            raise HTTPException(404, f"Gebiet '{name}' nicht gefunden")
        grid = RegionGrid.open(name)
        if not grid.meta.complete:
            raise HTTPException(409, f"Gebiet '{name}' ist noch nicht fertig vorbereitet")
        with self.lock:
            if self.cov is not None and self.cov.job.state == "running":
                raise HTTPException(409, "Im aktuellen Gebiet läuft eine Berechnung")
            fresh = not (config.projects_dir() / f"{name}.json").exists()
            store = ProjectStore(name, name)
            m = grid.meta
            w, s = _to_ll.transform(m.x0, m.y0)
            e, n = _to_ll.transform(m.x1, m.y1)
            if fresh and seed_from is not None:
                _seed_project(store, seed_from, grid)
            r_ = grid.meta.res
            q = store.project.compute.out_res / r_
            if q < 1 - 1e-6 or abs(q - round(q)) > 1e-6:
                store.update_settings("compute", {"out_res": float(r_)})
            self.name, self.grid, self.store = name, grid, store
            self.cov = CoverageService(grid, store)
            self.tiles = TileRenderer(grid, self.cov)
            self.load_state = {"result": None, "running": False, "error": None}
            self.bounds = [w, s, e, n]
            self.boundary = A.boundary(name)
            self.calib = {"report": None}
            cf = results_dir(store.project) / "calibration.json"
            if cf.exists():
                try:
                    self.calib["report"] = json.loads(cf.read_text())
                except ValueError:
                    pass
            _load_load_layer(self.cov, self.tiles)
            (config.data_dir() / "active_area.txt").write_text(name)


def _seed_project(store: ProjectStore, src: Project, grid) -> None:
    """Neues Gebiet übernimmt Parameter und die darin liegenden Repeater des bisherigen Gebiets."""
    with store.lock:
        p = store.project
        for sec in ("radio", "node", "env", "stats", "compute"):
            setattr(p, sec, type(getattr(src, sec))(**asdict(getattr(src, sec))))
        r = grid.meta.res
        if p.compute.out_res < r or abs(p.compute.out_res / r - round(p.compute.out_res / r)) > 1e-6:
            p.compute.out_res = float(r)
        m = grid.meta
        for rep in src.repeaters:
            x, y = E.to_utm(rep.lat, rep.lon)
            if m.x0 < x < m.x1 and m.y0 < y < m.y1:
                p.repeaters.append(type(rep)(**asdict(rep)))
        store.save()


def create_app(region: str | None = None) -> FastAPI:
    _migrate_legacy()
    ctx = Ctx()
    start = region
    if not start:
        f = config.data_dir() / "active_area.txt"
        start = f.read_text().strip() if f.exists() else None
    if not start or not RegionGrid.exists(start):
        done = [a for a in A.list_areas() if a["complete"]]
        start = done[0]["name"] if done else None
    if start:
        try:
            ctx.activate(start)
        except HTTPException as exc:
            log.warning("Gebiet %s nicht aktivierbar: %s", start, exc.detail)

    app = FastAPI(title="Funkmesh Repeater-Simulator", version=__version__)

    @app.get("/api/state")
    def state():
        base = {"version": __version__, "backend": E.backend(), "areas": A.list_areas(),
                "area_job": ctx.area_job.to_dict(), "germany_bbox": A.GERMANY_BBOX}
        if ctx.cov is None:
            return {**base, "region": None}
        p = ctx.store.project
        m = ctx.grid.meta
        sens = sensitivity_dbm(p.radio.sf, p.radio.bw_khz, p.radio.nf_db)
        title = (m.info or {}).get("title") or ctx.name.capitalize()
        return {
            **base,
            "region": {"name": ctx.name, "title": title, "bounds": ctx.bounds, "res": m.res, "sources": m.sources,
                       "boundary": ctx.boundary, "km2": round(m.nx * m.ny * m.res ** 2 / 1e6)},
            "project": p.to_dict(),
            "derived": {"sens_dbm": round(sens, 1), "bel_db": round(p.environment().bel_db(), 1),
                        "veg_am_db": round(p.environment().veg_am, 1),
                        "airtime_msg_ms": round(airtime_s(90, p.radio.sf, p.radio.bw_khz, p.radio.cr, p.radio.preamble) * 1000, 1)},
            "job": ctx.cov.job.to_dict(),
            "result": ctx.cov.result_meta,
            "result_version": ctx.cov.version,
            "load": ctx.load_state["result"],
        }

    # ---------------- Gebiete ----------------
    def _spec(body) -> A.AreaSpec:
        try:
            bbox = [float(v) for v in body["bbox"]]
            if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                raise ValueError("bbox")
            return A.AreaSpec(title=str(body.get("title") or "Gebiet").strip()[:80], bbox=tuple(bbox),
                              res=int(body.get("res", 10)), buffer_m=float(body.get("buffer_m", 5000)),
                              use_nrw=bool(body.get("use_nrw", True)), use_osm=bool(body.get("use_osm", True)),
                              default_building_m=float(body.get("default_building_m", 9.0)),
                              boundary=body.get("boundary"), name=str(body.get("name") or ""))
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(400, f"Ungültige Gebietsangabe: {exc}")

    @app.post("/api/areas/estimate")
    def area_estimate(body: dict):
        sp = _spec(body)
        pl = A.plan(sp)
        pl.pop("tiles", None)
        out_res = float(sp.res)
        radius = float(ctx.store.project.compute.radius_km) if ctx.store else 15.0
        pl["compute"] = A.estimate_compute(sp.bbox, out_res, radius, E.backend())
        pl["exists"] = A.area_dir(sp.name).exists()
        import shutil as _sh

        pl["disk_free_gb"] = round(_sh.disk_usage(config.data_dir()).free / 2**30, 1)
        return pl

    @app.post("/api/areas")
    def area_create(body: dict):
        sp = _spec(body)
        if A.area_dir(sp.name).exists() and (A.area_dir(sp.name) / "meta.json").exists():
            m = json.loads((A.area_dir(sp.name) / "meta.json").read_text())
            if m.get("complete"):
                raise HTTPException(409, f"Gebiet '{sp.name}' existiert bereits – anderen Namen wählen")
        seed = ctx.store.project if ctx.store else None

        def done(spec):
            # Erstes Gebiet automatisch aktivieren
            if ctx.cov is None:
                try:
                    ctx.activate(spec.name, seed)
                except HTTPException:
                    pass

        try:
            ctx.area_job.start(sp, on_done=done)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return ctx.area_job.to_dict()

    @app.post("/api/areas/{name}/resume")
    def area_resume(name: str):
        if not A.area_dir(name).exists():
            raise HTTPException(404, "Gebiet nicht gefunden")
        try:
            ctx.area_job.start(A.spec_from_meta(name))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return ctx.area_job.to_dict()

    @app.post("/api/areas/job/cancel")
    def area_cancel():
        ctx.area_job.stop.set()
        return ctx.area_job.to_dict()

    @app.post("/api/areas/{name}/activate")
    def area_activate(name: str):
        seed = ctx.store.project if ctx.store else None
        ctx.activate(name, seed)
        return {"ok": True}

    @app.delete("/api/areas/{name}")
    def area_delete(name: str):
        if name == ctx.name:
            raise HTTPException(409, "Das aktive Gebiet kann nicht gelöscht werden – erst ein anderes wählen")
        if ctx.area_job.state == "running" and ctx.area_job.name == name:
            raise HTTPException(409, "Gebiet wird gerade vorbereitet")
        A.delete_area(name)
        return {"ok": True}

    @app.get("/api/places")
    def places(q: str, kind: str = ""):
        if len(q.strip()) < 2:
            return []
        try:
            return A.search_places(q.strip(), country=kind == "country")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Ortssuche (Nominatim) nicht erreichbar: {exc}")

    # ---------------- Repeater ----------------
    @app.post("/api/repeaters")
    def add_rep(body: dict):
        return asdict(ctx.require().store.add_repeater(body))

    @app.put("/api/repeaters/{rid}")
    def upd_rep(rid: str, body: dict):
        try:
            return asdict(ctx.require().store.update_repeater(rid, body))
        except KeyError:
            raise HTTPException(404, "Repeater nicht gefunden")

    @app.delete("/api/repeaters/{rid}")
    def del_rep(rid: str):
        ctx.require().store.delete_repeater(rid)
        return {"ok": True}

    @app.post("/api/repeaters/bulk")
    def bulk(body: dict):
        """Mehrere Repeater auf einmal löschen / (de)aktivieren / Höhe setzen."""
        st = ctx.require().store
        ids = set(body.get("ids") or [])
        action = body.get("action")
        if action not in ("delete", "enable", "disable", "height"):
            raise HTTPException(400, "Aktion unbekannt")
        n = 0
        with st.lock:
            if action == "delete":
                before = len(st.project.repeaters)
                st.project.repeaters = [r for r in st.project.repeaters if r.id not in ids]
                n = before - len(st.project.repeaters)
            else:
                for r in st.project.repeaters:
                    if r.id in ids:
                        if action == "height":
                            r.height_m = float(body.get("height_m", r.height_m))
                        else:
                            r.enabled = action == "enable"
                        n += 1
            st.save()
        return {"changed": n}

    @app.put("/api/settings/{section}")
    def upd_settings(section: str, body: dict):
        if section not in ("radio", "node", "env", "stats", "compute"):
            raise HTTPException(400, "unbekannter Abschnitt")
        ctx.require().store.update_settings(section, body)
        return ctx.store.project.to_dict()[section]

    # ---------------- Import ----------------
    @app.get("/api/import")
    def import_list(source: str = "both", max_age_days: float = 30, margin_km: float = 5):
        """Repeater von map.meshcore.io und/oder mapme.sh im Gebiet (+Rand), nach Public Key zusammengeführt."""
        c = ctx.require()
        w, s, e, n = c.bounds
        dlat = margin_km / 111.0
        dlon = margin_km / (111.0 * math.cos(math.radians((s + n) / 2)))
        bb = (w - dlon, s - dlat, e + dlon, n + dlat)
        found: dict[str, dict] = {}
        errors = []
        if source in ("both", "meshcore"):
            from ..meshcore_import import repeaters_in_bounds as mc

            try:
                for x in mc(bb, max_age_days, freq_mhz=None):
                    x["sources"] = ["meshcore.io"]
                    found[(x.get("public_key") or uuid.uuid4().hex).lower()] = x
            except Exception as exc:  # noqa: BLE001
                errors.append(f"map.meshcore.io: {exc}")
        if source in ("both", "mapme"):
            from ..mapme_import import repeaters_in_bounds as mm

            try:
                for x in mm(bb, max_age_days):
                    k = x["public_key"]
                    if k in found:
                        found[k]["sources"].append("mapme.sh")
                        a = found[k].get("age_days")
                        if x["age_days"] is not None and (a is None or x["age_days"] < a):
                            found[k]["age_days"] = x["age_days"]
                    else:
                        x["sources"] = ["mapme.sh"]
                        found[k] = x
            except Exception as exc:  # noqa: BLE001
                errors.append(f"mapme.sh: {exc}")
        if not found and errors:
            raise HTTPException(502, " · ".join(errors))
        known = {(r.public_key or "").lower() for r in c.store.project.repeaters if r.public_key}
        m = c.grid.meta
        out = []
        for x in found.values():
            x["known"] = (x.get("public_key") or "").lower() in known
            xx, yy = E.to_utm(x["lat"], x["lon"])
            x["inside_data"] = m.x0 < xx < m.x1 and m.y0 < yy < m.y1
            out.append(x)
        return {"items": out, "errors": errors}

    @app.post("/api/import")
    def import_do(body: dict):
        c = ctx.require()
        items = body.get("items", [])
        height = float(body.get("height_m", 10))
        added = []
        known = {(r.public_key or "").lower() for r in c.store.project.repeaters if r.public_key}
        for it in items:
            pk = (it.get("public_key") or "").lower() or None
            if pk and pk in known:
                continue
            r = c.store.add_repeater({"name": it["name"], "lat": float(it["lat"]), "lon": float(it["lon"]),
                                      "height_m": height, "source": "+".join(it.get("sources") or ["import"]),
                                      "public_key": pk})
            if pk:
                known.add(pk)
            added.append(asdict(r))
        return added

    # Kompatibilität (alte Oberfläche)
    @app.post("/api/meshcore/import")
    def meshcore_import(body: dict):
        return import_do(body)

    # ---------------- Zurücksetzen ----------------
    @app.post("/api/reset")
    def reset(body: dict):
        """Projekt des aktiven Gebiets auf Werkseinstellungen: Repeater, Parameter, Ergebnisse,
        Auslastung, Kalibrierung und Messdaten werden gelöscht. Geodaten bleiben erhalten."""
        c = ctx.require()
        if c.cov.job.state == "running":
            raise HTTPException(409, "Berechnung läuft – erst abbrechen")
        with c.lock:
            rd = results_dir(c.store.project)
            shutil.rmtree(rd, ignore_errors=True)
            if body.get("cache"):
                shutil.rmtree(E.cache_dir(c.name), ignore_errors=True)
            pf = config.projects_dir() / f"{c.name}.json"
            pf.unlink(missing_ok=True)
            name = c.name
            c.cov = None
            c.activate(name)
            if c.grid.meta.res > c.store.project.compute.out_res:
                c.store.update_settings("compute", {"out_res": c.grid.meta.res})
        return {"ok": True}

    # ---------------- Punktabfrage ----------------
    @app.get("/api/point")
    def point(lat: float, lon: float):
        c = ctx.require()
        x, y = E.to_utm(lat, lon)
        return _clean(c.cov.point(x, y))

    @app.get("/api/profile")
    def profile(rid: str, lat: float, lon: float):
        """Höhenprofil Repeater -> Punkt (Gelände, Gebäude, Bewuchs, Sichtlinie, 1. Fresnelzone)."""
        c = ctx.require()
        grid, store, m = c.grid, c.store, c.grid.meta
        rep = next((r for r in store.project.repeaters if r.id == rid), None)
        if rep is None:
            raise HTTPException(404, "Repeater nicht gefunden")
        xa, ya = E.to_utm(rep.lat, rep.lon)
        xb, yb = E.to_utm(lat, lon)
        d = math.dist((xa, ya), (xb, yb))
        nsm = int(min(max(d / m.res, 2), 2000))
        t = np.linspace(0, 1, nsm + 1)
        xs, ys = xa + (xb - xa) * t, ya + (yb - ya) * t
        r, cc = m.to_rc(xs, ys)
        ri = np.clip(np.round(r).astype(int), 0, m.ny - 1)
        ci = np.clip(np.round(cc).astype(int), 0, m.nx - 1)
        terr = np.asarray(grid.terrain[ri, ci], float)
        bld = np.asarray(grid.bld[ri, ci], float) / 10
        veg = np.asarray(grid.veg[ri, ci], float) / 10
        env = store.project.environment()
        ae = 6371e3 * env.k_factor
        di = t * d
        bulge = di * (d - di) / (2 * ae)
        za = terr[0] + rep.height_m
        zb = terr[-1] + store.project.node.height_m
        los = za + (zb - za) * t
        fres = np.sqrt(np.maximum(env.lam * di * (d - di) / max(d, 1), 0))
        return {"d": di.round(1).tolist(), "terrain": (terr + bulge).round(2).tolist(),
                "building": bld.round(1).tolist(), "vegetation": veg.round(1).tolist(),
                "los": los.round(2).tolist(), "fresnel": fres.round(2).tolist(), "dist_m": round(d)}

    # ---------------- Berechnung ----------------
    @app.post("/api/compute")
    def compute():
        c = ctx.require()
        try:
            c.cov.start()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return c.cov.job.to_dict()

    @app.post("/api/compute/cancel")
    def cancel():
        ctx.require().cov.cancel()
        return ctx.cov.job.to_dict()

    @app.get("/api/job")
    def job():
        if ctx.cov is None:
            return {"job": None, "result_version": 0, "area_job": ctx.area_job.to_dict()}
        return {"job": ctx.cov.job.to_dict(), "result_version": ctx.cov.version, "area_job": ctx.area_job.to_dict()}

    @app.get("/api/links")
    def links():
        c = ctx.require()
        if not c.cov.result_meta:
            return []
        try:
            return c.cov.links()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, str(exc))

    @app.post("/api/cache/clear")
    def clear_cache():
        c = ctx.require()
        if c.cov.job.state == "running":
            raise HTTPException(409, "Berechnung läuft")
        n = 0
        for f in E.cache_dir(c.name).glob("*"):
            f.unlink()
            n += 1
        return {"deleted": n}

    # ---------------- Auslastung ----------------
    @app.post("/api/load")
    def load(body: dict):
        c = ctx.require()
        if c.load_state["running"]:
            raise HTTPException(409, "Simulation läuft bereits")
        if not c.cov.result_meta:
            raise HTTPException(400, "Zuerst die Ausleuchtung berechnen")
        c.load_state["running"] = True
        c.load_state["error"] = None
        try:
            res = _run_load(c.cov, c.store, c.grid, body, c.boundary)
            c.load_state["result"] = res
            _write_load_layer(c.cov, c.store, res)
            _load_load_layer(c.cov, c.tiles)
            return res
        except Exception as exc:  # noqa: BLE001
            log.exception("Lastsimulation")
            raise HTTPException(500, str(exc))
        finally:
            c.load_state["running"] = False

    # ---------------- Kalibrierung ----------------
    from .. import calibration as C

    def _setup(body):
        names = {f for f in C.Setup.__dataclass_fields__}
        dflt = C.Setup()
        out = {}
        for k, v in (body.get("setup") or {}).items():
            if k in names:
                cur = getattr(dflt, k)
                out[k] = (v in (True, 1, "1", "true", "on")) if isinstance(cur, bool) else type(cur)(v)
        return C.Setup(**out)

    def _mdir():
        d = results_dir(ctx.require().store.project) / "measurements"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _mindex():
        f = _mdir() / "index.json"
        return json.loads(f.read_text()) if f.exists() else []

    def _msave(idx):
        (_mdir() / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1))

    def _apply(sug):
        c = ctx.store
        c.update_settings("stats", {"calib_offset_db": sug["calib_offset_db"], "sigma_outdoor_db": sug["sigma_outdoor_db"]})
        c.update_settings("env", {"veg_gamma_db_m": sug["veg_gamma_db_m"], "diffraction": sug["diffraction"]})

    def _finish(rep, body):
        rep = _clean(rep)
        fit = rep.get("fit") or {}
        rep["auto_applied"] = False
        if body.get("auto_apply") and fit.get("suggest") and fit.get("reliable"):
            _apply(fit["suggest"])
            rep["auto_applied"] = True
        ctx.calib["report"] = rep
        (results_dir(ctx.store.project) / "calibration.json").write_text(json.dumps(rep))
        return rep

    @app.get("/api/measurements")
    def meas_list():
        return _mindex()

    @app.post("/api/measurements")
    def meas_add(body: dict):
        """Messdatei hinzufügen (CSV beliebiger Spalten oder JSON der Mapper-App)."""
        text = body.get("text") or ""
        name = str(body.get("name") or "Messung")[:120]
        if len(text) > 80_000_000:
            raise HTTPException(413, "Datei zu groß (max. 80 MB)")
        try:
            csv_text, n, warn = C.normalize_upload(text, name)
        except ValueError as exc:
            raise HTTPException(400, f"{name}: {exc}")
        idx = _mindex()
        mid = uuid.uuid4().hex[:10]
        (_mdir() / f"{mid}.csv").write_text(csv_text)
        e = {"id": mid, "name": name, "rows": n, "added": time.strftime("%Y-%m-%d %H:%M"), "enabled": True,
             "warnings": warn[:5]}
        idx.append(e)
        _msave(idx)
        return e

    @app.put("/api/measurements/{mid}")
    def meas_upd(mid: str, body: dict):
        idx = _mindex()
        for e in idx:
            if e["id"] == mid:
                if "enabled" in body:
                    e["enabled"] = bool(body["enabled"])
        _msave(idx)
        return idx

    @app.delete("/api/measurements/{mid}")
    def meas_del(mid: str):
        idx = [e for e in _mindex() if e["id"] != mid]
        (_mdir() / f"{mid}.csv").unlink(missing_ok=True)
        _msave(idx)
        return idx

    def _pool_text():
        parts = []
        for e in _mindex():
            f = _mdir() / f"{e['id']}.csv"
            if e.get("enabled", True) and f.exists():
                lines = f.read_text().splitlines()
                parts.append(lines if not parts else lines[1:])
        return "\n".join(l for p in parts for l in p)

    @app.post("/api/calibration")
    def calibration(body: dict):
        """Auswertung über alle aktiven Messdatensätze (oder ein mitgeschicktes CSV)."""
        c = ctx.require()
        text = body.get("csv") or _pool_text()
        if not text.strip():
            raise HTTPException(400, "Keine Messdaten – erst eine Messdatei hinzufügen")
        st = _setup(body)
        try:
            rep = C.run(c.grid, c.store.project, text, st, fit=bool(body.get("fit", True)))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        rep["datasets"] = [e["name"] for e in _mindex() if e.get("enabled", True)] if not body.get("csv") else ["(direkt)"]
        return _finish(rep, body)

    @app.post("/api/calibration/mapme")
    def calibration_mapme(body: dict):
        """Messdaten von mapme.sh für alle Repeater mit Public Key als Datensatz hinzufügen und auswerten."""
        from .. import mapme_import as M

        c = ctx.require()
        try:
            rows, info = M.calibration_rows(c.store.project.repeaters, float(body.get("max_dist_km", 40)),
                                            float(body.get("ambiguity_ratio", 2.0)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"mapme.sh nicht erreichbar: {exc}")
        if not rows:
            raise HTTPException(400, "Keine verwertbaren mapme.sh-Messungen für die Repeater dieses Gebiets")
        csv_text = M.rows_to_csv(rows)
        # vorhandenen mapme-Datensatz ersetzen; Messaufbau „Fahrzeug, Gerät innen“ je Zeile
        csv_text = "\n".join([csv_text.splitlines()[0] + ",setup"] +
                             [l + ",fahrzeug_innen" for l in csv_text.splitlines()[1:]])
        idx = [e for e in _mindex() if e.get("source") != "mapme.sh"]
        for f in _mdir().glob("*.csv"):
            if f.stem not in {e["id"] for e in idx}:
                f.unlink()
        norm, n, warn = C.normalize_upload(csv_text, "mapme.csv")
        mid = uuid.uuid4().hex[:10]
        (_mdir() / f"{mid}.csv").write_text(norm)
        idx.append({"id": mid, "name": f"mapme.sh ({time.strftime('%d.%m.%Y')})", "rows": n, "source": "mapme.sh",
                    "added": time.strftime("%Y-%m-%d %H:%M"), "enabled": True, "warnings": [],
                    "info": {k: info[k] for k in ("hexes_total", "dropped_ambiguous", "dropped_far")}})
        _msave(idx)
        rep = calibration({k: v for k, v in body.items() if k != "csv"})
        rep["source"] = _clean({"name": "mapme.sh", **info})
        ctx.calib["report"] = rep
        (results_dir(ctx.store.project) / "calibration.json").write_text(json.dumps(rep))
        return rep

    @app.get("/api/calibration")
    def calibration_get():
        return ctx.calib["report"] if ctx.cov is not None else None

    @app.post("/api/calibration/apply")
    def calibration_apply(body: dict):
        ctx.require()
        rep = ctx.calib["report"] or {}
        fit = rep.get("fit") or {}
        sug = body.get("suggest") or fit.get("suggest")
        if not sug:
            raise HTTPException(400, "Kein Fit-Ergebnis vorhanden")
        if not body.get("force") and fit.get("reliable") is False:
            raise HTTPException(409, "Fit nicht plausibel (Reststreuung zu groß) – erst Repeater-Höhen/Leistung prüfen")
        _apply(sug)
        return ctx.store.project.to_dict()

    # ---------------- Kacheln ----------------
    @app.get("/tiles/{layer}/{z}/{x}/{y}.png")
    def tile(layer: str, z: int, x: int, y: int, v: str = "0", a: str = ""):
        if ctx.tiles is None or z < 5 or z > 19 or (a and a != ctx.name):
            return Response(content=b"", status_code=204)
        png = ctx.tiles.render(layer, z, x, y, v)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/")
    def index():
        return FileResponse(config.STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
    # Mess-App (PWA) – für Tablets per HTTPS bereitstellen, siehe README
    app.mount("/mapper", _NoDotFiles(directory=config.MAPPER_DIR, html=True), name="mapper")
    return app


class _NoDotFiles(StaticFiles):
    """Liefert keine versteckten Dateien/Ordner (z. B. .git, .github) aus."""

    async def get_response(self, path, scope):
        if any(part.startswith(".") for part in Path(path).parts):
            raise HTTPException(404)
        return await super().get_response(path, scope)


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    if isinstance(d, float) and not math.isfinite(d):
        return None
    if isinstance(d, (np.floating,)):
        return None if not np.isfinite(d) else float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


# ------------------------------------------------------------------ Last ----

def _sample_users(grid, boundary, n, seed):
    """Zufällige Standorte innerhalb der Gebietsgrenze – bevorzugt auf Gebäuden; gibt es im
    Raster keine Einzelgebäude (große Gebiete), auf Bauflächen/überall."""
    from shapely import contains_xy, transform as shp_transform
    from shapely.geometry import shape
    from shapely.ops import unary_union

    t = Transformer.from_crs(4326, 25832, always_xy=True)
    g = unary_union([shape(f["geometry"]) for f in boundary["features"]])
    g = shp_transform(g, lambda xy: np.column_stack(t.transform(xy[:, 0], xy[:, 1])))
    minx, miny, maxx, maxy = g.bounds
    rng = np.random.default_rng(seed)
    m = grid.meta
    pts = []
    tries = 0
    any_cell = False
    while len(pts) < n and tries < 400:
        tries += 1
        if tries == 60 and not pts:
            any_cell = True  # keine Gebäude im Raster
        xs = rng.uniform(minx, maxx, 20000)
        ys = rng.uniform(miny, maxy, 20000)
        inside = contains_xy(g, xs, ys)
        xs, ys = xs[inside], ys[inside]
        r, c = m.to_rc(xs, ys)
        ok = (r >= 0) & (r < m.ny - 0.5) & (c >= 0) & (c < m.nx - 0.5)
        xs, ys, r, c = xs[ok], ys[ok], r[ok], c[ok]
        ri = np.clip(np.round(r).astype(int), 0, m.ny - 1)
        ci = np.clip(np.round(c).astype(int), 0, m.nx - 1)
        b = np.ones(len(xs), bool) if any_cell else np.asarray(grid.bld[ri, ci]) > 0
        for x, y in zip(xs[b], ys[b]):
            pts.append((x, y))
            if len(pts) >= n:
                break
    return pts


def _place_users(grid, boundary, n, seed, outdoor_share):
    """Endgeräte: Anteil ``outdoor_share`` draußen (nächste freie Zelle <= 30 m neben einem
    Gebäude, also Straße/Garten), Rest in Gebäuden. Rückgabe [(x, y, draußen)]."""
    from ..calibration import _snap_outdoor

    rng = np.random.default_rng(seed + 1)
    out = []
    for x, y in _sample_users(grid, boundary, n, seed):
        if rng.random() < outdoor_share:
            s = _snap_outdoor(grid, x, y, max_m=30.0)
            if s is not None:
                out.append((s[0], s[1], True))
                continue
        r, c = grid.meta.to_rc(x, y)
        inb = grid.bld[int(round(float(r))), int(round(float(c)))] > 0
        out.append((x, y, not inb))
    return out


def _run_load(cov, store, grid, body, boundary):
    from ..sim.traffic import LoadParams, simulate

    p = store.project
    lp = LoadParams(**{k: type(getattr(LoadParams(), k))(v) for k, v in body.items() if hasattr(LoadParams, k)})
    meta = cov.result_meta
    reps_by = {r.id: r for r in p.repeaters}
    ids = [i for i in meta["repeaters"] if i in reps_by]
    reps = [reps_by[i] for i in ids]
    caches = [E.CachedLoss(Path(c)) for i, c in zip(meta["repeaters"], meta["cache"]) if i in reps_by]
    nr = len(reps)
    sens = sensitivity_dbm(p.radio.sf, p.radio.bw_khz, p.radio.nf_db)
    rr = np.full((nr, nr), np.nan)
    idx = {r.id: i for i, r in enumerate(reps)}
    for lk in cov.links():
        if lk["a"] in idx and lk["b"] in idx:
            a, b = idx[lk["a"]], idx[lk["b"]]
            rr[a, b] = lk["rssi_ab"]
            rr[b, a] = lk["rssi_ba"]
    users = _place_users(grid, boundary, lp.users, lp.seed, lp.outdoor_share)
    k = meta["k"]
    m = grid.meta
    ur = np.full((len(users), nr), np.nan)
    node = p.node
    for u, (x, y, _o) in enumerate(users):
        r = int((m.y1 - y) / (m.res * k))
        c = int((x - m.x0) / (m.res * k))
        for j, cl in enumerate(caches):
            L = cl.at(r, c) + p.stats.calib_offset_db
            if np.isfinite(L):
                ur[u, j] = node.tx_dbm + node.gain_dbi - node.body_loss_db - reps[j].cable_db - L
    res = simulate(lp, p.radio, rr, ur, sens, [r.name for r in reps])
    for j, r in enumerate(reps):
        res["repeaters"][j]["id"] = r.id
    outdoor = np.array([u[2] for u in users], bool)
    cov_u = (np.nan_to_num(ur, nan=-999) >= sens).any(axis=1)
    res["users_outdoor"] = int(outdoor.sum())
    res["users_indoor"] = int((~outdoor).sum())
    res["covered_outdoor"] = int((cov_u & outdoor).sum())
    res["covered_indoor"] = int((cov_u & ~outdoor).sum())
    return res


def _write_load_layer(cov, store, res):
    """Kanalbelegung je Ort = Summe der Sendezeitanteile aller hörbaren Repeater [%]."""
    meta = cov.result_meta
    p = store.project
    rd = results_dir(p)
    ny, nx = meta["ny"], meta["nx"]
    sens = meta["sens_dbm"]
    duty = {r["id"]: r["duty"] for r in res["repeaters"]}
    reps = {r.id: r for r in p.repeaters}
    acc = np.lib.format.open_memmap(rd / "_load.npy", mode="w+", dtype=np.float32, shape=(ny, nx))
    for i in range(0, ny, 2000):
        acc[i:i + 2000] = 0
    for rid, cp in zip(meta["repeaters"], meta["cache"]):
        if rid not in duty or rid not in reps:
            continue
        cl = E.CachedLoss(Path(cp))
        rep = reps[rid]
        h, w = cl.shape
        h = min(h, ny - cl.r0)
        w = min(w, nx - cl.c0)
        for a in range(0, h, 512):
            b = min(a + 512, h)
            L = cl.rows(a, b)[:, :w] + p.stats.calib_offset_db
            heard = np.isfinite(L) & (rep.tx_dbm - rep.cable_db + p.node.gain_dbi - p.node.body_loss_db - L >= sens)
            sl = (slice(cl.r0 + a, cl.r0 + b), slice(cl.c0, cl.c0 + w))
            acc[sl] = np.asarray(acc[sl]) + heard * duty[rid]
    out = np.lib.format.open_memmap(rd / "_load_u8.npy", mode="w+", dtype=np.uint8, shape=(ny, nx))
    for i in range(0, ny, 2000):
        out[i:i + 2000] = np.clip(np.round(np.asarray(acc[i:i + 2000]) * 100), 0, 254).astype(np.uint8)
    out.flush()
    del out, acc
    (rd / "_load_u8.npy").replace(rd / "load.npy")
    (rd / "_load.npy").unlink(missing_ok=True)


def _load_load_layer(cov, tiles):
    if not cov.result_meta:
        tiles.load_arr = None
        return
    f = results_dir(cov.store.project) / "load.npy"
    if f.exists():
        a = np.load(f, mmap_mode="r")
        if a.shape == (cov.result_meta["ny"], cov.result_meta["nx"]):
            tiles.load_arr = a
            return
    tiles.load_arr = None
