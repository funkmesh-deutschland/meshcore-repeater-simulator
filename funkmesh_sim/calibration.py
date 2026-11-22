"""Kalibrierung des Ausbreitungsmodells an Messungen (RSSI/SNR mit GPS-Position).

Eingabe: CSV mit einer Zeile je empfangenem (oder nicht empfangenem) Paket. Spaltennamen sind
flexibel (deutsch/englisch, Groß-/Kleinschreibung egal):

* ``lat`` / ``latitude`` / ``breite``, ``lon`` / ``lng`` / ``longitude`` / ``laenge``
* ``rssi`` [dBm] – leer = Paket **nicht** empfangen (für die Trefferquote)
* ``snr`` [dB] (optional) – bei SNR < 0 liegt das Nutzsignal unter dem Rauschen; der vom
  SX126x gemeldete RSSI enthält dann das Rauschen. Korrektur nach Semtech:
  ``P_signal = RSSI + SNR`` für SNR < 0.
* ``repeater`` / ``node`` / ``hash`` / ``name`` / ``id`` / ``public_key`` – Name, Projekt-ID oder
  Public-Key-Präfix (z. B. der 1-Byte-Hash ``a3`` aus einem MeshCore-Trace). Mehrdeutige Präfixe
  werden dem nächstgelegenen passenden Repeater zugeordnet.
* ``direction`` (optional): ``down`` = gemessen am Endgerät (Standard), ``up`` = gemessen am Repeater
* ``height`` (optional): Messhöhe über Grund [m]

Ablauf: Für jede Messung wird das Profil Repeater -> Messpunkt mit dem gleichen Kernel wie die
Karte gerechnet (Punktmodell, Empfänger im Freien: GPS-Punkte in Gebäudezellen werden auf die
nächste freie Zelle <= 10 m verschoben). Residuum = gemessen - vorhergesagt.

**Zensierung**: Nahe der Empfindlichkeitsgrenze werden nur zufällig starke Pakete empfangen –
das verfälscht den Mittelwert nach oben. Für Bias und Fit zählen daher nur Messungen, deren
Vorhersage mindestens ``censor_margin_db`` über der Empfindlichkeit liegt.

Fit: Gitter über Bewuchsdämpfung γ und Beugungsmodell; je Variante wird der Bias abgezogen und
die Reststreuung (RMSE) verglichen. Ergebnis: Pegel-Offset, γ, Beugungsmodell, σ (Ortsvariabilität).
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from .grid import RegionGrid
from .radio import engine as E
from .radio.lora import sensitivity_dbm

MAX_PLAUSIBLE_RMSE_DB = 12.0

ALIASES = {
    "lat": ("lat", "latitude", "breite", "breitengrad", "gps_lat", "y"),
    "lon": ("lon", "lng", "long", "longitude", "laenge", "länge", "laengengrad", "gps_lon", "x"),
    "rssi": ("rssi", "rssi_dbm", "pegel", "signal"),
    "snr": ("snr", "snr_db"),
    "rep": ("repeater", "node", "hash", "repeater_hash", "name", "id", "public_key", "pubkey", "prefix", "rpt",
            "knoten", "relais", "repeater_id"),
    "dir": ("direction", "richtung", "dir"),
    "h": ("height", "hoehe", "höhe", "height_m", "rx_height"),
    "w": ("weight", "gewicht", "samples", "count", "anzahl"),
    "acc": ("gps_accuracy", "accuracy", "gps_acc", "genauigkeit"),
    "setup": ("setup", "aufbau"),
}

# Zusatzverlust je Messaufbau (Spalte ``setup`` der Funkmesh-Mapper-App)
SETUP_LOSS_DB = {"fuss": 4.0, "fahrzeug_innen": 8.0, "aussenantenne": 0.0}


@dataclass
class Setup:
    """Messaufbau (Standard = Projekt-Endgerät)."""
    rx_gain_dbi: float = 0.0
    rx_extra_loss_db: float = 4.0     # Körper (zu Fuß) bzw. ~6–8 dB Fahrzeug ohne Außenantenne
    rx_height_m: float = 1.5
    tx_dbm_node: float = 22.0         # nur für direction=up
    censor_margin_db: float = 5.0
    max_gps_acc_m: float = 30.0       # Messungen mit schlechterem GPS verwerfen
    snap_outdoor: bool = True
    max_points: int = 3000


@dataclass
class Meas:
    lat: float
    lon: float
    rssi: float | None
    snr: float | None
    rep_key: str
    direction: str
    height: float | None
    weight: float = 1.0
    acc: float | None = None
    setup: str | None = None


def parse_csv(text: str) -> tuple[list[Meas], list[str]]:
    warn = []
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rd = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not rd.fieldnames:
        raise ValueError("CSV ohne Kopfzeile")
    cols = {}
    low = {f.strip().lower(): f for f in rd.fieldnames}
    for key, al in ALIASES.items():
        for a in al:
            if a in low:
                cols[key] = low[a]
                break
    for need in ("lat", "lon", "rep"):
        if need not in cols:
            raise ValueError(f"Spalte für '{need}' fehlt (gefunden: {', '.join(rd.fieldnames)})")
    if "rssi" not in cols:
        raise ValueError("Spalte 'rssi' fehlt")

    def num(v):
        if v is None:
            return None
        v = str(v).strip().replace(",", ".") if str(v).count(",") == 1 and "." not in str(v) else str(v).strip()
        if v == "" or v.lower() in ("nan", "none", "null", "-"):
            return None
        return float(v)

    out = []
    for i, row in enumerate(rd):
        try:
            lat, lon = num(row[cols["lat"]]), num(row[cols["lon"]])
            if lat is None or lon is None:
                continue
            out.append(Meas(lat, lon, num(row[cols["rssi"]]), num(row.get(cols.get("snr"))) if "snr" in cols else None,
                            str(row[cols["rep"]]).strip(),
                            (str(row.get(cols["dir"], "down")).strip().lower() if "dir" in cols else "down") or "down",
                            num(row.get(cols["h"])) if "h" in cols else None,
                            min(max(num(row.get(cols["w"])) or 1.0, 0.0), 20.0) if "w" in cols else 1.0,
                            num(row.get(cols["acc"])) if "acc" in cols else None,
                            (str(row.get(cols["setup"]) or "").strip().lower() or None) if "setup" in cols else None))
        except (ValueError, KeyError) as exc:
            if len(warn) < 10:
                warn.append(f"Zeile {i + 2}: {exc}")
    return out, warn


def match_repeater(key: str, lat: float, lon: float, reps) -> object | None:
    k = key.strip().lower()
    if not k:
        return None
    for r in reps:
        if r.id.lower() == k or r.name.strip().lower() == k:
            return r
    cand = [r for r in reps if r.public_key and r.public_key.lower().startswith(k)]
    if not cand and len(k) >= 3:
        cand = [r for r in reps if k in r.name.lower()]
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]
    return min(cand, key=lambda r: (r.lat - lat) ** 2 + ((r.lon - lon) * math.cos(math.radians(lat))) ** 2)


def _snap_outdoor(grid: RegionGrid, x, y, max_m=10.0):
    m = grid.meta
    r, c = m.to_rc(x, y)
    ri, ci = int(round(float(r))), int(round(float(c)))
    if not (0 <= ri < m.ny and 0 <= ci < m.nx):
        return None
    if grid.bld[ri, ci] == 0:
        return x, y
    k = int(max_m / m.res)
    best = None
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            rr, cc = ri + dr, ci + dc
            if 0 <= rr < m.ny and 0 <= cc < m.nx and grid.bld[rr, cc] == 0:
                d = dr * dr + dc * dc
                if best is None or d < best[0]:
                    best = (d, rr, cc)
    if best is None:
        return None
    return m.x0 + (best[2] + 0.5) * m.res, m.y1 - (best[1] + 0.5) * m.res


def run(grid: RegionGrid, project, text: str, setup: Setup, fit: bool = True, progress=None) -> dict:
    meas, warn = parse_csv(text)
    if not meas:
        raise ValueError("Keine gültigen Messungen gefunden")
    reps = [r for r in project.repeaters]
    p = project
    sens = sensitivity_dbm(p.radio.sf, p.radio.bw_khz, p.radio.nf_db)
    rows = []
    unmatched = {}
    bad_gps = 0
    for mm in meas:
        if mm.acc is not None and mm.acc > setup.max_gps_acc_m:
            bad_gps += 1
            continue
        rep = match_repeater(mm.rep_key, mm.lat, mm.lon, reps)
        if rep is None:
            unmatched[mm.rep_key] = unmatched.get(mm.rep_key, 0) + 1
            continue
        x, y = E.to_utm(mm.lat, mm.lon)
        if setup.snap_outdoor:
            s = _snap_outdoor(grid, x, y)
            if s is None:
                continue
            x, y = s
        pt = grid.sample_point(x, y)
        if not pt.get("inside"):
            continue
        h = mm.height if mm.height is not None else setup.rx_height_m
        sig = mm.rssi
        if sig is not None and mm.snr is not None and mm.snr < 0:
            sig = sig + mm.snr
        rows.append({"m": mm, "rep": rep, "x": x, "y": y, "z": pt["terrain"] + h, "sig": sig})
    if not rows:
        raise ValueError("Keine Messung ließ sich einem Repeater im Projekt zuordnen" +
                         (f" (unbekannt: {', '.join(list(unmatched)[:8])})" if unmatched else ""))
    if len(rows) > setup.max_points:
        idx = np.random.default_rng(0).choice(len(rows), setup.max_points, replace=False)
        rows = [rows[i] for i in sorted(idx)]
        warn.append(f"Auf {setup.max_points} zufällige Messungen reduziert")

    def budget(L, row):
        rep, mm = row["rep"], row["m"]
        extra = SETUP_LOSS_DB.get(mm.setup, setup.rx_extra_loss_db) if mm.setup else setup.rx_extra_loss_db
        if mm.direction.startswith("up"):
            return setup.tx_dbm_node + setup.rx_gain_dbi - extra - rep.cable_db - L
        return rep.tx_dbm - rep.cable_db + setup.rx_gain_dbi - extra - L

    items = [(r["rep"], r["x"], r["y"], r["z"]) for r in rows]
    env0 = project.environment()
    off0 = project.stats.calib_offset_db
    if progress:
        progress(0.02, "Profile extrahieren …")
    batch = E.StripBatch(grid, items)

    def evaluate(env):
        L = batch.evaluate(env)
        pred = np.array([budget(L[i] + off0, rows[i]) for i in range(len(rows))])
        return pred

    if progress:
        progress(0.05, "Vorhersage mit aktuellen Parametern …")
    pred0 = evaluate(env0)
    meas_sig = np.array([np.nan if r["sig"] is None else r["sig"] for r in rows])
    wts = np.array([r["m"].weight for r in rows], float)
    got = np.isfinite(meas_sig)
    okp = np.isfinite(pred0)
    use = got & okp & (pred0 >= sens + setup.censor_margin_db)
    report = _stats(meas_sig, pred0, use, got, okp, sens, rows, wts)
    if bad_gps:
        warn.append(f"{bad_gps} Messungen mit GPS-Ungenauigkeit > {setup.max_gps_acc_m:g} m verworfen")
    report.update({"n_rows": len(meas), "n_matched": len(rows), "unmatched": unmatched, "warnings": warn,
                   "sens_dbm": round(sens, 1), "setup": asdict(setup), "current": {
                       "offset_db": off0, "veg_gamma_db_m": env0.veg_gamma_db_m, "diffraction": env0.diffraction}})
    report["points"] = [{"lat": r["m"].lat, "lon": r["m"].lon, "rep": r["rep"].name,
                         "meas": None if not got[i] else round(float(meas_sig[i]), 1),
                         "pred": None if not okp[i] else round(float(pred0[i]), 1),
                         "used": bool(use[i])} for i, r in enumerate(rows)]

    if fit and use.sum() >= 20:
        cands = []
        grid_g = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
        modes = ["delta_bullington", "deygout"]
        tot = len(grid_g) * len(modes)
        k = 0
        for mode in modes:
            for g in grid_g:
                k += 1
                if progress:
                    progress(0.1 + 0.85 * k / tot, f"Fit: {mode}, γ = {g}")
                env = replace(env0, veg_gamma_db_m=g, diffraction=mode)
                pr = evaluate(env)
                ok = use & np.isfinite(pr)
                res = meas_sig[ok] - pr[ok]
                w = wts[ok]
                bias = float(np.average(res, weights=w))
                rmse = float(np.sqrt(np.average((res - bias) ** 2, weights=w)))
                cands.append({"diffraction": mode, "veg_gamma_db_m": g, "bias_db": round(bias, 2),
                              "rmse_db": round(rmse, 2), "n": int(ok.sum())})
        best = min(cands, key=lambda c: c["rmse_db"])
        report["fit"] = {
            "candidates": cands,
            "best": best,
            "suggest": {
                "calib_offset_db": round(off0 - best["bias_db"], 1),
                "veg_gamma_db_m": best["veg_gamma_db_m"],
                "diffraction": best["diffraction"],
                "sigma_outdoor_db": round(max(best["rmse_db"], 2.0), 1),
            },
            # Plausibilität: realistische Ortsvariabilität liegt bei 5..10 dB. Deutlich mehr heißt,
            # die Daten passen nicht zum Messaufbau (Höhen, Zuordnung, Geräte) – nicht übernehmen.
            "reliable": best["rmse_db"] <= MAX_PLAUSIBLE_RMSE_DB,
            "max_plausible_rmse_db": MAX_PLAUSIBLE_RMSE_DB,
        }
    elif fit:
        report["fit"] = {"error": "Zu wenige verwertbare Messungen für einen Fit (mind. 20 über der Zensierungsgrenze)"}
    if progress:
        progress(1.0, "fertig")
    return report


def _stats(meas, pred, use, got, okp, sens, rows, wts=None):
    if wts is None:
        wts = np.ones(len(meas))
    res = meas[use] - pred[use]
    w = wts[use]
    out = {"n_used": int(use.sum()), "n_received": int(got.sum()), "n_not_received": int((~got).sum()),
           "weighted": bool(np.any(wts != 1.0))}
    if res.size:
        b = float(np.average(res, weights=w))
        out.update({"bias_db": round(b, 2), "std_db": round(float(np.sqrt(np.average((res - b) ** 2, weights=w))), 2),
                    "rmse_db": round(float(np.sqrt(np.average(res ** 2, weights=w))), 2),
                    "mae_db": round(float(np.average(np.abs(res), weights=w)), 2),
                    # robust gegen Ausreißer (Mapper auf Türmen, falsche Zuordnung, ...)
                    "median_db": round(float(np.median(res)), 2),
                    "mad_sigma_db": round(float(1.4826 * np.median(np.abs(res - np.median(res)))), 2)})
        # nach Repeater
        per = {}
        for i in np.nonzero(use)[0]:
            n = rows[i]["rep"].name
            per.setdefault(n, []).append(meas[i] - pred[i])
        out["per_repeater"] = sorted([{"name": n, "n": len(v), "bias_db": round(float(np.mean(v)), 1),
                                       "std_db": round(float(np.std(v)), 1)} for n, v in per.items()],
                                     key=lambda d: -d["n"])
        # nach Entfernung
        dist = np.array([math.dist(E.to_utm(r["rep"].lat, r["rep"].lon), (r["x"], r["y"])) for r in rows])
        bins = [0, 500, 1000, 2000, 5000, 10000, 1e9]
        db = []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = use & (dist >= a) & (dist < b)
            if sel.sum():
                rr = meas[sel] - pred[sel]
                db.append({"from_m": a, "to_m": None if b > 1e8 else b, "n": int(sel.sum()),
                           "bias_db": round(float(rr.mean()), 1), "std_db": round(float(rr.std()), 1)})
        out["per_distance"] = db
    # Trefferquote Versorgungsvorhersage (empfangen vs. Vorhersage über Empfindlichkeit)
    valid = okp
    tp = int((valid & got & (pred >= sens)).sum())
    fn = int((valid & got & (pred < sens)).sum())
    fp = int((valid & ~got & (pred >= sens)).sum())
    tn = int((valid & ~got & (pred < sens)).sum())
    out["detection"] = {"received_predicted": tp, "received_not_predicted": fn,
                        "lost_predicted": fp, "lost_not_predicted": tn}
    return out


# ------------------------------------------------------- Messdatensätze ----

NORM_COLS = ["lat", "lon", "rssi", "snr", "repeater", "direction", "height", "weight", "gps_accuracy", "setup"]


def _csv_line(vals) -> str:
    out = []
    for v in vals:
        if v is None:
            out.append("")
            continue
        s = f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
        out.append('"' + s.replace('"', '""') + '"' if any(c in s for c in ',;"\n') else s)
    return ",".join(out)


def normalize_upload(text: str, filename: str = "") -> tuple[str, int, list[str]]:
    """Beliebige Messdatei (CSV mit flexiblen Spalten oder JSON-Export der Funkmesh-Mapper-App)
    in ein einheitliches CSV überführen. Rückgabe (csv, Zeilen, Warnungen)."""
    t = text.lstrip("﻿").strip()
    if filename.lower().endswith(".json") or t.startswith("{") or t.startswith("["):
        import json

        try:
            d = json.loads(t)
        except ValueError as exc:
            raise ValueError(f"JSON nicht lesbar: {exc}")
        rows = d.get("rows") if isinstance(d, dict) else d
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise ValueError("JSON enthält keine Messzeilen (erwartet Funkmesh-Mapper-Export mit 'rows')")
        keys = sorted({k for r in rows for k in r})
        t = "\n".join([_csv_line(keys)] + [_csv_line([r.get(k) for k in keys]) for r in rows])
    meas, warn = parse_csv(t)
    if not meas:
        raise ValueError("Keine gültigen Messungen gefunden")
    lines = [",".join(NORM_COLS)]
    for m in meas:
        lines.append(_csv_line([m.lat, m.lon, m.rssi, m.snr, m.rep_key, m.direction, m.height, m.weight, m.acc, m.setup]))
    return "\n".join(lines), len(meas), warn
