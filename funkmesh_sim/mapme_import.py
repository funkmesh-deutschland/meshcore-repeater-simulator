"""Messdaten und Antennenhöhen von mapme.sh (MeshCore-Coverage-Map der Community) übernehmen.

Endpunkte (öffentlich, JSON):

* ``/api/node/<pubkey>/coverage`` – je H3-Zelle (Auflösung 10, ~76 m Kante) RSSI [dBm] und Anzahl
  Samples, gemessen von Mapper-Geräten beim Empfang des Repeaters (Downlink).
* ``/api/repeater/<pubkey>/stats`` – Standort, ``elevation_m``, ``antenna_height_m``.

Achtung Hash-Kollisionen: Mapper ordnen Pakete meist über den **1-Byte-Pfad-Hash** zu
(Präfix des Public Keys). Unter ``/coverage`` landen daher Messungen *aller* Repeater mit gleichem
Präfix. Wir behalten eine Zelle nur, wenn unser Repeater der nächstgelegene aller bekannten
Repeater (map.meshcore.io) mit gleichem 1-Byte-Präfix ist und der nächste Konkurrent mindestens
``ambiguity_ratio``-mal weiter entfernt liegt, und wenn sie innerhalb ``max_dist_km`` liegt.
"""
from __future__ import annotations

import json
import math
import time

import requests

from . import config

BASE = "https://mapme.sh"
UA = {"User-Agent": "funkmesh-repeater-simulator/0.1 (+https://funkmesh.de)"}


def _cached_get(path: str, max_age_s: int = 6 * 3600):
    d = config.data_dir() / "mapme"
    d.mkdir(parents=True, exist_ok=True)
    f = d / (path.strip("/").replace("/", "_").replace("?", "_") + ".json")
    if f.exists() and time.time() - f.stat().st_mtime < max_age_s:
        return json.loads(f.read_text())
    r = requests.get(BASE + path, headers=UA, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    f.write_bytes(r.content)
    time.sleep(0.2)  # höflich bleiben
    return r.json()


def repeater_stats(pubkey: str):
    return _cached_get(f"/api/repeater/{pubkey}/stats")


def coverage(pubkey: str):
    return _cached_get(f"/api/node/{pubkey}/coverage")


def _dist_km(lat1, lon1, lat2, lon2):
    k = math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * k)


def known_repeaters():
    """Alle Repeater der MeshCore-Karte (für die Kollisionsprüfung), unabhängig vom Alter."""
    from .meshcore_import import fetch_nodes

    out = []
    for x in fetch_nodes():
        if x.get("type") == 2 and x.get("public_key") and x.get("adv_lat") and x.get("adv_lon"):
            out.append((x["public_key"].lower(), float(x["adv_lat"]), float(x["adv_lon"])))
    return out


def calibration_rows(repeaters, max_dist_km: float = 40.0, ambiguity_ratio: float = 2.0,
                     min_samples: int = 1, progress=None) -> tuple[list[dict], dict]:
    """Messzeilen (lat, lon, rssi, samples, repeater-id) für alle Repeater mit Public Key."""
    import h3

    others = known_repeaters()
    rows = []
    info = {"repeaters": [], "hexes_total": 0, "dropped_ambiguous": 0, "dropped_far": 0}
    reps = [r for r in repeaters if r.public_key]
    for i, r in enumerate(reps):
        if progress:
            progress(i / max(len(reps), 1), r.name)
        try:
            cov = coverage(r.public_key)
        except Exception as exc:  # noqa: BLE001
            info["repeaters"].append({"name": r.name, "error": str(exc)})
            continue
        hexes = (cov or {}).get("hexes") or []
        pre = r.public_key[:2].lower()
        rivals = [(la, lo) for pk, la, lo in others if pk[:2] == pre and pk != r.public_key.lower()]
        kept = 0
        for h in hexes:
            info["hexes_total"] += 1
            if h.get("rssi") is None or (h.get("samples") or 0) < min_samples:
                continue
            lat, lon = h3.cell_to_latlng(h["h"])
            d = _dist_km(lat, lon, r.lat, r.lon)
            if d > max_dist_km:
                info["dropped_far"] += 1
                continue
            if rivals and min(_dist_km(lat, lon, la, lo) for la, lo in rivals) < ambiguity_ratio * d:
                info["dropped_ambiguous"] += 1
                continue
            rows.append({"lat": lat, "lon": lon, "rssi": float(h["rssi"]), "samples": int(h.get("samples") or 1),
                         "repeater": r.id, "hex": h["h"]})
            kept += 1
        info["repeaters"].append({"name": r.name, "hexes": len(hexes), "kept": kept, "rivals_same_prefix": len(rivals)})
    return rows, info


def height_suggestions(repeaters) -> list[dict]:
    """Antennenhöhen laut mapme.sh im Vergleich zum Projekt."""
    out = []
    for r in repeaters:
        if not r.public_key:
            continue
        try:
            st = repeater_stats(r.public_key)
        except Exception:  # noqa: BLE001
            continue
        if not st or st.get("antenna_height_m") is None:
            continue
        out.append({"id": r.id, "name": r.name, "current_m": r.height_m,
                    "mapme_m": float(st["antenna_height_m"]), "elevation_m": st.get("elevation_m"),
                    "samples": ((st.get("coverage") or {}).get("total_samples"))})
    return out


def rows_to_csv(rows) -> str:
    lines = ["lat,lon,rssi,repeater,weight"]
    for r in rows:
        lines.append(f"{r['lat']:.6f},{r['lon']:.6f},{r['rssi']:.1f},{r['repeater']},{r['samples']}")
    return "\n".join(lines)


def repeaters_in_bounds(bounds_lonlat, max_age_days: float | None = 30) -> list[dict]:
    """Repeater-Standorte von mapme.sh (``/api/nodes``, nur aktive Knoten; ``ago`` = Sekunden
    seit dem letzten Empfang, negativ bei vorgehender Geräteuhr)."""
    w, s, e, n = bounds_lonlat
    r = requests.get(BASE + "/api/nodes", headers=UA, timeout=120,
                     params={"minLat": s, "maxLat": n, "minLon": w, "maxLon": e})
    r.raise_for_status()
    out = []
    for x in r.json().get("nodes", []):
        if x.get("type") != "repeater" or not x.get("pk"):
            continue
        try:
            lat, lon = float(x["lat"]), float(x["lon"])
        except (TypeError, ValueError, KeyError):
            continue
        if lat == 0 and lon == 0:
            continue
        try:
            age = max(float(x.get("ago") or 0), 0.0) / 86400
        except ValueError:
            age = None
        if max_age_days is not None and age is not None and age > max_age_days:
            continue
        out.append({"public_key": x["pk"].lower(), "name": (x.get("name") or "?").strip(), "lat": lat, "lon": lon,
                    "age_days": None if age is None else round(age, 1), "params": {}, "source": "mapme.sh"})
    return out
