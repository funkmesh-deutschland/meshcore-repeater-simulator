"""Repeater von der öffentlichen MeshCore-Karte (map.meshcore.io) übernehmen.

API: ``GET https://map.meshcore.io/api/v1/nodes`` – Liste aller Knoten mit
``type`` (1 = Companion, 2 = Repeater, 3 = Room-Server, 4 = Sensor), ``adv_lat``/``adv_lon``,
``adv_name``, ``last_advert``, ``params`` (freq/bw/sf/cr). Höhenangaben gibt es dort nicht –
die Antennenhöhe muss nach dem Import gepflegt werden (Vorgabe: 10 m).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests

from . import config

URL = "https://map.meshcore.io/api/v1/nodes"
UA = {"User-Agent": "funkmesh-repeater-simulator/0.1"}


def fetch_nodes(max_cache_age_s: int = 3600) -> list[dict]:
    f = config.data_dir() / "meshcore_nodes.json"
    if f.exists() and time.time() - f.stat().st_mtime < max_cache_age_s:
        return json.loads(f.read_text())
    r = requests.get(URL, headers=UA, timeout=180)
    r.raise_for_status()
    f.write_bytes(r.content)
    return r.json()


def repeaters_in_bounds(bounds_lonlat, max_age_days: float | None = 30, types=(2,),
                        freq_mhz: float | None = None) -> list[dict]:
    w, s, e, n = bounds_lonlat
    now = datetime.now(timezone.utc)
    out = []
    for x in fetch_nodes():
        if x.get("type") not in types:
            continue
        lat, lon = x.get("adv_lat"), x.get("adv_lon")
        if lat is None or lon is None or (lat == 0 and lon == 0):
            continue
        if not (s <= lat <= n and w <= lon <= e):
            continue
        age = None
        try:
            t = datetime.fromisoformat(x["last_advert"].replace("Z", "+00:00"))
            age = (now - t).total_seconds() / 86400
        except Exception:  # noqa: BLE001
            pass
        if max_age_days is not None and (age is None or age > max_age_days or age < -1):
            continue
        prm = x.get("params") or {}
        if freq_mhz is not None and prm.get("freq") and abs(float(prm["freq"]) - freq_mhz) > 0.01:
            continue
        out.append({
            "public_key": x.get("public_key"),
            "name": (x.get("adv_name") or "?").strip(),
            "lat": float(lat), "lon": float(lon),
            "last_advert": x.get("last_advert"),
            "age_days": None if age is None else round(age, 1),
            "params": prm,
        })
    return out
