"""Kommandozeile: ``python -m funkmesh_sim prepare|serve|areas``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from . import config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="funkmesh-sim", description="Funkmesh Repeater-Simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="Gebiet anlegen und Geodaten aufbereiten (fortsetzbar)")
    p.add_argument("name", help="Gebietsname, z. B. coesfeld (mit regions/<name>.geojson) oder frei mit --bbox")
    p.add_argument("--bbox", help="west,süd,ost,nord in Grad (statt Grenzdatei); 'de' = ganz Deutschland")
    p.add_argument("--title", default=None)
    p.add_argument("--res", type=int, default=None, help="Rasterweite in m: 5, 10, 25, 50, 100")
    p.add_argument("--buffer", type=float, default=5000, help="Rand um das Gebiet in m")
    p.add_argument("--no-nrw", action="store_true", help="NRW-1-m-Daten nicht verwenden")
    p.add_argument("--no-osm", action="store_true", help="keine OSM-Einzelgebäude")
    p.add_argument("--workers", type=int, default=None)
    sub.add_parser("areas", help="vorhandene Gebiete auflisten")
    s = sub.add_parser("serve", help="Web-Oberfläche starten")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--region", default=None, help="Gebiet beim Start aktivieren (Standard: zuletzt benutztes)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "prepare":
        from . import areas as A

        boundary = None
        if a.bbox == "de":
            bbox = A.GERMANY_BBOX
        elif a.bbox:
            bbox = tuple(float(v) for v in a.bbox.split(","))
        else:
            f = config.REGIONS_DIR / f"{a.name}.geojson"
            if not f.exists():
                sys.exit(f"Weder --bbox noch Grenzdatei {f}")
            from shapely.geometry import shape

            boundary = json.loads(f.read_text())
            bbox = shape(boundary["features"][0]["geometry"]).bounds
        res = a.res or (50 if a.bbox == "de" else 10)
        spec = A.AreaSpec(title=a.title or a.name, bbox=bbox, res=res, buffer_m=a.buffer, use_nrw=not a.no_nrw,
                          use_osm=not a.no_osm, boundary=boundary, name=A.slug(a.name))
        pl = A.plan(spec)
        pl.pop("tiles")
        print(json.dumps(pl, indent=1, ensure_ascii=False), flush=True)
        last = [0.0]

        def prog(f, msg):
            if time.time() - last[0] > 5 or f >= 1:
                last[0] = time.time()
                print(f"[{f * 100:5.1f} %] {msg}", flush=True)

        ok = A.prepare(spec, prog, workers=a.workers)
        print("fertig" if ok else "unvollständig", flush=True)
    elif a.cmd == "areas":
        from . import areas as A

        for x in A.list_areas():
            print(f"{x['name']:<24} {x['title']:<30} {x['res']:>5g} m  {x['km2']:>7} km²  {x['disk_gb']:>6} GB  "
                  f"{'fertig' if x['complete'] else 'unvollständig'}")
    elif a.cmd == "serve":
        import uvicorn

        from .server.app import create_app

        uvicorn.run(create_app(a.region), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
