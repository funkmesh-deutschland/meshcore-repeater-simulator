"""Gebiete, deutschlandweite Datenaufbereitung und Messdatensätze (offline)."""
import json

import numpy as np

from funkmesh_sim import areas as A
from funkmesh_sim import calibration as C
from funkmesh_sim.data import generic as G


def test_slug_and_spec():
    assert A.slug("Kreis Lüdinghausen-Süd") == "kreis-luedinghausen-sued"
    sp = A.AreaSpec("Deutschland", A.GERMANY_BBOX, res=50)
    assert sp.name == "deutschland"


def test_plan_alignment_without_nrw():
    sp = A.AreaSpec("Test", (10.5, 52.46, 10.6, 52.52), res=10, buffer_m=0, use_nrw=False)
    pl = A.plan(sp)
    assert pl["x0"] % 50 == 0 and pl["y1"] % 50 == 0
    assert pl["nx"] > 600 and pl["ny"] > 600 and not pl["use_nrw"]
    assert pl["use_osm"]  # klein + 10 m
    big = A.plan(A.AreaSpec("DE", A.GERMANY_BBOX, res=50, buffer_m=0))
    assert not big["use_osm"] and big["disk_gb"] < 3


def test_osm_height_parsing():
    assert G._parse_height({"height": "12.5 m"}, 9) == 12.5
    assert G._parse_height({"building:levels": "3"}, 9) == 3 * 3.0 + 1.5
    assert G._parse_height({"building:levels": "2", "roof:levels": "1"}, 9) == 2 * 3.0 + 2.5 + 1.5
    assert G._parse_height({"building": "garage"}, 9) == 3.0
    assert G._parse_height({"building": "yes"}, 9) == 9


def test_fill_terrain_under_forest():
    # schiefe Ebene, in der Mitte ein 20 m hoher "Wald" im Oberflächenmodell
    y, x = np.mgrid[0:200, 0:200].astype(np.float32)
    terrain = 50 + 0.05 * x
    dsm = terrain.copy()
    forest = (abs(x - 100) < 30) & (abs(y - 100) < 30)
    dsm[forest] += 20
    filled = G._fill_terrain(dsm, ~forest, 10.0)
    err = filled[forest] - terrain[forest]
    assert abs(err.mean()) < 0.5 and np.abs(err).max() < 2.0


def test_normalize_mapper_json_and_csv():
    rows = [{"time": "2026-01-01T10:00:00Z", "lat": 51.9, "lon": 7.2, "rssi": -101, "snr": 5.5, "repeater": "a3",
             "direction": "down", "height": 1.5, "setup": "fuss", "gps_accuracy": 6, "kind": "trace"},
            {"time": "2026-01-01T10:00:30Z", "lat": 51.91, "lon": 7.21, "rssi": None, "repeater": "a3", "kind": "timeout"}]
    csv_text, n, _ = C.normalize_upload(json.dumps({"format": "funkmesh-mapper/1", "rows": rows}), "x.json")
    assert n == 2
    meas, _ = C.parse_csv(csv_text)
    assert meas[0].rssi == -101 and meas[0].setup == "fuss" and meas[0].acc == 6
    assert meas[1].rssi is None
    csv2, n2, _ = C.normalize_upload("breite;länge;pegel;knoten\n51,9;7,2;-99;BIL-01\n", "eigene.csv")
    assert n2 == 1 and C.parse_csv(csv2)[0][0].rssi == -99
