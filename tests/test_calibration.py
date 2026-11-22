import numpy as np
import pytest

from funkmesh_sim import calibration as C
from funkmesh_sim.radio.engine import Repeater


def reps():
    return [Repeater(id="r1", name="Funkmesh.de - BIL-01", lat=51.9769, lon=7.28846, public_key="a3ff01"),
            Repeater(id="r2", name="Relais Baumberge", lat=51.9602, lon=7.36961, public_key="a3bb02"),
            Repeater(id="r3", name="Other", lat=51.70, lon=7.10, public_key="c4")]


def test_parse_semicolon_german():
    txt = "Breite;Länge;RSSI;SNR;Repeater\n51,97;7,29;-101;5,5;BIL-01\n51,96;7,30;;;a3\n"
    m, w = C.parse_csv(txt)
    assert len(m) == 2 and m[0].lat == pytest.approx(51.97) and m[0].snr == 5.5
    assert m[1].rssi is None


def test_match_prefix_picks_nearest():
    r = reps()
    assert C.match_repeater("a3", 51.976, 7.29, r).id == "r1"   # nah an BIL-01
    assert C.match_repeater("a3", 51.960, 7.37, r).id == "r2"   # nah am Baumberge-Relais
    assert C.match_repeater("bil-01", 51.9, 7.3, r).id == "r1"  # Namensteil
    assert C.match_repeater("zz", 51.9, 7.3, r) is None


def test_fit_recovers_synthetic_truth():
    from funkmesh_sim.grid import RegionGrid
    if not RegionGrid.exists("coesfeld"):
        pytest.skip("Region coesfeld nicht aufbereitet")
    from dataclasses import replace

    from funkmesh_sim.project import Project
    from funkmesh_sim.radio import engine as E

    g = RegionGrid.open("coesfeld")
    p = Project()
    p.repeaters = reps()[:2]
    for r in p.repeaters:
        r.height_m = 15
    rng = np.random.default_rng(7)
    setup = C.Setup()
    truth = replace(p.environment(), veg_gamma_db_m=0.4)
    rows, items = [], []
    while len(items) < 400:
        rep = p.repeaters[len(items) % 2]
        lat = rep.lat + rng.uniform(-0.04, 0.04)
        lon = rep.lon + rng.uniform(-0.06, 0.06)
        x, y = E.to_utm(lat, lon)
        s = C._snap_outdoor(g, x, y)
        if s is None:
            continue
        x, y = s
        z = g.sample_point(x, y)["terrain"] + 1.5
        items.append((rep, x, y, z))
        rows.append((lat, lon, rep))
    L = E.StripBatch(g, items).evaluate(truth)
    lines = ["lat,lon,rssi,repeater"]
    for (lat, lon, rep), l in zip(rows, L):
        rssi = rep.tx_dbm - rep.cable_db + setup.rx_gain_dbi - setup.rx_extra_loss_db - l + 3.0 + rng.normal(0, 4)
        lines.append(f"{lat},{lon},{'' if rssi < -128 else round(rssi, 1)},{rep.name}")
    rep_ = C.run(g, p, "\n".join(lines), setup, fit=True)
    best = rep_["fit"]["best"]
    assert best["diffraction"] == "delta_bullington"
    assert abs(best["veg_gamma_db_m"] - 0.4) <= 0.1, best
    assert abs(rep_["fit"]["suggest"]["calib_offset_db"] + 3.0) < 1.5, rep_["fit"]["suggest"]
    assert 3.0 < best["rmse_db"] < 5.5


def test_mapper_csv_columns():
    txt = ("time,lat,lon,gps_accuracy,rssi,snr,repeater,direction,height,setup,kind\n"
           "2026-11-22T10:00:00Z,51.97,7.29,8,-104,3.5,76,down,1.5,fahrzeug_innen,rx\n"
           "2026-11-22T10:00:30Z,51.97,7.29,80,-99,4,76,down,1.5,fuss,rx\n"
           "2026-11-22T10:01:00Z,51.97,7.29,6,,,766acf,down,1.5,fuss,trace_timeout\n")
    m, _ = C.parse_csv(txt)
    assert len(m) == 3
    assert m[0].setup == "fahrzeug_innen" and m[0].acc == 8 and m[0].rssi == -104
    assert m[1].acc == 80
    assert m[2].rssi is None and m[2].rep_key == "766acf"
    assert C.SETUP_LOSS_DB[m[0].setup] == 8.0
