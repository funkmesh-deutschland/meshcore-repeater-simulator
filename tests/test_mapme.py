import h3

from funkmesh_sim import mapme_import as M
from funkmesh_sim.radio.engine import Repeater


def test_filters_hash_collisions_and_distance(monkeypatch):
    me = Repeater(id="me", name="Mein", lat=51.98, lon=7.29, public_key="76aa")
    near = h3.latlng_to_cell(51.985, 7.30, 10)          # ~1 km von mir
    rival_side = h3.latlng_to_cell(52.20, 7.60, 10)     # nah am Konkurrenten mit gleichem Präfix
    far = h3.latlng_to_cell(52.60, 7.29, 10)            # ~70 km
    monkeypatch.setattr(M, "coverage", lambda pk: {"hexes": [
        {"h": near, "rssi": -90, "samples": 3}, {"h": rival_side, "rssi": -60, "samples": 9},
        {"h": far, "rssi": -110, "samples": 1}]})
    monkeypatch.setattr(M, "known_repeaters", lambda: [("76aa", 51.98, 7.29), ("76bb", 52.21, 7.61), ("11cc", 51.99, 7.30)])
    rows, info = M.calibration_rows([me])
    assert [r["hex"] for r in rows] == [near]
    assert info["dropped_ambiguous"] == 1 and info["dropped_far"] == 1
    assert rows[0]["samples"] == 3 and "weight" in M.rows_to_csv(rows).splitlines()[0]
