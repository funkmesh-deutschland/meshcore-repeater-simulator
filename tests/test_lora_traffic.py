import numpy as np

from funkmesh_sim.project import Radio
from funkmesh_sim.radio.lora import airtime_s, bel_p2109, sensitivity_dbm
from funkmesh_sim.sim.traffic import LoadParams, simulate


def test_airtime_reference():
    # Semtech-Rechner: SF7/125 kHz/CR4-5, 8 Präambel, 20 Byte -> 56,6 ms
    assert abs(airtime_s(20, 7, 125, 5, 8) * 1000 - 56.6) < 0.2
    # SF12/125/CR4-5, LDRO aktiv, 20 Byte -> 1318,9 ms
    assert abs(airtime_s(20, 12, 125, 5, 8) * 1000 - 1318.9) < 1


def test_sensitivity_and_bel():
    assert abs(sensitivity_dbm(8, 62.5, 6) - (-130.0)) < 0.1
    assert 12 < bel_p2109(0.8686) < 16


def test_traffic_chain():
    # Kette aus 4 Repeatern, Nachbarn hören sich
    rr = np.full((4, 4), np.nan)
    for i in range(3):
        rr[i, i + 1] = rr[i + 1, i] = -100.0
    ur = np.full((20, 4), np.nan)
    ur[:, 0] = -110.0
    res = simulate(LoadParams(users=20, hours=0.5, chan_msgs_per_user_h=4), Radio(), rr, ur, -130.0, list("ABCD"))
    assert res["packets"] > 10
    assert res["flood_reach_mean"] > 0.6  # Floods erreichen das Kettenende
    assert all(r["duty"] < 0.5 for r in res["repeaters"])
