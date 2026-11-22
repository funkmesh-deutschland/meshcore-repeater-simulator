"""Plausibilitätstests des Ausbreitungskernels an synthetischen Szenen (reines Python)."""
import math

import numpy as np

from funkmesh_sim.radio import kernel as K

RES = 5.0
F = 869.618e6
LAM = 299792458.0 / F


def params(ztx_agl=30.0, ground=50.0):
    P = np.zeros(K.P_N)
    P[K.P_RES] = RES; P[K.P_STEP] = RES; P[K.P_TXX] = 2.5; P[K.P_TXY] = 7.5
    P[K.P_ZTX] = ground + ztx_agl; P[K.P_GTX] = ground; P[K.P_HRX] = 1.5; P[K.P_LAM] = LAM
    P[K.P_AE] = 6371e3 * 4 / 3; P[K.P_FGHZ] = F / 1e9; P[K.P_VGAM] = 0.25; P[K.P_VAM] = 29
    P[K.P_WALL] = 6; P[K.P_INT] = 0.5; P[K.P_BEL] = 14; P[K.P_MAXD] = 1e6; P[K.P_GANT] = 0
    P[K.P_NEL] = 1; P[K.P_EPSR] = 22; P[K.P_SIGMA] = 0.003; P[K.P_BLDMAX] = 80
    return P


def flat(n=2200):
    return (np.full((3, n), 50.0, np.float32), np.zeros((3, n), np.uint16), np.zeros((3, n), np.uint16))


def fspl(d):
    return 20 * math.log10(4 * math.pi * d / LAM)


def test_short_range_is_free_space():
    T, B, V = flat()
    L = K._cell_loss(T, B, V, params(), 2.5 + 300, 7.5)
    assert abs(L - fspl(300)) < 1.5


def test_flat_long_range_matches_spherical_earth():
    T, B, V = flat()
    P = params()
    for dk in (5.0, 10.0):
        L = K._cell_loss(T, B, V, P, 2.5 + dk * 1000, 7.5)
        ref = fspl(dk * 1000) + K._ldsph(dk, 30, 1.5, P[K.P_AE] / 1000, F / 1e9, LAM, 22, 0.003)
        assert abs(L - ref) < 2.0, (dk, L, ref)
        # zwischen Freiraum und 'plane earth + 15 dB'
        pe = 40 * math.log10(dk * 1000) - 20 * math.log10(30 * 1.5)
        assert fspl(dk * 1000) < L < pe + 15


def test_building_and_forest_add_loss():
    T, B, V = flat()
    P = params()
    x = 2.5 + 1000
    L0 = K._cell_loss(T, B, V, P, x, 7.5)
    B2 = B.copy(); c = int((x - 20) / RES); B2[:, c - 1:c + 1] = 100   # 10-m-Haus 20 m vor Empfänger
    Lb = K._cell_loss(T, B2, V, P, x, 7.5)
    assert 10 < Lb - L0 < 35
    V2 = V.copy(); V2[:, int((x - 100) / RES):int((x - 50) / RES)] = 150  # 50 m Wald, 15 m hoch
    Lv = K._cell_loss(T, B, V2, P, x, 7.5)
    assert 5 < Lv - L0 < 30


def test_indoor_adds_bel():
    T, B, V = flat()
    P = params()
    x = 2.5 + 1000
    B2 = B.copy(); c = int(x / RES); B2[:, c - 1:c + 2] = 80
    L0 = K._cell_loss(T, B, V, P, x, 7.5)
    Li = K._cell_loss(T, B2, V, P, x, 7.5)
    assert abs((Li - L0) - P[K.P_BEL]) < 3
