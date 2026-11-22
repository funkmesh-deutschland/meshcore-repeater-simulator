"""LoRa-Kenngrößen: Empfindlichkeit, Rauschen, Airtime (Semtech AN1200.13 / SX126x-Datenblatt)."""
from __future__ import annotations

import math

# Demodulator-SNR-Grenze je Spreading-Faktor [dB] (SX126x)
SNR_MIN = {5: -2.5, 6: -5.0, 7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


def noise_floor_dbm(bw_khz: float, nf_db: float) -> float:
    return -174.0 + 10.0 * math.log10(bw_khz * 1e3) + nf_db


def sensitivity_dbm(sf: int, bw_khz: float, nf_db: float = 6.0) -> float:
    return noise_floor_dbm(bw_khz, nf_db) + SNR_MIN[int(sf)]


def airtime_s(payload_bytes: int, sf: int, bw_khz: float, cr: int, preamble: int = 16,
              explicit_header: bool = True, crc: bool = True) -> float:
    """Sendedauer eines LoRa-Pakets. ``cr`` = 5..8 (für 4/5 .. 4/8)."""
    tsym = (2 ** sf) / (bw_khz * 1e3)
    de = 1 if tsym > 0.016 else 0
    ih = 0 if explicit_header else 1
    crv = cr - 4
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * (1 if crc else 0) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (crv + 4), 0)
    return (preamble + 4.25) * tsym + n_payload * tsym


def bel_p2109(f_ghz: float, elevation_deg: float = 0.0, building: str = "traditional", p: float = 0.5) -> float:
    """Gebäudeeintrittsdämpfung nach ITU-R P.2109-2 (Median bei p = 0.5)."""
    from statistics import NormalDist

    if building == "thermally_efficient":
        r, s, t, u, v, w, x, y, z = 28.19, -3.00, 8.48, 13.5, 3.8, 27.8, -2.9, 9.4, -2.1
    else:
        r, s, t, u, v, w, x, y, z = 12.64, 3.72, 0.96, 9.6, 2.0, 9.1, -3.0, 4.5, -2.0
    lf = math.log10(f_ghz)
    lh = r + s * lf + t * lf * lf
    le = 0.212 * abs(elevation_deg)
    mu1 = lh + le
    mu2 = w + x * lf
    s1 = u + v * lf
    s2 = y + z * lf
    q = NormalDist().inv_cdf(p)
    a = q * s1 + mu1
    b = q * s2 + mu2
    c = -3.0
    return 10.0 * math.log10(10 ** (0.1 * a) + 10 ** (0.1 * b) + 10 ** (0.1 * c))


def antenna_elements(gain_dbi: float) -> float:
    """Äquivalente Anzahl Halbwellendipole einer Kollinear-Antenne für einen Gewinn."""
    return max(1.0, 10 ** ((gain_dbi - 2.15) / 10.0))
