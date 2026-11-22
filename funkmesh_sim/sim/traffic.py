"""Auslastungssimulation (ereignisgesteuert) für ein MeshCore-Netz.

Modell:
* Endgeräte (Companions) werden zufällig auf Gebäude im Gebiet verteilt (Menschen sind dort,
  wo Häuser stehen). Ihre Pegel zu jedem Repeater stammen aus den berechneten Verlustrastern.
* Flood-Pakete (Kanalnachrichten, Adverts, Erstzustellung von Direktnachrichten) werden von
  jedem Repeater, der sie erstmals fehlerfrei empfängt, genau einmal wiederholt
  (MeshCore: Duplikaterkennung per Paket-Hash, Pfad wächst je Hop um 1 Byte).
* Wiederholung nach Zufallsverzögerung ``U(0, 5) * txdelay * Airtime``, vorher
  Listen-before-Talk (Kanal belegt -> Backoff), danach Sendepause ``af * Airtime``
  (MeshCore Airtime-Faktor).
* Empfang scheitert bei Halbduplex (Empfänger sendet selbst) oder Kollision: ein zeitlich
  überlappendes Signal ist nicht mindestens ``capture_db`` schwächer (LoRa-Capture-Effekt).
* Ergebnis je Repeater: Sendezeit-Anteil (Duty Cycle, EU-Grenze 10 % in 869,4–869,65 MHz),
  Kanalbelegung, Kollisionsverluste; global: Flood-Reichweite und Zustellquote an Endgeräte.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass

import numpy as np

from ..radio.lora import airtime_s


@dataclass
class LoadParams:
    users: int = 200
    outdoor_share: float = 0.3        # Anteil Endgeräte draußen (Rest in Gebäuden)
    chan_msgs_per_user_h: float = 2.0
    dm_per_user_h: float = 1.0
    dm_flood_share: float = 0.3       # Anteil Direktnachrichten ohne bekannten Pfad (-> Flood)
    user_adverts_per_h: float = 0.25
    repeater_advert_h: float = 12.0   # Flood-Advert-Intervall der Repeater
    msg_bytes: int = 90
    advert_bytes: int = 140
    ack_bytes: int = 24
    txdelay: float = 0.5
    airtime_factor: float = 1.0
    capture_db: float = 6.0
    max_hops: int = 64
    hours: float = 1.0
    seed: int = 1
    duty_limit: float = 0.10


def simulate(lp: LoadParams, radio, rr_rssi: np.ndarray, ur_rssi: np.ndarray, sens: float,
             names: list[str]) -> dict:
    """rr_rssi[i, j]: Pegel von Repeater i an Repeater j [dBm] (NaN = keine Verbindung).
    ur_rssi[u, j]: Pegel Endgerät u -> Repeater j (Uplink); Downlink wird symmetrisch
    mit gleicher Matrix angenähert (Leistungsdifferenz als Offset in ``ur_down``)."""
    rng = np.random.default_rng(lp.seed)
    nr = rr_rssi.shape[0]
    nu = ur_rssi.shape[0]
    T = lp.hours * 3600.0
    sf, bw, cr, pre = radio.sf, radio.bw_khz, radio.cr, radio.preamble

    def air(nbytes):
        return airtime_s(int(nbytes), sf, bw, cr, pre)

    # Sender-Indizes: 0..nr-1 Repeater, nr..nr+nu-1 Endgeräte
    # Pegelmatrix Sender -> Repeater
    S2R = np.full((nr + nu, nr), -999.0)
    S2R[:nr] = np.nan_to_num(rr_rssi, nan=-999.0)
    np.fill_diagonal(S2R[:nr], -999.0)
    S2R[nr:] = np.nan_to_num(ur_rssi, nan=-999.0)

    # ---- Paketerzeugung ----
    origins = []  # (t, sender, bytes, kind)
    lam_u = lp.chan_msgs_per_user_h + lp.dm_per_user_h * lp.dm_flood_share + lp.user_adverts_per_h
    for u in range(nu):
        nmsg = rng.poisson(lam_u * lp.hours)
        for t in rng.uniform(0, T, nmsg):
            r = rng.uniform(0, lam_u)
            if r < lp.chan_msgs_per_user_h:
                origins.append((t, nr + u, lp.msg_bytes, "msg"))
            elif r < lp.chan_msgs_per_user_h + lp.dm_per_user_h * lp.dm_flood_share:
                origins.append((t, nr + u, lp.msg_bytes, "dm"))
            else:
                origins.append((t, nr + u, lp.advert_bytes, "advert"))
    if lp.repeater_advert_h > 0:
        for i in range(nr):
            per = lp.repeater_advert_h * 3600
            t = rng.uniform(0, per)
            while t < T:
                origins.append((t, i, lp.advert_bytes, "advert"))
                t += per

    ev = []
    seq = 0
    for k, (t, s, b, kind) in enumerate(origins):
        heapq.heappush(ev, (t, seq, "tx_req", (s, k, b, 0, 0)))
        seq += 1

    tx_log = []          # (t0, t1, sender, pkt, hops)
    active = []          # Indizes laufender Sendungen
    seen = np.zeros((len(origins), nr), bool)
    busy_until = np.zeros(nr + nu)
    next_allowed = np.zeros(nr + nu)
    st_tx = np.zeros(nr)
    st_air = np.zeros(nr)
    st_rx_ok = np.zeros(nr)
    st_rx_coll = np.zeros(nr)
    st_rx_hd = np.zeros(nr)
    st_defer = np.zeros(nr)
    pending_rx = []      # (t1, tx_idx)

    def audible(i_tx, j):
        return S2R[tx_log[i_tx][2], j] >= sens

    while ev:
        t, _, kind, data = heapq.heappop(ev)
        if t > T * 1.2:
            break
        # abgelaufene Sendungen entfernen
        active = [a for a in active if tx_log[a][1] > t]
        if kind == "tx_req":
            s, pkt, nbytes, hops, tries = data
            ta = air(nbytes + hops)
            # LBT: Kanal am Sender belegt?
            if s < nr:
                ch_busy = any(S2R[tx_log[a][2], s] >= sens - 3 for a in active)
                if (ch_busy or t < next_allowed[s] or t < busy_until[s]) and tries < 20:
                    st_defer[s] += 1
                    delay = max(next_allowed[s] - t, 0) + rng.uniform(0, ta)
                    heapq.heappush(ev, (t + delay, seq, "tx_req", (s, pkt, nbytes, hops, tries + 1)))
                    seq += 1
                    continue
            idx = len(tx_log)
            tx_log.append((t, t + ta, s, pkt, hops))
            active.append(idx)
            busy_until[s] = t + ta
            next_allowed[s] = t + ta * (1 + lp.airtime_factor)
            if s < nr:
                st_tx[s] += 1
                st_air[s] += ta
                seen[pkt, s] = True
            heapq.heappush(ev, (t + ta, seq, "tx_end", (idx, nbytes)))
            seq += 1
        else:
            idx, nbytes = data
            t0, t1, s, pkt, hops = tx_log[idx]
            # Empfang an allen Repeatern auswerten
            p_sig = S2R[s]
            cand = np.nonzero(p_sig >= sens)[0]
            if cand.size == 0:
                continue
            overl = [a for a in range(max(0, idx - 400), len(tx_log))
                     if a != idx and tx_log[a][0] < t1 and tx_log[a][1] > t0]
            for j in cand:
                if j == s:
                    continue
                # Halbduplex: j hat während des Pakets gesendet
                hd = any(tx_log[a][2] == j for a in overl)
                if hd:
                    st_rx_hd[j] += 1
                    continue
                pj = p_sig[j]
                coll = any(S2R[tx_log[a][2], j] > pj - lp.capture_db for a in overl)
                if coll:
                    st_rx_coll[j] += 1
                    continue
                st_rx_ok[j] += 1
                if not seen[pkt, j]:
                    seen[pkt, j] = True
                    if hops + 1 < lp.max_hops:
                        ta = air(nbytes + hops + 1)
                        d = rng.uniform(0, 5) * lp.txdelay * ta
                        heapq.heappush(ev, (t1 + d, seq, "tx_req", (j, pkt, nbytes, hops + 1, 0)))
                        seq += 1

    # ---- Auswertung ----
    reach = seen.mean(axis=1) if len(origins) else np.zeros(0)
    rep_busy = np.zeros(nr)
    if tx_log:
        arr = np.array([(a[0], a[1], a[2]) for a in tx_log])
        for j in range(nr):
            heard = arr[S2R[arr[:, 2].astype(int), j] >= sens]
            rep_busy[j] = _union_len(heard[:, 0], heard[:, 1]) / T if len(heard) else 0.0
    per_rep = []
    for j in range(nr):
        rx_tot = st_rx_ok[j] + st_rx_coll[j] + st_rx_hd[j]
        per_rep.append({
            "name": names[j], "tx": int(st_tx[j]), "duty": round(st_air[j] / T, 4),
            "busy": round(rep_busy[j], 4), "rx_ok": int(st_rx_ok[j]),
            "rx_lost": round((st_rx_coll[j] + st_rx_hd[j]) / rx_tot, 3) if rx_tot else 0.0,
            "deferrals": int(st_defer[j]),
            "over_limit": bool(st_air[j] / T > lp.duty_limit),
        })
    users_covered = int((ur_rssi >= sens).any(axis=1).sum()) if nu else 0
    return {
        "params": asdict(lp),
        "airtime_msg_ms": round(air(lp.msg_bytes) * 1000, 1),
        "airtime_advert_ms": round(air(lp.advert_bytes) * 1000, 1),
        "packets": len(origins),
        "transmissions": len(tx_log),
        "flood_reach_mean": round(float(reach.mean()), 3) if reach.size else 0,
        "flood_reach_p10": round(float(np.percentile(reach, 10)), 3) if reach.size else 0,
        "users": nu, "users_covered": users_covered,
        "repeaters": per_rep,
    }


def _union_len(a, b):
    if len(a) == 0:
        return 0.0
    o = np.argsort(a)
    a, b = a[o], b[o]
    tot = 0.0
    cs, ce = a[0], b[0]
    for s, e in zip(a[1:], b[1:]):
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs
