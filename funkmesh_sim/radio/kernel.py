"""Ausbreitungs-Kernel: Pfadverlust Sender -> jede Zielzelle über das Höhenprofil.

Derselbe Python-Code wird mit numba einmal für CUDA (GPU) und einmal für die CPU
(parallel) übersetzt. Pro Zielzelle wird das Profil Sender->Empfänger im Raster der
Geodaten abgetastet (Schrittweite = Rasterweite) und daraus berechnet:

1. **Beugung (ITU-R P.526, Deygout, bis zu 3 Kanten)** für drei Profilvarianten
   - P1: Gelände
   - P2: Gelände + Gebäude (LoD2)
   - P3: Gelände + Gebäude + Bewuchs (Bewuchs als undurchlässig)
2. **Erdkrümmung / Bodeneinfluss**: Delta-Bullington-Korrektur nach ITU-R P.1812/P.452
   (sphärische Erdbeugung ``Ldsph`` minus Bullington auf glatter Erde) mit
   effektiven Antennenhöhen über der Ausgleichsfläche des Geländes.
3. **Bewuchsdämpfung (ITU-R P.833)** entlang des direkten Strahls:
   ``A = Am * (1 - exp(-d * gamma / Am))``.
4. **Gebäudedurchdringung** des direkten Strahls: Wanddämpfung je Durchgang + Innendämpfung/m.
5. **Gebäudeeintritt (ITU-R P.2109)**, wenn der Empfänger (bzw. Sender) im Gebäude steht.

Kombination::

    L_über  = min(Deygout(P3), Deygout(P2) + A_Bewuchs)          # über Dächer/Baumkronen
    L_durch = Deygout(P1) + A_Bewuchs + A_Gebäude                 # durch Gebäude hindurch
    L_zus   = -10 log10(10^(-L_über/10) + 10^(-L_durch/10)) + max(Ldsph - Lbs, 0)
    Lb      = FSPL(d_3D) + L_zus + BEL - G_Antenne(Elevation)

Rückgabe: effektiver Pfadverlust [dB] inkl. Richtdiagramm der Repeater-Antenne.
"""
from __future__ import annotations

import math
import types

import numpy as np

# Indizes im Parametervektor P
P_RES = 0        # Rasterweite der Geodaten [m]
P_STEP = 1       # Profil-Schrittweite [m]
P_TXX = 2        # Sender x im Fenster [m, von links]
P_TXY = 3        # Sender y im Fenster [m, von oben]
P_ZTX = 4        # Antennenhöhe Sender [m ü. NHN]
P_GTX = 5        # Geländehöhe am Sender [m ü. NHN]
P_TXIN = 6       # Sender im Gebäude (1/0)
P_HRX = 7        # Empfängerhöhe über Grund [m]
P_LAM = 8        # Wellenlänge [m]
P_AE = 9         # effektiver Erdradius [m]
P_FGHZ = 10      # Frequenz [GHz]
P_VGAM = 11      # spezifische Bewuchsdämpfung [dB/m]
P_VAM = 12       # max. Bewuchsdämpfung Am [dB]
P_WALL = 13      # Dämpfung je Außenwand [dB]
P_INT = 14       # Innendämpfung [dB/m]
P_BEL = 15       # Gebäudeeintrittsdämpfung [dB]
P_ORES = 16      # Auflösung des Ausgaberasters [m]
P_OX0 = 17       # x des ersten Ausgabezellmittelpunkts im Fenster [m]
P_OY0 = 18       # y des ersten Ausgabezellmittelpunkts im Fenster [m]
P_MAXD = 19      # max. Entfernung [m]
P_GANT = 20      # Antennengewinn [dBi]
P_NEL = 21       # äquivalente Elementzahl (Kollinear)
P_TILT = 22      # elektrischer Downtilt [Grad]
P_DMODE = 23     # 0 = Delta-Bullington (P.1812, Standard), 1 = Deygout 3 Kanten
P_EPSR = 24      # rel. Permittivität Boden
P_SIGMA = 25     # Leitfähigkeit Boden [S/m]
P_BLDMAX = 26    # max. Durchdringungsdämpfung [dB]
P_FIXRX = 27     # >0: feste Empfänger-Antennenhöhe ü. NHN (Punkt-zu-Punkt), sonst Grund + HRX
P_ZRXABS = 28
P_N = 32

NAN = math.nan


def _J(v):
    """Knife-edge-Beugungsverlust J(v) nach ITU-R P.526 (Gl. 31)."""
    if v <= -0.78:
        return 0.0
    a = v - 0.1
    return 6.9 + 20.0 * math.log10(math.sqrt(a * a + 1.0) + a)


def _bil(T, x, y, res):
    ny = T.shape[0]
    nx = T.shape[1]
    fx = x / res - 0.5
    fy = y / res - 0.5
    if fx < 0.0:
        fx = 0.0
    if fy < 0.0:
        fy = 0.0
    if fx > nx - 1.001:
        fx = nx - 1.001
    if fy > ny - 1.001:
        fy = ny - 1.001
    c0 = int(fx)
    r0 = int(fy)
    tx = fx - c0
    ty = fy - r0
    a = T[r0, c0] * (1.0 - tx) + T[r0, c0 + 1] * tx
    b = T[r0 + 1, c0] * (1.0 - tx) + T[r0 + 1, c0 + 1] * tx
    return a * (1.0 - ty) + b * ty


def _near(A, x, y, res):
    ny = A.shape[0]
    nx = A.shape[1]
    c = int(x / res)
    r = int(y / res)
    if c < 0:
        c = 0
    if r < 0:
        r = 0
    if c > nx - 1:
        c = nx - 1
    if r > ny - 1:
        r = ny - 1
    return A[r, c]


def _ldft(adft, d, hte, hre, f, epsr, sigma):
    """Erste-Term-Beugung über sphärische Erde (ITU-R P.452-17 §4.2.2.2), vertikale Pol.
    adft [km], d [km], h [m], f [GHz]."""
    q = 18.0 * sigma / f
    kh = 0.036 * (adft * f) ** (-1.0 / 3.0) * ((epsr - 1.0) ** 2 + q * q) ** (-0.25)
    k = kh * math.sqrt(epsr * epsr + q * q)
    k2 = k * k
    k4 = k2 * k2
    beta = (1.0 + 1.6 * k2 + 0.67 * k4) / (1.0 + 4.5 * k2 + 1.53 * k4)
    x = 21.88 * beta * (f / (adft * adft)) ** (1.0 / 3.0) * d
    yfac = 0.9575 * beta * (f * f / adft) ** (1.0 / 3.0)
    if x >= 1.6:
        fx = 11.0 + 10.0 * math.log10(x) - 17.6 * x
    else:
        fx = -20.0 * math.log10(x) - 5.6488 * x ** 1.425
    gmin = 2.0 + 20.0 * math.log10(k)
    b = beta * yfac * hte
    if b > 2.0:
        gt = 17.6 * math.sqrt(b - 1.1) - 5.0 * math.log10(b - 1.1) - 8.0
    else:
        gt = 20.0 * math.log10(b + 0.1 * b * b * b)
    if gt < gmin:
        gt = gmin
    b = beta * yfac * hre
    if b > 2.0:
        gr = 17.6 * math.sqrt(b - 1.1) - 5.0 * math.log10(b - 1.1) - 8.0
    else:
        gr = 20.0 * math.log10(b + 0.1 * b * b * b)
    if gr < gmin:
        gr = gmin
    return -fx - gt - gr


def _ldsph(d, hte, hre, ap, f, lam, epsr, sigma):
    """Sphärische Erdbeugung Ldsph (ITU-R P.452-17 §4.2.2.1). d, ap [km]; h [m]."""
    dlos = math.sqrt(2.0 * ap) * (math.sqrt(0.001 * hte) + math.sqrt(0.001 * hre))
    if d >= dlos:
        return _ldft(ap, d, hte, hre, f, epsr, sigma)
    c = (hte - hre) / (hte + hre)
    m = 250.0 * d * d / (ap * (hte + hre))
    if m < 1e-9:
        return 0.0
    arg = 1.5 * c * math.sqrt(3.0 * m / ((m + 1.0) ** 3))
    if arg > 1.0:
        arg = 1.0
    if arg < -1.0:
        arg = -1.0
    b = 2.0 * math.sqrt((m + 1.0) / (3.0 * m)) * math.cos(math.pi / 3.0 + math.acos(arg) / 3.0)
    dse1 = d / 2.0 * (1.0 + b)
    dse2 = d - dse1
    hse = ((hte - 500.0 * dse1 * dse1 / ap) * dse2 + (hre - 500.0 * dse2 * dse2 / ap) * dse1) / d
    hreq = 17.456 * math.sqrt(max(dse1 * dse2 * lam / d, 0.0))
    if hse > hreq:
        return 0.0
    sq = math.sqrt(hte) + math.sqrt(hre)
    aem = 500.0 * (d / sq) ** 2
    ld = _ldft(aem, d, hte, hre, f, epsr, sigma)
    if ld < 0.0:
        return 0.0
    return (1.0 - hse / hreq) * ld


def _deygout_smooth(d, hte, hre, ae, lam, mode, n):
    """Deygout (gleiche Methode wie für das reale Profil) über glatter, gekrümmter Erde.
    d [m], hte/hre effektive Höhen [m], ae [m]. Wird für die Delta-Korrektur benötigt, damit
    streifender Bodeneinfluss nicht doppelt gezählt wird. n = Stützstellen wie reales Profil."""
    vm = -1e30
    pm = 0
    gp = 0.0
    for i in range(1, n):
        di = d * i / n
        dd = d - di
        e = di * dd / (2.0 * ae)
        zl = hte + (hre - hte) * i / n
        v = (e - zl) * math.sqrt(2.0 * d / (lam * di * dd))
        if v > vm:
            vm = v
            pm = i
            gp = e
    jp = _J(vm)
    if jp <= 0.0:
        return 0.0
    dp = d * pm / n
    lm = -1e30
    rm = -1e30
    for i in range(1, n):
        if i == pm:
            continue
        di = d * i / n
        dd = d - di
        e = di * dd / (2.0 * ae)
        if i < pm:
            zs = hte + (gp - hte) * di / dp
            v = (e - zs) * math.sqrt(2.0 * dp / (lam * di * (dp - di)))
            if v > lm:
                lm = v
        else:
            s = di - dp
            L2 = d - dp
            zs = gp + (hre - gp) * s / L2
            v = (e - zs) * math.sqrt(2.0 * L2 / (lam * s * dd))
            if v > rm:
                rm = v
    return _deygout(jp, _J(lm), _J(rm), d / 1000.0, mode)


def _pattern_db(theta, g, nel, tilt_deg):
    """Vertikaldiagramm einer Kollinear-Antenne (N Halbwellendipole, Abstand lambda/2).
    theta = Elevation [rad]. Rückgabe Gewinn [dBi] (Maximum = g)."""
    st = math.sin(theta)
    ct = math.cos(theta)
    if abs(ct) < 1e-6:
        el = 0.0
    else:
        el = abs(math.cos(0.5 * math.pi * st) / ct)
    af = 1.0
    if nel > 1.0001:
        psi = math.pi * (st + math.sin(tilt_deg * math.pi / 180.0))
        s = math.sin(0.5 * psi)
        if abs(s) > 1e-9:
            af = abs(math.sin(0.5 * nel * psi) / (nel * s))
    v = el * af
    floor = 10.0 ** (-30.0 / 20.0)
    if v < floor:
        v = floor
    return g + 20.0 * math.log10(v)


def _deygout(jp, jt, jr, dkm, mode):
    if jp <= 0.0:
        return 0.0
    if mode < 0.5:
        return jp + jt + jr
    return jp + (1.0 - math.exp(-jp / 6.0)) * (jt + jr + 10.0 + 0.04 * dkm)


def _bull_luc(stim, srim, vmax, zt, zr, d, lam):
    """Bullington-Ersatzkante (ITU-R P.526-15 §4.5.3 / P.1812-7), ohne Korrekturterm. Meter."""
    if stim < (zr - zt) / d:
        return _J(vmax)
    den = stim + srim
    if den <= 0.0:
        return _J(vmax)
    dbp = (zr - zt + srim * d) / den
    if dbp <= 0.0 or dbp >= d:
        return _J(vmax)
    vb = (zt + stim * dbp - (zt * (d - dbp) + zr * dbp) / d) * math.sqrt(2.0 * d / (lam * dbp * (d - dbp)))
    return _J(vb)


def _bull_smooth(d, hte, hre, ae, lam, n):
    """Bullington (ohne Korrektur) über glatter, gekrümmter Erde mit effektiven Höhen. Meter."""
    stim = -1e30
    srim = -1e30
    vmax = -1e30
    for i in range(1, n):
        di = d * i / n
        dd = d - di
        e = di * dd / (2.0 * ae)
        s1 = (e - hte) / di
        if s1 > stim:
            stim = s1
        s2 = (e - hre) / dd
        if s2 > srim:
            srim = s2
        v = (e - (hte * dd + hre * di) / d) * math.sqrt(2.0 * d / (lam * di * dd))
        if v > vmax:
            vmax = v
    return _bull_luc(stim, srim, vmax, hte, hre, d, lam)


def _delta_sph(d, n, hst, hsr, hobs, aobt, aobr, gtx, grx, zt, zr, ae, lam, f, epsr, sigma, mode):
    """Delta-Korrektur max(Ldsph - L_glatt, 0) mit effektiven Höhen nach ITU-R P.1812-7 §4.3.4."""
    if hobs <= 0.0:
        hstp = hst
        hsrp = hsr
    else:
        sa = aobt + aobr
        if sa <= 0.0:
            sa = 1e-9
        hstp = hst - hobs * aobt / sa
        hsrp = hsr - hobs * aobr / sa
    if hstp > gtx:
        hstp = gtx
    if hsrp > grx:
        hsrp = grx
    hte = zt - hstp
    hre = zr - hsrp
    if hte < 1.0:
        hte = 1.0
    if hre < 1.0:
        hre = 1.0
    dkm = d / 1000.0
    if dkm < 0.05:
        return 0.0
    lsph = _ldsph(dkm, hte, hre, ae / 1000.0, f, lam, epsr, sigma)
    if mode < 0.5:
        luc = _bull_smooth(d, hte, hre, ae, lam, n)
        lbs = luc + (1.0 - math.exp(-luc / 6.0)) * (10.0 + 0.02 * dkm)
    else:
        lbs = _deygout_smooth(d, hte, hre, ae, lam, 0.0, n)
    dl = lsph - lbs
    if dl < 0.0:
        return 0.0
    return dl


def _cell_loss(T, B, V, P, rx_x, rx_y):
    res = P[P_RES]
    step = P[P_STEP]
    tx_x = P[P_TXX]
    tx_y = P[P_TXY]
    zt = P[P_ZTX]
    gtx = P[P_GTX]
    lam = P[P_LAM]
    ae = P[P_AE]
    mode = P[P_DMODE]
    dx = rx_x - tx_x
    dy = rx_y - tx_y
    d = math.sqrt(dx * dx + dy * dy)
    if d > P[P_MAXD]:
        return NAN
    hr_g = _bil(T, rx_x, rx_y, res)
    hb_rx = _near(B, rx_x, rx_y, res) * 0.1
    rx_in = hb_rx > 0.0
    if P[P_FIXRX] > 0.5:
        zr = P[P_ZRXABS]
        if zr > hr_g + hb_rx:
            rx_in = False
    else:
        zr = hr_g + P[P_HRX]
    tx_in = P[P_TXIN] > 0.5
    if d < 1.0:
        d = 1.0
    n = int(d / step) + 1
    if n < 2:
        n = 2
    ds = d / n

    # eigenes Gebäude von Sender / Empfänger (Innenraum) nicht als Hindernis werten
    i_lo = 1
    if tx_in:
        while i_lo < n:
            t = i_lo / n
            if _near(B, tx_x + dx * t, tx_y + dy * t, res) > 0:
                i_lo += 1
            else:
                break
    i_hi = n - 1
    if rx_in:
        while i_hi >= i_lo:
            t = i_hi / n
            if _near(B, tx_x + dx * t, tx_y + dy * t, res) > 0:
                i_hi -= 1
            else:
                break

    # ---------- ein Durchlauf: Bullington-Größen für 3 Profile, Hindernisse, Ausgleichsgerade ----
    vm1 = -1e30
    vm2 = -1e30
    vm3 = -1e30
    p1 = -1
    p2 = -1
    p3 = -1
    g1p = 0.0
    g2p = 0.0
    g3p = 0.0
    st1 = -1e30
    st2 = -1e30
    st3 = -1e30
    sr1 = -1e30
    sr2 = -1e30
    sr3 = -1e30
    ho1 = -1e30
    ho2 = -1e30
    ho3 = -1e30
    at1 = -1e30
    at2 = -1e30
    at3 = -1e30
    ar1 = -1e30
    ar2 = -1e30
    ar3 = -1e30
    sv1 = 0.0
    sv2 = 0.0
    h_prev = gtx
    d_prev = 0.0
    veglen = 0.0
    blen = 0.0
    ncross = 0
    prev_in = False
    for i in range(1, n + 1):
        t = i / n
        di = i * ds
        x = tx_x + dx * t
        y = tx_y + dy * t
        if i == n:
            ht = hr_g
        else:
            ht = _bil(T, x, y, res)
        sv1 += ds * (ht + h_prev)
        sv2 += ds * (ht * (2.0 * di + d_prev) + h_prev * (di + 2.0 * d_prev))
        h_prev = ht
        d_prev = di
        if i == n:
            continue
        inside = i >= i_lo and i <= i_hi
        hb = 0.0
        hv = 0.0
        if inside:
            hb = _near(B, x, y, res) * 0.1
            if hb <= 0.0:
                hv = _near(V, x, y, res) * 0.1
        d2 = d - di
        e = di * d2 / (2.0 * ae)
        zl = zt + (zr - zt) * t
        fr = math.sqrt(2.0 * d / (lam * di * d2))
        g1 = ht + e
        g2 = g1 + hb
        g3 = g2 + hv
        # Profil 1: Gelände
        v = (g1 - zl) * fr
        if v > vm1:
            vm1 = v
            p1 = i
            g1p = g1
        s = (g1 - zt) / di
        if s > st1:
            st1 = s
        s = (g1 - zr) / d2
        if s > sr1:
            sr1 = s
        hh = ht - zl
        if hh > ho1:
            ho1 = hh
        if hh / di > at1:
            at1 = hh / di
        if hh / d2 > ar1:
            ar1 = hh / d2
        # Profil 2: + Gebäude
        v = (g2 - zl) * fr
        if v > vm2:
            vm2 = v
            p2 = i
            g2p = g2
        s = (g2 - zt) / di
        if s > st2:
            st2 = s
        s = (g2 - zr) / d2
        if s > sr2:
            sr2 = s
        hh = ht + hb - zl
        if hh > ho2:
            ho2 = hh
        if hh / di > at2:
            at2 = hh / di
        if hh / d2 > ar2:
            ar2 = hh / d2
        # Profil 3: + Bewuchs
        v = (g3 - zl) * fr
        if v > vm3:
            vm3 = v
            p3 = i
            g3p = g3
        s = (g3 - zt) / di
        if s > st3:
            st3 = s
        s = (g3 - zr) / d2
        if s > sr3:
            sr3 = s
        hh = ht + hb + hv - zl
        if hh > ho3:
            ho3 = hh
        if hh / di > at3:
            at3 = hh / di
        if hh / d2 > ar3:
            ar3 = hh / d2
        # direkter Strahl durch Bewuchs / Gebäude
        if hv > 0.0 and zl > g2 and zl <= g3:
            veglen += ds
        inb = hb > 0.0 and zl <= g2 and zl > g1
        if inb:
            blen += ds
            if not prev_in:
                ncross += 1
        prev_in = inb

    hst = (2.0 * sv1 * d - sv2) / (d * d)
    hsr = (sv2 - sv1 * d) / (d * d)
    dkm = d / 1000.0

    if mode < 0.5:
        # ---------- Delta-Bullington (ITU-R P.1812-7) ----------
        lu1 = _bull_luc(st1, sr1, vm1, zt, zr, d, lam)
        lu2 = _bull_luc(st2, sr2, vm2, zt, zr, d, lam)
        lu3 = _bull_luc(st3, sr3, vm3, zt, zr, d, lam)
        # Korrekturterm (gerundete Hindernisse) nur aus dem Geländeanteil; Gebäude und
        # Baumkronen wirken als scharfe Kanten.
        corr = (1.0 - math.exp(-lu1 / 6.0)) * (10.0 + 0.02 * dkm)
        ld1 = lu1 + corr
        ld2 = lu2 + corr
        ld3 = lu3 + corr
    else:
        # ---------- Deygout, 3 Kanten: zweiter Durchlauf für die Nebenkanten ----------
        l1m = -1e30
        r1m = -1e30
        l2m = -1e30
        r2m = -1e30
        l3m = -1e30
        r3m = -1e30
        dp1 = p1 * ds
        dp2 = p2 * ds
        dp3 = p3 * ds
        need = vm1 > -0.78 or vm2 > -0.78 or vm3 > -0.78
        if need:
            for i in range(i_lo, i_hi + 1):
                t = i / n
                di = i * ds
                d2 = d - di
                x = tx_x + dx * t
                y = tx_y + dy * t
                ht = _bil(T, x, y, res)
                hb = _near(B, x, y, res) * 0.1
                hv = 0.0
                if hb <= 0.0:
                    hv = _near(V, x, y, res) * 0.1
                e = di * d2 / (2.0 * ae)
                g1 = ht + e
                g2 = g1 + hb
                g3 = g2 + hv
                if p1 > 0 and vm1 > -0.78:
                    if i < p1:
                        v = (g1 - (zt + (g1p - zt) * di / dp1)) * math.sqrt(2.0 * dp1 / (lam * di * (dp1 - di)))
                        if v > l1m:
                            l1m = v
                    elif i > p1:
                        s = di - dp1
                        L2 = d - dp1
                        v = (g1 - (g1p + (zr - g1p) * s / L2)) * math.sqrt(2.0 * L2 / (lam * s * d2))
                        if v > r1m:
                            r1m = v
                if p2 > 0 and vm2 > -0.78:
                    if i < p2:
                        v = (g2 - (zt + (g2p - zt) * di / dp2)) * math.sqrt(2.0 * dp2 / (lam * di * (dp2 - di)))
                        if v > l2m:
                            l2m = v
                    elif i > p2:
                        s = di - dp2
                        L2 = d - dp2
                        v = (g2 - (g2p + (zr - g2p) * s / L2)) * math.sqrt(2.0 * L2 / (lam * s * d2))
                        if v > r2m:
                            r2m = v
                if p3 > 0 and vm3 > -0.78:
                    if i < p3:
                        v = (g3 - (zt + (g3p - zt) * di / dp3)) * math.sqrt(2.0 * dp3 / (lam * di * (dp3 - di)))
                        if v > l3m:
                            l3m = v
                    elif i > p3:
                        s = di - dp3
                        L2 = d - dp3
                        v = (g3 - (g3p + (zr - g3p) * s / L2)) * math.sqrt(2.0 * L2 / (lam * s * d2))
                        if v > r3m:
                            r3m = v
        ld1 = _deygout(_J(vm1), _J(l1m), _J(r1m), dkm, 0.0)
        ld2 = _deygout(_J(vm2), _J(l2m), _J(r2m), dkm, 0.0)
        ld3 = _deygout(_J(vm3), _J(l3m), _J(r3m), dkm, 0.0)

    # ---------- sphärische Erde (Delta-Korrektur) je Profil ----------
    f = P[P_FGHZ]
    epsr = P[P_EPSR]
    sig = P[P_SIGMA]
    dl1 = _delta_sph(d, n, hst, hsr, ho1, at1, ar1, gtx, hr_g, zt, zr, ae, lam, f, epsr, sig, mode)
    if ho2 <= 0.0 and ho1 <= 0.0:
        dl2 = dl1
    else:
        dl2 = _delta_sph(d, n, hst, hsr, ho2, at2, ar2, gtx, hr_g, zt, zr, ae, lam, f, epsr, sig, mode)
    if ho3 <= 0.0 and ho2 <= 0.0:
        dl3 = dl2
    else:
        dl3 = _delta_sph(d, n, hst, hsr, ho3, at3, ar3, gtx, hr_g, zt, zr, ae, lam, f, epsr, sig, mode)
    ld1 += dl1
    ld2 += dl2
    ld3 += dl3

    # ---------- Bewuchs, Gebäudedurchdringung, Kombination ----------
    am = P[P_VAM]
    aveg = 0.0
    if veglen > 0.0:
        aveg = am * (1.0 - math.exp(-veglen * P[P_VGAM] / am))
    l_over = ld3
    if ld2 + aveg < l_over:
        l_over = ld2 + aveg
    if ncross > 0:
        abld = ncross * 2.0 * P[P_WALL] + blen * P[P_INT]
        if abld > P[P_BLDMAX]:
            abld = P[P_BLDMAX]
        l_thr = ld1 + aveg + abld
        lex = -10.0 * math.log10(10.0 ** (-0.1 * l_over) + 10.0 ** (-0.1 * l_thr))
    else:
        lex = l_over
    if rx_in:
        lex += P[P_BEL]
    if tx_in:
        lex += P[P_BEL]

    dz = zr - zt
    d3 = math.sqrt(d * d + dz * dz)
    fspl = 20.0 * math.log10(4.0 * math.pi * d3 / lam)
    theta = math.atan2(dz, d)
    gant = _pattern_db(theta, P[P_GANT], P[P_NEL], P[P_TILT])
    return fspl + lex - gant


_PY = {
    "_J": _J,
    "_bil": _bil,
    "_near": _near,
    "_ldft": _ldft,
    "_ldsph": _ldsph,
    "_deygout": _deygout,
    "_deygout_smooth": _deygout_smooth,
    "_bull_luc": _bull_luc,
    "_bull_smooth": _bull_smooth,
    "_delta_sph": _delta_sph,
    "_pattern_db": _pattern_db,
    "_cell_loss": _cell_loss,
}

def _cov_cpu(T, B, V, P, out_arr, row0, nrows):
    ox0 = P[P_OX0]
    oy0 = P[P_OY0]
    ores = P[P_ORES]
    for r in prange(nrows):  # noqa: F821
        row = row0 + r
        if row >= out_arr.shape[0]:
            continue
        y = oy0 + row * ores
        for col in range(out_arr.shape[1]):
            out_arr[row, col] = cell(T, B, V, P, ox0 + col * ores, y)  # noqa: F821


def _pts_cpu(T, B, V, P, xs, ys, out_arr):
    for i in prange(xs.shape[0]):  # noqa: F821
        out_arr[i] = cell(T, B, V, P, xs[i], ys[i])  # noqa: F821


def _batch_cpu(Tall, Ball, Vall, offs, lens, Pall, xs, ys, out_arr):
    """Viele Profilstreifen (3 x n, hintereinander abgelegt) in einem parallelen Aufruf."""
    for i in prange(offs.shape[0]):  # noqa: F821
        a = offs[i]
        b = a + lens[i]
        out_arr[i] = cell(Tall[:, a:b], Ball[:, a:b], Vall[:, a:b], Pall[i], xs[i], ys[i])  # noqa: F821


_cache: dict[str, dict] = {}


def _globals():
    g = {"math": math, "NAN": NAN, "__builtins__": __builtins__}
    for k, v in globals().items():
        if k.startswith("P_"):
            g[k] = v
    return g


_INT_NAMES = {"i", "n", "i_lo", "i_hi", "p1", "p2", "p3", "pm", "ncross", "c0", "r0", "c", "r",
              "nx", "ny", "row", "col", "k", "dr", "dc", "rn", "cn", "sr", "sc", "sn", "pr", "pc",
              "rr", "cc", "q", "nst"}


class _F32Transformer:
    """AST-Umschreibung für die GPU: alle Gleitkommarechnungen in float32.

    Consumer-GPUs rechnen FP64 nur mit 1/32..1/64 der FP32-Geschwindigkeit. Fließkomma-
    Konstanten, Divisionen, ``math.pi`` und Ganzzahl-Operanden in Mischrechnungen werden in
    ``f32(...)`` gekapselt. Alle Größen sind Meter/dB in Fensterkoordinaten (< 100 km), float32
    reicht dafür (Auflösung < 1 cm); die Tests vergleichen GPU gegen CPU (float64)."""

    def __new__(cls):
        import ast

        def is_int(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return True
            if isinstance(node, ast.Name) and node.id in _INT_NAMES:
                return True
            if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Div):
                return is_int(node.left) and is_int(node.right)
            return False

        def wrap(node):
            return ast.Call(func=ast.Name(id="f32", ctx=ast.Load()), args=[node], keywords=[])

        class T(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, float):
                    return wrap(node)
                return node

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr == "pi":
                    return wrap(node)
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == "NAN" and isinstance(node.ctx, ast.Load):
                    return wrap(node)
                return node

            def visit_BinOp(self, node):
                li, ri = is_int(node.left), is_int(node.right)
                node = self.generic_visit(node)
                if isinstance(node.op, ast.Div):
                    node.left = wrap(node.left)
                    node.right = wrap(node.right)
                elif li != ri:
                    if li:
                        node.left = wrap(node.left)
                    else:
                        node.right = wrap(node.right)
                return node

        return T()


def _f32_function(f, g):
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(f))
    tree = ast.parse(src)
    fdef = tree.body[0]
    fdef.body = [s for s in fdef.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))] or fdef.body
    tree = _F32Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    ns = {}
    exec(compile(tree, f"<f32:{f.__name__}>", "exec"), g, ns)
    fn = ns[f.__name__]
    return types.FunctionType(fn.__code__, g, f.__name__)


def build(target: str) -> dict:
    """Übersetzt alle Funktionen für ``target`` ('cuda' = float32 auf GPU, 'cpu' = float64)."""
    if target in _cache:
        return _cache[target]
    g = _globals()
    if target == "cuda":
        from numba import cuda, float32

        g["f32"] = float32

        def jit(f):
            return cuda.jit(device=True)(_f32_function(f, g))
    else:
        from numba import njit

        def jit(f):
            fn = types.FunctionType(f.__code__, g, f.__name__, f.__defaults__, f.__closure__)
            return njit(cache=False, fastmath=False, nogil=True)(fn)

    out = {}
    for name, f in _PY.items():
        out[name] = jit(f)
        g[name] = out[name]

    cell = out["_cell_loss"]
    if target == "cuda":
        from numba import cuda, float32 as f32

        @cuda.jit
        def coverage_kernel(T, B, V, P, out_arr, row0):
            col, r = cuda.grid(2)
            row = row0 + r
            if row < out_arr.shape[0] and col < out_arr.shape[1]:
                x = P[P_OX0] + f32(col) * P[P_ORES]
                y = P[P_OY0] + f32(row) * P[P_ORES]
                out_arr[row, col] = cell(T, B, V, P, x, y)

        @cuda.jit
        def points_kernel(T, B, V, P, xs, ys, out_arr):
            i = cuda.grid(1)
            if i < xs.shape[0]:
                out_arr[i] = cell(T, B, V, P, xs[i], ys[i])

        out["coverage_kernel"] = coverage_kernel
        out["points_kernel"] = points_kernel
    else:
        from numba import njit, prange

        g["cell"] = cell
        g["prange"] = prange
        out["coverage_cpu"] = njit(parallel=True, nogil=True)(types.FunctionType(_cov_cpu.__code__, g, "_cov"))
        out["points_cpu"] = njit(parallel=True, nogil=True)(types.FunctionType(_pts_cpu.__code__, g, "_pts"))
        out["batch_cpu"] = njit(parallel=True, nogil=True)(types.FunctionType(_batch_cpu.__code__, g, "_batch"))
    _cache[target] = out
    return out
