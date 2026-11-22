"""Straßenschlucht-Ausbreitung (Reflexion/Beugung um Gebäudeecken entlang von Straßenzügen).

Der Profil-Kernel erfasst die vertikale Beugung über Dächer. Zusätzlich gelangt Energie
horizontal *entlang* von Straßen und um Ecken in Seitenstraßen. Dafür wird das
Site-specific-Modell für UHF-Straßenschluchten aus **ITU-R P.1411** (NLoS um eine Ecke:
Reflexionspfad ``Lr`` und Beugungspfad ``Ld``, Gl. 'L_NLoS2') auf das Raster angewendet:

* jede Außenzelle kann als „Ecke“ (Sekundärquelle) dienen; ``x1`` = Weglänge bis zur Ecke,
  ``x2`` = Entfernung hinter der Ecke, ``alpha`` = Straßenwinkel, ``w1, w2`` = lokale
  Straßenbreiten (aus dem Abstand zur nächsten Gebäudekante),
* der Überschussverlust an der Ecke (gegenüber Freiraum) wird übernommen,
* Relaxation: pro Iteration breitet sich die Lösung um eine Zelle aus; jede Zelle behält das
  Minimum aus Direktpfad und allen Schlucht-Kandidaten,
* Innenräume (Gebäudezellen) werden zusätzlich über die Fassade versorgt:
  ``L_fassade + BEL + Innendämpfung * Weg``.
"""
from __future__ import annotations

import math
import types

import numpy as np
from scipy import ndimage


CANYON_DIST = 20.0  # m


def _fspl(d, lam):
    if d < 1.0:
        d = 1.0
    return 20.0 * math.log10(4.0 * math.pi * d / lam)


def _l1411(x1, x2, alpha, w1, w2, lam):
    """ITU-R P.1411 Site-specific NLoS2 (UHF, Straßenecke). Rückgabe Grundübertragungsdämpfung [dB]."""
    if x1 < 1.0:
        x1 = 1.0
    if x2 < 1.0:
        x2 = 1.0
    if alpha < 0.6:
        alpha = 0.6
    if alpha > math.pi:
        alpha = math.pi
    k = 20.0 * math.log10(4.0 * math.pi / lam)
    los = 20.0 * math.log10(x1 + x2) + k
    fa = 3.86 / alpha ** 3.5
    lr = 20.0 * math.log10(x1 + x2) + x1 * x2 * fa / (w1 * w2) + k
    da = (40.0 / (2.0 * math.pi)) * (math.atan(x2 / w2) + math.atan(x1 / w1) - math.pi / 2.0)
    ld = 10.0 * math.log10(x1 * x2 * (x1 + x2)) + 2.0 * da - 0.1 * (90.0 - alpha * 180.0 / math.pi) + k
    big = lr
    if ld < big:
        big = ld
    if big - lr > 60.0 or big - ld > 60.0:
        lsum = big
    else:
        lsum = -10.0 * math.log10(10.0 ** (-0.1 * lr) + 10.0 ** (-0.1 * ld))
    if lsum < los:
        return los
    return lsum


def _angdiff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)


def _relax_cell(L, SRC, DS, SS, TH, LO, SRCO, DSO, SSO, THO, OPEN, W, CHG, CHGN, r, c, txr, txc, res, lam, bel, gint):
    """Liest nur den alten Zustand (L, SRC, ...) und schreibt den neuen (LO, SRCO, ...) –
    Doppelpufferung, damit parallel laufende Nachbarzellen nie halbfertige Zustände sehen."""
    ny = L.shape[0]
    nx = L.shape[1]
    LO[r, c] = L[r, c]
    SRCO[r, c] = SRC[r, c]
    DSO[r, c] = DS[r, c]
    SSO[r, c] = SS[r, c]
    THO[r, c] = TH[r, c]
    # nur rechnen, wenn sich in der Nachbarschaft etwas geändert hat
    act = False
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = r + dr
            cc = c + dc
            if rr >= 0 and rr < ny and cc >= 0 and cc < nx:
                if CHG[rr, cc] > 0:
                    act = True
    if not act:
        return 0
    best = L[r, c]
    if not (best < 1e9):
        best = 1e9
    bsrc = SRC[r, c]
    bds = DS[r, c]
    bss = SS[r, c]
    bth = TH[r, c]
    improved = False
    copen = OPEN[r, c] > 0
    ccan = OPEN[r, c] > 1
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            rn = r + dr
            cn = c + dc
            if rn < 0 or rn >= ny or cn < 0 or cn >= nx:
                continue
            ln = L[rn, cn]
            if not (ln < 1e9):
                continue
            step = res * math.sqrt(1.0 * (dr * dr + dc * dc))
            nopen = OPEN[rn, cn] > 0
            if not copen:
                cand = ln + gint * step
                if nopen:
                    cand += bel
                if cand < best - 0.01:
                    best = cand
                    bsrc = -2
                    improved = True
                continue
            if not nopen or not ccan:
                continue
            sn = SRC[rn, cn]
            if sn == -2:
                continue
            # --- Fall B: Nachbar ist neue Ecke (nur innerhalb von Straßenschluchten) ---
            ncan = OPEN[rn, cn] > 1
            if sn == -1:
                vy = (rn - txr) * res
                vx = (cn - txc) * res
                x1 = math.sqrt(vx * vx + vy * vy)
            else:
                sr = sn // nx
                sc = sn - sr * nx
                vy = (rn - sr) * res
                vx = (cn - sc) * res
                x1 = DS[rn, cn] + math.sqrt(vx * vx + vy * vy)
            thin = math.atan2(vy, vx)
            thout = math.atan2(dr * 1.0, dc * 1.0)
            phi = _angdiff(thout, thin)
            scorner = ln - _fspl(x1, lam)
            cand = scorner + _l1411(x1, step, math.pi - phi, W[rn, cn], W[r, c], lam)
            # nur direkt versorgte Zellen sind Ecken (ein Eckpunkt je Pfad, wie P.1411 NLoS2)
            if ncan and sn == -1 and cand < best - 0.01:
                best = cand
                bsrc = rn * nx + cn
                bds = x1
                bss = scorner
                bth = thin
                improved = True
            # --- Fall A: Quelle des Nachbarn (Ecke) weiterverwenden ---
            if sn >= 0:
                sr = sn // nx
                sc = sn - sr * nx
                vy = (r - sr) * res
                vx = (c - sc) * res
                x2 = math.sqrt(vx * vx + vy * vy)
                if x2 < 0.5 * res:
                    continue
                # Sichtprüfung Ecke -> c über die ganze Linie (keine Gebäudezelle dazwischen)
                nst = int(x2 / (0.7 * res)) + 1
                vis = True
                for q in range(1, nst):
                    fq = q / nst
                    pr = int(math.floor(sr + (r - sr) * fq + 0.5))
                    pc = int(math.floor(sc + (c - sc) * fq + 0.5))
                    if OPEN[pr, pc] == 0:
                        vis = False
                        break
                if not vis:
                    continue
                phi = _angdiff(math.atan2(vy, vx), TH[rn, cn])
                cand = SS[rn, cn] + _l1411(DS[rn, cn], x2, math.pi - phi, W[sr, sc], W[r, c], lam)
                if cand < best - 0.01:
                    best = cand
                    bsrc = sn
                    bds = DS[rn, cn]
                    bss = SS[rn, cn]
                    bth = TH[rn, cn]
                    improved = True
    if improved:
        LO[r, c] = best
        SRCO[r, c] = bsrc
        DSO[r, c] = bds
        SSO[r, c] = bss
        THO[r, c] = bth
        CHGN[r, c] = 1
        return 1
    return 0


def _it_cpu(L, SRC, DS, SS, TH, LO, SRCO, DSO, SSO, THO, OPEN, W, CHG, CHGN, P):
    ny, nx = L.shape
    tot = 0
    for r in prange(ny):  # noqa: F821 (prange wird beim Übersetzen eingesetzt)
        for c in range(nx):
            tot += _relax_cell(L, SRC, DS, SS, TH, LO, SRCO, DSO, SSO, THO, OPEN, W, CHG, CHGN, r, c,
                               P[0], P[1], P[2], P[3], P[4], P[5])
    return tot


_PY = {"_fspl": _fspl, "_l1411": _l1411, "_angdiff": _angdiff, "_relax_cell": _relax_cell}
_cache: dict = {}


def _build(target):
    if target in _cache:
        return _cache[target]
    from .kernel import _f32_function

    g = {"math": math, "__builtins__": __builtins__}
    out = {}
    if target == "cuda":
        from numba import cuda, float32

        g["f32"] = float32
        for k, f in _PY.items():
            out[k] = cuda.jit(device=True)(_f32_function(f, g))
            g[k] = out[k]
        relax = out["_relax_cell"]

        @cuda.jit
        def it_kernel(L, SRC, DS, SS, TH, LO, SRCO, DSO, SSO, THO, OPEN, W, CHG, CHGN, P, counter):
            c, r = cuda.grid(2)
            if r < L.shape[0] and c < L.shape[1]:
                n = relax(L, SRC, DS, SS, TH, LO, SRCO, DSO, SSO, THO, OPEN, W, CHG, CHGN, r, c,
                          P[0], P[1], P[2], P[3], P[4], P[5])
                if n > 0:
                    cuda.atomic.add(counter, 0, 1)

        out["iter"] = it_kernel
    else:
        from numba import njit, prange

        for k, f in _PY.items():
            fn = types.FunctionType(f.__code__, g, k)
            out[k] = njit(nogil=True)(fn)
            g[k] = out[k]
        g["prange"] = prange

        out["iter"] = njit(parallel=True, nogil=True)(types.FunctionType(_it_cpu.__code__, g, "_it_cpu"))
    _cache[target] = out
    return out


def canyon_pass(loss: np.ndarray, bld_out: np.ndarray, tx_rc, res: float, env, cancel=None) -> np.ndarray:
    """Wendet die Schlucht-Relaxation auf ein Verlustraster an (in-place-sicher, gibt neues Array)."""
    from .. import memguard
    from .engine import backend

    oy, ox = loss.shape
    free = bld_out == 0
    # Straßenbreite ~ 2 x Abstand zur nächsten Gebäudezelle (begrenzt auf 6..40 m);
    # Schluchtzellen = frei und Gebäude in <= CANYON_DIST m Entfernung
    dist = ndimage.distance_transform_edt(free).astype(np.float32) * res
    W = np.clip(2.0 * dist, 6.0, 40.0).astype(np.float32)
    open_ = np.where(free, np.where(dist <= CANYON_DIST, 2, 1), 0).astype(np.uint8)
    del dist, free
    L = np.where(np.isfinite(loss), loss, np.float32(1e10)).astype(np.float32)
    SRC = np.full((oy, ox), -1, np.int32)
    SRC[open_ == 0] = -2
    DS = np.zeros((oy, ox), np.float32)
    SS = np.zeros((oy, ox), np.float32)
    TH = np.zeros((oy, ox), np.float32)
    CHG = np.isfinite(loss).astype(np.uint8)
    CHGN = np.zeros_like(CHG)
    P = np.array([tx_rc[0], tx_rc[1], res, env.lam, env.bel_db(), env.interior_db_m], np.float64)
    iters = max(1, int(env.canyon_max_m / res))
    memguard.require_ram(L.nbytes * 12, "Schlucht-Raster")
    state = [L, SRC, DS, SS, TH]
    if backend() == "cuda":
        from numba import cuda

        memguard.require_vram(L.nbytes * 12, "Schlucht-Raster GPU")
        fn = _build("cuda")["iter"]
        cur = [cuda.to_device(a) for a in state]
        nxt = [cuda.to_device(a) for a in state]
        dO, dW = cuda.to_device(open_), cuda.to_device(W)
        dC, dCN = cuda.to_device(CHG), cuda.to_device(CHGN)
        zero = np.zeros_like(CHG)
        dP = cuda.to_device(P.astype(np.float32))
        cnt = cuda.to_device(np.zeros(1, np.int32))
        tpb = (32, 8)
        blocks = ((ox + 31) // 32, (oy + 7) // 8)
        for _ in range(iters):
            if cancel is not None and cancel.is_set():
                break
            cnt.copy_to_device(np.zeros(1, np.int32))
            dCN.copy_to_device(zero)
            fn[blocks, tpb](*cur, *nxt, dO, dW, dC, dCN, dP, cnt)
            cur, nxt = nxt, cur
            dC, dCN = dCN, dC
            if int(cnt.copy_to_host()[0]) == 0:
                break
        L = cur[0].copy_to_host()
        del cur, nxt
    else:
        fn = _build("cpu")["iter"]
        cur = state
        nxt = [a.copy() for a in state]
        for _ in range(iters):
            if cancel is not None and cancel.is_set():
                break
            CHGN[:] = 0
            n = fn(*cur, *nxt, open_, W, CHG, CHGN, P)
            cur, nxt = nxt, cur
            CHG, CHGN = CHGN, CHG
            if n == 0:
                break
        L = cur[0]
    L = L.astype(np.float32)
    L[L > 1e9] = np.nan
    return L
