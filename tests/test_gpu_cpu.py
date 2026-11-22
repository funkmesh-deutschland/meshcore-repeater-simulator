"""GPU (float32) und CPU (float64) müssen dasselbe Ergebnis liefern."""
import numpy as np
import pytest

from funkmesh_sim.radio import kernel as K
from tests.test_kernel import params


def scene(seed=3, n=300):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    T = (60 + 15 * ndimage.gaussian_filter(rng.standard_normal((n, n)), 25) * 20).astype(np.float32)
    B = np.zeros((n, n), np.uint16)
    for _ in range(150):
        r, c = rng.integers(0, n - 4, 2)
        B[r:r + 3, c:c + 3] = rng.integers(40, 150)
    V = np.zeros((n, n), np.uint16)
    for _ in range(60):
        r, c = rng.integers(0, n - 10, 2)
        V[r:r + 8, c:c + 8] = rng.integers(50, 250)
    V[B > 0] = 0
    return T, B, V


def run(target, T, B, V, P, n_out):
    out = np.full((n_out, n_out), np.nan, np.float32)
    if target == "cpu":
        K.build("cpu")["coverage_cpu"](T, B, V, P, out, 0, n_out)
        return out
    from numba import cuda
    fn = K.build("cuda")["coverage_kernel"]
    d = [cuda.to_device(a) for a in (T, B, V, P.astype(np.float32), out)]
    fn[((n_out + 31) // 32, (n_out + 3) // 4), (32, 4)](*d, 0)
    return d[4].copy_to_host()


@pytest.mark.parametrize("mode", [0.0, 1.0])
def test_gpu_matches_cpu(mode):
    try:
        from numba import cuda
        if not cuda.is_available():
            pytest.skip("keine CUDA-GPU")
    except Exception:
        pytest.skip("numba-cuda fehlt")
    T, B, V = scene()
    P = params(ztx_agl=12.0, ground=float(T[150, 150]))
    P[K.P_TXX] = 750.0; P[K.P_TXY] = 750.0; P[K.P_ORES] = 10.0; P[K.P_OX0] = 5.0; P[K.P_OY0] = 5.0
    P[K.P_DMODE] = mode
    a = run("cpu", T, B, V, P, 150)
    b = run("cuda", T, B, V, P, 150)
    ok = np.isfinite(a)
    assert ok.mean() > 0.9
    diff = np.abs(a[ok] - b[ok])
    assert np.percentile(diff, 99) < 0.05, np.percentile(diff, [50, 99, 100])
