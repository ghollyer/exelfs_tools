"""Noise characterisation: local noise, variance-vs-signal, fixed-pattern tests."""
import itertools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def robust_sigma(x):
    """Gaussian-equivalent sigma from the median absolute deviation (1.4826 * MAD)."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def robust_clip(x, nsig=8, center=False):
    """Clip ``x`` to median ± ``nsig`` robust sigma.

    Merges the two notebook versions:
    - ``center=False`` (``rclip``): clip around the median, keep the offset.
    - ``center=True`` (inside ``residual``): subtract the median, then clip to ±nsig·σ.
    """
    m, s = np.median(x), robust_sigma(x)
    if center:
        return np.clip(x - m, -nsig * s, nsig * s)
    return np.clip(x, m - nsig * s, m + nsig * s)


def local_noise(r, win=51):
    """Robust local noise = sliding-window MAD of the residual."""
    w = sliding_window_view(r, win)
    med = np.median(w, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(w - med), axis=1)
    return np.pad(mad, win // 2, mode="edge")[:len(r)]


def residual(c, win=51, poly=3, clip=8, center=True, k=7, thr=6.0):
    """Deglitch -> Savitzky-Golay detrend -> robust-clip the residual.

    With defaults this is the ``residual()`` from the correlation-matrix cell
    (median-centred, clipped at ±8σ). ``center=False`` gives the ``rclip``
    behaviour; ``clip=None`` returns the raw residual (no clipping).
    """
    from .filters import deglitch, detrend
    cc, _ = deglitch(c, k, thr)
    _, r = detrend(cc, win, poly)
    if clip is None:
        return r
    return robust_clip(r, clip, center=center)


def var_vs_signal(c, win=51, nb=16, edge=100, low_pct=2, min_pts=15):
    """Binned local variance vs local signal (photon-transfer curve).

    Returns ``(signal, variance)`` medians for ``nb`` quantile bins of signal.
    ``edge`` channels are dropped at each end; signals below the ``low_pct``
    percentile are discarded; bins with ≤ ``min_pts`` points are skipped.
    """
    from .filters import deglitch, detrend
    Cc, _ = deglitch(c)
    S, r = detrend(Cc, win)
    sg = local_noise(r, win)
    sl = slice(edge, len(Cc) - edge)
    s, v = S[sl], sg[sl] ** 2
    m = s > np.percentile(s, low_pct)
    s, v = s[m], v[m]
    order = np.argsort(s)
    s, v = s[order], v[order]
    edges = np.quantile(s, np.linspace(0, 1, nb + 1))
    xs, ys = [], []
    for i in range(nb):
        mm = (s >= edges[i]) & (s < edges[i + 1])
        if mm.sum() > min_pts:
            xs.append(np.median(s[mm]))
            ys.append(np.median(v[mm]))
    return np.array(xs), np.array(ys)


def fit_gain(xs, ys):
    """Fit var = g·S + b to a ``var_vs_signal`` curve.

    Returns ``dict(gain, floor, floor_sigma, loglog_slope)``; Poisson noise
    gives gain ≈ 1 and log-log slope ≈ 1.
    """
    A = np.vstack([xs, np.ones_like(xs)]).T
    (g, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pos = (xs > 0) & (ys > 0)
    slope = np.polyfit(np.log(xs[pos]), np.log(ys[pos]), 1)[0]
    return dict(gain=g, floor=b, floor_sigma=np.sqrt(max(b, 0)), loglog_slope=slope)


def residual_correlation_matrix(counts_list, sl=slice(150, 3550), **residual_kw):
    """Channel-by-channel correlation of residuals between spectra.

    Fixed-pattern noise -> +1, independent random noise -> 0. Spectra are
    aligned by channel index and cut to ``sl``. ``residual_kw`` is passed to
    ``residual`` (e.g. ``center=False`` for the rclip variant).
    """
    R = np.vstack([residual(c, **residual_kw)[sl] for c in counts_list])
    return np.corrcoef(R)


def difference_sum_test(Ca, Cb, sl=slice(100, 3610), win=51, poly=3, verbose=True, label=""):
    """Photon-transfer-style difference/sum test on a same-axis repeat pair.

    Fixed pattern cancels in A-B and adds in A+B; random noise behaves
    identically in both (σ ≈ quadrature sum of the singles).
    """
    from scipy.signal import savgol_filter
    from .filters import deglitch
    Ca, _ = deglitch(Ca)
    Cb, _ = deglitch(Cb)
    ra = (Ca - savgol_filter(Ca, win, poly))[sl]
    rb = (Cb - savgol_filter(Cb, win, poly))[sl]
    na, nb = np.median(local_noise(ra)), np.median(local_noise(rb))
    nd, ns = np.median(local_noise(ra - rb)), np.median(local_noise(ra + rb))
    quad = np.sqrt(na**2 + nb**2)
    r = np.corrcoef(ra - np.median(ra), rb - np.median(rb))[0, 1]
    if verbose:
        print(f"{label}:" if label else "pair:")
        print(f"   single σ      : {na:.1f}, {nb:.1f}   quadrature(if independent) = {quad:.1f}")
        print(f"   σ(A-B)={nd:.1f}   σ(A+B)={ns:.1f}    (random -> both ~ {quad:.1f}; fixed -> σ(A-B) << σ(A+B))")
        print(f"   residual correlation r = {r:+.3f}\n")
    return dict(sigma_a=na, sigma_b=nb, sigma_diff=nd, sigma_sum=ns, quadrature=quad, r=r)
