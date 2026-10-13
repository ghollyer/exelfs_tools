"""Smoothing, binning, spike removal and detrending."""
import numpy as np
from scipy.signal import medfilt, savgol_filter

from .noise import robust_sigma


def boxcar_average(data, width):
    """Simple boxcar (moving average) filter with given window width.

    Edges are handled by padding with edge values; output has the input length.
    """
    if width <= 1:
        return data.copy()
    kernel = np.ones(width) / width
    pad = width // 2
    padded = np.pad(data, pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(data)]


def bin_spectrum(E, C, bin_width):
    """Bin a spectrum by ``bin_width`` channels: counts are summed, energies averaged.

    Trailing channels that don't fill a whole bin are dropped.
    """
    E = np.asarray(E, dtype=float)
    C = np.asarray(C, dtype=float)
    n = len(C) // bin_width * bin_width
    C_binned = C[:n].reshape(-1, bin_width).sum(axis=1)
    E_binned = E[:n].reshape(-1, bin_width).mean(axis=1)
    return E_binned, C_binned


def deglitch(c, k=7, thr=6.0):
    """Replace cosmic-ray spikes (|c-median| > thr*robust-sigma) with the local median.

    Returns ``(cleaned, n_replaced)``.
    """
    med = medfilt(c, k)
    r = c - med
    bad = np.abs(r) > thr * robust_sigma(r)
    out = c.copy()
    out[bad] = med[bad]
    return out, int(bad.sum())


def detrend(c, win=51, poly=3):
    """Remove the smooth signal with a Savitzky-Golay fit; return (smooth, residual)."""
    S = savgol_filter(c, win, poly)
    return S, c - S
