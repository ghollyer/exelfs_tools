"""Reusable EXELFS spectrum utilities (extracted from the 2026_08_19_EXELFS notebooks).

Usage from a notebook anywhere under the EXELFS folder:

    import sys; sys.path.insert(0, "/Users/george/Documents/PhD/Research/EXELFS")
    from exelfs_tools import *
"""
from .io import parse_msa, load_spectra
from .filters import boxcar_average, bin_spectrum, deglitch, detrend
from .noise import (
    robust_sigma, robust_clip, local_noise, residual, var_vs_signal,
    fit_gain, residual_correlation_matrix, difference_sum_test,
)
from .plotting import plot_msa, plot_correlation_matrix

__all__ = [
    "parse_msa", "load_spectra",
    "boxcar_average", "bin_spectrum", "deglitch", "detrend",
    "robust_sigma", "robust_clip", "local_noise", "residual", "var_vs_signal",
    "fit_gain", "residual_correlation_matrix", "difference_sum_test",
    "plot_msa", "plot_correlation_matrix",
]
