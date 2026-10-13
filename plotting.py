"""Plot helpers."""
import matplotlib.pyplot as plt
import numpy as np

from .filters import boxcar_average
from .io import parse_msa


def plot_msa(filepath, ax, boxcar_width, label, scale=1, normalize=False, x_offset=0, y_offset=0):
    """Plot a raw MSA spectrum and its boxcar-smoothed version on ``ax``.

    ``normalize`` maps counts to (counts - min) / max; ``y_offset`` is only
    applied when normalizing (as in the original). ``scale`` multiplies the
    counts. Returns ``(line_raw, line_smooth)``.
    """
    energy, counts = parse_msa(filepath)
    counts = counts * scale

    if normalize:
        counts = (counts - min(counts)) / max(counts)
        counts = counts + y_offset

    energy = energy + x_offset

    (line_raw,) = ax.plot(energy, counts, alpha=0.35, lw=0.8, label=label + " Raw")
    smoothed = boxcar_average(counts, boxcar_width)
    (line_smooth,) = ax.plot(energy, smoothed, lw=1.5, label=label + f" Boxcar (n={boxcar_width})")
    return line_raw, line_smooth


def plot_correlation_matrix(M, labels, ax=None, title="Residual correlation matrix",
                            fontsize=9, tick_fontsize=None, rotate=False):
    """Heatmap of a correlation matrix with values annotated in each cell.

    Text colour switches to white where |r| > 0.5 for legibility.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5) if len(labels) < 6 else (9, 7.5))
    n = len(labels)
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if rotate:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
    else:
        ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=fontsize,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
