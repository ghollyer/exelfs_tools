"""Plot helpers."""
import matplotlib.pyplot as plt
import numpy as np

from .filters import boxcar_average
from .io import parse_msa


def plot_msa(filepath, ax=None, boxcar_width=10, label="", scale=1, normalize=False,
             x_offset=0, y_offset=0, show=None, xlim=None, edge=None, title=None):
    """Plot a raw MSA spectrum and its boxcar-smoothed version.

    If ``ax`` is None a new figure is created, labelled and shown, so a single
    call produces a finished plot. Pass an existing ``ax`` to overlay several
    spectra; the figure is then only shown if ``show=True`` (call it on the
    last spectrum, or call ``plt.show()`` yourself).

    ``normalize`` maps counts to (counts - min) / max; ``y_offset`` is only
    applied when normalizing (as in the original). ``scale`` multiplies the
    counts. ``xlim`` is ``(left, right)``; ``edge`` draws a dashed reference
    line at that energy (eV). Returns ``(line_raw, line_smooth)``.
    """
    new_fig = ax is None
    if new_fig:
        _, ax = plt.subplots(figsize=(10, 6))
    if show is None:
        show = new_fig
    energy, counts = parse_msa(filepath)
    counts = counts * scale

    if normalize:
        counts = (counts - min(counts)) / max(counts)
        counts = counts + y_offset

    energy = energy + x_offset

    prefix = f"{label} " if label else ""
    (line_raw,) = ax.plot(energy, counts, alpha=0.35, lw=0.8, label=prefix + "Raw")
    smoothed = boxcar_average(counts, boxcar_width)
    (line_smooth,) = ax.plot(energy, smoothed, lw=1.5, color=line_raw.get_color(),
                             label=prefix + f"Boxcar (n={boxcar_width})")

    if edge is not None:
        ax.axvline(x=edge, color="gray", ls="--", lw=0.8, alpha=0.7, label=f"Edge ({edge:g} eV)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Energy Loss (eV)")
    ax.set_ylabel("Counts (normalized)" if normalize else "Counts")
    ax.legend(loc="upper right")

    if show:
        plt.show()
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
