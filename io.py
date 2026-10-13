"""Reading EMSA/MSA spectrum files."""
import glob
import os

import numpy as np


def parse_msa(filepath, return_header=False, encoding="latin-1"):
    """Read an EMSA/MSA spectrum file.

    Merges both notebook versions: the boxcar_plot version returned
    ``(energies, counts)``; the noise_analysis version also parsed the
    ``#KEY : value`` header and returned ``(hdr, energies, counts)``.

    Parameters
    ----------
    filepath : str
    return_header : bool
        If True, return ``(hdr, E, C)``; otherwise ``(E, C)``.
    encoding : str
        Latin-1 by default, which tolerates the non-UTF-8 bytes (e.g. µ, °)
        Gatan writes into headers.

    Data lines are ``energy, counts`` (a trailing comma or extra columns are
    tolerated; only the first two are used). Non-numeric lines are skipped.
    """
    hdr, energies, counts = {}, [], []
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if ":" in s:
                    k, v = s[1:].split(":", 1)
                    hdr[k.strip()] = v.strip()
                continue
            if "," in s:
                p = s.split(",")
                try:
                    energies.append(float(p[0]))
                    counts.append(float(p[1]))
                except ValueError:
                    continue
    E, C = np.array(energies), np.array(counts)
    return (hdr, E, C) if return_header else (E, C)


def load_spectra(root, pattern="**/*.msa", verbose=True):
    """Load every MSA file under ``root`` into ``{basename: dict(E, C, hdr)}``."""
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    spectra = {}
    for fp in files:
        h, E, C = parse_msa(fp, return_header=True)
        spectra[os.path.basename(fp)] = dict(E=E, C=C, hdr=h, path=fp)

    if verbose:
        print(f"{len(spectra)} spectra loaded\n")
        for n, d in spectra.items():
            E = d["E"]
            print(f"{n[:45]:45s} N={len(E)}  E=[{E.min():7.0f},{E.max():7.0f}]  "
                  f"offset={d['hdr'].get('OFFSET')}  mag={d['hdr'].get('MAGCAM')}")
    return spectra
