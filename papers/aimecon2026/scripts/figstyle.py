"""Shared publication style for the AIME-Con paper figures (single source of truth).

ACL geometry (measured from the template): ``\\textwidth`` ~= 6.3 in (full / two-column span, ``figure*``)
and ``\\columnwidth`` ~= 3.15 in. Author each figure at its FINAL physical width and include it WITHOUT
scaling -- e.g. a 6.3 in figure with ``\\includegraphics[width=\\textwidth]{...}`` lands at scale 1.0, so the
point sizes below render at face value (~10 pt) against the 11 pt body. Scaling a wider figure down to
``\\textwidth`` would shrink the fonts; that is what we avoid by sizing via :func:`figsize`.

Output: vector PDF, ``savefig.dpi = 300`` (only matters for any rasterised inset), with Type-42 (TrueType)
font embedding -- camera-ready checks reject Type-3 fonts, which is matplotlib's default for PDF/PS.

Colour: the Okabe-Ito colour-blind-safe qualitative palette. We never pair red with green; line style and
marker shape carry redundant encoding so the figures also read in grayscale.
"""
import matplotlib as mpl

# ACL physical widths, inches
TEXT_WIDTH = 6.3
COL_WIDTH = 3.15

# Okabe-Ito colour-blind-safe palette (https://jfly.uni-koeln.de/color/)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}


def apply():
    """Install the publication rcParams. Call once at import time in each figure script."""
    mpl.rcParams.update({
        # --- output: vector PDF, embed TrueType (no Type-3), 300 dpi for any raster ---
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # --- fonts: serif / STIX to match the Times body, ~10 pt (close to 11 pt, legible) ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        # --- a clean, light frame ---
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "lines.linewidth": 1.8, "axes.linewidth": 0.8,
    })


def figsize(width="text", aspect=0.5):
    """Figure size in inches at a target ACL width.

    :param width: ``"text"`` (6.3 in, full / ``figure*``), ``"column"`` (3.15 in), or a float (inches).
    :param aspect: height / width ratio.
    """
    w = {"text": TEXT_WIDTH, "column": COL_WIDTH}.get(width, None)
    if w is None:
        w = float(width)
    return (w, w * aspect)
