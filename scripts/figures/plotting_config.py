"""Shared figure style (Okabe-Ito colours, 7-inch two-column width) and output location.

Figures are written to $SRWC_FIGURES (default: ./figures_out); data are read from the
deposit at $SRWC_DEPOSIT.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

PLOT_DPI = 300
PLOT_FORMAT = "pdf"
FIG_WIDTH_INCHES = 7
ASPECT_RATIO = 0.618
FONT_FAMILY = "sans-serif"
FONT_DEF = "Arial"
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

DEPOSIT = os.environ.get("SRWC_DEPOSIT", "")
BASE_OUTPUT_DIR = os.environ.get("SRWC_FIGURES", os.path.abspath("figures_out"))
OUTPUT_PREFIX = BASE_OUTPUT_DIR + "/"


def setup_publication_style():
    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update({
        "figure.dpi": PLOT_DPI, "savefig.dpi": PLOT_DPI, "savefig.format": PLOT_FORMAT,
        "figure.figsize": (FIG_WIDTH_INCHES, FIG_WIDTH_INCHES * ASPECT_RATIO), "figure.autolayout": True,
        "font.family": FONT_FAMILY, "font.serif": [FONT_DEF, "Computer Modern", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 9, "legend.frameon": False, "lines.linewidth": 1.2,
        "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })


def save_fig(fig, filename, subfolder=None):
    output_dir = os.path.join(BASE_OUTPUT_DIR, subfolder) if subfolder else BASE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{filename}.{fmt}"), bbox_inches="tight",
                    dpi=PLOT_DPI if fmt == "png" else None)
        print(f"saved {filename}.{fmt} to {output_dir}")
