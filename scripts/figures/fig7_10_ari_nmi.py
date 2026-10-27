#!/usr/bin/env python3
"""
Figures 7 and 10: clustering quality (ARI, NMI) over gamma and K.

Layout per figure (one figure for NVA, one for ex-... setting):

  Row 1: [ARI vs Gamma]  [shared legend]  [NMI vs Gamma]
  Row 2: [    ARI + NMI vs K  (full width, fixed gamma)    ]

Color encodes K-value (row 1) / metric (row 2).
Line style encodes method (KMeans = solid, BGM = dashed).
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import plotting_config as pc

# ── Colour / style constants ───────────────────────────────────────────────
KS_GAMMA_DEFAULT = [3, 10, 30, 50, 70, 100]
FIXED_GAMMA_DEFAULT = 0.2

METRIC_COLORS = {'ARI': '#0072B2', 'NMI': '#D55E00'}
METHOD_LS    = {'kmeans': '-',  'bgm': '--'}
METHOD_MK    = {'kmeans': 's',  'bgm': 'D'}
METHOD_LABEL = {'kmeans': 'KMeans', 'bgm': 'BGM'}


# ── Subplot helpers ────────────────────────────────────────────────────────

def _plot_metric_vs_gamma(ax, df, metric, ks, k_colors,
                          linewidth=1.2, markersize=3.5):
    """Plot metric vs gamma for specified Ks, colour = K, style = method."""
    for method in ['kmeans', 'bgm']:
        df_m = df[df['Method'] == method]
        ls   = METHOD_LS[method]
        mk   = METHOD_MK[method]
        for k in ks:
            df_k = df_m[df_m['K'] == k].sort_values('Gamma')
            if df_k.empty:
                continue
            ax.plot(df_k['Gamma'], df_k[metric],
                    color=k_colors[k], linestyle=ls, marker=mk,
                    linewidth=linewidth, markersize=markersize)

    ax.set_xlabel('Gamma')
    ax.set_ylabel(metric)
    ax.set_xticks([0.2, 0.5, 0.8])


def _plot_combined_vs_k(ax, df, fixed_gamma=FIXED_GAMMA_DEFAULT,
                        linewidth=1.3, markersize=0):
    """Plot ARI and NMI vs K on the same axes.

    Colour = metric (ARI/NMI), line style = method (KMeans/BGM).
    """
    df_g = df[df['Gamma'] == fixed_gamma].sort_values('K')
    for metric in ['ARI', 'NMI']:
        for method in ['kmeans', 'bgm']:
            df_sub = df_g[df_g['Method'] == method]
            if df_sub.empty:
                continue
            mk = METHOD_MK[method] if markersize > 0 else None
            ax.plot(df_sub['K'], df_sub[metric],
                    color=METRIC_COLORS[metric],
                    linestyle=METHOD_LS[method],
                    marker=mk,
                    linewidth=linewidth, markersize=markersize)

    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Score')
    ax.set_xscale('linear')


def _build_row1_legend(fig, ks, k_colors):
    """Draw the shared legend as a figure-level legend centered between ARI and NMI."""
    handles = []

    # --- K-value colour patches ---
    handles.append(mpatches.Patch(color='none', label='$K$'))
    for k in ks:
        handles.append(mpatches.Patch(color=k_colors[k], label=f'${k}$'))

    handles.append(mpatches.Patch(color='none', label=''))  # spacer

    # --- Method line-style entries ---
    handles.append(mpatches.Patch(color='none', label='Method'))
    for method in ['kmeans', 'bgm']:
        handles.append(
            mlines.Line2D([], [], color='#555555',
                          linestyle=METHOD_LS[method],
                          marker=METHOD_MK[method], markersize=5,
                          label=METHOD_LABEL[method])
        )

    # Use figure-level legend centered on the gap between panels a and b
    fig.legend(handles=handles, loc='center', frameon=True,
               fontsize=10, ncol=1,
               handlelength=1.6, labelspacing=0.20,
               facecolor='white', edgecolor='#cccccc', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.76))  # Adjust Y to align with row 1 center


def _build_row2_legend(ax, fixed_gamma, loc='upper right'):
    """Embed the row-2 legend directly inside the combined K plot."""
    handles = []

    # Metric colours
    handles.append(mpatches.Patch(color='none', label='Metric'))
    for metric in ['ARI', 'NMI']:
        handles.append(mpatches.Patch(color=METRIC_COLORS[metric], label=metric))

    # Method line styles
    handles.append(mpatches.Patch(color='none', label='Method'))
    for method in ['kmeans', 'bgm']:
        handles.append(
            mlines.Line2D([], [], color='#555555',
                          linestyle=METHOD_LS[method], markersize=0,
                          label=METHOD_LABEL[method])
        )

    ax.legend(handles=handles, loc=loc, frameon=True,
              fontsize=9.5, ncol=2, handlelength=2.0,
              facecolor='white', edgecolor='#cccccc', framealpha=0.8)


# ── Main figure builder ────────────────────────────────────────────────────

def plot_ari_nmi_figure(df, filename,
                        ks_gamma, fixed_gamma=FIXED_GAMMA_DEFAULT,
                        fig_width=None,
                        hspace=0.03, wspace=0.22,
                        panel_x=-0.16, panel_y=1.04, panel_fs=12,
                        legend_loc_row2='upper right'):
    """Build the 2-row figure for a single setting (NVA or Extended)."""
    pc.setup_publication_style()
    
    # Slight increase for plot-specific legibility
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.labelsize': 11,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
    })

    if fig_width is None:
        fig_width = pc.FIG_WIDTH_INCHES

    fig = plt.figure(figsize=(fig_width, fig_width * 0.65), layout='constrained')

    # GridSpec: 2 rows × 2 cols
    # Row 0: [ARI, NMI], Row 1: [Combined]
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1, 1],
        hspace=hspace,
        wspace=wspace,
    )

    # Sequential colour ramp for K values (viridis, dark→light excluded edges)
    # Generate legend colors for the specific K values
    import matplotlib
    _cmap = matplotlib.colormaps['viridis'].resampled(len(ks_gamma))
    k_colors = {k: _cmap(i) for i, k in enumerate(ks_gamma)}

    ax_ari    = fig.add_subplot(gs[0, 0])
    ax_nmi    = fig.add_subplot(gs[0, 1])
    ax_comb   = fig.add_subplot(gs[1, :])   # full-width row 2

    # --- Row 1 ---
    _plot_metric_vs_gamma(ax_ari, df, 'ARI', ks=ks_gamma, k_colors=k_colors)
    ax_ari.text(panel_x, panel_y, 'a)', transform=ax_ari.transAxes,
                fontsize=panel_fs, fontweight='bold', va='top', ha='right')

    _plot_metric_vs_gamma(ax_nmi, df, 'NMI', ks=ks_gamma, k_colors=k_colors)

    _build_row1_legend(fig, ks=ks_gamma, k_colors=k_colors)

    # --- Row 2 ---
    _plot_combined_vs_k(ax_comb, df, fixed_gamma=fixed_gamma)
    ax_comb.text(panel_x / 3, panel_y, 'b)', transform=ax_comb.transAxes,
                 fontsize=panel_fs, fontweight='bold', va='top', ha='right')
    _build_row2_legend(ax_comb, fixed_gamma, loc=legend_loc_row2)

    pc.save_fig(fig, filename, subfolder='ARI_NMI')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot ARI & NMI vs Gamma (row 1) and combined vs K (row 2).')
    parser.add_argument('--purity_csv',
                        default=os.path.join(pc.DEPOSIT, 'curves', 'purity_metrics_extended.csv'))
    parser.add_argument('--ks-gamma', type=int, nargs='+', default=KS_GAMMA_DEFAULT)
    parser.add_argument('--fixed-gamma', type=float, default=FIXED_GAMMA_DEFAULT)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--fig-width', type=float, default=pc.FIG_WIDTH_INCHES)
    parser.add_argument('--hspace', type=float, default=0.03)
    parser.add_argument('--wspace', type=float, default=0.22)
    parser.add_argument('--panel-x', type=float, default=-0.17)
    parser.add_argument('--panel-y', type=float, default=1.05)
    args = parser.parse_args()

    if not os.path.exists(args.purity_csv):
        print(f'Error: CSV not found at {args.purity_csv}')
        return

    df = pd.read_csv(args.purity_csv)
    df = df[df['K'] > 1].copy()

    df_nva  = df[df['Setting'] == 'nva']
    df_full = df[df['Setting'] == 'ex-PUNCT_X_INTJ_SPACE_SYM']
    if df_full.empty:
        df_full = df[df['Setting'] == 'full']

    if df_nva.empty:
        print('Warning: NVA data is empty.')
    if df_full.empty:
        print('Warning: Extended data is empty.')

    # Define K-values for each setting
    ks_nva = [3, 6, 10, 30, 50, 100]
    ks_ext = [13, 20, 60, 100, 120, 200]

    suffix = f'_{args.suffix}' if args.suffix else ''

    kw = dict(
        fixed_gamma=args.fixed_gamma,
        fig_width=args.fig_width,
        hspace=args.hspace,
        wspace=args.wspace,
        panel_x=args.panel_x,
        panel_y=args.panel_y,
    )

    print('Plotting NVA figure …')
    plot_ari_nmi_figure(df_nva,  f'ari_nmi_nva{suffix}',
                        ks_gamma=ks_nva, **kw)

    print('Plotting Extended figure …')
    plot_ari_nmi_figure(df_full, f'ari_nmi_extended{suffix}',
                        ks_gamma=ks_ext, legend_loc_row2='center right', **kw)

    print('Done.')


if __name__ == '__main__':
    main()
