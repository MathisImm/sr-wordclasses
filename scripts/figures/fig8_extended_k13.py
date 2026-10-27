"""
Figure 8: UMAPs and purity matrices for the extended vocabulary, gamma = 0.2, K = 13.
Data come from the deposit through figdata.py.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import figdata
import plotting_config as pc
from purity_heatmap import plot_cluster_purity_heatmap

def plot_umap_comparison(
    coords_2d: np.ndarray,
    clusters_km: np.ndarray,
    clusters_bgm: np.ndarray,
    pos_tags: List[str],
    n_c_kmeans: int,
    n_c_bgm: int,
    n_pos: int,
    valid_pos_pre: List[str],
    save_path: str = "",
    subfolder: str = None,
    show_labels: bool = False,
    max_cols_clusters: int = 4,
    max_cols_pos: int = 4,
    legend_row_height: float = 0.25,
    top_overhead: float = 0.55,
    min_subplot_height: float = 1.4,
    width_divisor: float = 3.2,
    base_dot_size: float = 25.0,
    columnspacing: float = 1.0,
    labelspacing: float = 0.5,
    handletextpad: float = 0.8,
    viz_mode: str = "1x3",
    umap_title_fontsize: float = 10.0,
    umap_legend_fontsize: float = 8.0,
    axes_arr=None,
    wspace: float = 0.1,
    hspace: float = 0.4,
    legend_y_offset: float = -0.15,
    fig_height: float = None
):
    """
    Create comparison plots: KMeans, BGM, and/or POS tags.
    """
    import math

    def _ncol_cluster(n_c):
        return min(max_cols_clusters, n_c) if n_c > 0 else 1

    def _ncol_pos():
        return min(max_cols_pos, n_pos) if n_pos > 0 else 1

    _ncol_km_pre  = _ncol_cluster(n_c_kmeans)
    _ncol_bgm_pre = _ncol_cluster(n_c_bgm)
    _ncol_pos_pre = _ncol_pos()

    # --- Precompute axis limits ---
    x_vals = coords_2d[:, 0]
    y_vals = coords_2d[:, 1]
    x_pad = (x_vals.max() - x_vals.min()) * 0.05
    y_pad = (y_vals.max() - y_vals.min()) * 0.05
    x_lim = (x_vals.min() - x_pad, x_vals.max() + x_pad)
    y_lim = (y_vals.min() - y_pad, y_vals.max() + y_pad)

    # Dynamic dot size scaling
    max_k = max(n_c_kmeans, n_c_bgm)
    size_scale = math.sqrt(10.0 / max(10.0, max_k))
    dot_size = base_dot_size * size_scale

    unique_pos = sorted(set(valid_pos_pre)) if valid_pos_pre else []
    priority = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "AUX", "PROPN", "NUM", "CONJ", "CCONJ", "SCONJ", "PART", "PUNCT"]
    sorted_pos = sorted(unique_pos, key=lambda x: (priority.index(x) if x in priority else 100, x))
    
    tab20 = sns.color_palette("tab20", n_colors=20)
    base_palette = list(tab20[::2]) + list(tab20[1::2])
    if len(sorted_pos) > len(base_palette):
        extra_count = len(sorted_pos) - len(base_palette)
        base_palette += sns.color_palette("husl", n_colors=extra_count + 10)
    pos_color_map = {tag: base_palette[i] for i, tag in enumerate(sorted_pos)}
    
    import colorsys
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment
    pos_array = np.array(pos_tags)
    
    def generate_variant_colors(base_rgb, n):
        if n == 1: return [base_rgb]
        h, _, s = colorsys.rgb_to_hls(*base_rgb)
        hue_spread = min(0.12, 0.02 * n) 
        hues = (np.linspace(h - hue_spread, h + hue_spread, n)) % 1.0
        l_min, l_max = 0.35, 0.75
        if n == 2:
            variants = [colorsys.hls_to_rgb(hues[0], l_max, s), colorsys.hls_to_rgb(hues[1], l_min, s)]
        elif n == 3:
            variants = [colorsys.hls_to_rgb(hues[0], l_max, s), colorsys.hls_to_rgb(hues[1], l_min, s), colorsys.hls_to_rgb(hues[2], (l_min+l_max)/2, s)]
        else:
            l_vals = np.linspace(l_max, l_min, n)
            lightnesses = np.zeros(n)
            lightnesses[0::2] = l_vals[:(n+1)//2]
            lightnesses[1::2] = l_vals[(n+1)//2:][::-1]
            variants = [colorsys.hls_to_rgb(hues[i], lightnesses[i], s) for i in range(n)]
        return variants

    valid_mask = (clusters_km != -1) & (clusters_bgm != -1)
    ck_valid = clusters_km[valid_mask]
    cb_valid = clusters_bgm[valid_mask]
    uk = sorted(list(set(ck_valid))) if len(ck_valid) > 0 else []
    ub = sorted(list(set(cb_valid))) if len(cb_valid) > 0 else []
    meta_clusters, matched_k, matched_b = [], set(), set()
    
    if len(uk) > 0 and len(ub) > 0:
        mat = np.zeros((len(uk), len(ub)))
        for i, k_lbl in enumerate(uk):
            for j, b_lbl in enumerate(ub):
                mat[i, j] = np.sum((ck_valid == k_lbl) & (cb_valid == b_lbl))
        row_ind, col_ind = linear_sum_assignment(-mat)
        for r, c in zip(row_ind, col_ind):
            meta_clusters.append({'kmeans': uk[r], 'bgm': ub[c]})
            matched_k.add(uk[r]); matched_b.add(ub[c])
            
    for k_lbl in sorted(set(clusters_km)):
        if k_lbl != -1 and k_lbl not in matched_k: meta_clusters.append({'kmeans': k_lbl})
    for b_lbl in sorted(set(clusters_bgm)):
        if b_lbl != -1 and b_lbl not in matched_b: meta_clusters.append({'bgm': b_lbl})
            
    meta_pos = []
    for mc in meta_clusters:
        pos_list = []
        if 'kmeans' in mc: pos_list.extend(pos_array[(clusters_km == mc['kmeans']) & (pos_array != "UNK")])
        if 'bgm' in mc: pos_list.extend(pos_array[(clusters_bgm == mc['bgm']) & (pos_array != "UNK")])
        if len(pos_list) > 0:
            counts = pd.Series(pos_list).value_counts()
            if "UNK" in counts and len(counts) > 1: counts = counts.drop("UNK")
            maj_pos = counts.index[0] if len(counts) > 0 else "UNK"
        else: maj_pos = unique_pos[0] if unique_pos else "UNK"
        meta_pos.append(maj_pos)
        
    pos_to_mcs = defaultdict(list)
    for i, pos in enumerate(meta_pos): pos_to_mcs[pos].append(i)
    meta_colors = {}
    for pos, mc_indices in pos_to_mcs.items():
        base_rgb = pos_color_map.get(pos, (0.5, 0.5, 0.5))
        variants = generate_variant_colors(base_rgb, len(mc_indices))
        for i, mc_idx in enumerate(mc_indices): meta_colors[mc_idx] = variants[i]
            
    kmeans_color_map, bgm_color_map, kmeans_majority_map, bgm_majority_map = {}, {}, {}, {}
    for i, mc in enumerate(meta_clusters):
        c, p = meta_colors[i], meta_pos[i]
        if 'kmeans' in mc: kmeans_color_map[mc['kmeans']], kmeans_majority_map[mc['kmeans']] = c, p
        if 'bgm' in mc: bgm_color_map[mc['bgm']], bgm_majority_map[mc['bgm']] = c, p
    
    # Subplot: KMeans
    if axes_arr is not None:
        ax = axes_arr[0]
        for cluster_id in sorted(set(clusters_km)):
            if cluster_id == -1: continue
            mask = clusters_km == cluster_id
            color, maj_pos = kmeans_color_map.get(cluster_id, (0.5, 0.5, 0.5)), kmeans_majority_map.get(cluster_id, "UNK")
            label = f'C{cluster_id}' + (f' ({maj_pos})' if show_labels else '')
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[color], label=label, alpha=0.7, s=dot_size, edgecolors='none')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), fontsize=umap_legend_fontsize, ncol=_ncol_km_pre, frameon=True, columnspacing=columnspacing, labelspacing=labelspacing, handletextpad=handletextpad)
        ax.set_title("KMeans", fontsize=umap_title_fontsize, fontweight='bold')
        
        # Subplot: BGM
        ax = axes_arr[1]
        if ax is not None:
            for cluster_id in sorted(set(clusters_bgm)):
                if cluster_id == -1: continue
                mask = clusters_bgm == cluster_id
                color, maj_pos = bgm_color_map.get(cluster_id, (0.5, 0.5, 0.5)), bgm_majority_map.get(cluster_id, "UNK")
                label = f'C{cluster_id}' + (f' ({maj_pos})' if show_labels else '')
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[color], label=label, alpha=0.7, s=dot_size, edgecolors='none')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), fontsize=umap_legend_fontsize, ncol=_ncol_bgm_pre, frameon=True, columnspacing=columnspacing, labelspacing=labelspacing, handletextpad=handletextpad)
            ax.set_title("BGM", fontsize=umap_title_fontsize, fontweight='bold')
        
        # Subplot: POS Tags
        ax = axes_arr[2]
        for pos in sorted_pos:
            mask = np.array(pos_tags) == pos
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[pos_color_map[pos]], label=pos, alpha=0.7, s=dot_size, edgecolors='none')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), fontsize=umap_legend_fontsize, ncol=_ncol_pos_pre, frameon=True, columnspacing=columnspacing, labelspacing=labelspacing, handletextpad=handletextpad)
        ax.set_title("POS Tags", fontsize=umap_title_fontsize, fontweight='bold')
        
        for ax in axes_arr:
            if ax is None: continue
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)

def process_results(generate_umap=True, generate_matrices=True, target_k=13, head_name="g0.2", setting="full", 
                    show_labels=False, no_bgm_umap=False,
                    full_filter="ex-PUNCT_X_INTJ_SPACE_SYM",
                    max_cols_clusters=3, max_cols_pos=3, 
                    legend_row_height=0.25, top_overhead=0.55, 
                    min_subplot_height=1.4, width_divisor=3.2, 
                    base_dot_size=25.0,
                    columnspacing=0.5, labelspacing=0.5, handletextpad=0.2,
                    umap_title_fontsize=11.0, umap_legend_fontsize=9.0,
                    matrix_title_fontsize=11.0, matrix_label_fontsize=11.0, 
                    matrix_annot_fontsize=9.0,
                    combined_height_ratio_umap=1.2, combined_height_ratio_matrix=1.7,
                    combined_height_multiplier=2.4, combined_hspace=0.4,
                    combined_wspace_umap=0.1, combined_wspace_matrix=0.25,
                    show_matrix_colorbar=True, umap_legend_y_offset=-0.5,
                    combined_a_x=-0.1, combined_a_y=1.05,
                    combined_b_x=-0.4, combined_b_y=1.05,
                    matrix_cbar_label_freq='Token count',
                    matrix_cbar_label_frac='Fraction',
                    matrix_cbar_fontsize=None,
                    fig_height=None):
    

    print(f"\nProcessing Specialized Plot: {head_name} | Target K={target_k} | Setting={setting}")
    pc.setup_publication_style()
    
    df = figdata.assignments(head_name, setting, "kmeans", target_k)
    token_ids = df['token_id'].tolist()

    subfolder = f"{head_name}_{setting}_k{target_k}"
    umap_coords = figdata.umap_coords(head_name, setting, token_ids) if generate_umap else None
    clusters_kmeans = df['cluster'].values
    clusters_bgm = figdata.assignments(head_name, setting, "bgm", target_k).set_index('token_id').loc[token_ids, 'cluster'].values
    pos_tags = df['pos'].values
    
    # --- Combined Figure Composition ---
    if generate_umap and generate_matrices:
        print("  -> Creating 2x1 Combined Layout...")
        plt.rcParams['figure.autolayout'] = False
    
    # 2 rows for the main layout
    if fig_height is None:
        fig_height = pc.FIG_WIDTH_INCHES * pc.ASPECT_RATIO * combined_height_multiplier
    
    fig = plt.figure(figsize=(pc.FIG_WIDTH_INCHES, fig_height))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[combined_height_ratio_umap, combined_height_ratio_matrix],
                              hspace=combined_hspace)
    
    # Row 0: UMAPs (3 or 2 subplots)
    n_umap_cols = 2 if no_bgm_umap else 3
    gs_umap = gs_main[0, 0].subgridspec(1, n_umap_cols, wspace=combined_wspace_umap)
    ax_umap_km = fig.add_subplot(gs_umap[0, 0])
    if no_bgm_umap:
        ax_umap_bgm = None
        ax_umap_pos = fig.add_subplot(gs_umap[0, 1])
    else:
        ax_umap_bgm = fig.add_subplot(gs_umap[0, 1])
        ax_umap_pos = fig.add_subplot(gs_umap[0, 2])
    
    # Row 1: 2 Matrices
    # Simplified to 2 columns so they take more space.
    # Independent gap is still controlled by combined_wspace_matrix.
    gs_mat = gs_main[1, 0].subgridspec(1, 2, wspace=combined_wspace_matrix)
    ax_mat_km_freq = fig.add_subplot(gs_mat[0, 0])
    ax_mat_km_frac = fig.add_subplot(gs_mat[0, 1])
    
    # Labels 'a)' and 'b)'
    ax_umap_km.text(combined_a_x, combined_a_y, 'a)', transform=ax_umap_km.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
    ax_mat_km_freq.text(combined_b_x, combined_b_y, 'b)', transform=ax_mat_km_freq.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
    
    plot_umap_comparison(
        coords_2d=umap_coords, clusters_km=clusters_kmeans, clusters_bgm=clusters_bgm,
        pos_tags=pos_tags,
        n_c_kmeans=len(set(clusters_kmeans)) - (1 if -1 in clusters_kmeans else 0),
        n_c_bgm=len(set(clusters_bgm)) - (1 if -1 in clusters_bgm else 0),
        n_pos=len(set([p for p in pos_tags if p != "UNK"])),
        valid_pos_pre=[p for p in pos_tags if p != "UNK"],
        show_labels=show_labels,
        max_cols_clusters=max_cols_clusters, max_cols_pos=max_cols_pos,
        legend_row_height=legend_row_height, top_overhead=top_overhead,
        min_subplot_height=min_subplot_height, width_divisor=width_divisor,
        base_dot_size=base_dot_size, columnspacing=columnspacing, labelspacing=labelspacing, handletextpad=handletextpad,
        viz_mode="1x3", axes_arr=[ax_umap_km, ax_umap_bgm, ax_umap_pos],
        umap_title_fontsize=umap_title_fontsize, umap_legend_fontsize=umap_legend_fontsize,
        legend_y_offset=umap_legend_y_offset,
        fig_height=fig_height
    )
    
    
    # Custom formatter to strip leading zeros: .nn instead of 0.nn
    # We achieve this by converting to string and replacing '0.' with '.'
    # But Seaborn's fmt doesn't support a lambda. We'll use a hack by passing the fmt
    # that we'll implement in the library or pre-formatting if we had control.
    # Since I modified the library to accept fmt_frac, I can pass a format string.
    # However, standard python format strings don't easily strip leading zeros.
    # Actually, I can pre-calculate the labels for the heatmap!
    
    # Let's compute manual annotations for the fraction plot to get .nn format
    unique_clusters = sorted(set(clusters_kmeans))
    unique_pos = sorted(set(pos_tags))
    contingency = np.zeros((len(unique_clusters), len(unique_pos)))
    for i, cluster_id in enumerate(unique_clusters):
        for j, pos in enumerate(unique_pos):
            mask = (clusters_kmeans == cluster_id) & (np.array(pos_tags) == pos)
            contingency[i, j] = mask.sum()
    contingency_norm = contingency / (contingency.sum(axis=1, keepdims=True) + 1e-10)
    
    # Format matrices to strings
    annot_frac = []
    for row in contingency_norm:
        annot_row = []
        for val in row:
            s = f"{val:.2f}"
            if s.startswith("0."): s = s[1:]
            if s == "1.00": s = "1.0"
            annot_row.append(s)
        annot_frac.append(annot_row)
    annot_frac = np.array(annot_frac)

    plot_cluster_purity_heatmap(
        clusters_kmeans, pos_tags.tolist(), None,
        title_fontsize=matrix_title_fontsize, label_fontsize=matrix_label_fontsize, annot_fontsize=matrix_annot_fontsize,
        axes_arr=[ax_mat_km_freq, ax_mat_km_frac],
        wspace=combined_wspace_matrix, hspace=combined_hspace,
        show_colorbar=show_matrix_colorbar,
        show_ylabel=True,
        fmt_frac=annot_frac,
        cbar_label_freq=matrix_cbar_label_freq,
        cbar_label_frac=matrix_cbar_label_frac,
        cbar_fontsize=matrix_cbar_fontsize
    )
    
    if generate_umap and generate_matrices:
        save_name = f"umap_matrix_combined_{subfolder}"
        pc.save_fig(fig, save_name, subfolder=subfolder)
        plt.close(fig)
        print(f"  ✓ Saved combined plot to {subfolder}/{save_name}")

def main():
    import argparse
    # -------------------------------------------------------
    # ALL IMPORTANT FONT AND FIGURE PARAMS EXPOSED HERE
    # -------------------------------------------------------
    TARGET_K             = 13     # Number of clusters (K)
    HEAD_NAME            = "g0.2"   # Head name from the model (e.g., g0.2, g0.5)
    SETTING              = "full"   # Token subset ("nva" or "full")
    MAX_COLS_CLUSTERS    = 5      # Maximum columns for cluster legends
    MAX_COLS_POS         = 5      # Maximum columns for POS tag legend
    LEGEND_ROW_HEIGHT    = 0.15   # Estimated height per row in legends for layout (unused in some logic)
    TOP_OVERHEAD         = 0.55   # Vertical space overhead at the top
    MIN_SUBPLOT_HEIGHT   = 1.4    # Minimum height for each subplot
    WIDTH_DIVISOR        = 3.2    # Scale factor for legend width
    BASE_DOT_SIZE        = 10.0   # Base size for UMAP scatter dots
    COLUMNSPACING        = 0.15    # Horizontal whitespace between legend columns
    LABELSPACING         = 0.15    # Vertical whitespace between legend rows
    HANDLETEXTPAD        = 0.05   # Pad between legend handle and text
    UMAP_TITLE_FONTSIZE  = 10.0   # Font size for UMAP subplot titles
    UMAP_LEGEND_FONTSIZE = 6.0    # Font size for UMAP legend text
    MATRIX_TITLE_FONTSIZE = 8.0   # Font size for matrix titles
    MATRIX_LABEL_FONTSIZE = 7.0   # Font size for matrix axis labels
    MATRIX_ANNOT_FONTSIZE = 6.0   # Font size for matrix cell annotations    
    NO_BGM_UMAP          = True
    
    COMBINED_HEIGHT_RATIO_UMAP = 1.4   # Relative height of the UMAP row (top)
    COMBINED_HEIGHT_RATIO_MATRIX = 1.3 # Relative height of the Matrix row (bottom)
    COMBINED_HEIGHT_MULTIPLIER = 1.5   # Factor to scale the total figure height
    COMBINED_WSPACE_UMAP = 0.15         # Horizontal space between UMAPs
    COMBINED_WSPACE_MATRIX = 0.2      # Horizontal space between Matrices
    COMBINED_HSPACE      = 0.4         # Vertical space between UMAP and Matrix rows
    SHOW_MATRIX_COLORBAR = False        # Whether to show colorbars for heatmaps
    UMAP_LEGEND_Y_OFFSET = -0.1        # Vertical offset for UMAP legend 
    COMBINED_A_X         = -0.15        # X position of 'a)' label
    COMBINED_A_Y         = 1.08        # Y position of 'a)' label
    COMBINED_B_X         = -0.15        # X position of 'b)' label
    COMBINED_B_Y         = 1.07        # Y position of 'b)' label
    FIG_HEIGHT           = None        # Optional manual override
    
    MATRIX_CBAR_LABEL_FREQ = 'Token count'
    MATRIX_CBAR_LABEL_FRAC = 'Fraction'
    MATRIX_CBAR_FONTSIZE   = 7.0
    # -------------------------------------------------------

    parser = argparse.ArgumentParser(description="Specialized plotter for full k=13 g0.20 combined figure.")
    parser.add_argument("--no_umap", action="store_true")
    parser.add_argument("--no_matrices", action="store_true")
    parser.add_argument("--target_k", type=int, default=TARGET_K)
    parser.add_argument("--head_name", type=str, default=HEAD_NAME)
    parser.add_argument("--setting", type=str, choices=["nva", "full"], default=SETTING)
    parser.add_argument("--full-filter", type=str, default="ex-PUNCT_X_INTJ_SPACE_SYM",
                        help="Filter string for full-setting assignment files (empty = old unfiltered results)")
    parser.add_argument("--show_labels", action="store_true")
    parser.add_argument("--no_bgm_umap", action="store_true", help="Only include KMeans and POS Tags in UMAP row.")
    
    # Layout tuning arguments
    parser.add_argument("--max_cols_clusters", type=int, default=MAX_COLS_CLUSTERS)
    parser.add_argument("--max_cols_pos", type=int, default=MAX_COLS_POS)
    parser.add_argument("--legend_row_height", type=float, default=LEGEND_ROW_HEIGHT)
    parser.add_argument("--top_overhead", type=float, default=TOP_OVERHEAD)
    parser.add_argument("--min_subplot_height", type=float, default=MIN_SUBPLOT_HEIGHT)
    parser.add_argument("--width_divisor", type=float, default=WIDTH_DIVISOR)
    parser.add_argument("--base_dot_size", type=float, default=BASE_DOT_SIZE)
    parser.add_argument("--columnspacing", type=float, default=COLUMNSPACING)
    parser.add_argument("--labelspacing", type=float, default=LABELSPACING)
    parser.add_argument("--handletextpad", type=float, default=HANDLETEXTPAD)
    
    parser.add_argument("--umap_title_fontsize", type=float, default=UMAP_TITLE_FONTSIZE)
    parser.add_argument("--umap_legend_fontsize", type=float, default=UMAP_LEGEND_FONTSIZE)
    
    parser.add_argument("--matrix_title_fontsize", type=float, default=MATRIX_TITLE_FONTSIZE)
    parser.add_argument("--matrix_label_fontsize", type=float, default=MATRIX_LABEL_FONTSIZE)
    parser.add_argument("--matrix_annot_fontsize", type=float, default=MATRIX_ANNOT_FONTSIZE)
    
    parser.add_argument("--combined_height_ratio_umap", type=float, default=COMBINED_HEIGHT_RATIO_UMAP)
    parser.add_argument("--combined_height_ratio_matrix", type=float, default=COMBINED_HEIGHT_RATIO_MATRIX)
    parser.add_argument("--combined_height_multiplier", type=float, default=COMBINED_HEIGHT_MULTIPLIER)
    parser.add_argument("--combined_wspace_umap", type=float, default=COMBINED_WSPACE_UMAP)
    parser.add_argument("--combined_wspace_matrix", type=float, default=COMBINED_WSPACE_MATRIX)
    parser.add_argument("--combined_hspace", type=float, default=COMBINED_HSPACE)
    parser.add_argument("--show_matrix_colorbar", action="store_true", default=SHOW_MATRIX_COLORBAR)
    parser.add_argument("--umap_legend_y_offset", type=float, default=UMAP_LEGEND_Y_OFFSET)
    parser.add_argument("--combined_a_x", type=float, default=COMBINED_A_X)
    parser.add_argument("--combined_a_y", type=float, default=COMBINED_A_Y)
    parser.add_argument("--combined_b_x", type=float, default=COMBINED_B_X)
    parser.add_argument("--combined_b_y", type=float, default=COMBINED_B_Y)
    
    parser.add_argument("--matrix_cbar_label_freq", type=str, default=MATRIX_CBAR_LABEL_FREQ)
    parser.add_argument("--matrix_cbar_label_frac", type=str, default=MATRIX_CBAR_LABEL_FRAC)
    parser.add_argument("--matrix_cbar_fontsize", type=float, default=MATRIX_CBAR_FONTSIZE)
    parser.add_argument("--fig_height", type=float, default=FIG_HEIGHT)
    
    args = parser.parse_args()
    
    process_results(generate_umap=not args.no_umap, generate_matrices=not args.no_matrices, 
                    target_k=args.target_k, head_name=args.head_name, setting=args.setting, show_labels=args.show_labels,
                    full_filter=args.full_filter,
                    no_bgm_umap=args.no_bgm_umap or NO_BGM_UMAP,
                    max_cols_clusters=args.max_cols_clusters, max_cols_pos=args.max_cols_pos, 
                    legend_row_height=args.legend_row_height, top_overhead=args.top_overhead, 
                    min_subplot_height=args.min_subplot_height, width_divisor=args.width_divisor, 
                    base_dot_size=args.base_dot_size, columnspacing=args.columnspacing, labelspacing=args.labelspacing, handletextpad=args.handletextpad,
                    umap_title_fontsize=args.umap_title_fontsize, umap_legend_fontsize=args.umap_legend_fontsize,
                    matrix_title_fontsize=args.matrix_title_fontsize, matrix_label_fontsize=args.matrix_label_fontsize, matrix_annot_fontsize=args.matrix_annot_fontsize,
                    combined_height_ratio_umap=args.combined_height_ratio_umap, combined_height_ratio_matrix=args.combined_height_ratio_matrix,
                    combined_height_multiplier=args.combined_height_multiplier, 
                    combined_wspace_umap=args.combined_wspace_umap, 
                    combined_wspace_matrix=args.combined_wspace_matrix, 
                    combined_hspace=args.combined_hspace,
                    show_matrix_colorbar=args.show_matrix_colorbar, umap_legend_y_offset=args.umap_legend_y_offset,
                    combined_a_x=args.combined_a_x, combined_a_y=args.combined_a_y,
                    combined_b_x=args.combined_b_x, combined_b_y=args.combined_b_y,
                    matrix_cbar_label_freq=args.matrix_cbar_label_freq,
                    matrix_cbar_label_frac=args.matrix_cbar_label_frac,
                    matrix_cbar_fontsize=args.matrix_cbar_fontsize,
                    fig_height=args.fig_height)

if __name__ == "__main__":
    main()
