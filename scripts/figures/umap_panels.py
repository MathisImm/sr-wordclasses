"""UMAP panels coloured by KMeans clusters, BGM clusters and reference tags (Figures 5, 6, 8)."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plotting_config as pc


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
    top_overhead: float = 0.55,       # Vertical space for title
    min_subplot_height: float = 1.4,   # Minimum height of each subplot
    width_divisor: float = 3.2,        # Divisor to determine subplot width from total figure width
    base_dot_size: float = 25.0,       # Initial dot size (auto-scales with K)
    columnspacing: float = 1.0,        # Horizontal space between legend columns
    handletextpad: float = 0.8,        # Space between legend marker and text
    viz_mode: str = "1x3",             # "1x3", "1x2", "2x1", "3x1"
    umap_title_fontsize: float = 10.0,
    umap_legend_fontsize: float = 8.0,
    axes_arr=None,
    wspace: float = 0.1,
    hspace: float = 0.4,
    labelspacing: float = 0.5,         # Vertical space between legend entries
    legend_y_offset: float = -0.15,
    fig_height: float = None,
    use_panel_labels: bool = False,
    legend_markerscale: float = 1.0
):
    """
    Create comparison plots: KMeans, BGM, and/or POS tags.
    """

    
    # We determine if it's the NVA plot based on the subfolder string
    is_nva = "nva" in (subfolder or "")

    def _ncol_cluster(n_c):
        """Number of legend columns for a cluster subplot."""
        if n_c <= 4:
            return n_c
        elif is_nva:
            return min(max_cols_clusters, 4)
        else:
            return min(max_cols_clusters, max(1, n_c // 10 + 3))

    def _ncol_pos():
        if is_nva or n_c_kmeans == 3:
            return min(max_cols_pos, n_pos)
        else:
            return min(max_cols_pos, 4)

    _ncol_km_pre  = _ncol_cluster(n_c_kmeans)
    _ncol_bgm_pre = _ncol_cluster(n_c_bgm)
    _ncol_pos_pre = _ncol_pos()

    import math
    n_rows_km  = math.ceil(n_c_kmeans / _ncol_km_pre)  if _ncol_km_pre  > 0 else 1
    n_rows_bgm = math.ceil(n_c_bgm    / _ncol_bgm_pre) if _ncol_bgm_pre > 0 else 1
    n_rows_pos = math.ceil(n_pos       / _ncol_pos_pre) if _ncol_pos_pre > 0 else 1
    
    # Define which subplots to show based on viz_mode
    if viz_mode == "1x3":
        show_km, show_bgm, show_pos = True, True, True
        grid_rows, grid_cols = 1, 3
    elif viz_mode == "1x2":
        show_km, show_bgm, show_pos = True, False, True
        grid_rows, grid_cols = 1, 2
    elif viz_mode == "2x1":
        show_km, show_bgm, show_pos = True, False, True
        grid_rows, grid_cols = 2, 1
    elif viz_mode == "3x1":
        show_km, show_bgm, show_pos = True, True, True
        grid_rows, grid_cols = 3, 1
    else:
        raise ValueError(f"Unknown viz_mode: {viz_mode}")

    active_rows = []
    if show_km: active_rows.append(n_rows_km)
    if show_bgm: active_rows.append(n_rows_bgm)
    if show_pos: active_rows.append(n_rows_pos)
    
    if grid_rows == 1:
        max_legend_rows = max(active_rows) if active_rows else 1
    else:
        # For vertical layouts, we might need more complex height calculation
        # but let's assume each subplot gets its own legend space below it
        max_legend_rows = max(active_rows) if active_rows else 1

    # --- Precompute axis limits ---
    x_vals = coords_2d[:, 0]
    y_vals = coords_2d[:, 1]
    x_pad = (x_vals.max() - x_vals.min()) * 0.05
    y_pad = (y_vals.max() - y_vals.min()) * 0.05
    x_lim = (x_vals.min() - x_pad, x_vals.max() + x_pad)
    y_lim = (y_vals.min() - y_pad, y_vals.max() + y_pad)
    data_aspect = (y_lim[1] - y_lim[0]) / (x_lim[1] - x_lim[0])

    # --- Figure sizing ---
    fig_width = pc.FIG_WIDTH_INCHES
    if grid_cols > 1:
        subplot_width_in = fig_width / (width_divisor * (grid_cols / 3.0))
    else:
        subplot_width_in = fig_width * 0.8 # Use most of width for vertical single column
        
    subplot_height_in = max(min_subplot_height, subplot_width_in * data_aspect)
    
    if fig_height is None:
        if grid_rows == 1:
            legend_height_in = max_legend_rows * legend_row_height + 0.5
            fig_height = subplot_height_in + legend_height_in + top_overhead
        else:
            # For vertical, we multiply height components
            legend_per_subplot = 0.8 # approximate space for legend below each
            fig_height = (subplot_height_in + legend_per_subplot) * grid_rows + top_overhead

    if axes_arr is None:
        print(f"DEBUG: viz_mode={viz_mode}, fig_width={fig_width:.2f}, fig_height={fig_height:.2f}")
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height),
                                 gridspec_kw={'wspace': wspace, 'hspace': hspace})
        if grid_rows * grid_cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
    else:
        fig = None
        if isinstance(axes_arr, list):
            axes = np.array(axes_arr)
        else:
            axes = axes_arr.flatten()

    # Dynamic dot size scaling
    max_k = max(n_c_kmeans, n_c_bgm)
    size_scale = math.sqrt(10.0 / max(10.0, max_k))
    dot_size = base_dot_size * size_scale

    valid_pos  = valid_pos_pre
    unique_pos = sorted(set(valid_pos)) if valid_pos else []
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
    
    curr_ax_idx = 0
    # Subplot: KMeans
    if show_km:
        ax = axes[curr_ax_idx]
        for cluster_id in sorted(set(clusters_km)):
            if cluster_id == -1: continue
            mask = clusters_km == cluster_id
            color, maj_pos = kmeans_color_map.get(cluster_id, (0.5, 0.5, 0.5)), kmeans_majority_map.get(cluster_id, "UNK")
            label = f'C{cluster_id}' + (f' ({maj_pos})' if show_labels else '')
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[color], label=label, alpha=0.7, s=dot_size, edgecolors='none')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), fontsize=umap_legend_fontsize, 
                  ncol=_ncol_km_pre, frameon=True, columnspacing=columnspacing, 
                  handletextpad=handletextpad, labelspacing=labelspacing,
                  markerscale=legend_markerscale)
        if use_panel_labels:
            ax.set_title("a)", fontsize=umap_title_fontsize, fontweight='bold', loc='left')
        else:
            ax.set_title("KMeans", fontsize=umap_title_fontsize, fontweight='bold')
        curr_ax_idx += 1
    
    # Subplot: BGM
    if show_bgm:
        ax = axes[curr_ax_idx]
        for cluster_id in sorted(set(clusters_bgm)):
            if cluster_id == -1: continue
            mask = clusters_bgm == cluster_id
            color, maj_pos = bgm_color_map.get(cluster_id, (0.5, 0.5, 0.5)), bgm_majority_map.get(cluster_id, "UNK")
            label = f'C{cluster_id}' + (f' ({maj_pos})' if show_labels else '')
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[color], label=label, alpha=0.7, s=dot_size, edgecolors='none')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), fontsize=umap_legend_fontsize, 
                  ncol=_ncol_bgm_pre, frameon=True, columnspacing=columnspacing, 
                  handletextpad=handletextpad, labelspacing=labelspacing,
                  markerscale=legend_markerscale)
        if use_panel_labels:
            # Determine label based on which plots are shown
            label = "b)" if show_km else "a)"
            ax.set_title(label, fontsize=umap_title_fontsize, fontweight='bold', loc='left')
        else:
            ax.set_title("BGM", fontsize=umap_title_fontsize, fontweight='bold')
        curr_ax_idx += 1
    
    # Subplot: POS Tags
    if show_pos:
        ax = axes[curr_ax_idx]
        for pos in unique_pos:
            mask = np.array(pos_tags) == pos
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[pos_color_map[pos]], label=pos, alpha=0.7, s=dot_size, edgecolors='none')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), fontsize=umap_legend_fontsize, 
                  ncol=_ncol_pos_pre, frameon=True, columnspacing=columnspacing, 
                  handletextpad=handletextpad, labelspacing=labelspacing,
                  markerscale=legend_markerscale)
        if use_panel_labels:
            # Determine label based on which plots are shown
            idx = (1 if show_km else 0) + (1 if show_bgm else 0)
            label = chr(ord('a') + idx) + ")"
            ax.set_title(label, fontsize=umap_title_fontsize, fontweight='bold', loc='left')
        else:
            ax.set_title("POS Tags", fontsize=umap_title_fontsize, fontweight='bold')
        curr_ax_idx += 1
    
    for ax in axes[:curr_ax_idx]:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)

    if fig is not None:
        # Layout adjustment
        bottom_frac = (max_legend_rows * legend_row_height + 0.5) / fig_height if grid_rows == 1 else 0.15
        top_frac = 1.0 - (top_overhead / fig_height)
        plt.subplots_adjust(left=0.08, right=0.95, bottom=bottom_frac, top=top_frac, wspace=wspace, hspace=hspace)
        
        print(f"VERIFY: Actual Figure Size: {fig.get_size_inches()} inches")
        save_name = f"umap_{viz_mode}_{subfolder}"
        pc.save_fig(fig, save_path or save_name, subfolder=subfolder)
        plt.close(fig)

