"""Figure 4: UMAP of the gamma = 0.2 read-out before and after training (NVA and extended).

Trained panels: the embedding of figdata.py, which reproduces the archived coordinates,
attached by word. (For the extended vocabulary the original attached coordinates computed
in the K = 13 table's row order to the K = 120 table by row position. Both tables are
sorted by cluster, so each part-of-speech strand keeps nearly the same composition, but
individual words moved.)

Untrained panels: the network at initialisation. The original was unseeded; here the
initialisation is seeded (--seed), so the panel shows an equivalent, not identical,
structureless cloud. Features are softmax(logits) / (1 - gamma), as in the original.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from scipy.special import softmax
from typing import List, Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "src"))

import figdata
import plotting_config as pc
from srwc.model import SRNet


def untrained_coords(token_ids, seed, gamma=0.2):
    torch.manual_seed(seed)
    model = SRNet().eval()
    with torch.no_grad():
        z = model(torch.tensor(token_ids), gamma).numpy().astype(np.float64)
    emb = (softmax(z, axis=1) / (1 - gamma)).astype(np.float32)
    return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(emb)


def plot_combined_2x2(
    variant_data: Dict[str, Dict[str, Any]], # var_name -> {coords, pos_tags, titles, config}
    save_name: str,
    subfolder: str = "learning_progress",
    fig_width: float = pc.FIG_WIDTH_INCHES,
    fig_height: float = None, # If None, use width * fig_height_mult
    fig_height_mult: float = 1.1, # taller for 2 rows
    row_hspace: float = 0.35,     # Dist between Row 1 legend and Row 2 plots
    label_x: float = -0.15,
    label_y: float = 1.15,
    legend_markerscale: float = 1.0,
    show_labels: bool = True
):
    """Plot a 2x2 grid with NVA on top and Extended on bottom, with granular spacing control."""
    pc.setup_publication_style()
    
    if fig_height is None:
        fig_height = fig_width * fig_height_mult
    
    plt.rcParams['figure.autolayout'] = False
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    from matplotlib.gridspec import GridSpec
    # Outer GS for the two rows
    outer_gs = GridSpec(2, 1, hspace=row_hspace)
    
    var_names = ["nva", "extended"]
    row_labels = ["a)", "b)"]

    for row_idx, var_name in enumerate(var_names):
        data = variant_data[var_name]
        coords_list = data["coords"]
        pos_tags = data["pos_tags"]
        titles = data["titles"]
        cfg = data["config"]
        
        # Inner GS for (plots + legend)
        inner_gs = outer_gs[row_idx, 0].subgridspec(2, 2, height_ratios=[1.0, 0.15], hspace=cfg.get("legend_hspace", 0.05))
        
        # POS color mapping
        unique_pos = sorted(set(pos_tags))
        priority = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "AUX", "PROPN", "NUM", "CONJ", "CCONJ", "SCONJ", "PART", "PUNCT"]
        sorted_pos = sorted(unique_pos, key=lambda x: (priority.index(x) if x in priority else 100, x))
        
        tab20 = sns.color_palette("tab20", n_colors=20)
        base_palette = list(tab20[::2]) + list(tab20[1::2])
        if len(sorted_pos) > len(base_palette):
            extra_count = len(sorted_pos) - len(base_palette)
            base_palette += sns.color_palette("husl", n_colors=extra_count + 10)
        pos_color_map = {tag: base_palette[i] for i, tag in enumerate(sorted_pos)}
        
        handles, labels = [], []
        
        for col_idx in range(2):
            ax = fig.add_subplot(inner_gs[0, col_idx])
            coords = coords_list[col_idx]
            title = titles[col_idx]
            
            for pos in sorted_pos:
                mask = np.array(pos_tags) == pos
                scatter = ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[pos_color_map[pos]], label=pos, alpha=0.7, s=cfg["dot_size"], edgecolors='none'
                )
                if col_idx == 0:
                    handles.append(scatter)
                    labels.append(pos)
            
            ax.set_title(title, fontsize=cfg["umap_title_fontsize"], fontweight='bold')
            
            # Individual limits
            x_p = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
            y_p = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
            ax.set_xlim(coords[:, 0].min() - x_p, coords[:, 0].max() + x_p)
            ax.set_ylim(coords[:, 1].min() - y_p, coords[:, 1].max() + y_p)
            ax.grid(True, alpha=0.3)
            
            # Row label a), b) on top-left of the first plot in each row
            if show_labels and col_idx == 0:
                ax.text(label_x, label_y, row_labels[row_idx], transform=ax.transAxes, 
                        fontsize=12, fontweight='bold', va='top', ha='right')

        # Row Legend
        ax_leg = fig.add_subplot(inner_gs[1, :])
        ax_leg.axis('off')
        ax_leg.legend(
            handles, labels, loc='center', ncol=min(cfg["max_cols_pos"], len(sorted_pos)), 
            fontsize=cfg["umap_legend_fontsize"], frameon=True, 
            columnspacing=cfg["columnspacing"], handletextpad=cfg["handletextpad"],
            markerscale=legend_markerscale
        )
    
    # Global adjustment
    plt.subplots_adjust(bottom=0.05, top=0.92, wspace=0.15, left=0.08, right=0.95)
    pc.save_fig(fig, save_name, subfolder=subfolder)
    plt.close(fig)

def plot_learning_progress(
    coords_list: List[np.ndarray], # Now renamed to clarify these are UMAP coords
    pos_tags: List[str],
    titles: List[str],
    save_name: str,
    subfolder: str = "learning_progress",
    # Layout overrides from argparse
    umap_title_fontsize: float = 11.0,
    umap_legend_fontsize: float = 9.0,
    dot_size: float = 10.0,
    columnspacing: float = 0.5,
    handletextpad: float = 0.2,
    max_cols_pos: int = 3,
    fig_width: float = pc.FIG_WIDTH_INCHES,
    fig_height: float = None,
    fig_height_mult: float = 0.55,
    legend_hspace: float = 0.3,
    legend_markerscale: float = 1.0,
    wspace: float = 0.15,
    left_adj: float = 0.08,
    right_adj: float = 0.95,
    top_adj: float = 0.85,
    bottom_adj: float = 0.05,
    show_labels: bool = False
):
    """Plot precomputed UMAP coordinates with flexible publication styling."""
    pc.setup_publication_style()
    n_stages = len(coords_list)
    
    if fig_height is None:
        fig_height = fig_width * fig_height_mult
    
    plt.rcParams['figure.autolayout'] = False
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, n_stages, height_ratios=[1.0, 0.2], hspace=legend_hspace)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_stages)]
    
    # POS color mapping
    unique_pos = sorted(set(pos_tags))
    priority = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "AUX", "PROPN", "NUM", "CONJ", "CCONJ", "SCONJ", "PART", "PUNCT"]
    sorted_pos = sorted(unique_pos, key=lambda x: (priority.index(x) if x in priority else 100, x))
    
    tab20 = sns.color_palette("tab20", n_colors=20)
    base_palette = list(tab20[::2]) + list(tab20[1::2])
    if len(sorted_pos) > len(base_palette):
        extra_count = len(sorted_pos) - len(base_palette)
        base_palette += sns.color_palette("husl", n_colors=extra_count + 10)
    pos_color_map = {tag: base_palette[i] for i, tag in enumerate(sorted_pos)}
    
    alpha = 0.7
    handles, labels = [], []
    
    for i, (coords, title) in enumerate(zip(coords_list, titles)):
        ax = axes[i]
        for pos in sorted_pos:
            mask = np.array(pos_tags) == pos
            scatter = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[pos_color_map[pos]], label=pos, alpha=alpha, s=dot_size, edgecolors='none'
            )
            if i == 0:
                handles.append(scatter)
                labels.append(pos)
        
        ax.set_title(title, fontsize=umap_title_fontsize, fontweight='bold')
        
        # Individual limits for each subplot to fill the space
        x_p = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
        y_p = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
        ax.set_xlim(coords[:, 0].min() - x_p, coords[:, 0].max() + x_p)
        ax.set_ylim(coords[:, 1].min() - y_p, coords[:, 1].max() + y_p)
        
        # Ticks enabled as requested
        ax.grid(True, alpha=0.3)
        
        if show_labels:
            plot_labels = ['a)', 'b)', 'c)']
            ax.text(-0.1, 1.15, plot_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    ax_leg.legend(
        handles, labels, loc='center', ncol=min(max_cols_pos, len(sorted_pos)), 
        fontsize=umap_legend_fontsize, frameon=True, 
        columnspacing=columnspacing, handletextpad=handletextpad,
        markerscale=legend_markerscale
    )
    
    plt.subplots_adjust(bottom=bottom_adj, top=top_adj, wspace=wspace, left=left_adj, right=right_adj)
    pc.save_fig(fig, save_name, subfolder=subfolder)
    plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate UMAP learning progress plots.")
    parser.add_argument("--sr_head", type=str, default="g0.2")
    parser.add_argument("--seed", type=int, default=0, help="seed of the untrained initialisation")
    parser.add_argument("--show_subplot_labels", action="store_true") # Keep this for global override
    parser.add_argument("--combined", action="store_true", help="Generate 2x2 combined plot")
    parser.add_argument("--row_hspace", type=float, default=0.1, help="Dist between Row 1 legend and Row 2 plots")
    parser.add_argument("--label_x", type=float, default=-0.05, help="X coord for a) b) labels")
    parser.add_argument("--label_y", type=float, default=1.15, help="Y coord for a) b) labels")
    parser.add_argument("--fig_width", type=float, default=pc.FIG_WIDTH_INCHES, help="Figure width in inches")
    parser.add_argument("--fig_height", type=float, default=5.5, help="Figure height in inches (None=auto)")
    args = parser.parse_args()

    # --- PER-SETTING CONFIGURATIONS ---
    # Legend/Label settings below are per-variant. 
    # NOTE: In --combined mode, spacing/adjustment params (bottom_adj, legend_hspace etc.) 
    # are overridden by global args for better alignment across rows.
    CONFIGS = {
        "nva": {
            "umap_title_fontsize": 11.0,  # [Combined: YES] Top of each subplot
            "umap_legend_fontsize": 9.0,   # [Combined: YES] Legend below row
            "dot_size": 10.0,              # [Combined: YES] Scatter point size
            "columnspacing": 0.5,          # [Combined: YES] Space between legend columns
            "handletextpad": 0.2,          # [Combined: YES] Space between legend icon and text
            "max_cols_pos": 3,             # [Combined: YES] Max columns in legend
            "legend_hspace": 0.25,          # [Combined: YES] Dist between plot and legend
            "wspace": 0.15,                # [Combined: NO, fixed 0.15]
            "left_adj": 0.08,              # [Combined: NO, fixed 0.08]
            "right_adj": 0.95,             # [Combined: NO, fixed 0.95]
            "top_adj": 0.85,               # [Combined: NO, fixed 0.92]
            "bottom_adj": 0.05,            # [Combined: NO, fixed 0.05]
            "show_labels": False,
        },
        "extended": {
            "umap_title_fontsize": 11.0,
            "umap_legend_fontsize": 9.0,
            "dot_size": 10.0,
            "columnspacing": 0.15,
            "handletextpad": 0.05,
            "max_cols_pos": 7,
            "legend_hspace": 0.35,
            "wspace": 0.15,
            "left_adj": 0.08,
            "right_adj": 0.95,
            "top_adj": 0.85,
            "bottom_adj": 0.05,
            "show_labels": False,
        }
    }

    titles = ["Untrained", "Trained"]
    variants = [("nva", "nva", 3), ("extended", "ext", 120)]   # the tables the original used
    all_variant_results = {}
    for var_name, setting, k in variants:
        print(f"\nProcessing variant: {var_name}")
        df = figdata.assignments(args.sr_head, setting, "kmeans", k)
        token_ids = df["token_id"].tolist()
        pos_tags = df["pos"].tolist()
        all_stage_coords = [untrained_coords(token_ids, args.seed, float(args.sr_head[1:])),
                            figdata.umap_coords(args.sr_head, setting, token_ids)]
        
        config_var = CONFIGS[var_name].copy()
        if args.show_subplot_labels:
            config_var["show_labels"] = True
            
        # Pass dimensions if provided
        config_var["fig_width"] = args.fig_width
        config_var["fig_height"] = args.fig_height
        
        all_variant_results[var_name] = {
            "coords": all_stage_coords,
            "pos_tags": pos_tags,
            "titles": titles,
            "config": config_var
        }

        # Generate individual plots regardless
        plot_learning_progress(
            all_stage_coords, pos_tags, titles, 
            f"umap_learning_progress_{var_name}",
            **config_var
        )

    if args.combined and len(all_variant_results) == 2:
        print("\nGenerating combined 2x2 plot...")
        plot_combined_2x2(
            all_variant_results, 
            "umap_learning_progress_combined",
            row_hspace=args.row_hspace,
            label_x=args.label_x,
            label_y=args.label_y,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            show_labels=True # Always show a) b) for combined
        )

if __name__ == "__main__":
    main()
