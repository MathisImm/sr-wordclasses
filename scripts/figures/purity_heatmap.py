"""Cluster-by-tag purity heatmaps (counts and row fractions), as in Figures 5, 6 and 8."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_cluster_purity_heatmap(
    labels: np.ndarray,
    pos_tags: List[str],
    save_path: str = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    annot_fontsize: float = 10.0,
    axes_arr=None,
    fig_width: float = 18.0,
    fig_height: float = None,
    wspace: float = 0.3,
    hspace: float = 0.3,
    show_colorbar: bool = False,
    show_ylabel: bool = True,
    fmt_freq: str = '.0f',
    fmt_frac: str = '.2f',
    cbar_label_freq: str = 'Token count',
    cbar_label_frac: str = 'Fraction',
    cbar_fontsize: float = None
):
    """
    Create heatmap showing cluster-POS contingency table.
    
    Args:
        labels: Cluster labels
        pos_tags: POS tags
        save_path: Path to save figure
    """
    # Build contingency table
    unique_clusters = sorted(set(labels))
    unique_pos = sorted(set(pos_tags))
    
    contingency = np.zeros((len(unique_clusters), len(unique_pos)))
    
    for i, cluster_id in enumerate(unique_clusters):
        for j, pos in enumerate(unique_pos):
            mask = (labels == cluster_id) & (np.array(pos_tags) == pos)
            contingency[i, j] = mask.sum()
    
    # Create figure if no axes provided
    if axes_arr is None:
        fh = fig_height if fig_height is not None else max(8, len(unique_clusters) * 0.4)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fh))
        ax1, ax2 = axes[0], axes[1]
    else:
        fig = None
        ax1, ax2 = axes_arr[0], axes_arr[1]
    # Left: Frequencies
    annot1 = fmt_freq if isinstance(fmt_freq, (np.ndarray, list)) else True
    fmt1 = fmt_freq if not isinstance(fmt_freq, (np.ndarray, list)) else ""
    
    cb_fs = cbar_fontsize if cbar_fontsize is not None else label_fontsize
    sns.heatmap(
        contingency,
        xticklabels=unique_pos,
        yticklabels=[f'C{c}' for c in unique_clusters],
        annot=annot1,
        fmt=fmt1,
        cmap='YlOrRd',
        ax=ax1,
        annot_kws={'size': annot_fontsize},
        cbar=show_colorbar,
        cbar_kws={'label': cbar_label_freq}
    )
    if show_colorbar:
        ax1.collections[0].colorbar.ax.yaxis.label.set_size(cb_fs)
        ax1.collections[0].colorbar.ax.tick_params(labelsize=cb_fs)
    
    ax1.set_title("Frequencies", fontsize=title_fontsize, fontweight='bold')
    ax1.set_xlabel('POS Tag', fontsize=label_fontsize)
    ax1.tick_params(labelsize=label_fontsize)
    if show_ylabel:
        ax1.set_ylabel('Cluster', fontsize=label_fontsize)
    ax1.grid(False) # Ensure grid is off for heatmaps
    
    # Right: Normalized by cluster (purity)
    contingency_norm = contingency / (contingency.sum(axis=1, keepdims=True) + 1e-10)
    
    annot2 = fmt_frac if isinstance(fmt_frac, (np.ndarray, list)) else True
    fmt2 = fmt_frac if not isinstance(fmt_frac, (np.ndarray, list)) else ""

    sns.heatmap(
        contingency_norm,
        xticklabels=unique_pos,
        yticklabels=[f'C{c}' for c in unique_clusters],
        annot=annot2,
        fmt=fmt2,
        cmap='Blues',
        ax=ax2,
        annot_kws={'size': annot_fontsize},
        cbar=show_colorbar,
        cbar_kws={'label': cbar_label_frac}
    )
    if show_colorbar:
        ax2.collections[0].colorbar.ax.yaxis.label.set_size(cb_fs)
        ax2.collections[0].colorbar.ax.tick_params(labelsize=cb_fs)
    
    ax2.set_title("Fractions", fontsize=title_fontsize, fontweight='bold')
    ax2.set_xlabel('POS Tag', fontsize=label_fontsize)
    ax2.tick_params(labelsize=label_fontsize)
    if show_ylabel:
        ax2.set_ylabel('Cluster', fontsize=label_fontsize)
    ax2.grid(False) # Ensure grid is off for heatmaps
    
    if fig is not None:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        
        # Use publication config if available in the environment/imports
        try:
            import plotting_config as pc
            # We save directly using the config values to match pc.save_fig behavior
            if save_path:
                plt.savefig(save_path, dpi=pc.PLOT_DPI if save_path.endswith('.png') else None, bbox_inches='tight')
        except ImportError:
            if save_path:
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
        plt.close(fig)
        
        if save_path:
            print(f"  ✓ Saved purity heatmap: {save_path}")


