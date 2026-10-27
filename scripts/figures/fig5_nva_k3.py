"""
Figure 5: UMAPs, purity matrices and the SR flow network for NVA, gamma = 0.2, K = 3.
Data come from the deposit through figdata.py (see there for what differs in the UMAP panels).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import figdata
import plotting_config as pc
from purity_heatmap import plot_cluster_purity_heatmap
from umap_panels import plot_umap_comparison

def compute_inter_cluster_sr(X, labels, token_ids):
    n_tokens = X.shape[0]
    unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
    n_clusters = len(unique_clusters)
    
    if X.shape[1] > n_tokens:
        sr_matrix = X[:, token_ids]
    else:
        sr_matrix = X
        
    cluster_sr = np.zeros((n_clusters, n_clusters))
    
    for i, c_from in enumerate(unique_clusters):
        mask_from = (labels == c_from)
        for j, c_to in enumerate(unique_clusters):
            mask_to = (labels == c_to)
            if np.any(mask_from) and np.any(mask_to):
                block = sr_matrix[np.ix_(mask_from, mask_to)]
                cluster_sr[i, j] = block.sum() / mask_from.sum()
                
    row_sums = cluster_sr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cluster_sr = cluster_sr / row_sums
    
    return cluster_sr, unique_clusters

def draw_connectivity_heatmap(matrix, ax, k, labels=None, annot_fontsize=10, tick_fontsize=10, label_fontsize=12, linewidth=0.5, linecolor="none", show_colorbar=True):
    annot_size = max(4, annot_fontsize - int(k / 4))
    
    annot_data = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val >= 0.995: text = "1.0"
            elif val < 0.005: text = "0"
            else: text = f"{val:.2f}"
            annot_data[i, j] = text
                
    sns.heatmap(matrix, annot=annot_data, fmt='', cmap='Blues', ax=ax, square=True, 
                cbar=show_colorbar,
                cbar_kws={'label': 'Fraction', 'shrink': 0.8} if show_colorbar else None, 
                annot_kws={'size': annot_size},
                xticklabels=labels if labels else [f"c{i}" for i in range(k)],
                yticklabels=labels if labels else [f"c{i}" for i in range(k)],
                linewidths=linewidth, linecolor=linecolor)
    
    plt.setp(ax.get_yticklabels(), va="center", ma="center")
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xlabel("Target Cluster", fontsize=label_fontsize)
    ax.set_ylabel("Source Cluster", fontsize=label_fontsize)
    ax.grid(False)

def draw_static_network(matrix, ax, k, cluster_pos=None, threshold=0.01, edge_color_source=False, node_size=600, label_fontsize=10, legend_fontsize=8, net_margin=0.2):
    G = nx.DiGraph()
    num_clusters = matrix.shape[0]
    G.add_nodes_from(range(num_clusters))
    
    edges = []
    weights = []
    for i in range(num_clusters):
        row = matrix[i, :].copy()
        row[i] = 0 
        top_indices = np.argsort(row)[::-1][:3]
        
        for j in top_indices:
            if row[j] > threshold:
                edges.append((i, j))
                weights.append(row[j])
                G.add_edge(i, j, weight=row[j])
                
    if len(edges) == 0: return
        
    layout_k = 2.5 / (num_clusters ** 0.5) if num_clusters > 1 else 2.0
    pos = nx.spring_layout(G, k=layout_k, iterations=100, seed=42)
    
    if cluster_pos is not None:
        unique_pos_tags = sorted(list(set(cluster_pos.values())))
        pos_color_map = {tag: pc.OKABE_ITO[idx % len(pc.OKABE_ITO)] for idx, tag in enumerate(unique_pos_tags)}
        node_colors = [pos_color_map[cluster_pos.get(i, "Unknown")] if cluster_pos.get(i, "Unknown") in pos_color_map else '#1f77b4' for i in range(num_clusters)]
        legend_patches = [mpatches.Patch(color=color, label=tag) for tag, color in pos_color_map.items()]
    else:
        node_colors = ['#1f77b4'] * num_clusters
        legend_patches = []
        
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_size, alpha=0.9)
    
    max_w = max(weights) if weights else 1.0
    edge_widths = [2.0 * (w / max_w) for w in weights]
    
    if cluster_pos is not None:
        edge_colors = [node_colors[u] if edge_color_source else node_colors[v] for u, v in edges]
    else:
        edge_colors = ['gray'] * len(edges)
        
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges,
                           width=edge_widths, arrowsize=15,
                           edge_color=edge_colors, alpha=0.6,
                           connectionstyle='arc3,rad=0.1',
                           node_size=node_size)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=label_fontsize, font_color='white', font_family=pc.FONT_FAMILY)
    
    if legend_patches:
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Majority POS", fontsize=legend_fontsize)
    ax.margins(net_margin)
    ax.axis('off')

# --- Main Tri-Plot Assembly ---

def process_results(target_k=3, head_name="g0.2", setting="nva", show_labels=False,
                    max_cols_clusters=4, max_cols_pos=4, legend_row_height=0.25, top_overhead=0.55, 
                    min_subplot_height=1.4, width_divisor=3.2, base_dot_size=25.0,
                    columnspacing=1.0, handletextpad=0.8,
                    umap_title_fontsize=11.0, umap_legend_fontsize=9.0,
                    matrix_title_fontsize=11.0, matrix_label_fontsize=11.0, matrix_annot_fontsize=9.0,
                    net_font_size=10, net_label_size=10, net_legend_size=8, net_node_size=600, net_margin=0.2,
                    hm_font_size=11, hm_tick_size=10, hm_label_size=11, hm_annot_size=9,
                    umap_wspace=0.1, umap_hspace=0.4,
                    combined_height_ratio_umap=0.8, combined_height_ratio_matrix=0.6, combined_height_ratio_net=0.8,
                    combined_height_multiplier=1.8, combined_wspace=2.0, combined_hspace=0.6, combined_hspace_row23=0.0,
                    show_matrix_colorbar=False, umap_legend_y_offset=-0.15,
                    combined_row2_gap=1,
                    combined_a_x=-0.1, combined_a_y=1.15,
                    combined_b_x=-0.4, combined_b_y=1.15,
                    combined_c_x=-0.1, combined_c_y=1.11,
                    combined_d_x=-0.3, combined_d_y=1.1):
    

    print(f"\nProcessing Tri-Plot: {head_name} | Target K={target_k} | Setting={setting}")
    pc.setup_publication_style()
    
    df = figdata.assignments(head_name, setting, "kmeans", target_k)
    token_ids = df['token_id'].tolist()

    try:
        X_concat = figdata.features(head_name, setting, token_ids)
        subfolder = f"{head_name}_{setting}_k{target_k}"
        umap_coords = figdata.umap_coords(head_name, setting, token_ids)
        clusters_kmeans = df['cluster'].values
        clusters_bgm = figdata.assignments(head_name, setting, "bgm", target_k).set_index('token_id').loc[token_ids, 'cluster'].values
        pos_tags = df['pos'].values

        # SR Flow Data
        sr_matrix, unique_clusters = compute_inter_cluster_sr(X_concat, clusters_kmeans, token_ids)
        cluster_pos = {}
        if 'pos' in df.columns:
            for i, c in enumerate(unique_clusters):
                c_df = df[df['cluster'] == c]
                if not c_df.empty:
                    cluster_pos[i] = c_df['pos'].value_counts().idxmax()
                else:
                    cluster_pos[i] = "Unknown"

        # --- Plot Assembly ---
        print("  -> Assembling Tri-Row Figure...")
        plt.rcParams['figure.autolayout'] = False
        
        fig = plt.figure(figsize=(pc.FIG_WIDTH_INCHES, pc.FIG_WIDTH_INCHES * pc.ASPECT_RATIO * combined_height_multiplier))
        
        # 3 Rows, 12 Columns GridSpec for more fine-grained control
        gs = fig.add_gridspec(3, 12, height_ratios=[combined_height_ratio_umap, combined_height_ratio_matrix, combined_height_ratio_net],
                              wspace=combined_wspace, hspace=combined_hspace)
        
        # Row 1: UMAPs (0-4, 4-8, 8-12)
        ax_umap_km = fig.add_subplot(gs[0, 0:4])
        ax_umap_bgm = fig.add_subplot(gs[0, 4:8])
        ax_umap_pos = fig.add_subplot(gs[0, 8:12])
        
        # Row 2: Matrices
        # Calculate centering based on gap
        # Total columns = 12. Total width of 2 matrices = 8 or something.
        # Let's say each matrix is 4 wide. 4+gap+4 = 8+gap.
        # Start at (12 - (8+gap))/2
        r2_start = max(0, (12 - 8 - combined_row2_gap) // 2)
        ax_mat_km_freq = fig.add_subplot(gs[1, r2_start : r2_start+4])
        ax_mat_km_frac = fig.add_subplot(gs[1, r2_start+4+combined_row2_gap : r2_start+8+combined_row2_gap])
        
        # Row 3: Networks (0-4 Network, 7-12 Heatmap - gap in between)
        # We can apply the extra hspace here by shifting the subplots down if needed, 
        # or more cleanly by using a nested GridSpec or just adjusting the positions.
        # However, a simpler way is to just use subplots_adjust specifically or a spacer row.
        # But let's use the simplest: adjust the position of the subplots manually if hspace_row23 > 0
        ax_net_graph = fig.add_subplot(gs[2, 0:4])
        ax_net_heatmap = fig.add_subplot(gs[2, 7:12])
        
        if combined_hspace_row23 != 0:
            for ax in [ax_net_graph, ax_net_heatmap]:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 - combined_hspace_row23, pos.width, pos.height])
        
        # Subplot Labels
        ax_umap_km.text(combined_a_x, combined_a_y, 'a)', transform=ax_umap_km.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax_mat_km_freq.text(combined_b_x, combined_b_y, 'b)', transform=ax_mat_km_freq.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax_net_graph.text(combined_c_x, combined_c_y, 'c)', transform=ax_net_graph.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax_net_heatmap.text(combined_d_x, combined_d_y, 'd)', transform=ax_net_heatmap.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        
        # 1. Plot UMAPs
        plot_umap_comparison(
            coords_2d=umap_coords, clusters_km=clusters_kmeans, clusters_bgm=clusters_bgm,
            pos_tags=pos_tags,
            n_c_kmeans=len(set(clusters_kmeans)) - (1 if -1 in clusters_kmeans else 0),
            n_c_bgm=len(set(clusters_bgm)) - (1 if -1 in clusters_bgm else 0),
            n_pos=len(set([p for p in pos_tags if p != "UNK"])),
            valid_pos_pre=[p for p in pos_tags if p != "UNK"],
            save_path="temp", subfolder=subfolder,
            show_labels=show_labels,
            max_cols_clusters=max_cols_clusters, max_cols_pos=max_cols_pos,
            legend_row_height=legend_row_height, top_overhead=top_overhead,
            min_subplot_height=min_subplot_height, width_divisor=width_divisor,
            base_dot_size=base_dot_size, columnspacing=columnspacing, handletextpad=handletextpad,
            viz_mode="1x3", axes_arr=[ax_umap_km, ax_umap_bgm, ax_umap_pos],
            umap_title_fontsize=umap_title_fontsize, umap_legend_fontsize=umap_legend_fontsize,
            wspace=combined_wspace, hspace=combined_hspace, legend_y_offset=umap_legend_y_offset
        )
        
        # 2. Plot Matrices
        plot_cluster_purity_heatmap(
            clusters_kmeans, pos_tags.tolist(), None,
            title_fontsize=matrix_title_fontsize, label_fontsize=matrix_label_fontsize, annot_fontsize=matrix_annot_fontsize,
            axes_arr=[ax_mat_km_freq, ax_mat_km_frac],
            wspace=combined_wspace, hspace=combined_hspace,
            show_colorbar=show_matrix_colorbar,
            show_ylabel=True
        )
        
        # 3. Plot SR Networks
        draw_static_network(sr_matrix, ax_net_graph, target_k, cluster_pos=cluster_pos, node_size=net_node_size, label_fontsize=net_label_size, legend_fontsize=net_legend_size, net_margin=net_margin)
        
        # Prepare labels: ID\n(POS)
        matrix_labels = [f"{i}\n({cluster_pos[i]})" for i in range(sr_matrix.shape[0])]

        draw_connectivity_heatmap(sr_matrix, ax_net_heatmap, target_k, labels=matrix_labels, 
                                 annot_fontsize=hm_annot_size, tick_fontsize=hm_tick_size, 
                                 label_fontsize=hm_label_size, show_colorbar=False)
        # ax_net_heatmap.set_title(f"Inter-Cluster SR Heatmap (KMeans k={target_k})", fontsize=hm_font_size, fontweight='bold')
        
        save_name = f"tri_combined_{subfolder}"
        pc.save_fig(fig, save_name, subfolder=subfolder)
        plt.close(fig)
        print(f"  ✓ Saved '{save_name}.{pc.PLOT_FORMAT}' to {pc.OUTPUT_PREFIX}/{subfolder}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {subfolder}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate 3-row Tri-Plot: UMAPs, Matrices, and Flow Networks.")
    parser.add_argument("--target_k", type=int, default=3)
    parser.add_argument("--head_name", type=str, default="g0.2")
    parser.add_argument("--setting", type=str, choices=["nva", "full"], default="nva")
    parser.add_argument("--show_labels", action="store_true")

    # Layout tuning arguments
    parser.add_argument("--max_cols_clusters", type=int, default=3)
    parser.add_argument("--max_cols_pos", type=int, default=3)
    parser.add_argument("--legend_row_height", type=float, default=0.25)
    parser.add_argument("--top_overhead", type=float, default=0.55)
    parser.add_argument("--min_subplot_height", type=float, default=1.4)
    parser.add_argument("--width_divisor", type=float, default=3.2)
    parser.add_argument("--base_dot_size", type=float, default=25.0, help="Base dot size (auto-scales with K)")
    parser.add_argument("--columnspacing", type=float, default=0.5, help="Horizontal space between legend columns")
    parser.add_argument("--handletextpad", type=float, default=0.2, help="Space between legend marker and text")
    
    parser.add_argument("--umap_title_fontsize", type=float, default=11.0)
    parser.add_argument("--umap_legend_fontsize", type=float, default=9.0)
    parser.add_argument("--matrix_title_fontsize", type=float, default=11.0)
    parser.add_argument("--matrix_label_fontsize", type=float, default=11.0)
    parser.add_argument("--matrix_annot_fontsize", type=float, default=9.0)
    parser.add_argument("--net_font_size", type=int, default=10)
    parser.add_argument("--net_label_size", type=int, default=10)
    parser.add_argument("--net_legend_size", type=int, default=8)
    parser.add_argument("--net_node_size", type=int, default=600)
    parser.add_argument("--net_margin", type=float, default=0.2)
    parser.add_argument("--hm_font_size", type=int, default=11)
    parser.add_argument("--hm_tick_size", type=int, default=10)
    parser.add_argument("--hm_label_size", type=int, default=11)
    parser.add_argument("--hm_annot_size", type=int, default=9)
    
    parser.add_argument("--umap_wspace", type=float, default=0.1)
    parser.add_argument("--umap_hspace", type=float, default=0.4)
    parser.add_argument("--combined_height_ratio_umap", type=float, default=0.8)
    parser.add_argument("--combined_height_ratio_matrix", type=float, default=0.6)
    parser.add_argument("--combined_height_ratio_net", type=float, default=0.8)
    parser.add_argument("--combined_height_multiplier", type=float, default=1.8)
    parser.add_argument("--combined_wspace", type=float, default=2.0)
    parser.add_argument("--combined_hspace", type=float, default=0.6)
    parser.add_argument("--combined_hspace_row23", type=float, default=-0.02, help="Extra vertical space specifically between Row 2 and Row 3")
    parser.add_argument("--combined_row2_gap", type=int, default=1, help="Horizontal gap between Row 2 matrices (in grid columns, 0-4)")
    parser.add_argument("--show_matrix_colorbar", action="store_true", default=False)
    parser.add_argument("--umap_legend_y_offset", type=float, default=-0.15)
    parser.add_argument("--combined_a_x", type=float, default=-0.1)
    parser.add_argument("--combined_a_y", type=float, default=1.15)
    parser.add_argument("--combined_b_x", type=float, default=-0.4)
    parser.add_argument("--combined_b_y", type=float, default=1.15)
    parser.add_argument("--combined_c_x", type=float, default=-0.1)
    parser.add_argument("--combined_c_y", type=float, default=1.05)
    parser.add_argument("--combined_d_x", type=float, default=-0.3)
    parser.add_argument("--combined_d_y", type=float, default=1.05)
    
    args = parser.parse_args()
    
    process_results(
        target_k=args.target_k, head_name=args.head_name, setting=args.setting, show_labels=args.show_labels,
        max_cols_clusters=args.max_cols_clusters, max_cols_pos=args.max_cols_pos, 
        legend_row_height=args.legend_row_height, top_overhead=args.top_overhead, 
        min_subplot_height=args.min_subplot_height, width_divisor=args.width_divisor, 
        base_dot_size=args.base_dot_size, columnspacing=args.columnspacing, handletextpad=args.handletextpad,
        umap_title_fontsize=args.umap_title_fontsize, umap_legend_fontsize=args.umap_legend_fontsize,
        matrix_title_fontsize=args.matrix_title_fontsize, matrix_label_fontsize=args.matrix_label_fontsize, matrix_annot_fontsize=args.matrix_annot_fontsize,
        net_font_size=args.net_font_size, net_label_size=args.net_label_size, net_legend_size=args.net_legend_size, net_node_size=args.net_node_size,
        net_margin=args.net_margin,
        hm_font_size=args.hm_font_size, hm_tick_size=args.hm_tick_size, hm_label_size=args.hm_label_size, hm_annot_size=args.hm_annot_size,
        umap_wspace=args.umap_wspace, umap_hspace=args.umap_hspace,
        combined_height_ratio_umap=args.combined_height_ratio_umap, combined_height_ratio_matrix=args.combined_height_ratio_matrix, combined_height_ratio_net=args.combined_height_ratio_net,
        combined_height_multiplier=args.combined_height_multiplier, combined_wspace=args.combined_wspace, combined_hspace=args.combined_hspace,
        combined_hspace_row23=args.combined_hspace_row23,
        combined_row2_gap=args.combined_row2_gap,
        show_matrix_colorbar=args.show_matrix_colorbar, umap_legend_y_offset=args.umap_legend_y_offset,
        combined_a_x=args.combined_a_x, combined_a_y=args.combined_a_y,
        combined_b_x=args.combined_b_x, combined_b_y=args.combined_b_y,
        combined_c_x=args.combined_c_x, combined_c_y=args.combined_c_y,
        combined_d_x=args.combined_d_x, combined_d_y=args.combined_d_y
    )

if __name__ == "__main__":
    main()
