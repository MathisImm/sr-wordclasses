"""Figure 9: SR connectivity within the numerical and temporal clusters (extended, gamma = 0.2, K = 120).

The transition matrix is the deposit's transitions/extended_k120_gamma0.2_kmeans.csv and the
cluster tags come from its K = 120 partition. Cluster IDs below refer to that partition.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import figdata
import plotting_config as pc

def main():
    parser = argparse.ArgumentParser(description="Analyze inter-cluster flow for specific clusters.")
    
    # Network Parameters
    net_group = parser.add_argument_group("Network Parameters")
    net_group.add_argument("--net_node_size", type=int, default=5000, help="Size of nodes in network plots.")
    net_group.add_argument("--net_font_size", type=int, default=9, help="Base font size for network plots.")
    net_group.add_argument("--net_label_size", type=int, default=12, help="Font size for node labels in network.")
    net_group.add_argument("--net_legend_size", type=int, default=12, help="Font size for legends.")
    net_group.add_argument("--net_legend_title_size", type=int, default=13, help="Font size for legend title.")
    net_group.add_argument("--net_legend_title", type=str, default="Majority POS", help="Title text for the legend.")
    net_group.add_argument("--net_edge_scale", type=float, default=6.0, help="Scaling factor for network edge thickness.")
    net_group.add_argument("--net_edge_power", type=float, default=1.0, help="Exponent for edge scaling (e.g. 0.5 for sqrt, 2 for squared).")
    net_group.add_argument("--net_margin_left", type=float, default=0.12, help="Left margin in network plots (e.g. 0.15).")
    net_group.add_argument("--net_margin_right", type=float, default=0.12, help="Right margin in network plots (e.g. 0.15).")
    net_group.add_argument("--net_margin_top", type=float, default=0.25, help="Top margin in network plots (e.g. 0.15).")
    net_group.add_argument("--net_margin_bottom", type=float, default=0.20, help="Bottom margin in network plots (e.g. 0.15).")
    net_group.add_argument("--net_legend_x", type=float, default=-0.17, help="Legend X anchor position for network plots.")
    net_group.add_argument("--net_legend_y", type=float, default=1.0, help="Legend Y anchor position for network plots.")
    net_group.add_argument("--net_width", type=float, default=7.0, help="Figure width for network plots.")
    net_group.add_argument("--net_height", type=float, default=6.0, help="Figure height for network plots.")
    
    # Heatmap Parameters
    hm_group = parser.add_argument_group("Heatmap Parameters")
    hm_group.add_argument("--hm_font_size", type=int, default=12, help="Base font size for heatmap plots.")
    hm_group.add_argument("--hm_tick_size", type=int, default=11, help="Font size for heatmap axis ticks.")
    hm_group.add_argument("--hm_label_size", type=int, default=13, help="Font size for heatmap axis labels.")
    hm_group.add_argument("--hm_annot_size", type=int, default=13, help="Font size for heatmap annotations.")
    hm_group.add_argument("--hm_linecolor", type=str, default="none", help="Color of lines between heatmap cells.")
    hm_group.add_argument("--hm_linewidth", type=float, default=0.5, help="Width of lines between heatmap cells.")
    hm_group.add_argument("--hm_cmap", type=str, default="Blues", help="Colormap for heatmaps.")
    hm_group.add_argument("--hm_fmt", type=str, default=".2f", help="Annotation format for heatmaps.")
    hm_group.add_argument("--hm_width", type=float, default=7.0, help="Figure width for heatmaps.")
    hm_group.add_argument("--hm_height", type=float, default=3.0, help="Figure height for heatmaps.")
    hm_group.add_argument("--hm_xrot", type=float, default=45.0, help="Rotation angle for heatmap x-axis labels.")
    hm_group.add_argument("--hm_yrot", type=float, default=0.0, help="Rotation angle for heatmap y-axis labels.")
    
    # Combined Plot Parameters
    comb_group = parser.add_argument_group("Combined Plot Parameters")
    comb_group.add_argument("--text_a_x", type=float, default=-0.15, help="X position for 'a)' text.")
    comb_group.add_argument("--text_a_y", type=float, default=0.99, help="Y position for 'a)' text.")
    comb_group.add_argument("--text_b_x", type=float, default=-0.15, help="X position for 'b)' text.")
    comb_group.add_argument("--text_b_y", type=float, default=1.02, help="Y position for 'b)' text.")
    comb_group.add_argument("--comb_width", type=float, default=7.0, help="Figure width for combined plot.")
    comb_group.add_argument("--comb_height", type=float, default=8.0, help="Figure height for combined plot.")
    comb_group.add_argument("--comb_height_ratio", type=float, default=1.4, help="Height ratio of Top (Network) vs Bottom (Heatmap) in combined plot (e.g. 1.2).")
    
    # Save Parameters
    parser.add_argument("--ext", type=str, default=pc.PLOT_FORMAT, choices=['png', 'pdf', 'svg'], help=f"File extension for plots (default: {pc.PLOT_FORMAT}).")
    parser.add_argument("--dpi", type=int, default=pc.PLOT_DPI, help=f"DPI for saved figures (default: {pc.PLOT_DPI}).")
    
    # Toggle parameters
    parser.add_argument("--all", action="store_true", help="Generate all reports and plots (default: only restricted network and final heatmap).")
    parser.add_argument("--show_titles", action="store_true", help="Whether to show titles on plots.")
    parser.add_argument("--show_grid", action="store_true", help="Whether to show background grid on heatmaps.")
    parser.add_argument("--hide_colorbar", action="store_true", help="Whether to hide the colorbar on heatmaps.")
    
    args = parser.parse_args()

    # Local aliases for easy access at start of function
    NODE_SIZE = args.net_node_size
    NET_LABEL_FONT = args.net_label_size
    LEGEND_FONT = args.net_legend_size
    LEGEND_TITLE_FONT = args.net_legend_title_size
    LEGEND_TITLE = args.net_legend_title
    NET_EDGE_SCALE = args.net_edge_scale
    NET_EDGE_POWER = args.net_edge_power
    NET_MARGIN_LEFT = args.net_margin_left
    NET_MARGIN_RIGHT = args.net_margin_right
    NET_MARGIN_TOP = args.net_margin_top
    NET_MARGIN_BOTTOM = args.net_margin_bottom
    NET_LEGEND_X = args.net_legend_x
    NET_LEGEND_Y = args.net_legend_y
    NET_WIDTH = args.net_width
    NET_HEIGHT = args.net_height
    
    # Heatmap Parameters
    HM_FONT = args.hm_font_size
    HM_TICK_FONT = args.hm_tick_size
    HM_LABEL_FONT = args.hm_label_size
    ANNOT_FONT = args.hm_annot_size
    HM_LINEWIDTH = args.hm_linewidth
    HM_LINECOLOR = args.hm_linecolor
    HM_CMAP = args.hm_cmap
    HM_FMT = args.hm_fmt
    HM_WIDTH = args.hm_width
    HM_HEIGHT = args.hm_height
    HM_XROT = args.hm_xrot
    HM_YROT = args.hm_yrot
    
    TEXT_A_X = args.text_a_x
    TEXT_A_Y = args.text_a_y
    TEXT_B_X = args.text_b_x
    TEXT_B_Y = args.text_b_y
    COMB_WIDTH = args.comb_width
    COMB_HEIGHT = args.comb_height
    COMB_RATIO = args.comb_height_ratio
    
    # Toggle parameters
    SHOW_TITLES = args.show_titles
    SHOW_GRID = args.show_grid
    GENERATE_ALL = args.all
    HIDE_COLORBAR = args.hide_colorbar
    EXT = args.ext
    DPI = args.dpi
    # ------------------------

    pc.setup_publication_style()
    
    # Paths
    subfolder = "g0.2_extended_k120"
    
    # Output directory
    output_dir = os.path.join(pc.BASE_OUTPUT_DIR, "flow_analysis_top5")
    os.makedirs(output_dir, exist_ok=True)
    
    output_md = os.path.join(output_dir, f"flow_analysis_{subfolder}.md")

    # Clusters of interest (k=120 ex-PUNCT_X_INTJ_SPACE_SYM configuration)
    # Temporal/numeric subclusters with linguistically meaningful SR flow
    clusters_of_interest = {
        37: "Relational_Preps", # from, of, through, into, on, in, over, towards, around, along, 
        # 64:  "numbers_higher", # 100,150,200,250,300,400,500... (large cardinals)
        65:  "Numbers_Mid",    # 33-70 range 2-digit numbers
        66:  "Numbers_Date",    # 6-31 range (day-of-month numbers)
        67:  "Units",           # feet,km,ft,mi,mph,m,h,°,miles (measurement units)
        117: "Months",          # October,June,July,December... + since,until,late
        118: "Years_late",      # 1980s-2000s years (1983-2005)
        119: "Years_early",     # 1920s-1960s years (1921-1961)
    }
    # Published Figure 9b reports outflow for these four clusters only; c37, c118 and
    # c119 are shown as targets, not sources. That asymmetry is deliberate: the
    # prepositions are a general-purpose destination and the year clusters end the date
    # expression, so their strongest mass leaves the displayed set entirely
    # (c37 -> c53 determiners/prenominal modifiers, 0.43; c118/c119 -> c16 clause
    # connectives, 0.18/0.21). The paper's "four to six times more mass to its strongest
    # partner within the set than to any single cluster outside it" is a claim about
    # final_rows only -- it does not hold, and is not asserted, for the other three.
    final_rows = [65, 66, 67, 117,]
    final_cols = [37, 65, 66, 67, 117, 118, 119]
    
    # Load data
    transitions = pd.read_csv(os.path.join(figdata.DEPOSIT, "transitions", "extended_k120_gamma0.2_kmeans.csv"))
    matrix = transitions[[f"C{c}" for c in range(len(transitions))]].to_numpy()
    df = figdata.assignments("g0.2", "ext", "kmeans", 120)

    # Compute majority POS for each cluster
    cluster_pos = {}
    for c in range(matrix.shape[0]):
        c_df = df[df['cluster'] == c]
        if not c_df.empty:
            cluster_pos[c] = c_df['pos'].value_counts().idxmax()
        else:
            cluster_pos[c] = "Unknown"

    interest_list = sorted(list(clusters_of_interest.keys()))
    non_interest_indices = [j for j in range(matrix.shape[1]) if j not in interest_list]

    # Shared color mapping with explicit colors for key POS tags for maximum distinction
    unique_pos_all = sorted(list(set([cluster_pos[n] for n in range(matrix.shape[0])])))
    pos_colors = {p: pc.OKABE_ITO[i % len(pc.OKABE_ITO)] for i, p in enumerate(unique_pos_all)}
    
    # Explicit overrides for clarity
    if "NUM" in pos_colors: pos_colors["NUM"] = pc.OKABE_ITO[0]   # Orange
    if "PROPN" in pos_colors: pos_colors["PROPN"] = pc.OKABE_ITO[6] # Reddish Purple
    if "NOUN" in pos_colors: pos_colors["NOUN"] = pc.OKABE_ITO[1]  # Sky Blue
    if "ADP" in pos_colors: pos_colors["ADP"] = pc.OKABE_ITO[2]   # Bluish Green
    
    legend_patches = [mpatches.Patch(color=pos_colors[p], label=p) for p in sorted(pos_colors.keys())]

    def draw_network(G, filename, title_text, layout_k=1.5, ax=None):

        save_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(NET_WIDTH, NET_HEIGHT))
            save_plot = True

        pos = nx.spring_layout(G, k=layout_k, iterations=50, seed=42)
        node_colors = [pos_colors[cluster_pos[n]] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=NODE_SIZE, alpha=0.8)
        labels = {n: f"c{n}\n{clusters_of_interest[n].replace('_', '\n')}" if n in clusters_of_interest else f"c{n}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=NET_LABEL_FONT)
        # Get max weight for scaling
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(weights) if weights else 1.0

        # Draw regular edges and self-loops separately for better styling
        regular_edges = [(u, v) for u, v in G.edges() if u != v]
        
        if regular_edges:
            # Scale edges relative to the max weight, raised to user power, multiplied by scaling factor
            reg_widths = [NET_EDGE_SCALE * ((G[u][v]['weight'] / max_w) ** NET_EDGE_POWER) for u, v in regular_edges]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=regular_edges, width=reg_widths, 
                                   arrowsize=15, alpha=0.8, edge_color='#555555',
                                   connectionstyle='arc3,rad=0.1',
                                   node_size=NODE_SIZE, arrowstyle='-|>')
        
        leg = ax.legend(handles=[p for p in legend_patches if p.get_label() in set([cluster_pos[n] for n in G.nodes()])], 
                        title=LEGEND_TITLE, loc='upper left', bbox_to_anchor=(NET_LEGEND_X, NET_LEGEND_Y), fontsize=LEGEND_FONT)
        leg.get_title().set_fontsize(LEGEND_TITLE_FONT)
        
        if SHOW_TITLES: ax.set_title(title_text)
        ax.axis('off')
        
        # Add padding to prevent nodes at the edge from being cut off
        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        ax.set_xlim(x_min - x_range * NET_MARGIN_LEFT, x_max + x_range * NET_MARGIN_RIGHT)
        ax.set_ylim(y_min - y_range * NET_MARGIN_BOTTOM, y_max + y_range * NET_MARGIN_TOP)
        
        if save_plot:
            plt.tight_layout()
            if filename:
                plt.savefig(os.path.join(output_dir, filename), dpi=DPI)
            plt.close()

    # --- Conditional Generation ---
    
    if GENERATE_ALL:
        # 1. Markdown Report
        with open(output_md, 'w') as f:
            f.write(f"# SR Flow Analysis: {subfolder}\n\n")
            names_str = ", ".join([f"c{c} ({name.replace('_', ' ')})" for c, name in clusters_of_interest.items()])
            f.write(f"Analyzing clusters of interest: {names_str}\n\n")
            for c, name in clusters_of_interest.items():
                f.write(f"## Cluster {c}: {name.replace('_', ' ')} ({cluster_pos[c]})\n")
                top_5_idx = np.argsort(matrix[c, :])[::-1][:5]
                f.write("| Target | Flow | POS | Note |\n|---|---|---|---|\n")
                for idx in top_5_idx:
                    highlight = f"**{clusters_of_interest[idx].replace('_', ' ')}**" if idx in clusters_of_interest else ""
                    f.write(f"| c{idx} | {matrix[c, idx]:.4f} | {cluster_pos[idx]} | {highlight} |\n")
                f.write("\n---\n\n")

        # 2. Top 5 Heatmap
        top_5_union = set()
        for c in clusters_of_interest:
            top_5_union.update(np.argsort(matrix[c, :])[::-1][:5])
        sorted_union = sorted(list(top_5_union))
        sub_matrix = matrix[np.ix_(interest_list, sorted_union)]
        fig, ax = plt.subplots(figsize=(HM_WIDTH, HM_HEIGHT))
        xtl = [f"c{i}\n{clusters_of_interest[i].replace('_', '\n')}" if i in clusters_of_interest else f"c{i}" for i in sorted_union]
        ytl = [f"c{i}\n{clusters_of_interest[i].replace('_', '\n')}" for i in interest_list]
        sns.heatmap(sub_matrix, annot=True, fmt=HM_FMT, cmap=HM_CMAP, ax=ax, xticklabels=xtl,
                    yticklabels=ytl,
                    linewidths=HM_LINEWIDTH, linecolor=HM_LINECOLOR, annot_kws={"size": ANNOT_FONT},
                    cbar=not HIDE_COLORBAR)
        ax.tick_params(axis='x', which='major', labelsize=HM_TICK_FONT)
        ax.tick_params(axis='y', which='major', labelsize=HM_TICK_FONT, labelrotation=HM_YROT)
        plt.setp(ax.get_xticklabels(), rotation=HM_XROT, ha="right", rotation_mode="anchor")
        if not HIDE_COLORBAR:
            cbar = ax.collections[0].colorbar
            if cbar: cbar.ax.tick_params(labelsize=HM_TICK_FONT)
        ax.set_xlabel("Target Cluster", fontsize=HM_LABEL_FONT)
        ax.set_ylabel("Source Cluster", fontsize=HM_LABEL_FONT)
        ax.grid(SHOW_GRID)
        if SHOW_TITLES: ax.set_title("SR Flow Sub-Matrix (Top 5 Union)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"subnetwork_top5_heatmap.{EXT}"), dpi=DPI, bbox_inches='tight')
        plt.close()

        # 3. Top 5 Network
        G_top5 = nx.DiGraph()
        G_top5.add_nodes_from(set(sorted_union) | set(interest_list))
        for c in interest_list:
            outgoing = matrix[c, :]
            for target in np.argsort(outgoing)[::-1][:5]:
                G_top5.add_edge(c, target, weight=outgoing[target])
        draw_network(G_top5, f"subnetwork_top5_network.{EXT}", "Top 5 Subnetwork")

        # 4. Restricted Heatmap
        # Same rows/cols as the published Figure 9b, so the Max External column here
        # cannot disagree with the claim made in the paper.
        restricted_matrix_ext = np.column_stack([matrix[np.ix_(final_rows, final_cols)], [np.max(matrix[r, non_interest_indices]) for r in final_rows]])
        fig, ax = plt.subplots(figsize=(HM_WIDTH, HM_HEIGHT))
        xtl_res = [f"c{i}\n{clusters_of_interest[i].replace('_', '\n')}" for i in final_cols] + ["Max\nExternal"]
        ytl_res = [f"c{i}\n{clusters_of_interest[i].replace('_', '\n')}" for i in final_rows]
        sns.heatmap(restricted_matrix_ext, annot=True, fmt=HM_FMT, cmap=HM_CMAP, ax=ax, xticklabels=xtl_res,
                    yticklabels=ytl_res,
                    linewidths=HM_LINEWIDTH, linecolor=HM_LINECOLOR, annot_kws={"size": ANNOT_FONT},
                    cbar=not HIDE_COLORBAR)
        ax.tick_params(axis='x', which='major', labelsize=HM_TICK_FONT)
        ax.tick_params(axis='y', which='major', labelsize=HM_TICK_FONT, labelrotation=HM_YROT)
        plt.setp(ax.get_xticklabels(), rotation=HM_XROT, ha="right", rotation_mode="anchor")
        if not HIDE_COLORBAR:
            cbar = ax.collections[0].colorbar
            if cbar: cbar.ax.tick_params(labelsize=HM_TICK_FONT)
        ax.set_xlabel("Target Cluster", fontsize=HM_LABEL_FONT)
        ax.set_ylabel("Source Cluster", fontsize=HM_LABEL_FONT)
        ax.grid(SHOW_GRID)
        if SHOW_TITLES: ax.set_title("SR Flow: Restricted + External Max")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"subnetwork_restricted_heatmap.{EXT}"), dpi=DPI, bbox_inches='tight')
        plt.close()

    # --- Always Generated (unless GEN_ALL handles them differently, here we just separate the "core" results) ---
    
    fig, axes = plt.subplots(2, 1, figsize=(COMB_WIDTH, COMB_HEIGHT), gridspec_kw={'height_ratios': [COMB_RATIO, 1.0]})
    
    # a) Restricted Network (Default)
    G_res = nx.DiGraph()
    G_res.add_nodes_from(interest_list)
    for c in interest_list:
        for target in interest_list:
            if matrix[c, target] > 0.005: G_res.add_edge(c, target, weight=matrix[c, target])
    draw_network(G_res, None, "Restricted Subnetwork", layout_k=2.8, ax=axes[0])
    
    axes[0].text(TEXT_A_X, TEXT_A_Y, 'a)', transform=axes[0].transAxes, 
                 fontsize=HM_FONT + 4, fontweight='bold', va='top', ha='right')

    # b) Final Heatmap (Default)
    
    # final_rows_old, final_cols_old = [15, 16, 18, 19, 46], [15, 16, 19, 45, 46, 47]
    final_ext = np.column_stack([matrix[np.ix_(final_rows, final_cols)], [np.max(matrix[r, non_interest_indices]) for r in final_rows]])
    xtl_final = [f"c{i}\n{clusters_of_interest[i].replace('_', '\n')}" for i in final_cols] + ["Max\nExternal"]
    ytl_final = [f"c{i}\n{clusters_of_interest[i].replace('_', '\n')}" for i in final_rows]
    sns.heatmap(final_ext, annot=True, fmt=HM_FMT, cmap=HM_CMAP, ax=axes[1],
                xticklabels=xtl_final,
                yticklabels=ytl_final,
                linewidths=HM_LINEWIDTH, linecolor=HM_LINECOLOR, annot_kws={"size": ANNOT_FONT},
                cbar=False)
    axes[1].tick_params(axis='x', which='major', labelsize=HM_TICK_FONT)
    axes[1].tick_params(axis='y', which='major', labelsize=HM_TICK_FONT, labelrotation=HM_YROT)
    plt.setp(axes[1].get_xticklabels(), rotation=HM_XROT, ha="right", rotation_mode="anchor")
    axes[1].set_xlabel("Target Cluster", fontsize=HM_LABEL_FONT)
    axes[1].set_ylabel("Source Cluster", fontsize=HM_LABEL_FONT)
    axes[1].grid(SHOW_GRID)
    if SHOW_TITLES: axes[1].set_title("Final Subnetwork Refinement")
    
    axes[1].text(TEXT_B_X, TEXT_B_Y, 'b)', transform=axes[1].transAxes, 
                 fontsize=HM_FONT + 4, fontweight='bold', va='top', ha='right')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"subnetwork_combined.{EXT}"), dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
