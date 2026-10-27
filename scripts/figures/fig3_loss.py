#!/usr/bin/env python3
"""
Figure 3: training and validation losses, from the deposit's model/loss_history.json.
Generates the clean loss plot for the publication.
Extracts training and validation loss, applies master plot config.
"""
import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

# Use the publication styling config
import plotting_config as pc

def load_loss_history(checkpoint_dir):
    pt_path = os.path.join(checkpoint_dir, "loss_history.pt")
    json_path = os.path.join(checkpoint_dir, "loss_history.json")
    
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu")
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"No loss history found in {checkpoint_dir}")

def plot_publication_loss():
    legend_fontsize = pc.FONT_SIZE - 2 if hasattr(pc, 'FONT_SIZE') else 8

    # Setup styling based on config
    pc.setup_publication_style()
    
    # Checkpoint Dir
    checkpoint_dir = os.path.join(pc.DEPOSIT, "model")
    history = load_loss_history(checkpoint_dir)
    
    epochs = np.array(history['epochs'])
    train_loss = np.array(history['train_loss'])
    
    valid_val_indices = [i for i, v in enumerate(history['val_loss']) if v is not None]
    val_epochs = epochs[valid_val_indices]
    val_loss = np.array([history['val_loss'][i] for i in valid_val_indices])

    # We want a 2x2 grid for Mean and individual gammas.
    fig, axes = plt.subplots(2, 2, figsize=(pc.FIG_WIDTH_INCHES, pc.FIG_WIDTH_INCHES * 0.5))
    
    # Define consistent colors for train and val
    train_color = 'black'
    val_color = '#D55E00'  # Vermillion from OKABE_ITO

    # --- 1. Mean Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color=train_color, linestyle='-', linewidth=2, marker='o', markersize=4, label='Train Mean')
    if len(val_loss) > 0:
        ax.plot(val_epochs, val_loss, color=val_color, linestyle='--', linewidth=2, marker='s', markersize=4, label='Val Mean')
    ax.set_title("Mean Loss", fontweight='bold', y=1.0)
    ax.set_ylabel('KL Divergence')
    ax.legend(loc='upper right', frameon=True, fontsize=legend_fontsize)

    # --- Individual Gamma Heads ---
    gamma_keys = sorted([k for k in history.keys() if k.startswith('train_sr_loss_g')])
    
    # Map each gamma to a subplot index
    plot_indices = [(0, 1), (1, 0), (1, 1)]
    
    for i, g_key in enumerate(gamma_keys):
        if i >= 3: break # Only 3 spots left in 2x2 grid
        
        ax = axes[plot_indices[i]]
        g_val = g_key.split('_')[-1] # e.g., 'g0.2'
        
        train_g_loss = np.array(history[g_key])
        ax.plot(epochs, train_g_loss, color=train_color, linestyle='-', linewidth=1.6, marker='o', markersize=4, label=f'Train {g_val}')
        
        val_g_key = g_key.replace('train_', 'val_')
        if val_g_key in history:
            valid_val_g_indices = [j for j, v in enumerate(history[val_g_key]) if v is not None]
            val_g_epochs = epochs[valid_val_g_indices]
            val_g_loss = np.array([history[val_g_key][j] for j in valid_val_g_indices])
            if len(val_g_loss) > 0:
                ax.plot(val_g_epochs, val_g_loss, color=val_color, linestyle='--', linewidth=1.6, marker='s', markersize=4, label=f'Val {g_val}')
        
        ax.set_title(f"Gamma {g_val[1:]}", fontweight='bold', y=1.0)
        if plot_indices[i][1] == 0: # Left column
            ax.set_ylabel('KL Divergence')
        ax.legend(loc='upper right', frameon=True, fontsize=legend_fontsize)

    # General formatting for all axes
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        if max(epochs) <= 20:
            ax.set_xticks(epochs)
            
    plt.tight_layout(h_pad=0.25)
    pc.save_fig(fig, "Fig_Training_Loss_Curve")
    plt.close(fig)

if __name__ == "__main__":
    plot_publication_loss()
