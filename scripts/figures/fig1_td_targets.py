"""
Figure 1: TD learning targets for successor representations (1-step, n-step, TD(lambda)).

Illustrates the difference between 1-step TD, n-step TD and TD(λ) successor
representation updates.  For each method a horizontal token timeline is drawn
(top) together with a bar chart showing the explicit contribution weight of
each future feature φ(s_{t+k}) to the SR estimate ψ(s_t) (bottom).

Weight derivation
-----------------
  True SR:   ψ(s_t) = Σ_{k=0}^∞  γ^k · φ(s_{t+k})

  1-step:    explicitly uses k=0 only, then bootstraps from ψ(s_{t+1})
             target = φ(s_t) + γ · ψ(s_{t+1})

  n-step:    explicitly uses k=0…n−1, then bootstraps from ψ(s_{t+n})
             target = Σ_{k=0}^{n-1} γ^k · φ(s_{t+k})  +  γ^n · ψ(s_{t+n})

  TD(λ):     λ-weighted mix of all n-step returns; effective weight of
             φ(s_{t+k}) equals (γλ)^k  (no finite bootstrap cutoff)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import plotting_config as pc

# ── tuneable parameters ──────────────────────────────────────────────────────
N_SHOW   = 7       # number of future steps shown (t, t+1, …, t+N_SHOW)
N_STEP   = 4       # n for the n-step panel
LAMBDA   = 0.9     # λ for the TD(λ) panel
GAMMA    = 0.8     # discount factor used throughout
FIG_W    = 7.0     # inches — matches plotting_config.FIG_WIDTH_INCHES
FIG_H    = 5.4     # total figure height in inches
TIMELINE_Y_FRAC = 0.82   # fractional height for timeline inside each axes
BAR_ALPHA = 0.82

OUTPUT_DIR  = pc.BASE_OUTPUT_DIR
PNG_OUTPUT  = os.path.join(pc.BASE_OUTPUT_DIR, "sr_td_illustration.png")
# ─────────────────────────────────────────────────────────────────────────────


def _weights(method, N, gamma, n_step, lam):
    """Return explicit contribution weights w[k] for φ(s_{t+k}), k = 0…N.

    ARRIVAL REWARD semantics (used in this project):
      ψ(s_t) predicts what comes AFTER s_t, so φ(s_t) is NOT included.
      Weight at k=0 is always 0.

      1-step:   w[1] = 1,  rest 0   (bootstrap from ψ_{t+1})
      n-step:   w[k] = γ^{k-1}  for k = 1..n,  rest 0
      TD(λ):    w[k] = (γλ)^{k-1} for k = 1..N,  derived from
                G_t = φ_{t+1} + γ[(1-λ)ψ_{t+1} + λ G_{t+1}]
    """
    w = np.zeros(N + 1)
    if method == "1step":
        w[1] = 1.0           # only φ_{t+1}; ψ_{t+1} is the bootstrap
    elif method == "nstep":
        for k in range(1, n_step + 1):
            w[k] = gamma ** (k - 1)
    elif method == "lambda":
        for k in range(1, N + 1):
            w[k] = (gamma * lam) ** (k - 1)
    return w


def _bootstrap_x(method, n_step, N):
    """Return the x-position (bs_x) so the dashed line is drawn at x = bs_x - 0.5.

    Arrival-reward bootstrap positions:
      1-step:  dashed line at x = 1.5  →  bs_x = 2
      n-step:  dashed line at x = n+0.5  →  bs_x = n+1
      TD(λ):   no hard cutoff  →  None
    """
    if method == "1step":
        return 2               # dashed line at x = 1.5  (after explicit φ_{t+1})
    elif method == "nstep":
        return n_step + 1      # dashed line at x = n+0.5
    else:
        return None            # TD(λ): infinite horizon, no cutoff


def make_sr_illustration(
    filename="Fig_SR_Illustration",
    n_steps=N_STEP,
    lambda_val=LAMBDA,
    gamma=GAMMA,
    N=N_SHOW,
):
    pc.setup_publication_style()

    # Colour scheme (Okabe-Ito)
    C_1step  = pc.OKABE_ITO[4]   # blue
    C_nstep  = pc.OKABE_ITO[2]   # bluish-green
    C_lambda = pc.OKABE_ITO[0]   # orange

    methods = [
        dict(method="1step",  color=C_1step,
             title="1-Step TD",
             formula=(r"$G_t \;=\; \phi_{t+1}"
                      r"\;+\; \gamma\,\psi_{t+1}$")),
        dict(method="nstep",  color=C_nstep,
             title=f"{n_steps}-Step TD",
             formula=(r"$G_t \;=\; \sum_{k=1}^{n}"
                      r"\gamma^{k-1}\phi_{t+k} \;+\; \gamma^n\psi_{t+n}$")),
        dict(method="lambda", color=C_lambda,
             title=r"TD($\lambda$)",
             formula=(r"$G_t \;=\; \phi_{t+1}"
                      r"\;+\; \gamma\,[(1\!-\!\lambda)\psi_{t+1}"
                      r"\;+\;\lambda\,G_{t+1}]$")),
    ]

    positions   = np.arange(N + 1)
    node_labels = [r"$t$"] + [r"$t\!+\!%d$" % k for k in range(1, N + 1)]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={"hspace": 0.72},
    )

    for ax, cfg in zip(axes, methods):
        m       = cfg["method"]
        color   = cfg["color"]
        title   = cfg["title"]
        formula = cfg["formula"]
        w       = _weights(m, N, gamma, n_steps, lambda_val)
        bs_x    = _bootstrap_x(m, n_steps, N)

        # ── 1. Token timeline  (x=data coords, y=axes-fraction via get_xaxis_transform) ──
        tl_y = TIMELINE_Y_FRAC
        tr   = ax.get_xaxis_transform()   # x in data, y in [0,1] axes fraction

        # timeline rail — drawn via ax.plot with mixed transform (no axhline!)
        ax.plot(
            [positions[0] - 0.5, positions[-1] + 0.5], [tl_y, tl_y],
            color="#BBBBBB", lw=0.9, transform=tr, clip_on=True, zorder=1,
        )

        # token nodes + labels
        # k=0 is the *query* position (ψ computed here) — hollow with dark border
        for k in range(N + 1):
            if k == 0:
                nc, ec, lw_n, ms = "white", "#333333", 1.8, 8   # hollow = "query"
            elif w[k] > 1e-9:
                nc, ec, lw_n, ms = color, color, 1.5, 7          # filled = contributes
            else:
                nc, ec, lw_n, ms = "#EEEEEE", "#AAAAAA", 0.8, 7  # grey = no contribution
            ax.plot(k, tl_y, 'o', color=nc, markeredgecolor=ec,
                    markeredgewidth=lw_n, markersize=ms,
                    transform=tr, clip_on=False, zorder=5)
            ax.text(k, tl_y + 0.065, node_labels[k],
                    ha='center', va='bottom', fontsize=7.5,
                    transform=tr, clip_on=False)

        # ── span bracket: colored bar under timeline from t+1 to last contributing ──
        last_k = int(np.max(np.where(w > 1e-9))) if np.any(w > 1e-9) else 0
        if last_k >= 1:
            # for TD(λ) extend the line past the last shown node to signal continuation
            line_end = N + 0.6 if m == "lambda" else last_k
            ax.plot(
                [1, line_end], [tl_y, tl_y],
                color=color, lw=2.5, solid_capstyle='round',
                alpha=0.75, transform=tr, clip_on=False, zorder=4,
            )
            if m == "lambda":
                ax.text(line_end + 0.08, tl_y, "∞",
                        ha='left', va='center', fontsize=8, color=color,
                        alpha=0.85, transform=tr, clip_on=False)
            else:
                ax.annotate("",
                    xy=(last_k + 0.05, tl_y),
                    xytext=(last_k - 0.05, tl_y),
                    xycoords=tr, textcoords=tr,
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.15, head_length=0.08",
                        color=color, lw=1.5, alpha=0.75,
                    ),
                    clip_on=False,
                )

        # ── 2. Contribution weight bars ────────────────────────────────────────
        bars = ax.bar(positions, w, color=color, alpha=BAR_ALPHA, width=0.55,
                      zorder=3, linewidth=0)
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(1.2)
        # bar height annotations disabled

        # ── 3. Bootstrap marker ────────────────────────────────────────────────
        if bs_x is not None:
            ax.axvline(x=bs_x - 0.5, color='#999999',
                       linestyle=':', lw=1.1, alpha=0.85, zorder=2)
            # label just below the timeline rail, close to the dashed line
            ax.text(bs_x - 0.38, tl_y - 0.13,
                    "bootstrap →", fontsize=6.5,
                    color='#888888', va='top', ha='left', style='italic',
                    transform=ax.get_xaxis_transform(), clip_on=False)

        # ── 4. Axes cosmetics ──────────────────────────────────────────────────
        ax.set_xlim(-0.7, N + 0.7)
        ax.set_ylim(0, 1.28)
        ax.set_xticks(positions)
        ax.set_xticklabels(node_labels, fontsize=8)
        ax.set_yticks([0, 0.5, 1.0])
        ylabel = (r"Feature Weight $((\gamma\lambda)^{k-1})$"
                  if m == "lambda" else r"Feature Weight $(\gamma^{k-1})$")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.set_title(f"{title}   —   {formula}", fontsize=9, loc='left', pad=4)

        # γ / λ annotation — lower-right corner
        note_parts = [rf"$\gamma={gamma}$"]
        if m == "nstep":
            note_parts.append(f"$n={n_steps}$")
        if m == "lambda":
            note_parts.append(rf"$\lambda={lambda_val}$")
        ax.text(0.985, 0.22, ",  ".join(note_parts),
                transform=ax.transAxes, fontsize=7.5,
                ha='right', va='bottom', color='#666666')

    # ── no suptitle ───────────────────────────────────────────────────────────
    axes[-1].set_xlabel("Future Token Position", fontsize=9)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF → {pdf_path}")

    fig.savefig(PNG_OUTPUT, bbox_inches='tight', dpi=300)
    print(f"Saved PNG (300 dpi) → {PNG_OUTPUT}")

    plt.close(fig)


if __name__ == "__main__":
    make_sr_illustration()
