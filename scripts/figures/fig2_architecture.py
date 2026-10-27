#!/usr/bin/env python3
"""
Figure 2: architecture and training flow of the SR network.
Single-page, paper-ready figure (7 × ~5.5 in).

All layout knobs are collected in the PARAMETERS section below so the
figure can be tuned without touching any drawing code.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle

from plotting_config import OKABE_ITO, OUTPUT_PREFIX

# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS  — edit here; do not touch the drawing code below
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Colours ──────────────────────────────────────────────────────────────────
# (refer to these keys throughout; change hex here to retheme)
C = {
    'input'   : '#3D3D3D',       # dark grey  – input/output tensor boxes
    'embed'   : OKABE_ITO[0],    # orange     – embedding layer
    'backbone': OKABE_ITO[4],    # blue       – backbone ResNet blocks
    'head'    : OKABE_ITO[1],    # sky blue   – per-γ head blocks
    'linear'  : OKABE_ITO[2],    # green      – (kept for reference; ψ boxes now use 'psi')
    'psi'     : '#20898B',       # teal       – ψ boxes: softmax-normalized head output (sits between sky-blue heads and vermillion KL)
    'phi'     : '#996600',       # dark gold  – φ one-hot (no gradient)
    'ema'     : OKABE_ITO[6],    # purple     – EMA target network
    'loss'    : OKABE_ITO[5],    # vermillion – loss / gradient path
    'arrow'   : '#444444',       # dark grey  – default arrows
    'grey'    : '#777777',       # mid grey   – dim labels / annotations
    'lgrey'   : '#CCCCCC',       # light grey – divider line
    'bg'      : 'white',
    'text'    : '#111111',
    # ── ResidualBlock panel role-colours ──────────────────────────────────────
    'blk_norm': '#D0D0D5',       # slightly darker cool gray – LayerNorm (utility; more contrast vs panel bg)
    'blk_act' : '#C97A50',       # warm coral      – GELU (non-linearity)
    'blk_add' : '#C8C8CE',       # medium cool gray — matches LayerNorm palette, cohesive
    # blk_bg / blk_bg_ec defined as P_BLK_PANEL_* parameters below
}

# ─── Figure dimensions ────────────────────────────────────────────────────────
P_FIG_W     = 7      # inches – paper column width
P_FIG_H     = 5.5    # inches – height (content height; no blank space)
P_YLIM_LO   = 47     # lower y-axis limit (update boxes removed; total loss bottom ~51)
P_YLIM_HI   = 97     # upper y-axis limit (title removed)
# Resulting scale:  1 x-unit ≈ P_FIG_W/100 = 0.070 in
#                   1 y-unit ≈ P_FIG_H/(P_YLIM_HI-P_YLIM_LO) = 0.090 in

# ─── Font sizes (pt) — minimum 8pt ───────────────────────────────────────────
# Hierarchy: TITLE(11.5) > SEC/FORMULA(8.5) > BOX/SUB/DIM/ANNOT(8.0)
P_FS_TITLE    = 11.5  # main figure title / diamond label
P_FS_SEC      = 8.5   # section headers (◆ …)
P_FS_BOX      = 8.0   # primary text inside a box
P_FS_SUB      = 8.0   # sub-label text inside a box (second line)
P_FS_DIM      = 8.0   # tensor dimension annotations on arrows
P_FS_FORMULA  = 8.5   # math formulas in the right panel
P_FS_FOOTNOTE = 8.0   # footnote text at bottom

# ─── Column geometry ──────────────────────────────────────────────────────────
P_DIVIDER_X = 36.5   # x of dashed separator: detail panels LEFT, main flow RIGHT
P_CX        = 68     # centre-x of main flow (right column)
P_RX        = 19     # centre-x of detail panels (left column)

# ─── Head x-positions and labels ──────────────────────────────────────────────
# Three γ-heads; midpoint should equal P_CX for visual symmetry
P_HX        = [48, 68, 88]
P_GLABELS   = [r'$\gamma\!=\!0.2$', r'$\gamma\!=\!0.5$', r'$\gamma\!=\!0.8$']
P_PSI_LBLS  = [r'$\boldsymbol{\psi}_{\boldsymbol{0.2}}$', r'$\boldsymbol{\psi}_{\boldsymbol{0.5}}$', r'$\boldsymbol{\psi}_{\boldsymbol{0.8}}$']
P_KL_FORMULAS = [
    r'$\mathrm{KL}(G_{0.2} \,\|\, \psi_{0.2})$',
    r'$\mathrm{KL}(G_{0.5} \,\|\, \psi_{0.5})$',
    r'$\mathrm{KL}(G_{0.8} \,\|\, \psi_{0.8})$',
]

# ─── Main flow: y-positions (box / text centres) ──────────────────────────────
P_Y_TITLE    = 98.9   # (unused — title removed)
P_Y_INPUT    = 95.0   # input token IDs box
P_Y_PHI      = 95.0   # φ one-hot box — same row as input, above bracket
P_Y_EMBED    = 89.7   # embedding box
P_Y_BB       = 83.5   # backbone block
P_Y_HEAD     = 71.5   # γ-head blocks
P_Y_PSI      = 63.5   # ψ logit boxes
P_Y_KL       = 58.5   # KL loss boxes
P_Y_TOTAL    = 52.5   # total loss box

# ─── Main flow: box dimensions ────────────────────────────────────────────────
P_W_INPUT  = 30;   P_H_INPUT  = 2.5;  P_INPUT_CX = 78  # input box
P_W_PHI    = 17;   P_H_PHI    = 2.5;  P_PHI_CX  = 46   # φ box — slightly right of bracket spine
P_W_EMBED  = 22;   P_H_EMBED  = 2.5;  P_EMBED_CX = 68  # embedding — centred on backbone
P_W_BB     = 44;   P_H_BB     = 5.5;  P_BB_CX   = 68   # backbone centre-x  (= P_CX)
P_W_HEAD   = 18;   P_H_HEAD   = 8.0   # taller to fit 4 text lines
P_W_PSI    = 16;   P_H_PSI    = 2.5
P_W_KL     = 16.5; P_H_KL     = 2.5   # KL boxes (no subtitle — γ-specific formula covers it)

# ─── φ → target construction connection ──────────────────────────────────────
# Routing: φ LEFT edge → vertical DOWN to jog level → horizontal LEFT → arrowhead DOWN
#   into target box top (top-right area, clear of corner).
#   Vertical at x=37.5 clears all main-flow boxes (backbone left@46, head left@39).
#   Horizontal jog at y=62: in the gap below ψ-boxes (bottom@62.25) and above target top.
P_SHOW_PHI_RAIL   = True
P_PHI_RAIL_Y      = 95.0   # top of vertical leg (= φ box y)
P_PHI_RAIL_X      = 37.5   # x of horizontal exit leg (= φ box left edge: P_PHI_CX - P_W_PHI/2)
P_PHI_VERT_X      = 36.0   # x of vertical leg (clear of bracket spine at 37.5 and shadow arrowhead)
P_PHI_JOG_Y       = 63.5   # y of horizontal jog (higher = more arrowhead space below)
P_PHI_ENTRY_X     = 32.0   # x where arrow enters target box top (2.5 units from right edge)
P_PHI_TARGET_Y    = 61.8   # y of visual box top — arrowhead stops at the border
P_PHI_RAIL_LW     = 0.9    # linewidth of the dashed rail
P_PHI_RAIL_LS     = '--'   # linestyle: standard -- dashes for all legs

# ─── Right column: y-positions ────────────────────────────────────────────────
P_Y_SEC_BLOCK  = 96.2   # ResidualBlock section header
P_Y_BLK_IN     = 94.7   # "x ∈ R^d" input label
P_Y_BLK_LN     = 92.2   # LayerNorm box
P_Y_BLK_FC1    = 88.4   # Linear 1 box  (3.8-unit gap — unified)
P_Y_BLK_ACT    = 84.6   # GELU box      (3.8-unit gap)
P_Y_BLK_FC2    = 80.8   # Linear 2 box  (3.8-unit gap)
P_Y_BLK_ADD    = 77.0   # + residual    (3.8-unit gap)
P_Y_BLK_OUT    = 74.2   # "x_out" label (further below ADD to give arrow enough space)
P_Y_SEC_EMA    = 71.3   # EMA section header — 1.0-unit gap to box top (matches λ-return)
P_Y_EMA        = 68.5   # EMA box centre
P_Y_SEC_TARGET = 62.8   # target-construction section header — 1.5-unit gap to panel top
P_Y_TARGET     = 56.5   # target panel centre

# ─── Right column: box dimensions ─────────────────────────────────────────────
P_W_BLK   = 22;   P_H_BLK   = 2.2   # ResNetBlock component boxes
P_RES_OFF = 14.0                      # x-offset of skip-connection rail from BX
P_W_EMA   = 30;   P_H_EMA   = 3.0   # EMA box (compact: equation only)
P_W_TR    = 31   # target-formula panel

# ResidualBlock panel background — light gray container signals "legend/detail"
# rather than a live component in the forward pass.
P_BLK_PANEL_X  = 5.5     # left edge
P_BLK_PANEL_W  = 28.5    # width      (covers x=5.5 → 34.0, clears rail at 35.5)
P_BLK_PANEL_Y  = 73.5    # bottom edge (covers spread-out boxes to P_Y_BLK_OUT=75.0)
P_BLK_PANEL_H  = 24.0    # height      (covers y=73.5 → 97.5)
P_BLK_PANEL_FC = '#F4F4F4'  # fill colour (light gray)
P_BLK_PANEL_EC = '#EBEBEB'  # border colour (very light — subordinate reference panel)
P_BLK_PANEL_LW = 0.35       # border linewidth (lighter than main boxes)

P_SHOW_DIVIDER = False   # set True to re-enable the dashed column separator line

# ─── φ padlock position ───────────────────────────────────────────────────────
P_PHI_LOCK_X   = 46 + 17/2 - 0.9   # x: top-right inside φ box
P_PHI_LOCK_Y   = 95.0 + 2.5/2 - 0.7  # y: near box top
P_PHI_LOCK_SZ  = 1.1               # size in data units
P_EMA_LOCK_X      = 32.5     # x of padlock badge (top-right of EMA box)
P_EMA_LOCK_Y      = 68.8     # y of padlock centre
P_EMA_LOCK_SZ     = 1.7      # size in data units (whole padlock fits in H=3.0 box)
P_EMA_HATCH       = ''              # hatch disabled — PDF renders linewidth=0 hatches visibly
P_EMA_HATCH_ALPHA = 0.20            # opacity of hatch overlay

# ─── θ bracket (spans all trainable blocks: Embedding + Backbone + Heads) ─────
# Left-side bracket: spine just left of leftmost content, ticks point RIGHT.
# Keeps the shadow-copy arrow short (only 3-4 x-units to EMA right edge).
P_THETA_BRAC_X   = 37.5    # x of bracket spine (just left of leftmost head edge x≈39)
P_THETA_TOP_Y    = 90.95   # top tick y    (= P_Y_EMBED + P_H_EMBED/2)
P_THETA_BOT_Y    = 67.5    # bottom tick y (= P_Y_HEAD  - P_H_HEAD /2)
P_THETA_TICK     = 1.2     # outward tick length (pointing RIGHT toward content)
P_THETA_LW       = 1.0     # bracket linewidth
P_THETA_COL      = '#555555'  # bracket colour
P_THETA_LABEL    = r'$\theta_\mathrm{live}$'  # label shown rotated left of spine
P_THETA_LABEL_FS = 8.0     # label font size

# ─── Shadow-copy dashed arrow (θ bracket → EMA) ───────────────────────────────
# Short HORIZONTAL arrow from bracket spine → EMA right edge at EMA y-level.
P_SHADOW_START_X  = 37.5     # start x  (= P_THETA_BRAC_X)
P_SHADOW_START_Y  = 68.5     # start y  (= P_Y_EMA)
P_SHADOW_END_X    = 34.0     # end x    (right edge of EMA: P_RX + P_W_EMA/2)
P_SHADOW_END_Y    = 68.5     # end y    (same as start → purely horizontal arrow)
P_SHADOW_LW       = 0.9      # line width
P_SHADOW_COL      = '#555555'  # bracket colour (same as θ_live bracket)
P_SHADOW_LABEL    = ''              # suppress
P_SHADOW_LBL_X    = 37.0           # (unused when LABEL='')
P_SHADOW_LBL_Y    = 80.5           # (unused when LABEL='')
P_SHADOW_LBL_FS   = 5.3

# ─── Total-loss box and inference callout (bottom row) ────────────────────────
P_TOTAL_CX  = 60.0    # total loss box centre-x (shifted left to make room for callout)
P_W_TOTAL   = 22      # width  (was 36; narrowed to share row with callout)
P_H_TOTAL   = 3.0     # height (unchanged)
# Inference-only callout: white fill, thin dashed border
P_INFER_CX  = 83.5    # callout centre-x
P_INFER_W   = 18.0    # callout width
P_INFER_H   = 3.5     # callout height (2 lines)
P_INFER_COL = '#F4F4F4'        # fill (matches ResidualBlock panel bg — quiet, not popping)
P_INFER_EC  = '#999999'        # border colour
P_INFER_TC  = '#333333'        # text colour

# ─── Arrow / line global properties ──────────────────────────────────────────
P_LW_MAIN    = 1.2   # main flow arrows (left column)
P_LW_DETAIL  = 1.0   # arrows inside right-panel detail (ResNetBlock)
P_LW_DIV     = 0.7   # dashed column-divider
P_SHRINK_A   = 1.5   # arrowhead tail shrink
P_SHRINK_B   = 1.5   # arrowhead tip shrink

# ─── Text content ─────────────────────────────────────────────────────────────
# Edit strings here to change labels without hunting through drawing code.
P_FOOTNOTE  = ''   # leave empty to suppress; will appear at P_Y_FOOTNOTE = 41.5

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING CODE  — change parameters above, not here
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(P_FIG_W, P_FIG_H), facecolor=C['bg'])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(P_YLIM_LO, P_YLIM_HI)
ax.axis('off')

# ── Drawing primitives ────────────────────────────────────────────────────────
def box(cx, cy, w, h, color, text='', subtext='', subtext2='', subtext3='',
        ec='white', lw=1.2, alpha=1.0, tc='white', fontsize=None,
        ls='-', bold=True, sub_fs=None):
    fontsize = fontsize or P_FS_BOX
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.22',
        facecolor=color, edgecolor=ec, linewidth=lw,
        alpha=alpha, linestyle=ls, zorder=3, clip_on=False))
    sfs = sub_fs or (fontsize - 1.0)
    if subtext3:
        # 4-line layout: evenly spaced at ±1.55 and ±0.55
        for s, y_off, fs_, bld in [
                (text,     1.55,  fontsize, bold),
                (subtext,  0.55,  sfs,      False),
                (subtext2, -0.55, sfs,      False),
                (subtext3, -1.55, sfs,      False)]:
            if s:
                ax.text(cx, cy + y_off, s, ha='center', va='center',
                        fontsize=fs_, fontweight='bold' if bld else 'normal',
                        color=tc, alpha=0.9 if not bld else 1.0,
                        zorder=4, clip_on=False)
    elif subtext2:
        # 3-line layout: evenly spaced
        for s, y_off, fs_, bld in [
                (text,     1.1,  fontsize, bold),
                (subtext,  0.0,  sfs,      False),
                (subtext2, -1.1, sfs,      False)]:
            if s:
                ax.text(cx, cy + y_off, s, ha='center', va='center',
                        fontsize=fs_, fontweight='bold' if bld else 'normal',
                        color=tc, alpha=0.9 if not bld else 1.0,
                        zorder=4, clip_on=False)
    else:
        dy = 0.48 if subtext else 0
        if text:
            ax.text(cx, cy + dy, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold' if bold else 'normal',
                    color=tc, zorder=4, clip_on=False)
        if subtext:
            ax.text(cx, cy - 0.58, subtext, ha='center', va='center',
                    fontsize=sfs, color=tc, alpha=0.9, zorder=4, clip_on=False)


def arr(x1, y1, x2, y2, color=None, lw=None, style='->',
        conn='arc3,rad=0', label='', loff=(0.4, 0), label_color=None):
    color = color or C['arrow']
    lw    = lw    or P_LW_MAIN
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        connectionstyle=conn,
                        shrinkA=P_SHRINK_A, shrinkB=P_SHRINK_B),
        zorder=2, clip_on=False)
    if label:
        mx, my = (x1+x2)/2 + loff[0], (y1+y2)/2 + loff[1]
        ax.text(mx, my, label, ha='left', va='center',
                fontsize=P_FS_DIM, color=label_color or C['grey'],
                style='italic', zorder=5)


def padlock(cx, cy, sz=1.5, col=None, zo=6):
    """Draw a minimal padlock icon in data coordinates.
    sz  – overall size in data units (body width ≈ 0.7·sz, total height ≈ sz)
    """
    col = col or C['ema']
    bw  = sz * 0.68          # body width
    bh  = sz * 0.52          # body height
    by  = cy - sz * 0.28     # body bottom y
    # Body
    ax.add_patch(FancyBboxPatch(
        (cx - bw/2, by), bw, bh,
        boxstyle='round,pad=0.04', facecolor=col, edgecolor='white',
        linewidth=0.4, zorder=zo, clip_on=False))
    # Shackle (open-top U: semicircle above body)
    ax.add_patch(Arc(
        (cx, by + bh), bw * 0.55, sz * 0.62,
        theta1=0, theta2=180, color=col, lw=1.1, zorder=zo, clip_on=False))
    # Keyhole (small white dot)
    ax.add_patch(Circle(
        (cx, by + bh * 0.45), bw * 0.11,
        facecolor='white', edgecolor='none', zorder=zo + 1, clip_on=False))


def hline(x1, x2, y, color=None, lw=None, ls='-'):
    ax.plot([x1, x2], [y, y],
            color=color or C['arrow'], lw=lw or P_LW_MAIN, ls=ls, zorder=2)


def vline(x, y1, y2, color=None, lw=None, ls='-'):
    ax.plot([x, x], [y1, y2],
            color=color or C['arrow'], lw=lw or P_LW_MAIN, ls=ls, zorder=2)


def txt(x, y, s, fs=None, color=None, bold=False, italic=False,
        ha='center', va='center', zorder=5, bg=None):
    fs = fs or P_FS_BOX
    kw = dict(ha=ha, va=va, fontsize=fs,
              fontweight='bold' if bold else 'normal',
              fontstyle='italic' if italic else 'normal',
              color=color or C['text'], zorder=zorder, clip_on=False)
    if bg:
        kw['bbox'] = dict(boxstyle='round,pad=0.15', facecolor=bg,
                          edgecolor='none', alpha=0.85)
    ax.text(x, y, s, **kw)


def sec(x, y, s, color=None):
    txt(x, y, s, fs=P_FS_SEC, color=color or C['text'], bold=True)


# Title: centred over main flow column, dark colour
txt(P_CX, P_Y_TITLE, r'◆  SR ResNet Training Graph',
    fs=P_FS_TITLE, color='#1A1A2E', bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — main architecture flow
# ══════════════════════════════════════════════════════════════════════════════

# ── Input ─────────────────────────────────────────────────────────────────────
box(P_INPUT_CX, P_Y_INPUT, P_W_INPUT, P_H_INPUT, C['input'],
    'Input Token IDs',
    r'$x_t,\,x_{t+1},\!\ldots,\!x_{T-1} \in \{0,\ldots,V\!-\!1\}$')

# ── φ one-hot (no-gradient branch) ────────────────────────────────────────────
box(P_PHI_CX, P_Y_PHI, P_W_PHI, P_H_PHI, C['phi'],
    r'$\boldsymbol{\phi}_{t+1:T-1}$',
    r'$\mathrm{one\_hot}(x_{t+1:T-1})$',
    fontsize=P_FS_BOX, sub_fs=P_FS_SUB)
padlock(P_PHI_LOCK_X, P_PHI_LOCK_Y, sz=P_PHI_LOCK_SZ, col='white', zo=6)
# Horizontal arrow from input left edge → φ right edge, label centred in gap
arr(P_INPUT_CX - P_W_INPUT/2, P_Y_PHI, P_PHI_CX + P_W_PHI/2, P_Y_PHI,
    color=C['grey'], lw=1.0)
ax.text((P_INPUT_CX - P_W_INPUT/2 + P_PHI_CX + P_W_PHI/2) / 2,
        P_Y_PHI - 0.95,
        r'$x_{t+1:T-1}$',
        ha='center', va='center', fontsize=P_FS_DIM, color=C['grey'])

# ── Embedding ─────────────────────────────────────────────────────────────────
box(P_EMBED_CX, P_Y_EMBED, P_W_EMBED, P_H_EMBED, C['embed'],
    r'Embedding$(V \to d)$')
arr(P_EMBED_CX, P_Y_INPUT - P_H_INPUT/2,
    P_EMBED_CX, P_Y_EMBED + P_H_EMBED/2,
    color=C['grey'],
    label=r'$x_t$', loff=(0.35, 0.0))

# ── Backbone ──────────────────────────────────────────────────────────────────
box(P_BB_CX, P_Y_BB, P_W_BB, P_H_BB, C['backbone'],
    'Backbone',
    r'$8\times$ ResidualBlock$(d)$')
arr(P_EMBED_CX, P_Y_EMBED - P_H_EMBED/2,
    P_EMBED_CX, P_Y_BB + P_H_BB/2,
    color=C['grey'])

# ── Branch to heads ───────────────────────────────────────────────────────────
Y_BRANCH = P_Y_BB - P_H_BB/2 - 1.0
vline(P_BB_CX, P_Y_BB - P_H_BB/2, Y_BRANCH, color=C['grey'])
hline(P_HX[0], P_HX[2], Y_BRANCH, color=C['grey'])

# ── γ-Heads ───────────────────────────────────────────────────────────────────
for hx, gl in zip(P_HX, P_GLABELS):
    arr(hx, Y_BRANCH, hx, P_Y_HEAD + P_H_HEAD/2, color=C['grey'])
    box(hx, P_Y_HEAD, P_W_HEAD, P_H_HEAD, C['head'],
        f'Head  {gl}',
        r'$7\times$ ResidualBlock$(d)$',
        r'$+$ Linear$(d\!\to\!V)$',
        r'$+$ softmax')

# ── ψ logits ──────────────────────────────────────────────────────────────────
for hx, pl in zip(P_HX, P_PSI_LBLS):
    arr(hx, P_Y_HEAD - P_H_HEAD/2, hx, P_Y_PSI + P_H_PSI/2, color=C['grey'])
    box(hx, P_Y_PSI, P_W_PSI, P_H_PSI, C['psi'],
        pl, '', fontsize=P_FS_BOX)

# ── KL loss ───────────────────────────────────────────────────────────────────
for hx, kl_lbl in zip(P_HX, P_KL_FORMULAS):
    arr(hx, P_Y_PSI - P_H_PSI/2, hx, P_Y_KL + P_H_KL/2, color=C['grey'])
    box(hx, P_Y_KL, P_W_KL, P_H_KL, C['loss'], kl_lbl, fontsize=P_FS_BOX)

# ── G_γ → KL: single horizontal arrow from target panel right edge to leftmost KL box ──
_src_x = P_RX + P_W_TR/2          # right edge of target panel
arr(_src_x, P_Y_KL, P_HX[0] - P_W_KL/2, P_Y_KL,
    color=C['loss'], lw=0.85)

# ── Total loss (same convergence style as backbone → heads) ──────────────────
y_conv = P_Y_KL - P_H_KL/2 - 1.0
for hx in P_HX:
    vline(hx, P_Y_KL - P_H_KL/2, y_conv, color=C['grey'])
hline(P_HX[0], P_HX[2], y_conv, color=C['grey'])
# Straight vertical arrow from P_TOTAL_CX (which lies on the hline: HX[0]≤60≤HX[2])
arr(P_TOTAL_CX, y_conv, P_TOTAL_CX, P_Y_TOTAL + P_H_TOTAL/2, color=C['grey'])
box(P_TOTAL_CX, P_Y_TOTAL, P_W_TOTAL, P_H_TOTAL, C['loss'],
    r'$\mathcal{L} = \sum_{\gamma} \mathrm{KL}(G_\gamma \,\|\, \psi_\gamma)$',
    '', fontsize=P_FS_BOX)

# ── Inference-only callout (same row as total loss, to the right) ─────────────
box(P_INFER_CX, P_Y_TOTAL, P_INFER_W, P_INFER_H,
    P_INFER_COL,
    'Inference:',
    r'$\mathrm{SR}_\gamma = \psi_\gamma \cdot (1\!-\!\gamma)^{-1}$',
    ec=P_INFER_EC, lw=0.8, ls='--', tc=P_INFER_TC, fontsize=P_FS_BOX, sub_fs=P_FS_FORMULA)

# ── φ → target rail ───────────────────────────────────────────────────────────
# 4-leg dashed path + solid arrowhead entering target box from the top.
# Leg 0: short horizontal LEFT exit from φ box left edge, clearing the bracket spine
# Leg 1: vertical DOWN along clear x (left of bracket at 37.5)
# Leg 2: horizontal LEFT jog to target entry column
# Leg 3: solid arrowhead DOWN into target box top-right area
if P_SHOW_PHI_RAIL:
    _lw  = P_PHI_RAIL_LW
    _ls  = P_PHI_RAIL_LS
    _col = C['phi']
    # Leg 0: horizontal exit from φ left edge to vertical rail x
    hline(P_PHI_VERT_X, P_PHI_RAIL_X, P_PHI_RAIL_Y, color=_col, lw=_lw, ls=_ls)
    # Leg 1: vertical from φ box level down to jog y
    vline(P_PHI_VERT_X, P_PHI_JOG_Y, P_PHI_RAIL_Y, color=_col, lw=_lw, ls=_ls)
    # Leg 2: horizontal jog left to entry x
    hline(P_PHI_ENTRY_X, P_PHI_VERT_X, P_PHI_JOG_Y, color=_col, lw=_lw, ls=_ls)
    # Leg 3: dashed line + arrowhead stopping at target box visual top
    ax.annotate('',
        xy=(P_PHI_ENTRY_X, P_PHI_TARGET_Y),
        xytext=(P_PHI_ENTRY_X, P_PHI_JOG_Y),
        arrowprops=dict(arrowstyle='->', color=_col, lw=_lw,
                        linestyle='--',
                        connectionstyle='arc3,rad=0', shrinkA=0, shrinkB=0),
        zorder=4, clip_on=False)

# ── θ bracket: left-side bracket spanning Embedding + Backbone + Heads ───────
_bx = P_THETA_BRAC_X
_bls = (0, (4, 2))   # same dash pattern as φ rail and shadow arrow
vline(_bx, P_THETA_BOT_Y, P_THETA_TOP_Y, color=P_THETA_COL, lw=P_THETA_LW, ls=_bls)
# Ticks point RIGHT (inward toward main content)
hline(_bx, _bx + P_THETA_TICK, P_THETA_TOP_Y, color=P_THETA_COL, lw=P_THETA_LW, ls=_bls)
hline(_bx, _bx + P_THETA_TICK, P_THETA_BOT_Y, color=P_THETA_COL, lw=P_THETA_LW, ls=_bls)
# Rotated label on RIGHT side of spine, vertically centred — describes the whole bracketed region
ax.text(_bx + 1.4, (P_THETA_TOP_Y + P_THETA_BOT_Y) / 2,
        P_THETA_LABEL, ha='center', va='center',
        fontsize=P_THETA_LABEL_FS, fontweight='bold',
        color=P_THETA_COL, rotation=90, zorder=5, clip_on=False,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.9))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — detail panels
# ══════════════════════════════════════════════════════════════════════════════
if P_SHOW_DIVIDER:
    vline(P_DIVIDER_X, P_YLIM_LO + 3, 96, color=C['lgrey'], lw=P_LW_DIV, ls='--')

# ── A. ResidualBlock detail panel ────────────────────────────────────────────
# Panel background at zorder=1 (behind everything) signals "legend/detail"
ax.add_patch(FancyBboxPatch(
    (P_BLK_PANEL_X, P_BLK_PANEL_Y), P_BLK_PANEL_W, P_BLK_PANEL_H,
    boxstyle='round,pad=0.3',
    facecolor=P_BLK_PANEL_FC, edgecolor=P_BLK_PANEL_EC,
    linewidth=P_BLK_PANEL_LW, alpha=0.97, zorder=1, clip_on=False))

sec(P_RX, P_Y_SEC_BLOCK, '◆  Residual Block', color=C['backbone'])
BX = P_RX

txt(BX, P_Y_BLK_IN, r'$x \in \mathbb{R}^{d}$', fs=P_FS_BOX, bold=True)

# Skip-connection: x_in → RIGHT → DOWN → LEFT → + (all FancyArrowPatch for uniform dotted style)
_resx = BX + P_RES_OFF
_skip_kw = dict(color=C['arrow'], linestyle=':', linewidth=P_LW_DETAIL,
                zorder=4, clip_on=False, shrinkA=0, shrinkB=0)
ax.add_patch(FancyArrowPatch(                                            # right leg
    (BX + 4, P_Y_BLK_IN), (_resx, P_Y_BLK_IN),
    arrowstyle='-', mutation_scale=7, **_skip_kw))
ax.add_patch(FancyArrowPatch(                                            # down leg
    (_resx, P_Y_BLK_IN), (_resx, P_Y_BLK_ADD),
    arrowstyle='-', mutation_scale=7, **_skip_kw))
ax.add_patch(FancyArrowPatch(                                            # left leg → arrowhead
    (_resx, P_Y_BLK_ADD), (BX + 5, P_Y_BLK_ADD),
    arrowstyle='->', mutation_scale=7, **_skip_kw))

# (colour, text-colour, label, sublabel) per role:
#   LayerNorm  → cool light gray + dark text  (utility, not a learned weight)
#   Linear W₁/W₂ → backbone blue + white text (matches main-flow blue blocks)
#   GELU       → warm coral + white text       (non-linearity, distinct category)
_comps = [
    (P_Y_BLK_LN,  C['blk_norm'],  '#555555', r'LayerNorm$(d)$',    ''),
    (P_Y_BLK_FC1, C['backbone'],  'white',   r'Linear$(d \to d)$', ''),
    (P_Y_BLK_ACT, C['blk_act'],   'white',   r'GELU',              ''),
    (P_Y_BLK_FC2, C['backbone'],  'white',   r'Linear$(d \to d)$', ''),
]
arr(BX, P_Y_BLK_IN - 0.3, BX, P_Y_BLK_LN + 1.1, color=C['grey'], lw=P_LW_DETAIL)
for i, (by, bc, btc, bl, bsl) in enumerate(_comps):
    bec = '#A0A0A8' if i == 0 else 'white'   # LayerNorm: thin medium-gray border for contrast
    box(BX, by, P_W_BLK, P_H_BLK, bc, bl, bsl, fontsize=P_FS_BOX, sub_fs=P_FS_SUB, tc=btc, ec=bec)
    if i < len(_comps) - 1:
        arr(BX, by - 1.1, BX, _comps[i+1][0] + 1.1,
            color=C['grey'], lw=P_LW_DETAIL)

arr(BX, P_Y_BLK_FC2 - 1.1, BX, P_Y_BLK_ADD + 1.0, color=C['grey'], lw=P_LW_DETAIL)
# + box: dark navy — structural wiring, not a learned layer
box(BX, P_Y_BLK_ADD, 10, 2.0, C['blk_add'], r'$\boldsymbol{+}$', '', fontsize=10, tc='#555555', ec='#A0A0A8')
arr(BX, P_Y_BLK_ADD - 1.0, BX, P_Y_BLK_OUT + 0.4, color=C['grey'], lw=P_LW_DETAIL)
txt(BX, P_Y_BLK_OUT, r'$x_\mathrm{out} \in \mathbb{R}^{d}$', fs=P_FS_BOX, bold=True)

# ── B. EMA target network ─────────────────────────────────────────────────────

# Shadow-copy arrow: use arr() for a clean visible arrowhead (FancyArrowPatch
# loses its arrowhead on short dashed paths at this scale)
ax.add_patch(FancyArrowPatch(
    (P_SHADOW_START_X, P_SHADOW_START_Y), (P_SHADOW_END_X, P_SHADOW_END_Y),
    arrowstyle='->', color=P_SHADOW_COL,
    linestyle='--', linewidth=P_SHADOW_LW,
    mutation_scale=8, zorder=4, clip_on=False))
if P_SHADOW_LABEL:
    txt(P_SHADOW_LBL_X, P_SHADOW_LBL_Y, P_SHADOW_LABEL,
        fs=P_SHADOW_LBL_FS, color=P_SHADOW_COL, italic=True)

sec(P_RX, P_Y_SEC_EMA, '◆  EMA Target Network', color=C['ema'])

box(P_RX, P_Y_EMA, P_W_EMA, P_H_EMA, C['ema'],
    r'$\theta_\mathrm{ema} \leftarrow \alpha\,\theta_\mathrm{ema} + (1\!-\!\alpha)\,\theta_\mathrm{live}$',
    '', fontsize=P_FS_BOX, alpha=0.88, ls='-', ec='white', lw=1.5)

# Frozen indicator: hatch overlay + padlock badge
if P_EMA_HATCH:
    ax.add_patch(FancyBboxPatch(
        (P_RX - P_W_EMA/2, P_Y_EMA - P_H_EMA/2), P_W_EMA, P_H_EMA,
        boxstyle='round,pad=0.22', facecolor='none', edgecolor='none',
        hatch=P_EMA_HATCH, alpha=P_EMA_HATCH_ALPHA, linewidth=0,
        zorder=4, clip_on=False))
padlock(P_EMA_LOCK_X, P_EMA_LOCK_Y, sz=P_EMA_LOCK_SZ,
        col='white', zo=6)

arr(P_RX, P_Y_EMA - P_H_EMA/2, P_RX, P_Y_SEC_TARGET + 0.5,
    color=C['ema'], lw=P_LW_MAIN,
    label=r'$\psi_\mathrm{ema}$', loff=(0.5, 0), label_color=C['ema'])

# ── C. λ-return target construction ──────────────────────────────────────────
sec(P_RX, P_Y_SEC_TARGET, r'◆  $\lambda$-Return Target  $G_\gamma$', color=C['loss'])

ax.add_patch(FancyBboxPatch(
    (P_RX - P_W_TR/2, P_Y_TARGET - 2.5),
    P_W_TR, 7.5,                         # asymmetric: content spans -2.1..+4.2 from centre
    boxstyle='round,pad=0.3',
    facecolor='#FFF3F0', edgecolor=C['loss'],
    linewidth=1.2, alpha=0.96, zorder=2))

txt(P_RX, P_Y_TARGET + 4.2, r'Boundary  $(t = T\!-\!2)$:',
    fs=P_FS_BOX, bold=True)
txt(P_RX, P_Y_TARGET + 2.7,
    r'$G_{T-2} = (1\!-\!\gamma)\,\phi_{T-1} + \gamma\,\psi_\mathrm{ema,T-1}$',
    fs=7.8)

txt(P_RX, P_Y_TARGET + 1.0, r'General  $(t < T\!-\!2)$:',
    fs=P_FS_BOX, bold=True)
txt(P_RX, P_Y_TARGET - 0.35,
    r'$G_t = (1\!-\!\gamma)\,\phi_{t+1}$',
    fs=7.8)
txt(P_RX, P_Y_TARGET - 1.85,
    r'$\quad+\;\gamma\,[(1\!-\!\lambda)\,\psi_\mathrm{ema,t+1} + \lambda\,G_{t+1}]$',
    fs=7.8)

# φ entry: the arrowhead on the rail (drawn above) arrives at the G_t formula level

# ── Footnote (optional) ───────────────────────────────────────────────────────
if P_FOOTNOTE:
    txt(50, P_YLIM_LO + 1.5, P_FOOTNOTE, fs=P_FS_FOOTNOTE, color=C['lgrey'])

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out_dir = os.path.join(OUTPUT_PREFIX, 'architecture')
os.makedirs(out_dir, exist_ok=True)

# Compute tight bbox once so PDF and PNG are pixel-identical in layout
renderer = fig.canvas.get_renderer()
tight_bbox = fig.get_tightbbox(renderer)

for fmt, dpi in [('pdf', None), ('png', 600)]:
    fpath = os.path.join(out_dir, f'architecture_infographic.{fmt}')
    fig.savefig(fpath, bbox_inches=tight_bbox, dpi=dpi, facecolor=C['bg'])
    print(f'Saved: {fpath}')

plt.close(fig)
