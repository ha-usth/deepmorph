"""
visualize_architecture.py - DeepMorph Network Architecture Diagram

Illustrates tensor dimensions at every stage:
  Stage 1: MAE Pretraining (Encoder + Decoder)
  Stage 2: Fine-tuning  (Heatmap head / Regression head)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ─── palette ──────────────────────────────────────────────────────────────────
C = dict(
    input    = "#2196F3",   # blue
    embed    = "#9C27B0",   # purple
    mask     = "#FF5722",   # deep-orange
    enc      = "#1976D2",   # dark blue
    latent   = "#0097A7",   # teal
    dec      = "#388E3C",   # green
    pred     = "#F57C00",   # orange
    loss     = "#C62828",   # red
    head_h   = "#7B1FA2",   # violet  – heatmap head
    head_r   = "#00695C",   # teal    – regression head
    arrow    = "#37474F",   # dark gray
    bg_enc   = "#E3F2FD",
    bg_dec   = "#E8F5E9",
    bg_stage2= "#F3E5F5",
    text     = "#212121",
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, label, sublabel="", fontsize=8, alpha=0.92,
        text_color="white", radius=0.012):
    """Draw a rounded rectangle with centred text."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor="white",
                          linewidth=1.4, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.022, sublabel,
                ha="center", va="center", fontsize=6.5,
                color=text_color, alpha=0.88, zorder=4,
                style="italic")


def arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.4,
          style="->", label="", fs=7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.012, my, label, ha="left", va="center",
                fontsize=fs, color=color, zorder=5)


def bracket_label(ax, x, y_top, y_bot, text, color="#555"):
    """Vertical brace + label on the right side."""
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="]-[", color=color,
                                lw=1.2, mutation_scale=8))
    ax.text(x + 0.018, (y_top + y_bot)/2, text,
            ha="left", va="center", fontsize=7, color=color, rotation=90,
            fontweight="bold")


def section_bg(ax, x0, y0, x1, y1, color, label, fontsize=8):
    rect = mpatches.FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                                   boxstyle="round,pad=0.008",
                                   facecolor=color, edgecolor="#BDBDBD",
                                   linewidth=1, alpha=0.35, zorder=1)
    ax.add_patch(rect)
    ax.text((x0+x1)/2, y1 - 0.018, label,
            ha="center", va="top", fontsize=fontsize,
            fontweight="bold", color="#424242", zorder=2, alpha=0.8)


# ─── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

fig.suptitle(
    "DeepMorph — Network Architecture & Tensor Dimensions\n"
    "MAE ViT-Base  (img=224², patch=16², mask=75%)",
    fontsize=14, fontweight="bold", color=C["text"], y=0.98
)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MAE  (left 62% of canvas)
# ══════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.01, 0.03, 0.63, 0.94, C["bg_enc"], "Stage 1 — MAE Pre-training", fontsize=9)

# ── Encoder sub-section ──────────────────────────────────────────
section_bg(ax, 0.02, 0.52, 0.41, 0.91, C["bg_enc"], "Encoder", fontsize=8)

# Node positions (x, y)
nodes_enc = {
    "input":   (0.08, 0.84),
    "patch":   (0.08, 0.74),
    "posemb":  (0.08, 0.64),
    "mask":    (0.08, 0.545),
    "clstok":  (0.22, 0.545),
    "concat":  (0.22, 0.64),
    "enc_blk": (0.22, 0.74),
    "enc_norm":(0.22, 0.84),
}

# ── Input image
box(ax, *nodes_enc["input"], 0.10, 0.038, C["input"],
    "Input Image", "(B, 3, 224, 224)")

# ── Patch Embedding
arrow(ax, nodes_enc["input"][0], nodes_enc["input"][1] - 0.020,
           nodes_enc["patch"][0],  nodes_enc["patch"][1]  + 0.020)
box(ax, *nodes_enc["patch"], 0.10, 0.038, C["embed"],
    "Patch Embed", "Conv2d 16×16 / 16  →  (B, 196, 768)")

# ── Positional Embedding
arrow(ax, nodes_enc["patch"][0], nodes_enc["patch"][1] - 0.020,
           nodes_enc["posemb"][0], nodes_enc["posemb"][1] + 0.020)
box(ax, *nodes_enc["posemb"], 0.10, 0.038, C["embed"],
    "+ Pos Embed", "sin-cos 2D  →  (B, 196, 768)")

# ── Random Masking  (75 %)
arrow(ax, nodes_enc["posemb"][0], nodes_enc["posemb"][1] - 0.020,
           nodes_enc["mask"][0],   nodes_enc["mask"][1]   + 0.020)
box(ax, *nodes_enc["mask"], 0.10, 0.038, C["mask"],
    "Random Mask 75%", "keep 49/196  →  (B, 49, 768)")

# ── CLS token (side branch)
box(ax, *nodes_enc["clstok"], 0.10, 0.038, C["embed"],
    "CLS Token", "expand  →  (B, 1, 768)")

# ── Concat CLS + visible patches
arrow(ax, nodes_enc["mask"][0],   nodes_enc["mask"][1]   + 0.020,
           nodes_enc["concat"][0], nodes_enc["concat"][1] - 0.020,
           label="cat")
arrow(ax, nodes_enc["clstok"][0], nodes_enc["clstok"][1] + 0.020,
           nodes_enc["concat"][0], nodes_enc["concat"][1] - 0.020,
           label="")
box(ax, *nodes_enc["concat"], 0.10, 0.038, C["embed"],
    "Concat [CLS|patches]", "(B, 50, 768)")

# ── Encoder Transformer ×12
arrow(ax, nodes_enc["concat"][0],  nodes_enc["concat"][1]  + 0.020,
           nodes_enc["enc_blk"][0], nodes_enc["enc_blk"][1] - 0.020)
box(ax, *nodes_enc["enc_blk"], 0.10, 0.050, C["enc"],
    "Transformer Block ×12",
    "LN → MHA(12h) → LN → MLP(768→3072→768)\n(B, 50, 768) → (B, 50, 768)")

# ── Encoder LayerNorm
arrow(ax, nodes_enc["enc_blk"][0],  nodes_enc["enc_blk"][1]  + 0.026,
           nodes_enc["enc_norm"][0], nodes_enc["enc_norm"][1] - 0.020)
box(ax, *nodes_enc["enc_norm"], 0.10, 0.038, C["latent"],
    "LayerNorm → Latent", "(B, 50, 768)")

# ── Latent label
ax.text(0.305, 0.845, "LATENT\n(B, 50, 768)",
        ha="center", va="center", fontsize=8,
        fontweight="bold", color=C["latent"],
        bbox=dict(boxstyle="round,pad=0.05", fc="#E0F7FA", ec=C["latent"], lw=1.2))
arrow(ax, nodes_enc["enc_norm"][0] + 0.053, nodes_enc["enc_norm"][1],
           0.292, 0.845, color=C["latent"])

# ── Decoder sub-section ──────────────────────────────────────────
section_bg(ax, 0.02, 0.06, 0.62, 0.50, C["bg_dec"], "Decoder  (pretraining only)", fontsize=8)

nodes_dec = {
    "dec_emb":  (0.12, 0.44),
    "mask_tok": (0.30, 0.34),
    "restore":  (0.12, 0.34),
    "add_pos":  (0.12, 0.24),
    "dec_blk":  (0.12, 0.155),
    "dec_norm": (0.12, 0.08),
    "dec_pred": (0.30, 0.08),
    "loss":     (0.48, 0.08),
}

# ── Decoder Embed
arrow(ax, 0.22, nodes_enc["enc_norm"][1] - 0.019,
           nodes_dec["dec_emb"][0], nodes_dec["dec_emb"][1] + 0.020,
           label="", color=C["latent"])
# vertical down from latent column
ax.annotate("", xy=(0.12, 0.46), xytext=(0.22, 0.83),
            arrowprops=dict(arrowstyle="->", color=C["latent"], lw=1.4,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
box(ax, *nodes_dec["dec_emb"], 0.12, 0.038, C["dec"],
    "Decoder Embed", "Linear(768→512)  →  (B, 50, 512)")

# ── Mask Tokens
box(ax, *nodes_dec["mask_tok"], 0.12, 0.038, C["mask"],
    "Mask Tokens", "147 masked pos  →  (B, 147, 512)")

# ── Restore / Unshuffle
arrow(ax, nodes_dec["dec_emb"][0],   nodes_dec["dec_emb"][1]   - 0.020,
           nodes_dec["restore"][0],   nodes_dec["restore"][1]   + 0.020)
arrow(ax, nodes_dec["mask_tok"][0],   nodes_dec["mask_tok"][1]  - 0.020,
           nodes_dec["restore"][0] + 0.065, nodes_dec["restore"][1] + 0.020,
           label="unshuffle")
box(ax, *nodes_dec["restore"], 0.12, 0.038, C["dec"],
    "Restore Full Seq", "cat+gather  →  (B, 197, 512)")

# ── Add Decoder Pos Embed
arrow(ax, nodes_dec["restore"][0], nodes_dec["restore"][1] - 0.020,
           nodes_dec["add_pos"][0], nodes_dec["add_pos"][1] + 0.020)
box(ax, *nodes_dec["add_pos"], 0.12, 0.038, C["dec"],
    "+ Decoder Pos Embed", "(B, 197, 512)")

# ── Decoder Transformer ×8
arrow(ax, nodes_dec["add_pos"][0], nodes_dec["add_pos"][1] - 0.020,
           nodes_dec["dec_blk"][0], nodes_dec["dec_blk"][1] + 0.026)
box(ax, *nodes_dec["dec_blk"], 0.12, 0.052, C["dec"],
    "Transformer Block ×8",
    "LN → MHA(16h) → LN → MLP(512→2048→512)\n(B, 197, 512) → (B, 197, 512)")

# ── Decoder LayerNorm
arrow(ax, nodes_dec["dec_blk"][0],  nodes_dec["dec_blk"][1]  - 0.027,
           nodes_dec["dec_norm"][0], nodes_dec["dec_norm"][1] + 0.020)
box(ax, *nodes_dec["dec_norm"], 0.12, 0.038, C["dec"],
    "LayerNorm", "(B, 197, 512)")

# ── Decoder Pred
arrow(ax, nodes_dec["dec_norm"][0] + 0.062, nodes_dec["dec_norm"][1],
           nodes_dec["dec_pred"][0] - 0.063, nodes_dec["dec_pred"][1])
box(ax, *nodes_dec["dec_pred"], 0.12, 0.038, C["pred"],
    "Decoder Pred (Linear)", "512 → 768  →  (B, 196, 768)\nremove CLS  →  (B, 196, 768)")

# ── Loss
arrow(ax, nodes_dec["dec_pred"][0] + 0.063, nodes_dec["dec_pred"][1],
           nodes_dec["loss"][0] - 0.065, nodes_dec["loss"][1])
box(ax, *nodes_dec["loss"], 0.12, 0.050, C["loss"],
    "MAE Loss (MSE)",
    "Target: patchify(imgs) → (B, 196, 768)\nnorm per-patch, loss on masked only")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Fine-tuning  (right 36%)
# ══════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.65, 0.03, 0.99, 0.94, C["bg_stage2"],
           "Stage 2 — Fine-tuning (Landmark Detection)", fontsize=9)

# ── Frozen encoder (shared)
box(ax, 0.82, 0.82, 0.22, 0.040, C["enc"],
    "MAE Encoder (frozen/fine-tuned)", "forward_features  →  (B, 197, 768)")
ax.text(0.82, 0.77, "↓  same weights from Stage 1",
        ha="center", fontsize=7, color="#555", style="italic")

# ─────────────────────────────────────────────
# Heatmap Head branch  (left half of stage 2)
# ─────────────────────────────────────────────
hx = 0.71   # centre-x for heatmap branch

section_bg(ax, 0.655, 0.12, 0.78, 0.75, "#FCE4EC",
           "Heatmap Head", fontsize=7.5)

nodes_hm = [
    (hx, 0.69, "(B, 196, 768)  drop CLS"),
    (hx, 0.60, "reshape  →  (B, 768, 14, 14)"),
    (hx, 0.50, "ConvT2d(768→256, 2×2)\n(B, 256, 28, 28)"),
    (hx, 0.40, "Conv2d(256→256, 3×3)\n(B, 256, 28, 28)"),
    (hx, 0.30, "ConvT2d(256→128, 2×2)\n(B, 128, 56, 56)"),
    (hx, 0.20, "Conv2d(128→128, 3×3)\n(B, 128, 56, 56)"),
    (hx, 0.13, "Conv2d(128→K, 1×1)\nHeatmaps (B, 15, 56, 56)"),
]
labels_hm = [
    "Drop CLS Token", "Reshape 1D→2D",
    "Upsample ×2", "Refine",
    "Upsample ×2", "Refine",
    "Predict Heatmaps",
]
colors_hm = [C["latent"], C["embed"], C["head_h"], C["head_h"],
             C["head_h"], C["head_h"], C["pred"]]

arrow(ax, 0.82, 0.80, hx, 0.71, color=C["enc"])
for i, (x, y, sub) in enumerate(nodes_hm):
    box(ax, x, y, 0.105, 0.046, colors_hm[i], labels_hm[i], sub, fontsize=7)
    if i < len(nodes_hm) - 1:
        arrow(ax, x, y - 0.023, x, nodes_hm[i+1][1] + 0.023)

# ─────────────────────────────────────────────
# Regression Head branch  (right half of stage 2)
# ─────────────────────────────────────────────
rx = 0.93

section_bg(ax, 0.875, 0.12, 0.995, 0.75, "#E8F5E9",
           "Regression Head", fontsize=7.5)

nodes_rg = [
    (rx, 0.69, "CLS  (B, 768)\n+ GAP patches  (B, 768)"),
    (rx, 0.58, "Concat  →  (B, 1536)"),
    (rx, 0.48, "Linear(1536→512)\nLayerNorm + GELU + Drop"),
    (rx, 0.38, "Linear(512→256)\nLayerNorm + GELU + Drop"),
    (rx, 0.28, "Linear(256→K×2)\n(B, 30)"),
    (rx, 0.18, "Reshape + Sigmoid\nCoords  (B, 15, 2)"),
]
labels_rg = [
    "Split Tokens", "Concat",
    "MLP Layer 1", "MLP Layer 2",
    "MLP Layer 3", "Output Coords",
]
colors_rg = [C["latent"], C["embed"], C["head_r"], C["head_r"],
             C["head_r"], C["pred"]]

arrow(ax, 0.82, 0.80, rx, 0.71, color=C["enc"])
for i, (x, y, sub) in enumerate(nodes_rg):
    box(ax, x, y, 0.105, 0.055, colors_rg[i], labels_rg[i], sub, fontsize=7)
    if i < len(nodes_rg) - 1:
        arrow(ax, x, y - 0.027, x, nodes_rg[i+1][1] + 0.027)

# ══════════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C["input"],  "Input / Data"),
    (C["embed"],  "Embedding / Positional"),
    (C["mask"],   "Masking / Mask Token"),
    (C["enc"],    "Encoder Block"),
    (C["latent"], "Latent / Features"),
    (C["dec"],    "Decoder Block"),
    (C["pred"],   "Prediction Layer"),
    (C["loss"],   "Loss Function"),
    (C["head_h"], "Heatmap Head Conv"),
    (C["head_r"], "Regression Head MLP"),
]
lx, ly = 0.655, 0.98
ax.text(lx, ly, "Legend:", fontsize=7.5, fontweight="bold", color=C["text"],
        va="top", ha="left")
for i, (col, lbl) in enumerate(legend_items):
    xi = lx + (i % 5) * 0.068
    yi = ly - 0.035 - (i // 5) * 0.030
    patch = mpatches.FancyBboxPatch((xi, yi - 0.010), 0.012, 0.018,
                                     boxstyle="round,pad=0.002",
                                     facecolor=col, edgecolor="white",
                                     linewidth=0.8, zorder=5)
    ax.add_patch(patch)
    ax.text(xi + 0.015, yi - 0.001, lbl, fontsize=6.0,
            va="center", color=C["text"])

# ══════════════════════════════════════════════════════════════════════════════
# Key numbers annotation box
# ══════════════════════════════════════════════════════════════════════════════
info = (
    "Key numbers (ViT-Base)\n"
    "Patches : 14×14 = 196\n"
    "Visible  : 196×0.25 = 49\n"
    "Masked  : 196×0.75 = 147\n"
    "Encoder depth : 12 blocks\n"
    "Decoder depth : 8 blocks\n"
    "Encoder embed : 768\n"
    "Decoder embed : 512\n"
    "MLP ratio  : 4×\n"
    "Patch pixels : 16²×3 = 768\n"
    "Total params : ~86 M"
)
ax.text(0.355, 0.87, info, ha="left", va="top", fontsize=7,
        color="#212121", family="monospace",
        bbox=dict(boxstyle="round,pad=0.08", fc="#FFFDE7",
                  ec="#F9A825", lw=1.2, alpha=0.95),
        zorder=6)

# ── Stage divider
ax.axvline(x=0.635, color="#78909C", lw=1.5, ls="--", alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "architecture_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved: {out}")
plt.close(fig)
