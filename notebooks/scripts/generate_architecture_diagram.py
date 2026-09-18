from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIG_W = 15.0
FIG_H = 7.8

BORDER = "#003366"
TEXT = "#1F2937"
MUTED = "#4B5563"
SECTION_BLUE = "#EAF4FF"
SECTION_PURPLE = "#F3EEFF"
SECTION_GREEN = "#ECF9EF"
BOX_WHITE = "#FFFFFF"
BOX_ORANGE = "#FFF1DD"


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    *,
    fill=BOX_WHITE,
    title_size=10.5,
    body_size=9.0,
    title_y=0.67,
    body_y=0.33,
):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        linewidth=1.6,
        edgecolor=BORDER,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
        family="Arial",
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=MUTED,
        family="Arial",
        zorder=4,
    )


def add_section(ax, x, y, w, h, title, fill):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.5,
        edgecolor=BORDER,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.16,
        y + h - 0.20,
        title,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=BORDER,
        family="Arial",
        zorder=2,
    )


def ortho_arrow(ax, start, end, *, mid_x=None, mid_y=None, color=BORDER, lw=1.8):
    sx, sy = start
    ex, ey = end
    if mid_x is not None:
        ax.plot([sx, mid_x], [sy, sy], color=color, lw=lw, zorder=2)
        ax.plot([mid_x, mid_x], [sy, ey], color=color, lw=lw, zorder=2)
        tail = (mid_x, ey)
    elif mid_y is not None:
        ax.plot([sx, sx], [sy, mid_y], color=color, lw=lw, zorder=2)
        ax.plot([sx, ex], [mid_y, mid_y], color=color, lw=lw, zorder=2)
        tail = (ex, mid_y)
    else:
        tail = start
    arrow = FancyArrowPatch(
        tail,
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        zorder=2,
    )
    ax.add_patch(arrow)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig1_architecture_diagram.png"
    svg_path = out_dir / "fig1_architecture_diagram.svg"

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.default"] = "regular"

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    ax.text(
        FIG_W / 2,
        7.45,
        "Architecture of the GoEmotions DeBERTa-v3 Multi-Label Emotion Classifier",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        family="Arial",
    )
    ax.text(
        FIG_W / 2,
        7.18,
        "Official GoEmotions split, DeBERTa-v3 tokenization, clipped ASL training, validation-time threshold calibration, and token-level XAI",
        ha="center",
        va="center",
        fontsize=9.3,
        color=MUTED,
        family="Arial",
    )

    # Section containers with >= 0.5in separation.
    add_section(ax, 0.55, 0.85, 3.75, 5.90, "Section 1. Data and Tokens", SECTION_BLUE)
    add_section(ax, 4.95, 0.85, 5.20, 5.90, "Section 2. Architecture and Optimization", SECTION_PURPLE)
    add_section(ax, 10.80, 0.85, 3.65, 5.90, "Section 3. Predictions and XAI", SECTION_GREEN)

    # Left section blocks.
    add_box(
        ax,
        1.08,
        5.16,
        2.70,
        1.02,
        "Input Reddit Text",
        "GoEmotions official split\n43,410 train | 5,426 val | 5,427 test",
        title_size=10.2,
        body_size=8.4,
        title_y=0.68,
        body_y=0.30,
    )
    add_box(
        ax,
        1.08,
        3.78,
        2.70,
        0.92,
        "Tokenizer",
        "28 labels -> 7 macro labels\nDeBERTa-v3 tokenizer, max length = 128",
        body_size=8.2,
        title_y=0.66,
        body_y=0.29,
    )
    add_box(
        ax,
        1.08,
        2.38,
        2.70,
        0.92,
        "Sub-token Embeddings",
        "input_ids + attention_mask\nbatched contextual token sequence",
        body_size=8.4,
        title_y=0.66,
        body_y=0.29,
    )

    # Center section blocks.
    add_box(
        ax,
        5.45,
        4.92,
        4.10,
        1.15,
        "DeBERTa-v3-base\nEncoder Backbone",
        "contextual transformer encoder\ndisentangled attention over token sequence",
        title_size=10.0,
        body_size=8.8,
        title_y=0.65,
        body_y=0.30,
    )
    add_box(
        ax,
        6.52,
        3.62,
        2.05,
        0.86,
        "Logits Generation",
        "linear head -> 7 logits",
        fill=BOX_WHITE,
        title_y=0.64,
        body_y=0.31,
    )
    add_box(
        ax,
        5.35,
        2.02,
        2.10,
        1.12,
        "Clipped Asymmetric Loss\n+ Class Weighting",
        "$\\mathcal{L}_{ASL}$ for training\n$\\gamma_{neg}, \\gamma_{pos}, clip, \\mathbf{w}_c$",
        fill=BOX_ORANGE,
        title_size=9.3,
        body_size=8.6,
        title_y=0.66,
        body_y=0.28,
    )
    add_box(
        ax,
        7.58,
        2.02,
        2.10,
        1.12,
        "Validation Threshold\nTuning",
        "validation n = 5,426\ncoordinate ascent for $t_c$",
        fill=BOX_ORANGE,
        title_size=8.8,
        body_size=8.1,
        title_y=0.66,
        body_y=0.28,
    )

    # Right section blocks.
    add_box(
        ax,
        11.28,
        4.36,
        2.68,
        1.00,
        "7 Macro-Emotion Output",
        "7 sigmoid scores $\\mathbf{p}_i$\nmulti-hot decision using $t_c$",
        fill=BOX_WHITE,
        title_size=9.9,
        body_size=8.3,
    )
    add_box(
        ax,
        11.28,
        2.78,
        2.68,
        1.12,
        "Captum Integrated Gradients",
        "token attribution heatmaps\n$n_{steps}=32$ explanation tracing",
        fill=BOX_WHITE,
        title_size=10.0,
        body_size=8.4,
    )
    add_box(
        ax,
        11.28,
        1.20,
        2.68,
        0.96,
        "Token Attributions &\nFaithfulness (AOPC)",
        "visual heatmaps + deletion AOPC\nexplanation metrics for reporting",
        fill=BOX_WHITE,
        title_size=9.5,
        body_size=8.1,
        title_y=0.64,
        body_y=0.27,
    )

    # Orthogonal connectors.
    ortho_arrow(ax, (2.43, 5.16), (2.43, 4.70))
    ortho_arrow(ax, (2.43, 3.78), (2.43, 3.30))
    ortho_arrow(ax, (3.78, 2.84), (5.45, 5.50), mid_x=4.55)
    ortho_arrow(ax, (7.57, 4.92), (7.57, 4.48))
    ortho_arrow(ax, (7.55, 3.62), (7.55, 3.14))
    ortho_arrow(ax, (8.63, 4.05), (8.63, 3.14))
    ortho_arrow(ax, (9.68, 2.58), (11.28, 4.86), mid_x=10.35)
    ortho_arrow(ax, (12.62, 4.36), (12.62, 3.90))
    ortho_arrow(ax, (12.62, 2.78), (12.62, 2.16))

    # Small notes for publication clarity.
    ax.text(7.50, 0.42, "Directional flow: dataset -> tokenization -> encoder -> logits -> training/calibration -> predictions -> XAI evidence.", ha="center", va="center", fontsize=8.3, color=MUTED, family="Arial")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
