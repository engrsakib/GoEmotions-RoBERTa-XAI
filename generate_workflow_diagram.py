from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "workflow_diagram.png"


def add_box(ax, xy, width, height, text, *, fc, ec="#5f6b7a", fontsize=11, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#1f2933",
        wrap=True,
    )
    return patch


def add_arrow(ax, start, end, *, color="#64748b", lw=1.8, mutation=14, style="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    return arrow


def center_right(x, y, w, h):
    return (x + w, y + h / 2)


def center_left(x, y, w, h):
    return (x, y + h / 2)


def center_bottom(x, y, w, h):
    return (x + w / 2, y)


def center_top(x, y, w, h):
    return (x + w / 2, y + h)


def main():
    fig, ax = plt.subplots(figsize=(20, 11), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(
        12,
        13.35,
        "GoEmotions-RoBERTa-XAI Workflow",
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color="#0f172a",
    )
    ax.text(
        12,
        12.82,
        "Draw.io-style view of data flow, training stages, dual tracks, and deployment artifacts",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )

    # Inputs
    raw = (0.8, 10.3, 3.1, 1.15)
    kaggle = (0.8, 8.7, 3.1, 1.0)
    config = (0.8, 7.25, 3.1, 1.0)
    add_box(ax, raw[:2], raw[2], raw[3], "Raw GoEmotions Data\nReddit comments + labels", fc="#dbeafe", weight="bold")
    add_box(ax, kaggle[:2], kaggle[2], kaggle[3], "Kaggle / Local Paths\nmixed-source artifact discovery", fc="#e0f2fe")
    add_box(ax, config[:2], config[2], config[3], "Configs & Profiles\ntrain_config.yaml\nmodel_profiles.yaml", fc="#ede9fe")

    # Central stage chain
    stages = [
        ((4.8, 10.15, 2.4, 1.2), "Stage 0\nBootstrap"),
        ((7.7, 10.15, 2.55, 1.2), "Stage 1\nData Pipeline"),
        ((10.75, 10.15, 2.3, 1.2), "Stage 2\nEDA"),
        ((13.6, 10.15, 2.7, 1.2), "Stage 3\nBaselines"),
        ((16.85, 10.15, 2.5, 1.2), "Stage 4\nTrain"),
        ((19.85, 10.15, 2.55, 1.2), "Stage 5\nEvaluate"),
    ]
    colors = ["#fee2e2", "#dcfce7", "#fef3c7", "#fae8ff", "#e0f2fe", "#dbeafe"]
    for (box, label), color in zip(stages, colors):
        add_box(ax, box[:2], box[2], box[3], label, fc=color, weight="bold")

    for idx in range(len(stages) - 1):
        a = stages[idx][0]
        b = stages[idx + 1][0]
        add_arrow(ax, center_right(*a), center_left(*b))

    # Stage 6 and 7 below eval to keep room
    stage6 = (18.0, 8.15, 2.8, 1.15)
    stage7 = (21.2, 8.15, 2.0, 1.15)
    add_box(ax, stage6[:2], stage6[2], stage6[3], "Stage 6\nXAI", fc="#fde68a", weight="bold")
    add_box(ax, stage7[:2], stage7[2], stage7[3], "Stage 7\nExport", fc="#bbf7d0", weight="bold")
    add_arrow(ax, center_bottom(*stages[-1][0]), center_top(*stage6))
    add_arrow(ax, center_right(*stage6), center_left(*stage7))

    # Side output boxes
    processed = (7.25, 7.8, 3.0, 1.1)
    figures = (10.7, 7.8, 3.0, 1.1)
    checkpoints = (14.25, 7.8, 3.0, 1.1)
    metrics = (19.45, 6.4, 3.2, 1.1)
    export = (21.0, 4.8, 2.8, 1.2)
    add_box(ax, processed[:2], processed[2], processed[3], "Processed Splits\ntrain/validation/test\nlabel_map.json", fc="#dcfce7")
    add_box(ax, figures[:2], figures[2], figures[3], "EDA & Paper Figures\nclass distribution\npublication PNGs", fc="#fef3c7")
    add_box(ax, checkpoints[:2], checkpoints[2], checkpoints[3], "Checkpoints\nHF Trainer artifacts", fc="#e0f2fe")
    add_box(ax, metrics[:2], metrics[2], metrics[3], "Metrics & Thresholds\nmacro/micro-F1\nHamming loss\ngap analysis", fc="#dbeafe")
    add_box(ax, export[:2], export[2], export[3], "Saved Model Export\nweights + thresholds.json", fc="#bbf7d0", weight="bold")

    add_arrow(ax, center_bottom(*stages[1][0]), center_top(*processed), style="->", mutation=12)
    add_arrow(ax, center_bottom(*stages[2][0]), center_top(*figures), style="->", mutation=12)
    add_arrow(ax, center_bottom(*stages[4][0]), center_top(*checkpoints), style="->", mutation=12)
    add_arrow(ax, center_bottom(*stages[5][0]), center_top(*metrics), style="->", mutation=12)
    add_arrow(ax, center_bottom(*stage7), center_top(*export))

    # Inputs to chain
    add_arrow(ax, center_right(*raw), center_left(*stages[1][0]))
    add_arrow(ax, center_right(*kaggle), (5.15, 10.45), style="->", mutation=12)
    add_arrow(ax, center_right(*config), (5.15, 7.75), style="->", mutation=12)
    add_arrow(ax, (5.15, 7.75), (17.8, 7.75), style="->", mutation=12, lw=1.4, color="#8b5cf6")
    add_arrow(ax, (17.8, 7.75), center_left(*stage6), style="->", mutation=12, lw=1.4, color="#8b5cf6")

    # Dual tracks
    ax.text(5.0, 5.85, "Dual Research / Deployment Tracks", fontsize=15, weight="bold", color="#0f172a")
    track_a = (4.8, 3.75, 7.6, 1.35)
    track_b = (13.0, 3.75, 7.8, 1.35)
    add_box(
        ax,
        track_a[:2],
        track_a[2],
        track_a[3],
        "Track A: IEEE Multilabel Research\n7-macro multi-hot labels\nDeBERTa-v3 + asymmetric loss\nthreshold-tuned benchmark reporting",
        fc="#dbeafe",
        weight="bold",
    )
    add_box(
        ax,
        track_b[:2],
        track_b[2],
        track_b[3],
        "Track B: Production Inference\nsingle-macro clean rows\nRoBERTa / distillation path\nFastAPI-ready explainable serving",
        fc="#dcfce7",
        weight="bold",
    )

    add_arrow(ax, (18.1, 9.9), center_top(*track_a), style="->", mutation=12)
    add_arrow(ax, center_bottom(*track_a), (9.0, 2.65), style="->", mutation=12)
    add_arrow(ax, center_bottom(*track_b), center_top(*export), style="->", mutation=12)
    ax.text(9.0, 2.32, "Teacher logits / evaluation results", ha="center", fontsize=9.5, color="#475569")
    add_arrow(ax, (9.0, 2.55), center_left(*track_b), style="->", mutation=12, color="#0ea5e9")

    # Notes
    note1 = (0.85, 5.0, 3.0, 1.25)
    note2 = (21.0, 2.7, 2.8, 1.25)
    add_box(ax, note1[:2], note1[2], note1[3], "Key data ops\nmap 28 -> 7 labels\nclean / dedup / split\noptional balance & augmentation", fc="#fff7ed", ec="#f59e0b")
    add_box(ax, note2[:2], note2[2], note2[3], "Deployment handoff\npackages/model/\nsaved_emotion_model/\nheatmaps + uncertain rejection", fc="#ecfccb", ec="#84cc16")
    add_arrow(ax, center_right(*note1), center_left(*processed), style="->", mutation=11, lw=1.3, color="#f59e0b")
    add_arrow(ax, center_top(*note2), center_bottom(*export), style="->", mutation=11, lw=1.3, color="#84cc16")

    ax.text(
        0.8,
        0.65,
        "Based on notebooks/docs/03-kaggle-setup.md and notebooks/docs/IEEE_METHODOLOGY.md",
        fontsize=10,
        color="#64748b",
    )

    plt.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
