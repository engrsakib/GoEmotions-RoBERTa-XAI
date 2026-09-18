from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
SVG_PATH = ROOT / "workflow_linear_diagram.svg"
PNG_PATH = ROOT / "workflow_diagram.png"
MERMAID_PATH = ROOT / "workflow_linear_diagram.mmd"

COLORS = {
    "navy": "#0F2D52",
    "blue": "#DCEBFA",
    "green": "#DFF3E5",
    "gold": "#FFF3CD",
    "peach": "#FCE6D8",
    "purple": "#EEE3FF",
    "gray": "#F6F8FB",
    "text": "#1F2937",
    "muted": "#5B6472",
}


def draw_box(ax, x, y, w, h, title, lines, fill, edge=None, fontsize=10, title_size=11.5):
    edge = edge or COLORS["navy"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        lines,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["muted"],
        zorder=3,
    )


def draw_arrow(ax, start, end, color=None, lw=1.6, style="-|>", linestyle="-", rad=0.0):
    color = color or COLORS["navy"]
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)


def add_lane(ax, y, h, label):
    lane = FancyBboxPatch(
        (0.03, y),
        0.94,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=0.8,
        edgecolor="#D7DEE8",
        facecolor="#FBFCFE",
        zorder=0,
    )
    ax.add_patch(lane)
    ax.text(
        0.05,
        y + h - 0.035,
        label,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["navy"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )


def build_figure():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.965,
        "GoEmotions-RoBERTa-XAI Thesis Workflow",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.937,
        "Original research-pipeline view of data preparation, transformer training, calibration, XAI, and thesis deliverables",
        ha="center",
        va="center",
        fontsize=9.6,
        color=COLORS["muted"],
    )

    add_lane(ax, 0.67, 0.21, "Lane 1: Data Sources and Experiment Setup")
    add_lane(ax, 0.37, 0.23, "Lane 2: Core Modeling and Optimization")
    add_lane(ax, 0.10, 0.20, "Lane 3: Evaluation, XAI, and Thesis Outputs")

    draw_box(ax, 0.05, 0.72, 0.15, 0.11, "GoEmotions Inputs", "Reddit comments\nemotion labels", COLORS["purple"])
    draw_box(ax, 0.23, 0.72, 0.15, 0.11, "Project Setup", "configs, profiles,\nartifact paths", COLORS["blue"])
    draw_box(ax, 0.41, 0.70, 0.17, 0.14, "Data Preparation", "cleaning\nmacro-label mapping\ntrain/val/test split", COLORS["green"])
    draw_box(ax, 0.61, 0.70, 0.16, 0.14, "Tokenizer + Loader", "batch creation\nattention masks\nmulti-hot labels", COLORS["blue"])
    draw_box(ax, 0.80, 0.70, 0.15, 0.14, "Experiment Registry", "benchmark runs\nm1, m2, m3, m6\ntracked exports", COLORS["gray"])

    draw_box(ax, 0.13, 0.40, 0.20, 0.11, "DeBERTa-v3 Encoder", "contextual text features\nsequence representation", COLORS["purple"])
    draw_box(ax, 0.38, 0.40, 0.19, 0.11, "Multi-Label Head", "7 macro emotion logits\nsigmoid probabilities", COLORS["blue"])
    draw_box(ax, 0.62, 0.45, 0.13, 0.08, "ASL", "gamma / clip", COLORS["gold"], title_size=11)
    draw_box(ax, 0.79, 0.45, 0.13, 0.08, "Class Weights", "minority boost", COLORS["peach"], title_size=10.5)
    draw_box(ax, 0.62, 0.36, 0.30, 0.12, "Threshold Optimization", "validation-only per-class tuning\nbest decision thresholds", COLORS["green"])

    draw_box(ax, 0.06, 0.15, 0.18, 0.10, "Benchmark Comparison", "BERT / RoBERTa /\nDeBERTa baselines", COLORS["blue"])
    draw_box(ax, 0.29, 0.15, 0.18, 0.10, "Metrics and Analysis", "Macro-F1 80.74%\nMicro-F1 85.10%\ngap + hamming loss", COLORS["gold"])
    draw_box(ax, 0.52, 0.15, 0.18, 0.10, "XAI Heatmaps", "Integrated Gradients\ntoken-level attribution", COLORS["green"])
    draw_box(ax, 0.75, 0.15, 0.19, 0.10, "Thesis Outputs", "figures\npaper draft\npresentation Q&A", COLORS["purple"])

    draw_arrow(ax, (0.20, 0.775), (0.23, 0.775))
    draw_arrow(ax, (0.38, 0.775), (0.41, 0.775))
    draw_arrow(ax, (0.58, 0.775), (0.61, 0.775))
    draw_arrow(ax, (0.77, 0.775), (0.80, 0.775))

    draw_arrow(ax, (0.69, 0.70), (0.23, 0.51))
    draw_arrow(ax, (0.33, 0.455), (0.38, 0.455))
    draw_arrow(ax, (0.57, 0.455), (0.62, 0.49))
    draw_arrow(ax, (0.57, 0.455), (0.79, 0.49))
    draw_arrow(ax, (0.57, 0.455), (0.62, 0.42))

    draw_arrow(ax, (0.77, 0.42), (0.61, 0.25))
    draw_arrow(ax, (0.24, 0.17), (0.29, 0.17))
    draw_arrow(ax, (0.47, 0.17), (0.52, 0.17))
    draw_arrow(ax, (0.70, 0.17), (0.75, 0.17))

    ax.text(
        0.03,
        0.035,
        "Design note: this figure uses an original layout tailored to the GoEmotions-RoBERTa-XAI thesis pipeline.",
        ha="left",
        va="center",
        fontsize=8.2,
        color="#6B7280",
    )

    return fig


def build_mermaid() -> str:
    mermaid = dedent(
        """
        flowchart LR
            A["GoEmotions Inputs"] --> B["Data Preparation"]
            C["Project Setup"] --> B
            B --> D["Tokenizer + Loader"]
            D --> E["DeBERTa-v3 Encoder"]
            E --> F["Multi-Label Head"]
            F --> G["ASL + Class Weights"]
            G --> H["Threshold Optimization"]
            H --> I["Benchmark Comparison"]
            H --> J["Metrics and Analysis"]
            H --> K["XAI Heatmaps"]
            J --> L["Thesis Outputs"]
            K --> L
            I --> L

            classDef data fill:#DCEBFA,stroke:#0F2D52,stroke-width:2px,color:#1F2937;
            classDef model fill:#EEE3FF,stroke:#0F2D52,stroke-width:2px,color:#1F2937;
            classDef result fill:#DFF3E5,stroke:#0F2D52,stroke-width:2px,color:#1F2937;

            class A,B,C,D data;
            class E,F,G,H model;
            class I,J,K,L result;
        """
    ).strip()
    MERMAID_PATH.write_text(mermaid + "\n", encoding="utf-8")
    return mermaid


def main():
    fig = build_figure()
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(PNG_PATH, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    mermaid = build_mermaid()
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {MERMAID_PATH}")
    print("\nMermaid source:\n")
    print(mermaid)


if __name__ == "__main__":
    main()
