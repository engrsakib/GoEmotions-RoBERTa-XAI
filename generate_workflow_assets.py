from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
SVG_PATH = ROOT / "workflow_linear_diagram.svg"
PNG_PATH = ROOT / "workflow_diagram.png"
PPTX_PATH = ROOT / "workflow_linear_diagram.pptx"
MERMAID_PATH = ROOT / "workflow_linear_diagram.mmd"

SLIDE_W = 16.0
SLIDE_H = 9.0

COLORS = {
    "navy": "#0F2D52",
    "blue": "#DCEBFA",
    "green": "#DFF3E5",
    "gold": "#FFF3CD",
    "peach": "#FCE6D8",
    "purple": "#EEE3FF",
    "gray": "#F6F8FB",
    "white": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#5B6472",
    "lane": "#FBFCFE",
    "lane_border": "#D7DEE8",
}


@dataclass(frozen=True)
class LaneSpec:
    key: str
    label: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class BoxSpec:
    key: str
    title: str
    body: str
    x: float
    y: float
    w: float
    h: float
    fill: str
    title_size: float = 12.0
    body_size: float = 10.0
    align_left: bool = False


@dataclass(frozen=True)
class ArrowSpec:
    start: str
    end: str
    start_anchor: str = "right"
    end_anchor: str = "left"
    style: str = "solid"
    rad: float = 0.0


LANES = [
    LaneSpec("top", "Input and Preparation", 0.03, 0.67, 0.94, 0.21),
    LaneSpec("middle", "Training and Calibration", 0.03, 0.37, 0.94, 0.23),
    LaneSpec("bottom", "Evaluation, XAI, and Thesis Outputs", 0.03, 0.10, 0.94, 0.20),
]

BOXES = [
    BoxSpec(
        "raw",
        "GoEmotions Inputs",
        "Reddit comments\n28 emotion labels",
        0.05,
        0.71,
        0.17,
        0.12,
        "purple",
    ),
    BoxSpec(
        "prep",
        "Map + Clean",
        "28 -> 7 macro mapping\ndedup + filtering\nstratified or official split",
        0.29,
        0.69,
        0.20,
        0.15,
        "green",
    ),
    BoxSpec(
        "batches",
        "Token Batches",
        "train / val / test\nmax length 128",
        0.55,
        0.71,
        0.16,
        0.12,
        "blue",
    ),
    BoxSpec(
        "setup",
        "Experiment Setup",
        "model profiles\ntrain config\nartifact paths",
        0.77,
        0.69,
        0.16,
        0.15,
        "gray",
    ),
    BoxSpec(
        "core",
        "DeBERTa-v3 Multilabel Training",
        "encoder backbone\n7-label sigmoid head",
        0.24,
        0.41,
        0.28,
        0.14,
        "purple",
        title_size=13.0,
    ),
    BoxSpec(
        "asl",
        "Asymmetric Loss",
        "gamma-/gamma+\nclip",
        0.58,
        0.47,
        0.15,
        0.08,
        "gold",
        title_size=10.8,
        body_size=8.6,
    ),
    BoxSpec(
        "weights",
        "Class Weights",
        "inverse freq.\nminority labels",
        0.76,
        0.47,
        0.17,
        0.08,
        "peach",
        title_size=10.8,
        body_size=8.6,
    ),
    BoxSpec(
        "thresholds",
        "Threshold Tuning",
        "validation-only\nper-class search",
        0.58,
        0.35,
        0.34,
        0.12,
        "green",
        title_size=12.5,
    ),
    BoxSpec(
        "benchmark",
        "Benchmark Comparison",
        "M1 BERT-base\nM2 RoBERTa-base\nM3/M6 DeBERTa",
        0.07,
        0.15,
        0.17,
        0.10,
        "blue",
        body_size=9.0,
    ),
    BoxSpec(
        "metrics",
        "Metrics",
        "Macro-F1 80.74%\nMicro-F1 85.10%\nHamming 0.0471",
        0.29,
        0.15,
        0.18,
        0.10,
        "gold",
        body_size=9.0,
    ),
    BoxSpec(
        "xai",
        "Integrated Gradients",
        "token-level attribution\nheatmaps",
        0.51,
        0.15,
        0.19,
        0.10,
        "green",
        body_size=9.0,
    ),
    BoxSpec(
        "outputs",
        "Thesis Outputs",
        "figures\npaper draft\nexport + slides",
        0.75,
        0.15,
        0.19,
        0.10,
        "purple",
        body_size=9.0,
    ),
]

ARROWS = [
    ArrowSpec("raw", "prep"),
    ArrowSpec("prep", "batches"),
    ArrowSpec("setup", "batches", start_anchor="left", end_anchor="right"),
    ArrowSpec("batches", "core", start_anchor="bottom", end_anchor="top"),
    ArrowSpec("core", "asl"),
    ArrowSpec("core", "weights"),
    ArrowSpec("core", "thresholds", start_anchor="right", end_anchor="left"),
    ArrowSpec("thresholds", "metrics", start_anchor="bottom_left", end_anchor="top_right"),
    ArrowSpec("thresholds", "xai", start_anchor="bottom", end_anchor="top"),
    ArrowSpec("metrics", "outputs", start_anchor="right_low", end_anchor="left_low"),
    ArrowSpec("xai", "outputs", start_anchor="right_low", end_anchor="left_low"),
]


def rgb(hex_color: str) -> RGBColor:
    value = COLORS.get(hex_color, hex_color).lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def color_value(name: str) -> str:
    return COLORS.get(name, name)


def get_anchor(box: BoxSpec, anchor: str) -> tuple[float, float]:
    if anchor == "left":
        return box.x, box.y + box.h / 2
    if anchor == "right":
        return box.x + box.w, box.y + box.h / 2
    if anchor == "top":
        return box.x + box.w / 2, box.y + box.h
    if anchor == "bottom":
        return box.x + box.w / 2, box.y
    if anchor == "top_right":
        return box.x + box.w * 0.72, box.y + box.h
    if anchor == "bottom_left":
        return box.x + box.w * 0.28, box.y
    if anchor == "left_low":
        return box.x, box.y + box.h * 0.30
    if anchor == "right_low":
        return box.x + box.w, box.y + box.h * 0.30
    raise ValueError(f"Unknown anchor: {anchor}")


def box_map() -> dict[str, BoxSpec]:
    return {box.key: box for box in BOXES}


def draw_box(ax, box: BoxSpec) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=COLORS["navy"],
        facecolor=COLORS[box.fill],
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        box.x + box.w / 2,
        box.y + box.h * 0.67,
        box.title,
        ha="center",
        va="center",
        fontsize=box.title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=3,
    )
    ax.text(
        box.x + box.w / 2,
        box.y + box.h * 0.33,
        box.body,
        ha="center",
        va="center",
        fontsize=box.body_size,
        color=COLORS["muted"],
        zorder=3,
    )


def draw_arrow(ax, arrow: ArrowSpec, boxes: dict[str, BoxSpec]) -> None:
    start = get_anchor(boxes[arrow.start], arrow.start_anchor)
    end = get_anchor(boxes[arrow.end], arrow.end_anchor)
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.6,
        color=COLORS["navy"],
        linestyle="--" if arrow.style == "dashed" else "-",
        connectionstyle=f"arc3,rad={arrow.rad}",
        zorder=1,
    )
    ax.add_patch(patch)


def build_matplotlib_figure():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "GoEmotions-RoBERTa-XAI Thesis Pipeline",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.935,
        "Reference-inspired layout grounded in the actual data, training, calibration, XAI, and export flow",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    for lane in LANES:
        rect = FancyBboxPatch(
            (lane.x, lane.y),
            lane.w,
            lane.h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0.8,
            edgecolor=COLORS["lane_border"],
            facecolor=COLORS["lane"],
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            lane.x + 0.02,
            lane.y + lane.h - 0.035,
            lane.label,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=COLORS["navy"],
            bbox={"facecolor": COLORS["white"], "edgecolor": "none", "pad": 1.2},
        )

    boxes = box_map()
    for arrow in ARROWS:
        draw_arrow(ax, arrow, boxes)
    for box in BOXES:
        draw_box(ax, box)

    ax.text(
        0.03,
        0.035,
        "Primary workflow source: run_pipeline stages + IEEE methodology + trainer/XAI modules.",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#6B7280",
    )
    return fig


def add_pptx_textbox(slide, left, top, width, height, text, *, font_size, bold=False, color="text", align=PP_ALIGN.CENTER):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def build_pptx() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb("white")

    add_pptx_textbox(slide, Inches(0.5), Inches(0.18), Inches(15.0), Inches(0.42), "GoEmotions-RoBERTa-XAI Thesis Pipeline", font_size=23, bold=True)
    add_pptx_textbox(
        slide,
        Inches(0.75),
        Inches(0.56),
        Inches(14.5),
        Inches(0.28),
        "Reference-inspired layout grounded in the actual data, training, calibration, XAI, and export flow",
        font_size=10.5,
        color="muted",
    )

    for lane in LANES:
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(lane.x * SLIDE_W),
            Inches((1 - lane.y - lane.h) * SLIDE_H),
            Inches(lane.w * SLIDE_W),
            Inches(lane.h * SLIDE_H),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb("lane")
        shape.line.color.rgb = rgb("lane_border")
        shape.line.width = Pt(0.8)
        add_pptx_textbox(
            slide,
            Inches((lane.x + 0.02) * SLIDE_W),
            Inches((1 - lane.y - lane.h + 0.01) * SLIDE_H),
            Inches(3.8),
            Inches(0.26),
            lane.label,
            font_size=11,
            bold=True,
            color="navy",
            align=PP_ALIGN.LEFT,
        )

    for box in BOXES:
        left = Inches(box.x * SLIDE_W)
        top = Inches((1 - box.y - box.h) * SLIDE_H)
        width = Inches(box.w * SLIDE_W)
        height = Inches(box.h * SLIDE_H)
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(box.fill)
        shape.line.color.rgb = rgb("navy")
        shape.line.width = Pt(1.4)
        add_pptx_textbox(slide, left + Inches(0.05), top + Inches(0.10), width - Inches(0.10), Inches(0.28), box.title, font_size=box.title_size, bold=True)
        add_pptx_textbox(slide, left + Inches(0.05), top + height * 0.45, width - Inches(0.10), Inches(0.34), box.body, font_size=box.body_size, color="muted")

    boxes = box_map()
    for arrow in ARROWS:
        sx, sy = get_anchor(boxes[arrow.start], arrow.start_anchor)
        ex, ey = get_anchor(boxes[arrow.end], arrow.end_anchor)
        connector = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Inches(sx * SLIDE_W),
            Inches((1 - sy) * SLIDE_H),
            Inches(ex * SLIDE_W),
            Inches((1 - ey) * SLIDE_H),
        )
        connector.line.color.rgb = rgb("navy")
        connector.line.width = Pt(1.8)
        if arrow.style == "dashed":
            try:
                connector.line.dash_style = 2
            except Exception:
                pass
        try:
            connector.line.end_arrowhead = True
        except Exception:
            pass

    add_pptx_textbox(
        slide,
        Inches(0.45),
        Inches(8.18),
        Inches(15.0),
        Inches(0.20),
        "Canonical generator output for SVG, PNG, and PPTX.",
        font_size=9,
        color="muted",
        align=PP_ALIGN.LEFT,
    )
    prs.save(PPTX_PATH)


def build_mermaid() -> str:
    mermaid = dedent(
        """
        flowchart LR
            rawInput["GoEmotions Inputs"] --> prep["Map + Clean"]
            prep --> tokenBatches["Token Batches"]
            setup["Experiment Setup"] --> tokenBatches
            tokenBatches --> coreTrain["DeBERTa-v3 Multilabel Training"]
            coreTrain --> asl["Asymmetric Loss"]
            coreTrain --> classWeights["Class Weights"]
            coreTrain --> thresholds["Threshold Tuning"]
            thresholds --> metrics["Metrics"]
            thresholds --> xai["Integrated Gradients"]
            setup --> benchmark["Benchmark Comparison"]
            benchmark --> outputs["Thesis Outputs"]
            metrics --> outputs
            xai --> outputs
        """
    ).strip()
    MERMAID_PATH.write_text(mermaid + "\n", encoding="utf-8")
    return mermaid


def main() -> None:
    fig = build_matplotlib_figure()
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight", facecolor=COLORS["white"])
    fig.savefig(PNG_PATH, format="png", dpi=300, bbox_inches="tight", facecolor=COLORS["white"])
    plt.close(fig)
    build_pptx()
    mermaid = build_mermaid()
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PPTX_PATH}")
    print(f"Wrote {MERMAID_PATH}")
    print("\nMermaid source:\n")
    print(mermaid)


if __name__ == "__main__":
    main()
