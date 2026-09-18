from html import escape
from pathlib import Path
from textwrap import dedent

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
PPTX_PATH = ROOT / "workflow_linear_diagram.pptx"
SVG_PATH = ROOT / "workflow_linear_diagram.svg"
MERMAID_PATH = ROOT / "workflow_linear_diagram.mmd"

SOFT_BLUE = "E6F2FF"
DARK_NAVY = "004080"
TEXT_DARK = "1F2937"
BG = "FFFFFF"
PANEL_PINK = "F9F2FB"
PANEL_GREEN = "EFFAF1"
PANEL_LAVENDER = "F3ECFF"
PANEL_GOLD = "FFF8D8"
WARM_ACCENT = "FFB84D"
LIGHT_ORANGE = "FF9F2F"
LIGHT_BLUE = "D7E7FF"
LIGHT_GREEN = "C8F1C5"
LIGHT_PURPLE = "DCC7FF"
GRAY_BOX = "F8FAFC"

COMMENT_CARDS = [
    "I finally\nfeel calm",
    "This update\nmade me smile",
    "I am nervous\nabout tomorrow",
    "Proud of\nthis result",
    "Still a little\nsad today",
    "Grateful for\nthe support",
    "That joke was\nactually funny",
    "I love this\ncommunity",
    "Frustrated, but\nstill hopeful",
]

SAMPLE_TEXT = 'reddit comment sample ...\n"I feel worried but hopeful\nabout the final exam"'


def rgb(hex_color: str) -> RGBColor:
    return RGBColor(int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def style_shape(shape, fill_hex: str, line_hex: str = DARK_NAVY, line_width: float = 1.5) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill_hex)
    shape.line.color.rgb = rgb(line_hex)
    shape.line.width = Pt(line_width)


def add_textbox(slide, left, top, width, height, text, *, font_size, bold=False, color=TEXT_DARK, align=PP_ALIGN.CENTER):
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


def add_round_box(slide, left, top, width, height, fill_hex, *, line_hex=DARK_NAVY, line_width=1.5):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    style_shape(shape, fill_hex, line_hex=line_hex, line_width=line_width)
    return shape


def add_arrow(slide, x1, y1, x2, y2, *, width=2.0, color=DARK_NAVY):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(width)
    try:
        connector.line.end_arrowhead = True
    except Exception:
        pass
    return connector


def add_comment_grid(slide, left, top):
    add_round_box(slide, left, top, Inches(2.35), Inches(2.85), PANEL_LAVENDER)
    card_w = Inches(0.63)
    card_h = Inches(0.63)
    gap = Inches(0.06)
    idx = 0
    for row in range(3):
        for col in range(3):
            cx = left + Inches(0.12) + col * (card_w + gap)
            cy = top + Inches(0.14) + row * (card_h + gap)
            card = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, cx, cy, card_w, card_h)
            style_shape(card, "EAD5FF", line_hex="5B2C83", line_width=1.0)
            add_textbox(
                slide,
                cx + Inches(0.03),
                cy + Inches(0.04),
                card_w - Inches(0.06),
                card_h - Inches(0.08),
                COMMENT_CARDS[idx],
                font_size=7.2,
                bold=True,
                color="4A235A",
            )
            idx += 1
    add_textbox(slide, left + Inches(0.12), top + Inches(2.28), Inches(2.12), Inches(0.46), "GoEmotions corpus\n+ 7 macro labels", font_size=13.0, bold=True)


def add_patch_stack(slide, left, top, label):
    for offset in (Inches(0.10), Inches(0.05), Inches(0.0)):
        card = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, left + offset, top + offset, Inches(0.52), Inches(0.52))
        style_shape(card, "C78CF2", line_hex="5B2C83", line_width=1.0)
    add_textbox(slide, left - Inches(0.05), top + Inches(0.56), Inches(0.68), Inches(0.18), label, font_size=8.3, color="4A235A")


def add_core_model(slide, left, top):
    add_round_box(slide, left, top, Inches(4.55), Inches(2.95), PANEL_PINK, line_hex="D1B3D8", line_width=1.2)
    add_round_box(slide, left + Inches(0.18), top + Inches(0.16), Inches(1.55), Inches(0.38), PANEL_GOLD, line_hex="B07D12", line_width=1.1)
    add_round_box(slide, left + Inches(2.00), top + Inches(0.16), Inches(1.55), Inches(0.38), "FFF0E2", line_hex="C96A1B", line_width=1.1)
    add_textbox(slide, left + Inches(0.24), top + Inches(0.18), Inches(1.43), Inches(0.28), "Asymmetric Loss\n(ASL)", font_size=10.7, bold=True, color="6B4E00")
    add_textbox(slide, left + Inches(2.08), top + Inches(0.18), Inches(1.39), Inches(0.28), "Class Weights\n+ Threshold Tune", font_size=10.2, bold=True, color="8A4B08")

    encoder_left = left + Inches(0.52)
    encoder_top = top + Inches(0.82)
    bar_w = Inches(0.17)
    heights = [1.05, 0.88, 0.70, 0.50]
    for i, h in enumerate(heights):
        x = encoder_left + i * Inches(0.30)
        y = encoder_top + (Inches(1.18) - Inches(h))
        bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, x, y, bar_w, Inches(h))
        style_shape(bar, LIGHT_BLUE, line_hex="6D8CCF", line_width=1.0)
    bottleneck = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        left + Inches(1.98),
        top + Inches(1.53),
        Inches(0.38),
        Inches(0.32),
    )
    style_shape(bottleneck, "C9D8F7", line_hex="6D8CCF", line_width=1.0)
    for i, h in enumerate(reversed(heights)):
        x = left + Inches(2.62) + i * Inches(0.30)
        y = encoder_top + (Inches(1.18) - Inches(h))
        bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, x, y, bar_w, Inches(h))
        style_shape(bar, LIGHT_BLUE, line_hex="6D8CCF", line_width=1.0)

    add_textbox(slide, left + Inches(0.88), top + Inches(2.20), Inches(2.80), Inches(0.40), "DeBERTa-v3 multilabel training", font_size=13.8, bold=True)
    add_textbox(slide, left + Inches(1.06), top + Inches(2.48), Inches(2.44), Inches(0.24), "backbone encoder + 7-label sigmoid head", font_size=10.1, color=DARK_NAVY)


def add_logits_block(slide, left, top):
    size = Inches(0.24)
    gap = Inches(0.10)
    coords = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
    for col, row in coords:
        sq = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            left + col * (size + gap),
            top + row * (size + gap),
            size,
            size,
        )
        style_shape(sq, LIGHT_ORANGE, line_hex="C96A1B", line_width=1.0)


def add_probability_vector(slide, left, top):
    add_round_box(slide, left, top, Inches(2.02), Inches(0.82), GRAY_BOX, line_hex="8A8F98", line_width=1.2)
    for i in range(5):
        circ = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, left + Inches(0.16) + i * Inches(0.28), top + Inches(0.24), Inches(0.15), Inches(0.15))
        style_shape(circ, WARM_ACCENT, line_hex="C98712", line_width=1.0)
    add_textbox(slide, left + Inches(1.58), top + Inches(0.16), Inches(0.30), Inches(0.22), "...", font_size=18, bold=True)
    add_textbox(slide, left + Inches(0.18), top + Inches(0.50), Inches(1.64), Inches(0.18), "7-class calibrated probabilities", font_size=9.2)


def add_fusion_block(slide, left, top):
    add_round_box(slide, left, top, Inches(3.02), Inches(1.76), PANEL_GREEN, line_hex="4C9B4C", line_width=1.6)
    add_textbox(slide, left + Inches(0.28), top + Inches(0.16), Inches(2.46), Inches(0.42), "Prediction + XAI\ninference block", font_size=14, bold=True)
    add_textbox(slide, left + Inches(0.36), top + Inches(0.58), Inches(2.30), Inches(0.34), "[ Calibrated scores ; Labels ;\nImportant tokens ]", font_size=10.0, bold=True, color="2A5E2A")
    for i in range(5):
        sq = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, left + Inches(0.48) + i * Inches(0.34), top + Inches(1.02), Inches(0.20), Inches(0.18))
        style_shape(sq, LIGHT_GREEN, line_hex="68AA68", line_width=1.0)
    for i in range(4):
        sq = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, left + Inches(0.64) + i * Inches(0.38), top + Inches(1.30), Inches(0.22), Inches(0.18))
        style_shape(sq, LIGHT_PURPLE, line_hex="896BC8", line_width=1.0)
    add_textbox(slide, left + Inches(2.45), top + Inches(1.02), Inches(0.34), Inches(0.22), "...", font_size=18, bold=True)
    add_textbox(slide, left + Inches(2.45), top + Inches(1.30), Inches(0.34), Inches(0.22), "...", font_size=18, bold=True)


def add_stack_panel(slide, left, top):
    add_round_box(slide, left, top, Inches(1.55), Inches(1.92), PANEL_GOLD, line_hex="708B2A", line_width=1.2)
    header = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, left, top, Inches(1.55), Inches(0.26))
    style_shape(header, "7D9430", line_hex="708B2A", line_width=1.0)
    add_textbox(slide, left + Inches(0.08), top + Inches(0.02), Inches(1.38), Inches(0.20), "Metrics + XAI stack", font_size=9.0, bold=True, color="FFFFFF")
    y = top + Inches(0.36)
    for label in ("Threshold check", "IG heatmap", "Metric summary"):
        add_round_box(slide, left + Inches(0.22), y, Inches(1.10), Inches(0.34), "F6FAE8", line_hex="A5B86C", line_width=1.0)
        add_textbox(slide, left + Inches(0.28), y + Inches(0.04), Inches(0.98), Inches(0.22), label, font_size=8.5, bold=True, color="41581F")
        y += Inches(0.44)


def add_prediction_row(slide, left, top):
    add_round_box(slide, left, top, Inches(1.58), Inches(0.56), "F4F8FF", line_hex="B7C4DE", line_width=1.0)
    for i in range(4):
        sq = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, left + Inches(0.12) + i * Inches(0.28), top + Inches(0.18), Inches(0.16), Inches(0.16))
        style_shape(sq, "E1EAFE", line_hex="7D95C9", line_width=0.9)
    add_textbox(slide, left + Inches(1.28), top + Inches(0.10), Inches(0.16), Inches(0.18), "...", font_size=16, bold=True)
    add_textbox(slide, left + Inches(0.12), top - Inches(0.24), Inches(1.34), Inches(0.18), "Per-class decisions", font_size=9.5, bold=True, color=DARK_NAVY)


def add_final_report(slide, left, top):
    add_round_box(slide, left, top, Inches(2.80), Inches(0.78), GRAY_BOX, line_hex="6E83B7", line_width=1.6)
    icon = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left + Inches(0.12), top + Inches(0.15), Inches(0.38), Inches(0.36))
    style_shape(icon, "F4F8FF", line_hex="6E83B7", line_width=1.0)
    add_textbox(slide, left + Inches(0.62), top + Inches(0.12), Inches(2.02), Inches(0.40), "Final thesis outputs\n(labels, scores, figures)", font_size=11.8, bold=True, align=PP_ALIGN.LEFT)


def build_pptx():
    prs = Presentation()
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(BG)

    add_textbox(slide, Inches(0.45), Inches(0.20), Inches(15.0), Inches(0.44), "GoEmotions-RoBERTa-XAI Thesis Pipeline", font_size=23, bold=True)
    add_textbox(slide, Inches(0.90), Inches(0.62), Inches(14.2), Inches(0.30), "Idea-style publication diagram adapted to the actual preprocessing, training, calibration, and XAI workflow", font_size=11.2, color=DARK_NAVY)

    add_comment_grid(slide, Inches(0.18), Inches(1.62))
    add_textbox(slide, Inches(2.80), Inches(1.30), Inches(2.38), Inches(0.24), "Clean, map, split, tokenize", font_size=11.1, bold=True)
    add_textbox(slide, Inches(5.02), Inches(1.30), Inches(1.90), Inches(0.24), "Mini-batch stream", font_size=11.1, bold=True)
    add_patch_stack(slide, Inches(3.10), Inches(1.74), "Train")
    add_patch_stack(slide, Inches(4.30), Inches(1.74), "Val")
    add_textbox(slide, Inches(5.54), Inches(1.86), Inches(0.34), Inches(0.20), "...", font_size=18, bold=True)
    add_patch_stack(slide, Inches(5.96), Inches(1.74), "Test")

    add_core_model(slide, Inches(3.10), Inches(2.36))
    add_logits_block(slide, Inches(4.68), Inches(5.48))
    add_round_box(slide, Inches(3.95), Inches(6.56), Inches(2.86), Inches(0.92), PANEL_GOLD, line_hex="D3B35A", line_width=1.3)
    add_textbox(slide, Inches(4.10), Inches(6.76), Inches(2.56), Inches(0.38), "Threshold tuning layer\n(validation-only per-class search)", font_size=11.8, bold=True, color="6B4E00")
    add_probability_vector(slide, Inches(7.20), Inches(6.58))
    add_fusion_block(slide, Inches(11.42), Inches(5.18))
    add_textbox(slide, Inches(11.25), Inches(7.56), Inches(3.32), Inches(0.22), "Input text tokens for inference", font_size=10.0, color=DARK_NAVY)
    add_round_box(slide, Inches(11.36), Inches(7.15), Inches(3.08), Inches(0.54), "F6F4EC", line_hex="9C8E68", line_width=1.1)
    add_textbox(slide, Inches(11.52), Inches(7.23), Inches(2.76), Inches(0.34), SAMPLE_TEXT, font_size=9.4, bold=True, align=PP_ALIGN.LEFT, color="554C33")
    add_stack_panel(slide, Inches(12.18), Inches(3.12))
    add_prediction_row(slide, Inches(12.16), Inches(2.18))
    add_final_report(slide, Inches(11.28), Inches(0.92))

    add_arrow(slide, Inches(2.58), Inches(2.32), Inches(3.05), Inches(2.02))
    add_arrow(slide, Inches(4.56), Inches(2.28), Inches(4.56), Inches(2.72))
    add_arrow(slide, Inches(5.37), Inches(5.33), Inches(5.37), Inches(5.80))
    add_arrow(slide, Inches(5.37), Inches(6.18), Inches(5.37), Inches(6.52))
    add_arrow(slide, Inches(6.83), Inches(7.02), Inches(7.18), Inches(7.02))
    add_arrow(slide, Inches(9.26), Inches(7.00), Inches(11.00), Inches(7.00))
    add_arrow(slide, Inches(11.42), Inches(6.08), Inches(11.03), Inches(6.08))
    add_arrow(slide, Inches(12.90), Inches(7.15), Inches(12.90), Inches(6.93))
    add_arrow(slide, Inches(12.95), Inches(5.18), Inches(12.95), Inches(4.44))
    add_arrow(slide, Inches(12.96), Inches(3.12), Inches(12.96), Inches(2.74))
    add_arrow(slide, Inches(12.96), Inches(2.18), Inches(12.96), Inches(1.72))

    footer = slide.shapes.add_textbox(Inches(0.48), Inches(8.25), Inches(15.0), Inches(0.22))
    footer_tf = footer.text_frame
    footer_tf.clear()
    footer_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = footer_tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = "Editable vector diagram generated with python-pptx and adapted to the actual thesis pipeline."
    run.font.name = "Arial"
    run.font.size = Pt(9.5)
    run.font.color.rgb = rgb(DARK_NAVY)

    prs.save(PPTX_PATH)


def svg_text(x: int, y: int, lines: list[str], *, size: int, weight: str = "400", fill: str = TEXT_DARK, anchor: str = "middle", line_gap: int | None = None) -> str:
    if line_gap is None:
        line_gap = size + 6
    tspans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_gap
        tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="#{fill}">{"".join(tspans)}</text>'


def build_svg():
    width = 3200
    height = 1800
    parts: list[str] = []
    parts.append(f'<rect width="100%" height="100%" fill="#{BG}"/>')
    parts.append(svg_text(width // 2, 72, ["GoEmotions-RoBERTa-XAI Thesis Pipeline"], size=42, weight="700"))
    parts.append(svg_text(width // 2, 118, ["Idea-style publication diagram adapted to the actual preprocessing, training, calibration, and XAI workflow"], size=20, fill=DARK_NAVY))

    parts.append(f'<rect x="60" y="280" rx="28" ry="28" width="500" height="600" fill="#{PANEL_LAVENDER}" stroke="#{DARK_NAVY}" stroke-width="4"/>')
    card_x0 = 88
    card_y0 = 314
    idx = 0
    for row in range(3):
        for col in range(3):
            x = card_x0 + col * 134
            y = card_y0 + row * 134
            parts.append(f'<rect x="{x}" y="{y}" width="116" height="116" fill="#EAD5FF" stroke="#5B2C83" stroke-width="2"/>')
            parts.append(svg_text(x + 58, y + 44, COMMENT_CARDS[idx].split("\n"), size=17, weight="700", fill="4A235A"))
            idx += 1
    parts.append(svg_text(310, 824, ["GoEmotions corpus", "+ 7 macro labels"], size=28, weight="700"))

    parts.append(svg_text(720, 210, ["Clean, map, split, tokenize"], size=22, weight="700"))
    parts.append(svg_text(1140, 210, ["Mini-batch stream"], size=22, weight="700"))
    for x, y, label in ((700, 300, "Train"), (920, 300, "Val"), (1400, 300, "Test")):
        for offset in (22, 10, 0):
            parts.append(f'<rect x="{x + offset}" y="{y + offset}" width="104" height="104" fill="#C78CF2" stroke="#5B2C83" stroke-width="2"/>')
        parts.append(svg_text(x + 52, y + 140, [label], size=16, fill="4A235A"))
    parts.append(svg_text(1250, 368, ["..."], size=34, weight="700"))

    parts.append(f'<rect x="700" y="470" rx="24" ry="24" width="760" height="520" fill="#{PANEL_PINK}" stroke="#D1B3D8" stroke-width="3"/>')
    parts.append(f'<rect x="730" y="500" rx="18" ry="18" width="255" height="64" fill="#{PANEL_GOLD}" stroke="#B07D12" stroke-width="2"/>')
    parts.append(f'<rect x="1032" y="500" rx="18" ry="18" width="285" height="64" fill="#FFF0E2" stroke="#C96A1B" stroke-width="2"/>')
    parts.append(svg_text(857, 528, ["Asymmetric Loss", "(ASL)"], size=21, weight="700", fill="6B4E00"))
    parts.append(svg_text(1174, 528, ["Class Weights", "+ Threshold Tune"], size=20, weight="700", fill="8A4B08"))
    for x, y, w, h in ((792, 700, 26, 180), (840, 730, 26, 150), (888, 760, 26, 120), (936, 790, 26, 90)):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#{LIGHT_BLUE}" stroke="#6D8CCF" stroke-width="2"/>')
    parts.append(f'<rect x="1032" y="800" rx="10" ry="10" width="60" height="44" fill="#C9D8F7" stroke="#6D8CCF" stroke-width="2"/>')
    for x, y, w, h in ((1140, 790, 26, 90), (1188, 760, 26, 120), (1236, 730, 26, 150), (1284, 700, 26, 180)):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#{LIGHT_BLUE}" stroke="#6D8CCF" stroke-width="2"/>')
    parts.append(svg_text(1080, 930, ["DeBERTa-v3 multilabel training"], size=31, weight="700"))
    parts.append(svg_text(1080, 968, ["backbone encoder + 7-label sigmoid head"], size=18, fill=DARK_NAVY))

    for x, y in ((980, 1042), (1040, 1042), (1100, 1042), (980, 1102), (1040, 1102), (1100, 1102), (1160, 1102)):
        parts.append(f'<rect x="{x}" y="{y}" width="38" height="38" fill="#{LIGHT_ORANGE}" stroke="#C96A1B" stroke-width="2"/>')
    parts.append(f'<rect x="848" y="1180" rx="20" ry="20" width="462" height="128" fill="#{PANEL_GOLD}" stroke="#D3B35A" stroke-width="3"/>')
    parts.append(svg_text(1079, 1230, ["Threshold tuning layer", "(validation-only per-class search)"], size=24, weight="700", fill="6B4E00"))
    parts.append(f'<rect x="1420" y="1192" rx="18" ry="18" width="330" height="118" fill="#{GRAY_BOX}" stroke="#8A8F98" stroke-width="3"/>')
    for i in range(5):
        cx = 1470 + i * 48
        parts.append(f'<circle cx="{cx}" cy="1244" r="14" fill="#{WARM_ACCENT}" stroke="#C98712" stroke-width="2"/>')
    parts.append(svg_text(1690, 1238, ["..."], size=30, weight="700"))
    parts.append(svg_text(1586, 1288, ["7-class calibrated probabilities"], size=18))

    parts.append(f'<rect x="2290" y="960" rx="24" ry="24" width="520" height="300" fill="#{PANEL_GREEN}" stroke="#4C9B4C" stroke-width="4"/>')
    parts.append(svg_text(2550, 1034, ["Prediction + XAI", "inference block"], size=30, weight="700"))
    parts.append(svg_text(2550, 1108, ["[ Calibrated scores ; Labels ;", "Important tokens ]"], size=21, weight="700", fill="2A5E2A"))
    for i in range(5):
        x = 2360 + i * 58
        parts.append(f'<rect x="{x}" y="1155" width="34" height="30" fill="#{LIGHT_GREEN}" stroke="#68AA68" stroke-width="2"/>')
    for i in range(4):
        x = 2392 + i * 64
        parts.append(f'<rect x="{x}" y="1210" width="36" height="30" fill="#{LIGHT_PURPLE}" stroke="#896BC8" stroke-width="2"/>')
    parts.append(svg_text(2748, 1176, ["..."], size=30, weight="700"))
    parts.append(svg_text(2748, 1231, ["..."], size=30, weight="700"))

    parts.append(f'<rect x="2430" y="560" rx="20" ry="20" width="250" height="330" fill="#{PANEL_GOLD}" stroke="#708B2A" stroke-width="3"/>')
    parts.append(f'<rect x="2430" y="560" width="250" height="44" fill="#7D9430" stroke="#708B2A" stroke-width="2"/>')
    parts.append(svg_text(2555, 590, ["Metrics + XAI stack"], size=16, weight="700", fill="FFFFFF"))
    for y, text in ((640, "Threshold check"), (714, "IG heatmap"), (788, "Metric summary")):
        parts.append(f'<rect x="2465" y="{y}" rx="16" ry="16" width="180" height="48" fill="#F6FAE8" stroke="#A5B86C" stroke-width="2"/>')
        parts.append(svg_text(2555, y + 30, [text], size=15, weight="700", fill="41581F"))

    parts.append(f'<rect x="2418" y="418" rx="16" ry="16" width="200" height="68" fill="#F4F8FF" stroke="#B7C4DE" stroke-width="2"/>')
    for i in range(4):
        x = 2440 + i * 38
        parts.append(f'<rect x="{x}" y="448" width="24" height="24" fill="#E1EAFE" stroke="#7D95C9" stroke-width="2"/>')
    parts.append(svg_text(2520, 380, ["Per-class decisions"], size=16, weight="700", fill=DARK_NAVY))
    parts.append(svg_text(2594, 448, ["..."], size=24, weight="700"))

    parts.append(f'<rect x="2190" y="160" rx="18" ry="18" width="620" height="126" fill="#{GRAY_BOX}" stroke="#6E83B7" stroke-width="4"/>')
    parts.append(f'<rect x="2220" y="194" rx="12" ry="12" width="78" height="72" fill="#F4F8FF" stroke="#6E83B7" stroke-width="2"/>')
    parts.append(svg_text(2510, 212, ["Final thesis outputs", "(labels, scores, figures)"], size=26, weight="700"))

    parts.append(svg_text(2510, 1402, ["Input text tokens for inference"], size=18, fill=DARK_NAVY))
    parts.append(f'<rect x="2280" y="1310" rx="18" ry="18" width="540" height="98" fill="#F6F4EC" stroke="#9C8E68" stroke-width="2"/>')
    parts.append(svg_text(2550, 1356, ['reddit comment sample ...', '"I feel worried but hopeful', 'about the final exam"'], size=18, weight="700", fill="554C33"))

    arrows = [
        (560, 540, 690, 420),
        (1012, 402, 1012, 462),
        (1080, 990, 1080, 1030),
        (1080, 1118, 1080, 1170),
        (1310, 1248, 1415, 1248),
        (1750, 1248, 2290, 1248),
        (2290, 1120, 2110, 1120),
        (2555, 960, 2555, 900),
        (2555, 560, 2555, 490),
        (2555, 418, 2555, 290),
        (2550, 1310, 2550, 1265),
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#{DARK_NAVY}" stroke-width="5" marker-end="url(#arrowhead)"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrowhead" markerWidth="14" markerHeight="12" refX="10" refY="6" orient="auto">
      <polygon points="0 0, 12 6, 0 12" fill="#{DARK_NAVY}" />
    </marker>
  </defs>
  {''.join(parts)}
</svg>
'''
    SVG_PATH.write_text(svg, encoding="utf-8")


def build_mermaid() -> str:
    mermaid = dedent(
        """
        flowchart LR
            A["GoEmotions Corpus + 7 Macro Labels"] --> B["Clean, Map, Split, Tokenize"]
            B --> C["DeBERTa-v3 Multilabel Training"]
            C --> D["7 Macro Logits"]
            D --> E["Threshold Tuning Layer"]
            E --> F["7-class Calibrated Probabilities"]
            F --> G["Prediction + XAI Inference Block"]
            H["Input Text Tokens"] --> G
            G --> I["Metrics + XAI Stack"]
            I --> J["Per-class Decisions"]
            J --> K["Final Thesis Outputs"]

            classDef process fill:#E6F2FF,stroke:#004080,stroke-width:2px,color:#1F2937;
            classDef focus fill:#EFFAF1,stroke:#4C9B4C,stroke-width:2px,color:#1F2937;
            classDef method fill:#F9F2FB,stroke:#004080,stroke-width:2px,color:#1F2937;

            class A,B,D,E,F,H,I,J,K process;
            class G focus;
            class C method;
        """
    ).strip()
    MERMAID_PATH.write_text(mermaid + "\n", encoding="utf-8")
    return mermaid


def main():
    build_pptx()
    build_svg()
    mermaid = build_mermaid()
    print(f"Wrote {PPTX_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {MERMAID_PATH}")
    print("\nMermaid source:\n")
    print(mermaid)


if __name__ == "__main__":
    main()
