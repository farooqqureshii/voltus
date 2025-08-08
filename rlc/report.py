from __future__ import annotations

from io import BytesIO
from typing import Dict, Optional

import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors


def _figure_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 600) -> bytes:
    return fig.to_image(format="png", width=width, height=height, scale=2, engine="kaleido")


def build_pdf_report(
    title: str,
    author: str,
    params: Dict[str, float],
    resonance_metrics: Dict[str, float],
    figures: Dict[str, Optional[go.Figure]],
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Paragraph(f"{author}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Parameters table
    data = [["Parameter", "Value"]]
    for k, v in params.items():
        data.append([k, f"{v:.6g}"])
    table = Table(data, hAlign="LEFT", colWidths=[1.5 * inch, 3.5 * inch])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#0b1220")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#e5e7eb")),
        ])
    )
    story.append(Paragraph("<b>Parameters</b>", styles["Heading2"]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Resonance metrics
    data2 = [["Metric", "Value"]]
    for k, v in resonance_metrics.items():
        data2.append([k, f"{v:.6g}"])
    table2 = Table(data2, hAlign="LEFT", colWidths=[1.5 * inch, 3.5 * inch])
    table2.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#0b1220")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#e5e7eb")),
        ])
    )
    story.append(Paragraph("<b>Resonance Metrics</b>", styles["Heading2"]))
    story.append(table2)
    story.append(Spacer(1, 0.25 * inch))

    # Figures
    for name, fig in figures.items():
        if fig is None:
            continue
        try:
            img_bytes = _figure_to_png_bytes(fig)
            img = Image(BytesIO(img_bytes))
            img._restrictSize(6.5 * inch, 4.8 * inch)
            story.append(Paragraph(f"<b>{name.title()}</b>", styles["Heading2"]))
            story.append(img)
            story.append(Spacer(1, 0.15 * inch))
        except Exception:
            # Skip figure on failure
            continue

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


