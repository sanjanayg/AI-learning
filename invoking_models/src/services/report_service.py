"""
ReportService
=============
Generates a professional, multi-page PDF Chat Summary Report using
reportlab.platypus (SimpleDocTemplate + Paragraph + Spacer).

Compared to the raw canvas API, platypus handles:
- Automatic text wrapping  — no manual line splitting needed
- Automatic page breaks    — content flows naturally across pages
- Style inheritance        — consistent typography via ParagraphStyle

PDF Sections (in order):
  1. Title banner
  2. Chat name + timestamp
  3. Executive Summary
  4. Topics Discussed
  5. Key User Questions
  6. Key Assistant Responses
  7. Decisions Made
  8. Errors / Issues Discussed
  9. Action Items / Next Steps
"""

import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# ── Brand colours (matching the frontend palette) ─────────────────────────────
PRIMARY_BLUE   = colors.HexColor("#0072FF")
ACCENT_CYAN    = colors.HexColor("#00C6FF")
DARK_BG        = colors.HexColor("#11151C")
LIGHT_BG       = colors.HexColor("#F0F6FF")
BORDER_COLOR   = colors.HexColor("#C5D8F5")
TEXT_DARK      = colors.HexColor("#1A2A3A")
TEXT_MUTED     = colors.HexColor("#5A7A99")

# ── Page margins ──────────────────────────────────────────────────────────────
LEFT_MARGIN  = 20 * mm
RIGHT_MARGIN = 20 * mm
TOP_MARGIN   = 18 * mm
BOTTOM_MARGIN = 18 * mm


class ReportService:
    """
    Converts a structured JSON summary produced by ChatSummaryService into a
    polished, multi-page A4 PDF report.

    Usage:
        pdf_bytes = ReportService.generate_pdf(
            chat_name="Support Session 12",
            summary_json='{ "executive_summary": "...", ... }',
        )
        # pdf_bytes is a BytesIO — pass to StreamingResponse directly.
    """

    # ── Styles ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_styles() -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()

        styles: dict[str, ParagraphStyle] = {}

        styles["title"] = ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=4,
            leading=28,
        )

        styles["subtitle"] = ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=0,
        )

        styles["meta"] = ParagraphStyle(
            "meta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_MUTED,
            alignment=TA_LEFT,
            spaceAfter=2,
        )

        styles["section_header"] = ParagraphStyle(
            "section_header",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=PRIMARY_BLUE,
            spaceBefore=14,
            spaceAfter=6,
            leading=16,
        )

        styles["body"] = ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_DARK,
            spaceAfter=6,
            leading=15,
        )

        styles["bullet"] = ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_DARK,
            spaceAfter=4,
            leading=14,
            leftIndent=14,
            bulletIndent=0,
        )

        styles["empty"] = ParagraphStyle(
            "empty",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            textColor=TEXT_MUTED,
            spaceAfter=4,
            leftIndent=14,
        )

        return styles

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def generate_pdf(cls, chat_name: str, summary_json: str) -> BytesIO:
        """
        Build the full PDF report and return it as a BytesIO buffer positioned
        at offset 0 (ready for StreamingResponse).

        Parameters
        ----------
        chat_name    : str  — display name of the chat session
        summary_json : str  — JSON string produced by ChatSummaryService

        Returns
        -------
        BytesIO : in-memory PDF bytes, seeked to position 0
        """
        try:
            data = json.loads(summary_json)
        except (json.JSONDecodeError, ValueError):
            logger.warning("summary_json was not valid JSON; treating as raw text.")
            data = {
                "executive_summary": summary_json,
                "topics_discussed": [],
                "key_user_questions": [],
                "key_assistant_responses": [],
                "decisions_made": [],
                "errors_issues": [],
                "action_items": [],
            }

        buffer = BytesIO()
        styles = cls._build_styles()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=LEFT_MARGIN,
            rightMargin=RIGHT_MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title=f"Chat Summary Report — {chat_name}",
            author="RAG Portal",
            subject="Automated Chat Summary",
        )

        story = []

        # ── Title banner ───────────────────────────────────────────────────────
        story.extend(cls._build_title_banner(chat_name, styles))

        # ── Metadata row ──────────────────────────────────────────────────────
        now_str = datetime.now(timezone.utc).strftime("%B %d, %Y  %H:%M UTC")
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Chat:</b>  {cls._escape(chat_name)}", styles["meta"]))
        story.append(Paragraph(f"<b>Generated:</b>  {now_str}", styles["meta"]))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR))
        story.append(Spacer(1, 6))

        # ── Sections ──────────────────────────────────────────────────────────
        sections = [
            ("Executive Summary",        data.get("executive_summary", ""), "text"),
            ("Topics Discussed",          data.get("topics_discussed", []),  "list"),
            ("Key User Questions",        data.get("key_user_questions", []), "list"),
            ("Key Assistant Responses",   data.get("key_assistant_responses", []), "list"),
            ("Decisions Made",            data.get("decisions_made", []),    "list"),
            ("Errors / Issues Discussed", data.get("errors_issues", []),     "list"),
            ("Action Items / Next Steps", data.get("action_items", []),      "list"),
        ]

        for section_title, content, content_type in sections:
            story.extend(
                cls._build_section(section_title, content, content_type, styles)
            )

        # ── Footer note ───────────────────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR))
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "This report was automatically generated by the RAG Portal Chat Summary engine.",
                styles["empty"],
            )
        )

        doc.build(story, onFirstPage=cls._add_page_number, onLaterPages=cls._add_page_number)

        buffer.seek(0)
        logger.info("PDF report generated for chat '%s'.", chat_name)
        return buffer

    # ── Section builders ───────────────────────────────────────────────────────

    @classmethod
    def _build_title_banner(cls, chat_name: str, styles: dict) -> list:
        """Render a dark gradient-style banner table as the report header."""
        page_width = A4[0] - LEFT_MARGIN - RIGHT_MARGIN

        title_para   = Paragraph("Chat Summary Report", styles["title"])
        subtitle_para = Paragraph(
            cls._escape(chat_name), styles["subtitle"]
        )

        banner_table = Table(
            [[title_para], [subtitle_para]],
            colWidths=[page_width],
        )
        banner_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
                ("TOPPADDING",    (0, 0), (-1, -1), 16),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
                ("LEFTPADDING",   (0, 0), (-1, -1), 20),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
                ("LINEBELOW", (0, 0), (-1, 0), 2, ACCENT_CYAN),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ])
        )
        return [banner_table]

    @classmethod
    def _build_section(
        cls,
        title: str,
        content: Any,
        content_type: str,
        styles: dict,
    ) -> list:
        """Build a titled section with either paragraph text or a bullet list."""
        elements = []
        elements.append(Paragraph(cls._escape(title), styles["section_header"]))

        if content_type == "text":
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                elements.append(Paragraph(cls._escape(text), styles["body"]))
            else:
                elements.append(Paragraph("None identified.", styles["empty"]))

        elif content_type == "list":
            items = content if isinstance(content, list) else [str(content)]
            non_empty = [i for i in items if str(i).strip()]
            if non_empty:
                for item in non_empty:
                    elements.append(
                        Paragraph(
                            f"\u2022&nbsp;&nbsp;{cls._escape(str(item))}",
                            styles["bullet"],
                        )
                    )
            else:
                elements.append(Paragraph("None identified.", styles["empty"]))

        return elements

    # ── Page number callback ──────────────────────────────────────────────────

    @staticmethod
    def _add_page_number(canvas, doc) -> None:
        """Draw a centred page number at the bottom of every page."""
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(TEXT_MUTED)
        page_num_text = f"Page {doc.page}"
        canvas.drawCentredString(A4[0] / 2.0, 10 * mm, page_num_text)
        canvas.restoreState()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _escape(text: str) -> str:
        """
        Escape HTML special characters for reportlab Paragraph rendering.
        reportlab XML parser will choke on raw < > & characters.
        """
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
