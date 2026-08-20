"""Executive Meeting PDF Report Generator matching Enterprise Briefing layout."""

import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

from ai_engine.config import (
    SUMMARY_FOLDER,
    TOPICS_FOLDER,
    ACTION_ITEMS_FOLDER,
    DECISIONS_FOLDER,
    OPEN_QUESTIONS_FOLDER,
    KEY_INSIGHTS_FOLDER,
    PDF_REPORT_FOLDER,
    REPORT_FOLDER,
)
from ai_engine.nlp.metadata_extractor import MetadataExtractor


class PDFReportGenerator:

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=self.styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0B192C"),
            alignment=0,
            spaceAfter=4,
        )

        self.category_subtitle_style = ParagraphStyle(
            "CategorySubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            alignment=0,
            spaceAfter=8,
        )

        self.pill_left_style = ParagraphStyle(
            "PillLeft",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1E293B"),
        )

        self.pill_right_style = ParagraphStyle(
            "PillRight",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1E293B"),
            alignment=2,
        )

        self.section_heading_text_style = ParagraphStyle(
            "SectionHeadingText",
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0B192C"),
        )

        self.sub_heading_style = ParagraphStyle(
            "SubHeading",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=8,
            spaceAfter=4,
        )

        self.body_style = ParagraphStyle(
            "ExecutiveBody",
            parent=self.styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )

        self.bullet_style = ParagraphStyle(
            "ExecutiveBullet",
            parent=self.styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor("#1E293B"),
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=4,
        )

        self.action_card_text_style = ParagraphStyle(
            "ActionCardText",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
        )

    def read_file(self, path):
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def clean_lines(self, text):
        cleaned = []
        for line in text.splitlines():
            line = re.sub(r"^=+.*?=+\s*$", "", line)
            line = re.sub(r"^-{3,}\s*$", "", line)
            line = line.strip()
            if not line:
                continue
            cleaned.append(line)
        return cleaned

    def make_section_header(self, title_text):
        t = Table(
            [[
                "",
                Paragraph(f"<b>{html.escape(title_text)}</b>", self.section_heading_text_style)
            ]],
            colWidths=[4, 536]
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#1D4ED8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (1, 0), (1, 0), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return t

    def make_action_card(self, index, item):
        if isinstance(item, dict):
            owner = item.get("owner", "Not specified")
            task = item.get("task", "")
            deadline = item.get("deadline", "Not specified")
            priority = item.get("priority", "Medium")
            context = item.get("context", "")

            p_content = f"<b>{index}. {html.escape(task)}</b><br/>"
            p_content += f"<font size='8.5' color='#475569'>Owner: <b>{html.escape(owner)}</b>"
            if deadline and str(deadline).strip().lower() not in {"not specified", "n/a", "none", "unspecified", "not mentioned", "no deadline", ""}:
                p_content += f" | Deadline: <b>{html.escape(deadline)}</b>"
            p_content += f" | Priority: <b>{html.escape(priority)}</b></font>"
            if context:
                p_content += f"<br/><font size='8.5' color='#64748B'>Context: {html.escape(context)}</font>"
        else:
            clean_text = re.sub(r"^[\s•\-\*\d\.]+", "", str(item)).strip()
            clean_text = re.sub(r"\*\*", "", clean_text).strip()
            p_content = f"<b>{index}. {html.escape(clean_text)}</b>"

        p = Paragraph(p_content, self.action_card_text_style)

        card_table = Table([[p]], colWidths=[540])
        card_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor("#F59E0B")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        return card_table

    def generate(self, output_name="meeting_001"):
        PDF_REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

        json_path = REPORT_FOLDER / f"{output_name}_report.json"
        report_data = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except Exception:
                pass

        summary_raw = report_data.get("executive_summary") or self.read_file(SUMMARY_FOLDER / f"{output_name}_summary.txt")
        topics_data = report_data.get("discussion_topics", [])
        key_discussion_data = report_data.get("key_discussion", [])
        decisions_data = report_data.get("key_decisions", [])
        actions_data = report_data.get("action_items", [])
        open_questions = report_data.get("open_questions") or self.clean_lines(self.read_file(OPEN_QUESTIONS_FOLDER / f"{output_name}_open_questions.txt"))
        key_insights = report_data.get("key_insights") or self.clean_lines(self.read_file(KEY_INSIGHTS_FOLDER / f"{output_name}_key_insights.txt"))

        extractor = MetadataExtractor()
        transcript = extractor.read_transcript(f"{output_name}.txt")
        metadata = extractor.extract(transcript, meeting_id=output_name)

        output = PDF_REPORT_FOLDER / f"{output_name}_report.pdf"
        document = SimpleDocTemplate(
            str(output),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )
        story = []

        # Header Title & Pill Info Bar
        story.append(Paragraph("Enterprise Meeting Analysis Report", self.title_style))
        story.append(Spacer(1, 4))

        duration_val = metadata.get("duration_minutes", "N/A")
        pill_table = Table(
            [[
                Paragraph(f"<b>Meeting Duration:</b> {html.escape(str(duration_val))}", self.pill_left_style),
                Paragraph("<b>Generated by:</b> MeetingMind AI", self.pill_right_style),
            ]],
            colWidths=[270, 270]
        )
        pill_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(pill_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph("Executive Briefing & Synthesis", self.category_subtitle_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=12, spaceBefore=0))

        # 1. Executive Summary Section
        story.append(self.make_section_header("Executive Summary"))
        story.append(Spacer(1, 6))
        summary_clean = re.sub(r"^=+.*?=+\s*$", "", summary_raw, flags=re.MULTILINE).strip()
        for block in summary_clean.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("Key Strategic Outcomes:") or block.startswith("•") or "\n•" in block:
                lines = block.splitlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("•") or line.startswith("-"):
                        clean_line = re.sub(r"^[\s•\-\*]+", "", line).strip()
                        story.append(Paragraph(f"&bull; &nbsp; {html.escape(clean_line)}", self.bullet_style))
                    else:
                        story.append(Paragraph(f"<b>{html.escape(line)}</b>", self.body_style))
            else:
                story.append(Paragraph(html.escape(block), self.body_style))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10, spaceBefore=2))

        # 2. Main Discussion Topics
        if topics_data:
            story.append(self.make_section_header("Main Discussion Topics"))
            story.append(Spacer(1, 6))
            for idx, topic in enumerate(topics_data[:4], start=1):
                if isinstance(topic, dict):
                    title = topic.get("title", "")
                    details = topic.get("details", [])
                    story.append(Paragraph(f"<b>{idx}. {html.escape(title)}</b>", self.sub_heading_style))
                    if details:
                        clean_d = re.sub(r"^[\s•\-\*\d\.]+", "", str(details[0])).strip()
                        clean_d = re.sub(r"^(?:Topic|EXPLANATION|Explanation|Summary|Detail)[:\-\s]+", "", clean_d, flags=re.IGNORECASE).strip()
                        story.append(Paragraph(f"&bull; &nbsp; {html.escape(clean_d)}", self.bullet_style))
                    story.append(Spacer(1, 3))
                else:
                    clean_item = re.sub(r"^[\s•\-\*\d\.]+", "", str(topic)).strip()
                    clean_item = re.sub(r"^(?:Topic|EXPLANATION|Explanation|Summary|Detail)[:\-\s]+", "", clean_item, flags=re.IGNORECASE).strip()
                    if " - " in clean_item:
                        parts = clean_item.split(" - ", 1)
                        story.append(Paragraph(f"<b>{idx}. {html.escape(parts[0].strip())}</b>", self.sub_heading_style))
                        story.append(Paragraph(f"&bull; &nbsp; {html.escape(parts[1].strip())}", self.bullet_style))
                    else:
                        story.append(Paragraph(f"&bull; &nbsp; {html.escape(clean_item)}", self.bullet_style))
                    story.append(Spacer(1, 3))

            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10, spaceBefore=2))

        # Key Discussion Section (Limit to 2-3 discussions max)
        if key_discussion_data:
            story.append(self.make_section_header("Key Discussion"))
            story.append(Spacer(1, 6))
            total_discussions = 0
            for group in key_discussion_data[:3]:
                if total_discussions >= 3:
                    break
                t_name = group.get("topic", "General Discussion")
                story.append(Paragraph(f"<b>Topic: {html.escape(t_name)}</b>", self.sub_heading_style))
                for pt in group.get("points", [])[:1]:
                    if total_discussions >= 3:
                        break
                    ts = pt.get("timestamp", "00:00")
                    pt_type = pt.get("type", "Fact").upper()
                    txt = pt.get("text", "")
                    b_line = f"<font size='8.5' color='#475569'><b>[{html.escape(ts)}] [{html.escape(pt_type)}]</b></font> &nbsp; {html.escape(txt)}"
                    story.append(Paragraph(f"&bull; &nbsp; {b_line}", self.bullet_style))
                    total_discussions += 1
                story.append(Spacer(1, 3))
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10, spaceBefore=2))

        # 4. Action Items Section
        if actions_data:
            story.append(self.make_section_header("Action Items"))
            story.append(Spacer(1, 6))
            for idx, act in enumerate(actions_data[:5], start=1):
                story.append(self.make_action_card(idx, act))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10, spaceBefore=2))




        # Footer
        story.append(Spacer(1, 16))
        story.append(
            Paragraph(
                "<para align='center'><font color='#64748B' size='8'>Confidential - For Internal Use Only | Generated by MeetingMind AI</font></para>",
                self.styles["Normal"],
            )
        )

        document.build(story)
        print("\n========== EXECUTIVE PDF REPORT GENERATED ==========\n")
        print(output)


if __name__ == "__main__":
    generator = PDFReportGenerator()
    generator.generate()
