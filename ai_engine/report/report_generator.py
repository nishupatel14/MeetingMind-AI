"""
Meeting Report Generator
"""

import json
import re

from ai_engine.config import (
    SUMMARY_FOLDER,
    TOPICS_FOLDER,
    ACTION_ITEMS_FOLDER,
    DECISIONS_FOLDER,
    OPEN_QUESTIONS_FOLDER,
    KEY_INSIGHTS_FOLDER,
    KEY_DISCUSSION_FOLDER,
    REPORT_FOLDER,
)

from ai_engine.nlp.metadata_extractor import MetadataExtractor
from ai_engine.utils.topic_parser import parse_topics


class ReportGenerator:

    def __init__(self):
        try:
            from .pdf_report_generator import PDFReportGenerator
            self.pdf = PDFReportGenerator()
        except Exception:
            self.pdf = None

    def read_file(self, path):
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def clean_lines(self, text):
        cleaned = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("==========") or line.startswith("-------------"):
                continue
            line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
            if line:
                cleaned.append(line)
        return cleaned

    def parse_topics_file(self, raw_topics, transcript=""):
        """Parse topics text into list of dicts [{title, details}]."""
        if not raw_topics:
            return []

        # If already JSON, load directly
        if isinstance(raw_topics, str) and raw_topics.strip().startswith(("[", "{")):
            try:
                parsed = json.loads(raw_topics)
                if isinstance(parsed, list) and parsed:
                    res = []
                    for item in parsed:
                        if isinstance(item, dict):
                            res.append(item)
                        elif isinstance(item, str) and " - " in item:
                            parts = item.split(" - ", 1)
                            res.append({"title": parts[0].strip(), "details": [parts[1].strip()]})
                        elif isinstance(item, str):
                            res.append({"title": item.strip(), "details": []})
                    if res:
                        return res
            except Exception:
                pass

        topics = []
        current_title = None
        current_details = []

        for line in raw_topics.splitlines():
            line = line.strip()
            if not line or line.startswith("=========="):
                if current_title and current_details:
                    topics.append({"title": current_title, "details": current_details})
                    current_title = None
                    current_details = []
                continue

            if line.startswith("TOPIC:"):
                if current_title and current_details:
                    topics.append({"title": current_title, "details": current_details})
                current_title = line[6:].strip()
                current_details = []
                continue

            detail_match = re.match(r'^\s*[\-•\*]\s*(.*)', line)
            if detail_match and current_title:
                current_details.append(detail_match.group(1).strip())
            elif not current_title and " - " in line:
                parts = line.lstrip("•-* ").split(" - ", 1)
                topics.append({"title": parts[0].strip(), "details": [parts[1].strip()]})
            elif current_title and len(line.split()) >= 4:
                current_details.append(line)

        if current_title and current_details:
            topics.append({"title": current_title, "details": current_details})

        if not topics:
            parsed = parse_topics(raw_topics, transcript=transcript)
            for t in parsed:
                if isinstance(t, dict):
                    topics.append(t)
                elif " - " in str(t):
                    parts = str(t).split(" - ", 1)
                    topics.append({"title": parts[0].strip(), "details": [parts[1].strip()]})
                else:
                    topics.append({"title": str(t), "details": ["Key discussion details covered during meeting."]})

        return topics

    def parse_decisions_file(self, text):
        """Parse decisions into structured list of dicts."""
        if not text:
            return []

        # If already JSON, load directly
        if isinstance(text, str) and text.strip().startswith(("[", "{")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed:
                    res = []
                    for item in parsed:
                        if isinstance(item, dict):
                            res.append(item)
                        elif isinstance(item, str):
                            res.append({"decision": item.strip(), "context": "", "responsible": "Not specified"})
                    if res:
                        return res
            except Exception:
                pass

        decisions = []
        current = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("=========="):
                if current.get("decision"):
                    decisions.append(current)
                    current = {}
                continue

            l_clean = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if l_clean.lower().startswith("decision:"):
                if current.get("decision"):
                    decisions.append(current)
                current = {"decision": l_clean[9:].strip(), "context": "", "responsible": "Not specified"}
            elif l_clean.lower().startswith("context:") and current:
                current["context"] = l_clean[8:].strip()
            elif l_clean.lower().startswith("responsible:") and current:
                current["responsible"] = l_clean[12:].strip()

        if current.get("decision"):
            decisions.append(current)

        if not decisions:
            lines = self.clean_lines(text)
            for l in lines:
                decisions.append({"decision": l, "context": "", "responsible": "Not specified"})

        return decisions

    def parse_actions_file(self, text):
        """Parse action items into structured list of dicts."""
        if not text:
            return []

        # If already JSON, load directly
        if isinstance(text, str) and text.strip().startswith(("[", "{")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed:
                    res = []
                    for item in parsed:
                        if isinstance(item, dict):
                            res.append(item)
                        elif isinstance(item, str):
                            single_m = re.match(r'^(.*?)\s*-\s*([^:]+):\s*(.*)$', item)
                            if single_m:
                                res.append({
                                    "owner": single_m.group(2).strip(),
                                    "task": single_m.group(1).strip(),
                                    "context": single_m.group(3).strip(),
                                    "deadline": "Not specified",
                                    "priority": "Medium"
                                })
                            else:
                                res.append({"owner": "Not specified", "task": item.strip(), "deadline": "Not specified", "priority": "Medium", "context": ""})
                    if res:
                        return res
            except Exception:
                pass

        actions = []
        current = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("=========="):
                if current.get("task"):
                    actions.append(current)
                    current = {}
                continue

            single_m = re.match(r'^(.*?)\s*-\s*([^:]+):\s*(.*)$', line)
            if single_m and not line.lower().startswith(("owner:", "task:", "deadline:", "priority:", "context:")):
                if current.get("task"):
                    actions.append(current)
                    current = {}
                actions.append({
                    "owner": single_m.group(2).strip(),
                    "task": single_m.group(1).strip(),
                    "context": single_m.group(3).strip(),
                    "deadline": "Not specified",
                    "priority": "Medium"
                })
                continue

            l_clean = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            l_lower = l_clean.lower()

            if l_lower.startswith("owner:") or l_lower.startswith("person:"):
                if current.get("task"):
                    actions.append(current)
                val = l_clean.split(":", 1)[1].strip()
                current = {
                    "owner": val if val and val.lower() not in {"unknown", "none", "n/a", "unspecified"} else "Not specified",
                    "task": "",
                    "deadline": "Not specified",
                    "priority": "Medium",
                    "context": ""
                }
            elif l_lower.startswith("task:") and current:
                current["task"] = l_clean[5:].strip()
            elif l_lower.startswith("deadline:") and current:
                current["deadline"] = l_clean[9:].strip()
            elif l_lower.startswith("priority:") and current:
                current["priority"] = l_clean[9:].strip().capitalize()
            elif l_lower.startswith("context:") and current:
                current["context"] = l_clean[8:].strip()

        if current.get("task"):
            actions.append(current)

        if not actions:
            lines = self.clean_lines(text)
            for l in lines:
                actions.append({
                    "owner": "Not specified",
                    "task": l,
                    "deadline": "Not specified",
                    "priority": "Medium",
                    "context": ""
                })

        return actions

    def generate(self, output_name="meeting_001"):
        summary = self.read_file(SUMMARY_FOLDER / f"{output_name}_summary.txt")

        extractor = MetadataExtractor()
        transcript = extractor.read_transcript(f"{output_name}.txt")

        raw_topics = self.read_file(TOPICS_FOLDER / f"{output_name}_topics.txt")
        topics = self.parse_topics_file(raw_topics, transcript=transcript)

        actions_text = self.read_file(ACTION_ITEMS_FOLDER / f"{output_name}_action_items.txt")
        actions = self.parse_actions_file(actions_text)

        decisions_text = self.read_file(DECISIONS_FOLDER / f"{output_name}_decisions.txt")
        decisions = self.parse_decisions_file(decisions_text)

        open_questions = []
        
        key_disc_path = KEY_DISCUSSION_FOLDER / f"{output_name}_key_discussion.json"
        key_discussion = []
        if key_disc_path.exists():
            try:
                with open(key_disc_path, "r", encoding="utf-8") as f:
                    key_discussion = json.load(f)
            except Exception:
                pass

        future_topics_file = FUTURE_TOPICS_FOLDER / f"{output_name}_future_topics.txt"
        future_topics = []
        if future_topics_file.exists():
            for line in self.read_file(future_topics_file).splitlines():
                line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
                if line and not line.startswith("="):
                    future_topics.append(line)

        metadata = extractor.extract(transcript, meeting_id=output_name)

        report_information = {
            "generated_by": "Enterprise Briefing AI",
            "version": "2.0",
            "report_type": "Refined Meeting Summary",
        }

        report = {
            "report_information": report_information,
            "metadata": metadata,
            "summary": summary,
            "executive_summary": summary,
            "topics": topics,
            "discussion_topics": topics,
            "key_discussion": key_discussion,
            "key_decisions": decisions,
            "decisions": decisions,
            "action_items": actions,
            "future_topics": future_topics,
            "open_questions": [],
        }

        REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
        output = REPORT_FOLDER / f"{output_name}_report.json"

        with open(output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        print("\n========== REPORT GENERATED ==========")
        print(output)
        return report


if __name__ == "__main__":
    ReportGenerator().generate()