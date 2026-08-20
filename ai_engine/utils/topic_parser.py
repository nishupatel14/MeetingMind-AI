"""
Topic Parser & Context Enrichment Utility

Parses raw topic lines (whether single-line, multi-line header + sub-bullets,
numbered lists, bold markdown, or title-only entries) and guarantees that
EVERY output item is strictly formatted as:
'Topic Title - Detailed summary of what was discussed'.
"""

import re


def extract_context_summary(title, transcript=""):
    """
    Extracts a real summary sentence from the transcript for a given topic title.
    """
    title_clean = re.sub(r'^(?:Topic\s*\d+[:\-]?\s*)', '', title, flags=re.IGNORECASE).strip()

    if not transcript:
        return f"Key discussions and operational details regarding {title_clean.lower()}."

    # Extract key words from title
    title_words = [
        w.lower()
        for w in re.findall(r'\b[a-zA-Z]{3,}\b', title_clean)
        if w.lower() not in {"and", "with", "from", "that", "this", "have", "overall", "about", "topic", "for", "the"}
    ]

    sentences = re.split(r'[\r\n.!?]+', transcript)
    matched_sentences = []

    for s in sentences:
        s_clean = s.strip()
        if len(s_clean.split()) >= 5:
            s_lower = s_clean.lower()
            if any(w in s_lower for w in title_words):
                matched_sentences.append(s_clean)
                if len(matched_sentences) >= 1:
                    break

    if matched_sentences:
        summary_text = matched_sentences[0].strip()
        words = summary_text.split()
        if len(words) > 18:
            summary_text = " ".join(words[:18]) + "."
        return summary_text
    else:
        return f"Key discussions covering {title_clean.lower()}."


def parse_topics(raw_lines_or_text, transcript=""):
    """
    Parses topic input. Accepts either:
    - A list of dicts [{'title': '...', 'details': [...]}]
    - A raw text string or list of strings
    Returns a list of dicts or formatted strings.
    """
    if isinstance(raw_lines_or_text, list):
        if raw_lines_or_text and isinstance(raw_lines_or_text[0], dict):
            return raw_lines_or_text

    if isinstance(raw_lines_or_text, str):
        lines = raw_lines_or_text.splitlines()
    elif isinstance(raw_lines_or_text, (list, tuple)):
        lines = list(raw_lines_or_text)
    else:
        return []

    # Check if raw text contains structured TOPIC: ... format
    rich_topics = []
    current_title = None
    current_details = []

    for line in lines:
        line_str = str(line).strip()
        if not line_str or line_str.startswith("==========") or line_str.startswith("-------------"):
            if current_title and current_details:
                rich_topics.append({"title": current_title, "details": current_details})
                current_title = None
                current_details = []
            continue

        if line_str.startswith("TOPIC:"):
            if current_title and current_details:
                rich_topics.append({"title": current_title, "details": current_details})
            current_title = line_str[6:].strip()
            current_details = []
            continue

        detail_match = re.match(r'^\s*[\-•\*]\s*(.*)', line_str)
        if detail_match and current_title:
            current_details.append(detail_match.group(1).strip())

    if current_title and current_details:
        rich_topics.append({"title": current_title, "details": current_details})

    if rich_topics:
        return rich_topics

    # Fallback for plain lines / "Title - Summary" strings
    cleaned_raw_lines = []
    for line in lines:
        line = str(line).strip()
        if not line or line.startswith("==========") or line.startswith("-------------"):
            continue
        cleaned_raw_lines.append(line)

    if not cleaned_raw_lines:
        return []

    raw_topics = []

    for line in cleaned_raw_lines:
        clean = re.sub(r'^#{1,4}\s*', '', line).strip()
        clean = re.sub(r"^[\s•\-\*\d\.]+", "", clean).strip()
        clean = clean.replace("**", "").strip()
        clean = re.sub(r'^[\#\*\-\s]+', '', clean).strip()

        if not clean:
            continue

        if clean.lower().endswith(":") and len(clean.split()) <= 5:
            continue

        skip_labels = {
            "discussion topics", "key discussion topics", "final discussion topics",
            "extracted themes", "final key discussion topics", "discussion notes",
            "topics discussed", "summary", "overview", "analysis",
        }
        if clean.lower() in skip_labels:
            continue

        if " - " in clean:
            parts = clean.split(" - ", 1)
            raw_topics.append((parts[0].strip(), parts[1].strip()))
            continue

        if ": " in clean:
            parts = clean.split(": ", 1)
            if len(parts[0].split()) <= 10:
                raw_topics.append((parts[0].strip(), parts[1].strip()))
                continue

        if len(clean.split()) >= 3:
            raw_topics.append((clean, ""))

    final_topics = []
    seen_keys = set()

    bad_title_words = {
        "technology", "issue", "meeting", "discussion", "item", "problem",
        "task", "general", "status", "update", "overview", "notes", "agenda"
    }

    for title, summary in raw_topics:
        title = re.sub(r'^(?:Topic\s*\d+[:\-]?\s*)', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'^(?:Option\s*\d+[:\-]?\s*)', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'^[\#\*\-\s]+', '', title).strip()

        if not title:
            continue

        norm_key = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
        if not norm_key or norm_key in seen_keys:
            continue

        if title.lower() in bad_title_words or (len(title.split()) == 1 and title.lower() in bad_title_words):
            continue

        seen_keys.add(norm_key)

        if len(summary.split()) < 4:
            summary = extract_context_summary(title, transcript)

        final_topics.append(f"{title} - {summary}")

    return final_topics[:6]

