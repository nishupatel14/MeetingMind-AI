"""
Key Discussion Detector

Extracts key points per topic from meeting transcript with timestamps
and categorizes each point as Fact, Concern, or Opinion.
"""

import json
import re
from ai_engine.config import KEY_DISCUSSION_FOLDER
from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner


class KeyDiscussionDetector:

    def __init__(self):
        self.cleaner = TranscriptCleaner()

    def format_timestamped_transcript(self, transcript_data):
        """
        Converts transcript input into string format with [MM:SS] timestamps.
        """
        if isinstance(transcript_data, list):
            lines = []
            for idx, seg in enumerate(transcript_data):
                if isinstance(seg, dict) and "text" in seg:
                    start_sec = int(float(seg.get("start", idx * 5)))
                    mm = start_sec // 60
                    ss = start_sec % 60
                    ts_str = f"[{mm:02d}:{ss:02d}]"
                    text = str(seg["text"]).strip()
                    if text:
                        lines.append(f"{ts_str} {text}")
                elif isinstance(seg, str) and seg.strip():
                    est_mm = (idx * 5) // 60
                    est_ss = (idx * 5) % 60
                    lines.append(f"[{est_mm:02d}:{est_ss:02d}] {seg.strip()}")
            if lines:
                return "\n".join(lines)

        raw_text = str(transcript_data)
        lines = []
        for idx, line in enumerate(raw_text.splitlines()):
            line = line.strip()
            if line:
                est_mm = (idx * 5) // 60
                est_ss = (idx * 5) % 60
                lines.append(f"[{est_mm:02d}:{est_ss:02d}] {line}")
        return "\n".join(lines)

    def detect(self, transcript_data):
        """
        Detect key discussion points per topic with timestamp and category (Fact / Concern / Opinion).
        """
        timestamped_text = self.format_timestamped_transcript(transcript_data)
        if not timestamped_text or len(timestamped_text.split()) < 15:
            return []

        words = timestamped_text.split()
        text_to_analyze = " ".join(words[:1500]) if len(words) > 1500 else timestamped_text

        prompt = f"""You are MeetingMind AI, an executive meeting analyst.

Review the timestamped meeting transcript below.

Your task: Extract key discussion highlights grouped by MAXIMUM 2 TO 3 MAIN TOPICS ONLY.

RULES FOR EACH POINT:
1. Extract MAXIMUM 2 to 3 main high-level topic groups.
2. For each topic, extract MAXIMUM 1 key timestamped point.
3. Include the timestamp (e.g. [01:25]) from the transcript where the point was discussed.
4. Classify each point as: [Fact], [Concern], or [Opinion]
5. State specific facts, figures, or key statements mentioned.

OUTPUT FORMAT:
TOPIC: [Main Topic Name]
- [01:25] [Fact]: [Specific statement with facts/figures]

Do NOT use markdown formatting like ** bold **.

Transcript:
{text_to_analyze}

Key Discussion Points:
"""
        output = ActionModelLoader.generate(prompt)

        results = self._parse_discussion_output(output)

        # Fallback: If LLM output fails to parse or returns empty, extract directly from timestamped transcript
        if not results:
            results = self._fallback_extract_from_transcript(timestamped_text)

        return results

    def _parse_discussion_output(self, raw_output):
        """Parse structured text output into top 2-3 topics with max 1 timestamped point each."""
        if not raw_output or not isinstance(raw_output, str):
            return []

        results = []
        current_topic = None
        current_points = []

        for raw_line in raw_output.splitlines():
            line = raw_line.replace("**", "").replace("##", "").strip()
            if not line:
                if current_topic and current_points:
                    results.append({
                        "topic": current_topic,
                        "points": current_points[:1]
                    })
                    current_topic = None
                    current_points = []
                continue

            # Flexible Topic Match: "TOPIC: ...", "Topic 1: ...", "1. TOPIC: ...", "Main Topic: ..."
            topic_match = re.match(r'^(?:\d+[\.\)]\s*)?(?:TOPIC|Topic|Main Topic)\s*\d*[:\-\.]\s*(.*)', line, re.IGNORECASE)
            if topic_match:
                if current_topic and current_points:
                    results.append({
                        "topic": current_topic,
                        "points": current_points[:1]
                    })
                title = topic_match.group(1).strip()
                if title:
                    current_topic = title
                    current_points = []
                    continue

            # Point Match with explicit [01:25] and [Fact/Concern/Opinion]
            point_match = re.search(r'\[?(\d{1,2}:\d{2})\]?\s*\[?(Fact|Concern|Opinion)\]?[:\-\s]+(.*)', line, re.IGNORECASE)
            if point_match:
                ts = point_match.group(1).strip()
                p_type = point_match.group(2).strip().capitalize()
                text = point_match.group(3).strip()
                if current_topic is None:
                    current_topic = "General Discussion"
                if len(current_points) < 1 and len(text.split()) >= 3:
                    current_points.append({
                        "timestamp": ts,
                        "type": p_type,
                        "text": text
                    })
            else:
                # Bullet Match fallback for lines starting with -, *, bullet, or number
                bullet_match = re.match(r'^(?:[\-•\*]|\d+[\.\)])\s*(.*)', line)
                if bullet_match:
                    content = bullet_match.group(1).strip()
                    ts_match = re.search(r'\[?(\d{1,2}:\d{2})\]?', content)
                    ts = ts_match.group(1) if ts_match else "00:00"

                    p_type = "Fact"
                    if any(w in content.lower() for w in ["concern", "risk", "issue", "challenge", "problem", "delay"]):
                        p_type = "Concern"
                    elif any(w in content.lower() for w in ["suggest", "think", "opinion", "believe", "proposal", "recommend"]):
                        p_type = "Opinion"

                    clean_text = re.sub(r'\[?\d{1,2}:\d{2}\]?\s*', '', content).strip()
                    clean_text = re.sub(r'^(?:Fact|Concern|Opinion)[:\-\s]*', '', clean_text, flags=re.IGNORECASE).strip()

                    if len(clean_text.split()) >= 4:
                        if current_topic is None:
                            current_topic = "General Discussion"
                        if len(current_points) < 1:
                            current_points.append({
                                "timestamp": ts,
                                "type": p_type,
                                "text": clean_text
                            })

        if current_topic and current_points:
            results.append({
                "topic": current_topic,
                "points": current_points[:1]
            })

        return results[:3]

    def _fallback_extract_from_transcript(self, timestamped_text):
        """Emergency fallback: Extract timestamped key discussion points directly from transcript."""
        lines = [l.strip() for l in timestamped_text.splitlines() if l.strip()]
        if not lines:
            return []

        # Select up to 2 distinct timestamped segments evenly distributed across the meeting
        selected_segments = []
        if len(lines) <= 2:
            selected_segments = lines
        else:
            step = len(lines) // 2
            selected_segments = [lines[0], lines[min(step, len(lines) - 1)]]

        results = []
        for idx, line in enumerate(selected_segments, start=1):
            match = re.search(r'\[?(\d{1,3}:\d{2})\]?\s*(.*)', line)
            if match:
                ts = match.group(1)
                txt = match.group(2).strip()
                if len(txt.split()) >= 3:
                    p_type = "Fact"
                    if any(w in txt.lower() for w in ["risk", "issue", "worry", "concern", "problem"]):
                        p_type = "Concern"
                    elif any(w in txt.lower() for w in ["think", "feel", "believe", "suggest"]):
                        p_type = "Opinion"

                    results.append({
                        "topic": f"Key Discussion Highlight #{idx}",
                        "points": [{
                            "timestamp": ts,
                            "type": p_type,
                            "text": txt
                        }]
                    })

        return results

    def save(self, filename, discussion_data):
        KEY_DISCUSSION_FOLDER.mkdir(parents=True, exist_ok=True)
        output = KEY_DISCUSSION_FOLDER / filename
        with open(output, "w", encoding="utf-8") as f:
            json.dump(discussion_data, f, indent=4, ensure_ascii=False)
        print(f"\nKey discussion saved to:\n{output}")
