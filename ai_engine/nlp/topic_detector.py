"""
AI Discussion Topic Detector

Extracts major discussion topics from meeting transcripts with detailed
explanations of what was discussed, examples given, and business context.
Produces structured output: [{title, details: [...]}, ...]
"""

print("TOPIC GENERATOR STARTED")

import re
from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    TOPICS_FOLDER,
)

from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner
from ai_engine.nlp.report_validator import ReportValidator


class TopicDetector:

    def __init__(self):
        self.cleaner = TranscriptCleaner()

    def read_transcript(self, filename):
        path = TRANSCRIPT_FOLDER / filename
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    # --------------------------------------------------------
    # Phase 1: Extract key discussion points from each chunk
    # --------------------------------------------------------

    def _extract_points_from_chunk(self, chunk, keywords_str):
        """Extract specific discussion points from a transcript chunk."""
        prompt = f"""You are MeetingMind AI, an executive meeting analyst.

Read this transcript portion carefully and extract the KEY DISCUSSION POINTS.

Key transcript concepts: {keywords_str}

For each point, write:
- WHAT was discussed (the specific topic or subject)
- IMPORTANT DETAILS mentioned (examples, data, names, problems, proposals)

RULES:
1. Extract 2 to 5 specific discussion points from this portion.
2. Write each point as a clear, detailed sentence (not just a keyword).
3. Include specific examples, names, numbers mentioned in the transcript.
4. Do NOT invent information. Only use what is actually in the transcript.
5. Do NOT use markdown formatting. Plain text only.
6. Format each point on a new line starting with a dash (-).

Transcript portion:
{chunk}

Key Discussion Points:
"""
        output = ActionModelLoader.generate(prompt)
        points = []
        for line in output.splitlines():
            line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
            line = line.replace("**", "").strip()
            if line and len(line.split()) >= 5:
                points.append(line)
        return points

    # --------------------------------------------------------
    # Phase 2: Group related points into major topics
    # --------------------------------------------------------

    def _group_and_synthesize(self, all_points, keywords_str, transcript):
        """Group related discussion points into 3-4 major strategic topics with 1 clear summary line."""
        points_text = "\n".join(f"- {p}" for p in all_points)

        prompt = f"""You are MeetingMind AI, an expert meeting analyst.

Below are key discussion points extracted from a meeting transcript.

Your task: Synthesize and group these points into EXACTLY 3 to 4 MAIN STRATEGIC DISCUSSION TOPICS.

RULES:
1. Combine minor sub-points into 3 to 4 broad, professional topic categories.
2. For each topic, write:
   TOPIC: [Clear descriptive title - high-level business topic category]
   EXPLANATION: [Write a complete, detailed 1-2 sentence explanation of WHAT was specifically discussed, proposed, or analyzed during the meeting for this topic. Do NOT write short phrases or 'Topic:' headers.]
3. Keep each topic focused on major strategic themes only.
4. Do NOT invent information not present in the discussion points.
5. Do NOT use markdown formatting (no **, no ###). Plain text only.
6. Separate each topic with a blank line.

Discussion Points:
{points_text}

Grouped Discussion Topics:
"""
        output = ActionModelLoader.generate(prompt)
        return self._parse_rich_topics(output, transcript)

    # --------------------------------------------------------
    # Parse structured topic output
    # --------------------------------------------------------

    def _parse_rich_topics(self, raw_output, transcript=""):
        """Parse LLM output into structured topic dicts (max 4 topics, 1 full explanation each)."""
        topics = []
        current_title = None
        current_details = []

        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                if current_title and current_details:
                    topics.append({
                        "title": current_title,
                        "details": current_details[:1]
                    })
                    current_title = None
                    current_details = []
                continue

            # Remove markdown formatting
            line = re.sub(r'^#{1,4}\s*', '', line).strip()
            line = line.replace("**", "").strip()

            # Check if this is a topic title line
            title_match = re.match(r'^(?:TOPIC|Topic)\s*\d*[:\-\.]\s*(.*)', line)
            if title_match:
                if current_title and current_details:
                    topics.append({
                        "title": current_title,
                        "details": current_details[:1]
                    })
                current_title = title_match.group(1).strip()
                current_details = []
                continue

            # Check for numbered topic (e.g. "1. Topic Name" or "1) Topic Name")
            numbered_match = re.match(r'^\d+[\.\)]\s+([A-Z].*)', line)
            if numbered_match and not line.startswith("-") and len(line.split()) <= 12:
                if current_title and current_details:
                    topics.append({
                        "title": current_title,
                        "details": current_details[:1]
                    })
                current_title = numbered_match.group(1).strip()
                current_details = []
                continue

            # Check for EXPLANATION line
            exp_match = re.match(r'^(?:EXPLANATION|Explanation|DETAIL|Detail)[:\-\s]+(.*)', line, re.IGNORECASE)
            if exp_match and current_title:
                exp_text = exp_match.group(1).strip()
                exp_text = re.sub(r'^(?:Topic|TOPIC)[:\-\s]+', '', exp_text, flags=re.IGNORECASE).strip()
                if len(exp_text.split()) >= 4:
                    current_details.append(exp_text)
                continue

            # Detail bullet
            detail_match = re.match(r'^[\-•\*]\s*(.*)', line)
            if detail_match and current_title:
                detail_text = detail_match.group(1).strip()
                detail_text = re.sub(r'^(?:Topic|EXPLANATION|Explanation|Detail)[:\-\s]+', '', detail_text, flags=re.IGNORECASE).strip()
                if len(detail_text.split()) >= 4:
                    current_details.append(detail_text)
            elif current_title and len(line.split()) >= 5:
                clean_line = re.sub(r'^(?:Topic|EXPLANATION|Explanation|Detail)[:\-\s]+', '', line, flags=re.IGNORECASE).strip()
                current_details.append(clean_line)

        # Don't forget last topic
        if current_title and current_details:
            topics.append({
                "title": current_title,
                "details": current_details[:1]
            })

        # Validate topic titles are grounded in transcript
        if transcript:
            t_words = ReportValidator.get_transcript_words(transcript)
            validated = []
            for topic in topics:
                title_lower = topic["title"].lower()
                if title_lower in ReportValidator.VAGUE_TOPIC_PATTERNS:
                    continue
                validated.append(topic)
            if validated:
                topics = validated

        return topics[:4]

    # --------------------------------------------------------
    # Main detection entry point
    # --------------------------------------------------------

    def detect_topics(self, transcript):
        """Main entry point — returns list of topic dicts with title + details."""
        print("ENTERED detect_topics()")

        cleaned_transcript = self.cleaner.clean(transcript)
        top_keywords = ReportValidator.extract_keywords(cleaned_transcript, top_n=15)
        keywords_str = ", ".join(top_keywords) if top_keywords else "N/A"

        word_count = len(cleaned_transcript.split())
        print(f"[TopicDetector] Transcript: {word_count} words")

        # Split into chunks for point extraction
        if word_count < 600:
            chunks = [cleaned_transcript]
        else:
            chunks = self.cleaner.split_text(cleaned_transcript, chunk_size=500)

        print(f"[TopicDetector] Extracting points from {len(chunks)} chunks...")

        # Phase 1: Extract key points from each chunk
        all_points = []
        for idx, chunk in enumerate(chunks, 1):
            if len(chunk.split()) < 25:
                continue
            print(f"[TopicDetector] Processing chunk {idx}/{len(chunks)}...")
            points = self._extract_points_from_chunk(chunk, keywords_str)
            all_points.extend(points)

        if not all_points:
            return [{
                "title": "General Meeting Discussion",
                "details": ["Key discussions and operational points were covered during the meeting."]
            }]

        # Remove near-duplicate points
        unique_points = []
        seen_normalized = set()
        for p in all_points:
            norm = re.sub(r'[^a-z0-9\s]', '', p.lower()).strip()
            short_key = " ".join(norm.split()[:8])
            if short_key not in seen_normalized:
                seen_normalized.add(short_key)
                unique_points.append(p)

        print(f"[TopicDetector] Extracted {len(unique_points)} unique discussion points")

        # Phase 2: Group into major topics
        topics = self._group_and_synthesize(unique_points, keywords_str, transcript)

        if not topics:
            # Fallback: use points as a single topic
            return [{
                "title": "Meeting Discussion Overview",
                "details": unique_points[:5]
            }]

        print(f"[TopicDetector] Final: {len(topics)} grouped discussion topics")
        for t in topics:
            print(f"  -> {t['title']} ({len(t['details'])} details)")

        return topics

    # --------------------------------------------------------
    # Backward-compatible flat topic list (for old callers)
    # --------------------------------------------------------

    def detect_topics_flat(self, transcript):
        """Returns flat list of 'Title - Summary' strings for backward compatibility."""
        rich = self.detect_topics(transcript)
        flat = []
        for topic in rich:
            summary = topic["details"][0] if topic["details"] else "Key discussions covered."
            flat.append(f"{topic['title']} - {summary}")
        return flat

    # --------------------------------------------------------
    # File I/O
    # --------------------------------------------------------

    def detect_file(self, transcript_file):
        transcript = self.read_transcript(transcript_file)
        topics = self.detect_topics(transcript)
        output_name = transcript_file.replace(".txt", "_topics.txt")
        self.save(output_name, topics)
        return topics

    def save(self, filename, topics):
        TOPICS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = TOPICS_FOLDER / filename

        with open(output, "w", encoding="utf-8") as file:
            file.write("========== DISCUSSION TOPICS ==========\n\n")
            if not topics:
                file.write("• General Meeting Discussion - Overview of key topics discussed.\n")
            elif isinstance(topics[0], dict):
                for topic in topics:
                    file.write(f"TOPIC: {topic['title']}\n")
                    for detail in topic.get("details", []):
                        file.write(f"  - {detail}\n")
                    file.write("\n")
            else:
                for topic in topics:
                    file.write(f"• {topic}\n")

        print(f"\nDiscussion topics saved to:\n{output}")


if __name__ == "__main__":
    detector = TopicDetector()
    topics = detector.detect_file("meeting_001.txt")
    print("\n========== DISCUSSION TOPICS ==========\n")
    for topic in topics:
        if isinstance(topic, dict):
            print(f"📌 {topic['title']}")
            for d in topic.get("details", []):
                print(f"   - {d}")
            print()
        else:
            print(f"• {topic}")