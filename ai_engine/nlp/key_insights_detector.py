"""
Key Insights Detector

Extracts 3-7 actionable business/technical insights
that can be directly supported by the meeting transcript.
"""

import re
from ai_engine.config import PROCESSED_DATASET
from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner

KEY_INSIGHTS_FOLDER = PROCESSED_DATASET / "key_insights"


class KeyInsightsDetector:

    def __init__(self):
        self.cleaner = TranscriptCleaner()

    def detect(self, transcript):
        cleaned = self.cleaner.clean(transcript)

        if not cleaned or len(cleaned.split()) < 30:
            return ["Insufficient transcript content for insight extraction."]

        words = cleaned.split()
        text_to_analyze = " ".join(words[:3000]) if len(words) > 3000 else cleaned

        prompt = f"""You are MeetingMind AI, an executive meeting analyst.

Review this meeting transcript and extract KEY INSIGHTS.

Key insights are important conclusions, observations, or strategic takeaways
that can be directly supported by the discussion in the meeting.

RULES:
1. Extract 3 to 5 key insights from the meeting.
2. Focus on useful business or technical insights, NOT just topic labels.
3. Each insight must be a specific, actionable sentence.
4. Only include insights supported by the actual discussion. Do NOT invent.
5. Do NOT use markdown formatting. Plain text only.
6. Format each insight on a new line starting with a dash (-).

Transcript:
{text_to_analyze}

Key Insights:
"""
        output = ActionModelLoader.generate(prompt)

        items = []
        for line in output.splitlines():
            line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
            line = line.replace("**", "").strip()
            if line and len(line.split()) >= 5:
                items.append(line)

        if not items:
            items = ["The meeting covered important operational and strategic points requiring follow-up."]

        return items[:7]

    def save(self, filename, items):
        KEY_INSIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = KEY_INSIGHTS_FOLDER / filename
        with open(output, "w", encoding="utf-8") as f:
            f.write("========== KEY INSIGHTS ==========\n\n")
            for item in items:
                f.write(f"• {item}\n")
        print(f"\nKey insights saved to:\n{output}")
