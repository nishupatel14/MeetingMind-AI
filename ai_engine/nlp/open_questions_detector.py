"""
Open Questions / Unresolved Issues Detector

Identifies unresolved questions, pending research, and unassigned items
from the meeting transcript.
"""

import re
from ai_engine.config import PROCESSED_DATASET
from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner

OPEN_QUESTIONS_FOLDER = PROCESSED_DATASET / "open_questions"


class OpenQuestionsDetector:

    def __init__(self):
        self.cleaner = TranscriptCleaner()

    def detect(self, transcript):
        cleaned = self.cleaner.clean(transcript)

        if not cleaned or len(cleaned.split()) < 30:
            return ["No major unresolved issues were identified."]

        # Use last portion if transcript is very long
        words = cleaned.split()
        text_to_analyze = " ".join(words[:3000]) if len(words) > 3000 else cleaned

        prompt = f"""You are MeetingMind AI, an executive meeting analyst.

Review this meeting transcript and identify UNRESOLVED ISSUES or OPEN QUESTIONS.

These are things that were:
- Discussed but NOT resolved or finalized
- Questions asked but NOT answered during the meeting
- Research or information still needed
- Responsibilities NOT yet assigned
- Proposals requiring further validation or approval

RULES:
1. List 2 to 5 open questions or unresolved issues.
2. Each item must be a clear, specific sentence.
3. Only include items actually discussed. Do NOT invent issues.
4. If nothing is unresolved, write: No major unresolved issues were identified.
5. Do NOT use markdown formatting. Plain text only.
6. Format each item on a new line starting with a dash (-).

Transcript:
{text_to_analyze}

Open Questions / Unresolved Issues:
"""
        output = ActionModelLoader.generate(prompt)

        items = []
        for line in output.splitlines():
            line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
            line = line.replace("**", "").strip()
            if line and len(line.split()) >= 4 and "no major unresolved" not in line.lower():
                items.append(line)

        if not items:
            items = ["No major unresolved issues were identified."]

        return items[:5]

    def save(self, filename, items):
        OPEN_QUESTIONS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = OPEN_QUESTIONS_FOLDER / filename
        with open(output, "w", encoding="utf-8") as f:
            f.write("========== OPEN QUESTIONS / UNRESOLVED ISSUES ==========\n\n")
            for item in items:
                f.write(f"• {item}\n")
        print(f"\nOpen questions saved to:\n{output}")
