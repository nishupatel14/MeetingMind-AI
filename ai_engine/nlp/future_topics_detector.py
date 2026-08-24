"""
AI Suggested Topics for Future Meetings Detector

Extracts forward-looking agenda items, unaddressed topics, or strategic priorities 
recommended for future meetings using Qwen 2.5 LLM.
"""

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    FUTURE_TOPICS_FOLDER,
)

from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner
import re


class FutureTopicsDetector:

    def __init__(self):
        self.cleaner = TranscriptCleaner()

    def read_transcript(self, filename):
        path = TRANSCRIPT_FOLDER / filename
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def detect(self, transcript, decisions=None, actions=None, open_questions=None):
        cleaned_transcript = self.cleaner.clean(transcript)

        if not cleaned_transcript or len(cleaned_transcript.split()) < 20:
            return ["No specific future meeting topics were identified."]

        context_extra = ""
        if open_questions:
            context_extra += "\nUnresolved Questions / Open Issues:\n" + "\n".join(f"- {q}" for q in open_questions)
        if decisions:
            context_extra += "\nKey Decisions Made:\n" + "\n".join(f"- {d.get('decision', d) if isinstance(d, dict) else d}" for d in decisions)

        prompt = f"""You are MeetingMind AI, an expert executive meeting analyst.

Review the meeting transcript and context below to identify 3 to 4 SUGGESTED TOPICS FOR FUTURE MEETINGS.

These must be forward-looking agenda items, follow-up milestones, or strategic priorities discussed or logically required by this meeting.

RULES:
1. Extract 3 to 4 forward-looking future meeting agenda items.
2. Format each item as a concise, action-oriented phrase (12 to 24 words).
   Examples:
   - Finalizing the Investment Roadmap and Product Development Priorities (AI/ML, Cybersecurity, ESG).
   - Implementing the New Internal System for Tracking Requirements, Discussions, and Updates (CRM/System Integration).
   - Detailed Execution Strategy for Content Creation and Lead Nurturing Campaigns.
3. Base them 100% on the transcript text. Do NOT invent outside generic topics.
4. Plain text only. No markdown formatting.

Transcript:
{cleaned_transcript[:3000]}
{context_extra}

Suggested Topics for Future Meetings:
"""

        output = ActionModelLoader.generate(prompt)

        lines = []
        for line in output.splitlines():
            line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
            line = re.sub(r"\*\*", "", line).strip()
            if line and len(line.split()) >= 3:
                lines.append(line)

        if not lines:
            lines = ["Follow up on open questions and action items assigned during this session."]

        return lines[:4]

    def save(self, filename, topics):
        FUTURE_TOPICS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = FUTURE_TOPICS_FOLDER / filename

        with open(output, "w", encoding="utf-8") as file:
            file.write("========== SUGGESTED TOPICS FOR FUTURE MEETINGS ==========\n\n")
            for topic in topics:
                file.write(f"• {topic}\n")

        print(f"\nFuture topics saved to:\n{output}")


if __name__ == "__main__":
    detector = FutureTopicsDetector()
    sample_text = "In our next meeting, we need to finalize the investment roadmap and test CRM integration."
    res = detector.detect(sample_text)
    print(res)
