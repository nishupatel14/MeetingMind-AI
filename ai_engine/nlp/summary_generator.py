"""
AI Executive Summary Generator
"""

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    SUMMARY_FOLDER,
)

from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner
import re


class SummaryGenerator:

    def __init__(self):
        self.cleaner = TranscriptCleaner()

    def read_transcript(self, filename):
        path = TRANSCRIPT_FOLDER / filename
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def split_text(self, text, chunk_size=500):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def summarize(self, transcript):
        print("ENTERED summarize()")

        if not transcript or not transcript.strip():
            return (
                "========== EXECUTIVE SUMMARY ==========\n\n"
                "No transcript content available for summarization."
            )

        cleaned_text = self.cleaner.clean(transcript)
        word_count = len(cleaned_text.split())

        if word_count < 600:
            # Direct single-pass summarization for standard meetings
            content_to_summarize = cleaned_text
        else:
            # Multi-chunk extraction for long meetings
            chunks = self.split_text(cleaned_text, chunk_size=500)
            print(f"Summarizing long transcript in {len(chunks)} chunks...")
            chunk_points = []
            for idx, chunk in enumerate(chunks, 1):
                if len(chunk.split()) < 20:
                    continue
                chunk_prompt = f"""
Summarize the key discussion points from this part of the meeting transcript in 2 clear sentences.

Transcript part:
{chunk}

Summary:
"""
                summary_part = ActionModelLoader.generate(chunk_prompt)
                if summary_part:
                    chunk_points.append(summary_part)
            content_to_summarize = "\n".join(chunk_points)

        final_prompt = f"""
You are MeetingMind AI, an expert executive assistant.

Write a concise, professional Executive Summary of the meeting content below.

STRICT FORMAT & LENGTH RULES:
1. Paragraph 1 (Overview): Write 2 short, high-level sentences stating the primary goal, background context, and main focus of the meeting.
2. Bullet Points (Key Outcomes): Write 2 to 3 concise bullet points (starting with •) summarizing the primary strategic takeaways, core decisions, and next steps.
3. Keep the TOTAL length under 120 words.
4. Base EVERY detail strictly on the provided text. Do NOT invent facts.

Meeting Content:
{content_to_summarize}

Executive Summary:
"""

        result = ActionModelLoader.generate(final_prompt)

        # Post-processing: Ensure clean formatting
        result = re.sub(r'^(?:Executive Summary[:\-\s]*|\*\*Executive Summary\*\*[:\-\s]*)', '', result, flags=re.IGNORECASE).strip()
        
        # Format into clean overview + bullet points
        lines = [l.strip() for l in result.splitlines() if l.strip()]
        overview_lines = []
        bullet_lines = []

        for line in lines:
            if line.startswith(("•", "-", "*", "1.", "2.", "3.")):
                clean_b = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
                if clean_b:
                    bullet_lines.append(f"• {clean_b}")
            else:
                overview_lines.append(line)

        overview_text = " ".join(overview_lines).strip()
        if not overview_text and bullet_lines:
            overview_text = "The meeting focused on key operational strategies and project coordination."

        final_summary = overview_text
        if bullet_lines:
            final_summary += "\n\nKey Strategic Outcomes:\n" + "\n".join(bullet_lines[:3])

        return (
            "========== EXECUTIVE SUMMARY ==========\n\n"
            + final_summary
        )

    def generate(self, transcript_file):

        transcript = self.read_transcript(transcript_file)

        summary = self.summarize(transcript)

        output_name = transcript_file.replace(
            ".txt",
            "_summary.txt",
        )

        self.save(output_name, summary)

        return summary

    def save(self, filename, summary):

        SUMMARY_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = SUMMARY_FOLDER / filename

        with open(output, "w", encoding="utf-8") as file:

            file.write(summary)

        print(f"\nSummary saved to:\n{output}")

if __name__ == "__main__":

    generator = SummaryGenerator()

    summary = generator.generate(
        "meeting_001.txt"
    )

    print("\n========== EXECUTIVE SUMMARY ==========\n")

    print(summary)