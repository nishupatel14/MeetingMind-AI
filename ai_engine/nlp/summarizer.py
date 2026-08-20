"""
Meeting Summarizer

Reads a transcript, cleans it, splits it into chunks,
summarizes each chunk, and generates one final meeting summary.
"""

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    SUMMARY_FOLDER,
)

from ai_engine.nlp.summarizer_model import SummarizerLoader
from ai_engine.nlp.transcript_cleaner import TranscriptCleaner


class MeetingSummarizer:

    def __init__(self):
        self.model = SummarizerLoader.get_model()
        self.cleaner = TranscriptCleaner()

    def read_transcript(self, filename):

        transcript_path = TRANSCRIPT_FOLDER / filename

        with open(transcript_path, "r", encoding="utf-8") as file:
            return file.read()

    def split_text(self, text, chunk_size=250):

        words = text.split()

        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def summarize_chunks(self, chunks):

        summaries = []

        total = len(chunks)

        for index, chunk in enumerate(chunks, start=1):

            print(f"Summarizing chunk {index}/{total}...")

            result = self.model(
                chunk,
                max_new_tokens=120,
                min_length=40,
                do_sample=False,
            )

            summaries.append(result[0]["summary_text"])

        return summaries

    def save_summary(self, filename, summary):

        SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)

        output_file = SUMMARY_FOLDER / filename

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(summary)

        print(f"\nSummary saved to:\n{output_file}")


if __name__ == "__main__":

    summarizer = MeetingSummarizer()

    transcript = summarizer.read_transcript("meeting_001.txt")

    transcript = summarizer.cleaner.clean(transcript)

    chunks = summarizer.split_text(transcript)

    summaries = summarizer.summarize_chunks(chunks)

    final_summary = "\n\n".join(summaries)

    print("\n========== FINAL SUMMARY ==========\n")

    print(final_summary)

    summarizer.save_summary(
        "meeting_001_summary.txt",
        final_summary,
    )
    