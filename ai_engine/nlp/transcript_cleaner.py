"""
Transcript Cleaner

Cleans the Whisper transcript before summarization.
"""

import re

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
)


class TranscriptCleaner:

    def __init__(self):

        self.filler_words = [
            "um",
            "uh",
            "hmm",
            "erm",
            "ah",
            "you know",
            "you can say",
            "you can search it out",
            "like",
            "yeah",
            "okay",
            "ok",
            "right",
            "actually",
            "basically",
            "literally",
            "well",
            "so",
            "i mean",
            "kind of",
            "sort of",
            "whatever it is",
            "as such",
            "in terms of",
            "you know what i mean",
            "that's what the thing is",
            "simple example is",
            "you can see",
        ]

    def read_transcript(self, filename):

        path = TRANSCRIPT_FOLDER / filename

        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def remove_fillers(self, text):

        for word in self.filler_words:

            pattern = r"\b" + re.escape(word) + r"\b"

            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE,
            )

        return text

    def remove_duplicate_words(self, text):

        text = re.sub(
            r"\b(\w+)( \1\b)+",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

        return text

    def remove_extra_spaces(self, text):

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def clean(self, text):

        text = self.remove_fillers(text)

        text = self.remove_duplicate_words(text)

        text = self.remove_extra_spaces(text)

        return text

    # ------------------------------------
    # NEW FUNCTION
    # ------------------------------------

    def split_text(self, text, chunk_size=350):

        words = text.split()

        chunks = []

        for i in range(0, len(words), chunk_size):

            chunks.append(
                " ".join(words[i:i + chunk_size])
            )

        return chunks


if __name__ == "__main__":

    cleaner = TranscriptCleaner()

    transcript = cleaner.read_transcript("meeting_001.txt")

    cleaned = cleaner.clean(transcript)

    print(cleaned[:1500])
    