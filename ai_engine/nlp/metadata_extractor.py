"""
Meeting Metadata Extractor
"""

import re

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
)


class MetadataExtractor:

    def read_transcript(self, filename):

        with open(
            TRANSCRIPT_FOLDER / filename,
            "r",
            encoding="utf-8",
        ) as file:

            return file.read()

    def extract(self, transcript, meeting_id="Unknown"):

        words = transcript.split()

        sentences = re.split(r"[.!?]+", transcript)

        metadata = {

            "meeting_id": meeting_id,

            "language": "English",

            "word_count": len(words),

            "sentence_count": len(
                [s for s in sentences if s.strip()]
        ),

        "reading_time": f"{round(len(words) / 200, 1)} min",

        "duration_minutes": f"{round(len(words) / 130, 1)} min",

}

        return metadata


if __name__ == "__main__":

    extractor = MetadataExtractor()

    transcript = extractor.read_transcript(
        "meeting_001.txt"
    )

    metadata = extractor.extract(
        transcript,
        "meeting_001",
    )

    print("\n========== METADATA ==========\n")

    for key, value in metadata.items():

        print(f"{key}: {value}")