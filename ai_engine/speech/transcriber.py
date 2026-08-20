"""
Speech-to-Text Transcriber

Converts meeting audio into text and JSON transcripts
using faster-whisper (CTranslate2 engine) on CPU.

Faster-whisper yields segments as a streaming generator,
allowing live transcript updates during long audio files.
"""

import json
import time

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_VAD_FILTER,
    WHISPER_MIN_SILENCE_DURATION_MS,
)
from ai_engine.speech.whisper_model import WhisperLoader


class MeetingTranscriber:
    """
    Handles speech-to-text transcription using faster-whisper.
    """

    def __init__(self):
        """
        Load faster-whisper model (CPU, INT8).
        """
        self.model = WhisperLoader.get_model()

    def transcribe(self, audio_path, on_segment=None):
        """
        Transcribe an audio file using faster-whisper streaming segments.

        Parameters
        ----------
        audio_path : str
            Path to WAV audio file.
        on_segment : callable, optional
            Callback invoked with each transcribed text segment in real-time.

        Returns
        -------
        tuple
            transcript_lines (list[str]), transcript_json (list[dict])
        """

        print("\nStarting transcription...")
        start_time = time.time()

        transcript_lines = []
        transcript_json = []
        detected_lang = WHISPER_LANGUAGE

        try:
            # faster-whisper returns a generator — segments stream in real-time
            segments, info = self.model.transcribe(
                audio_path,
                language=WHISPER_LANGUAGE,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters={
                    "min_silence_duration_ms": WHISPER_MIN_SILENCE_DURATION_MS,
                },
            )


            detected_lang = info.language if info.language else "en"

            print("=" * 50)
            print(f"Detected language : {detected_lang}")
            print(f"Audio duration    : {info.duration:.1f} seconds ({info.duration/60:.1f} min)")
            print("=" * 50)

            for segment in segments:
                text = str(segment.text).strip()
                start_ts = float(segment.start)
                end_ts = float(segment.end)

                if text:
                    print(f"[{start_ts:.2f} - {end_ts:.2f}] {text}")
                    transcript_lines.append(text)

                    if on_segment is not None:
                        on_segment(text)

                    transcript_json.append(
                        {
                            "start": round(start_ts, 2),
                            "end": round(end_ts, 2),
                            "text": text,
                        }
                    )

        except Exception as e:
            print(f"\n[MeetingTranscriber] Exception during transcription: {e}")
            raise e

        end_time = time.time()
        elapsed = round(end_time - start_time, 2)

        print("\n========== Transcription Complete ==========")
        print(f"Language          : {detected_lang}")
        print(f"Segments          : {len(transcript_lines)}")
        print(f"Processing Time   : {elapsed} seconds ({elapsed/60:.1f} min)")
        print("============================================\n")

        return transcript_lines, transcript_json

    def save_txt(self, transcript, filename):
        """
        Save transcript as TXT.
        """
        TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)

        output_file = TRANSCRIPT_FOLDER / f"{filename}.txt"

        with open(output_file, "w", encoding="utf-8") as file:
            for line in transcript:
                file.write(line + "\n")

        print(f"TXT Saved : {output_file}")

    def save_json(self, transcript_json, filename):
        """
        Save transcript as JSON.
        """
        TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)

        output_file = TRANSCRIPT_FOLDER / f"{filename}.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(
                transcript_json,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"JSON Saved: {output_file}")


if __name__ == "__main__":

    AUDIO_FILE = "datasets/processed/wav/meeting_001.wav"

    transcriber = MeetingTranscriber()

    transcript, transcript_json = transcriber.transcribe(AUDIO_FILE)

    transcriber.save_txt(transcript, "meeting_001")
    transcriber.save_json(transcript_json, "meeting_001")

    print("\nMeeting transcription completed successfully!")
