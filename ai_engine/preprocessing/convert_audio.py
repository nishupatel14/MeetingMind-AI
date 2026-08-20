"""
Audio Converter
"""

from pathlib import Path

from pydub import AudioSegment


class AudioConverter:

    def convert(self, input_file, output_file):

        print("\nConverting audio...")

        audio = AudioSegment.from_file(input_file)

        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audio.export(
            output_file,
            format="wav",
        )

        print("Audio converted successfully.")

        return output_file


if __name__ == "__main__":

    converter = AudioConverter()

    converter.convert(
        "datasets/raw/meetings/meeting_001.mp3",
        "datasets/processed/wav/meeting_001.wav",
    )
