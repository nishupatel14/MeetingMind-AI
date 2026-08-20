"""
Noise Reduction
"""

from pathlib import Path

import noisereduce as nr
import soundfile as sf


class NoiseReducer:

    def reduce_noise(
        self,
        input_audio,
        output_audio,
    ):

        print("\nReducing background noise...")

        audio, sample_rate = sf.read(input_audio)

        cleaned_audio = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
        )

        Path(output_audio).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            output_audio,
            cleaned_audio,
            sample_rate,
        )

        print("Noise reduction completed.")

        return output_audio


if __name__ == "__main__":

    reducer = NoiseReducer()

    reducer.reduce_noise(
        "datasets/processed/wav/meeting_001.wav",
        "datasets/processed/wav/meeting_001_clean.wav",
    )
    