from pathlib import Path

import librosa


class AudioAnalyzer:
    def __init__(self, audio_path: str):
        self.audio_path = Path(audio_path)

    def analyze(self) -> None:
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        y, sr = librosa.load(str(self.audio_path), sr=None, mono=True)
        duration = len(y) / sr
        channels = 1

        print(f"Audio file: {self.audio_path}")
        print(f"Duration: {duration:.2f}s")
        print(f"Sample rate: {sr} Hz")
        print(f"Channels: {channels}")


if __name__ == "__main__":

    audio_path = "datasets/processed/wav/meeting_001.wav"

    analyzer = AudioAnalyzer(audio_path)

    analyzer.analyze()
    