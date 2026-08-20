import subprocess
import sys
from pathlib import Path


def test_audio_info_script_runs():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "ai_engine/preprocessing/audio_info.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Duration:" in result.stdout
    assert "Sample rate:" in result.stdout

