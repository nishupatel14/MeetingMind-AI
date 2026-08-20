import sys
import os

# Force UTF-8 output on Windows to prevent charmap encoding errors
# for any Unicode characters printed anywhere in the application.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
from pathlib import Path
import re
import shutil
from threading import Lock
from uuid import uuid4
from ai_engine.nlp.metadata_extractor import MetadataExtractor

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

from ai_engine.pipeline import MeetingPipeline
from ai_engine.audio_converter import convert_to_wav

from ai_engine.config import (
    RAW_DATASET,
    WAV_FOLDER,
    TRANSCRIPT_FOLDER,
    SUMMARY_FOLDER,
    TOPICS_FOLDER,
    ACTION_ITEMS_FOLDER,
    DECISIONS_FOLDER,
    PDF_REPORT_FOLDER,
    REPORT_FOLDER,
)

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

live_transcript = {
    "is_transcribing": False,
    "text": "",
    "segment_count": 0,
    "message": "Waiting for an upload.",
}

live_transcript_lock = Lock()


# -------------------------------------------------------
# Live Transcript
# -------------------------------------------------------

def reset_live_transcript(message):
    with live_transcript_lock:
        live_transcript.update({
            "is_transcribing": True,
            "text": "",
            "segment_count": 0,
            "message": message,
        })


def add_live_segment(segment):
    with live_transcript_lock:
        live_transcript["text"] += segment + "\n"
        live_transcript["segment_count"] += 1
        live_transcript["message"] = "Transcribing audio..."


def finish_live_transcript(message):
    with live_transcript_lock:
        live_transcript["is_transcribing"] = False
        live_transcript["message"] = message


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def read_output_file(path):

    if path.exists():
        return path.read_text(encoding="utf-8")

    return "Not Available"


def sanitize_meeting_name(stem):
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', stem).strip()
    cleaned = cleaned.strip(". ")
    return cleaned if cleaned else "meeting"


def generate_unique_report_id(base_name, extension):
    candidate = base_name
    count = 1
    while True:
        report_exists = (REPORT_FOLDER / f"{candidate}_report.json").exists()
        wav_exists = (WAV_FOLDER / f"{candidate}.wav").exists()
        upload_exists = (UPLOAD_FOLDER / f"{candidate}{extension}").exists()

        if not (report_exists or wav_exists or upload_exists):
            return candidate

        candidate = f"{base_name} ({count})"
        count += 1


def get_report_id():

    report_id = request.args.get("id", "")

    if not report_id or ".." in report_id or "/" in report_id or "\\" in report_id:
        return None

    return report_id


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/transcription-status")
def transcription_status():

    with live_transcript_lock:
        return jsonify(live_transcript)


@app.route("/upload", methods=["POST"])
def upload_file():

    print("\n==============================")
    print("UPLOAD REQUEST RECEIVED")
    print("==============================")

    reset_live_transcript("Preparing uploaded audio...")

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "message": "No file uploaded."
        })

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "Please select an audio file."
        })

    extension = Path(file.filename).suffix.lower()

    if extension not in {".mp3", ".wav", ".m4a", ".flac"}:

        return jsonify({
            "success": False,
            "message": "Unsupported file format."
        })

    raw_stem = Path(file.filename).stem
    base_name = sanitize_meeting_name(raw_stem)
    report_id = generate_unique_report_id(base_name, extension)

    filename = report_id + extension

    upload_path = UPLOAD_FOLDER / filename

    file.save(upload_path)

    RAW_DATASET.mkdir(parents=True, exist_ok=True)
    shutil.copy(upload_path, RAW_DATASET / filename)

    try:

        wav_path = WAV_FOLDER / f"{report_id}.wav"

        with live_transcript_lock:
            live_transcript["message"] = "Converting audio..."

        convert_to_wav(upload_path, wav_path)

        pipeline = MeetingPipeline()

        pipeline.run(
            str(wav_path),
            output_name=report_id,
            on_transcript_update=add_live_segment,
        )

        finish_live_transcript("Report generated successfully.")

    except Exception as e:

        import traceback

        traceback.print_exc()

        finish_live_transcript(str(e))

        return jsonify({
            "success": False,
            "message": str(e)
        })

    return jsonify({
        "success": True,
        "report_id": report_id,
    })


# -------------------------------------------------------
# Result Page
# -------------------------------------------------------

@app.route("/result/<report_id>")
def result(report_id):

    report_file = REPORT_FOLDER / f"{report_id}_report.json"

    if not report_file.exists():
        return "Report not found.", 404

    with open(report_file, "r", encoding="utf-8") as file:
        report = json.load(file)

    return render_template(
        "result.html",
        report_id=report_id,
        summary=report.get("executive_summary", ""),
        topics=report.get("discussion_topics", []),
        key_discussion=report.get("key_discussion", []),
        actions=report.get("action_items", []),
        decisions=report.get("key_decisions", []),
        open_questions=report.get("open_questions", []),
        metadata=report.get("metadata", {}),
    )


# -------------------------------------------------------
# Downloads
# -------------------------------------------------------

@app.route("/download/pdf")
def download_pdf():

    report_id = get_report_id()

    if report_id is None:
        return "Report not found.", 404

    return send_file(
        PDF_REPORT_FOLDER / f"{report_id}_report.pdf",
        as_attachment=True,
    )


@app.route("/download/summary")
def download_summary():

    report_id = get_report_id()

    if report_id is None:
        return "Report not found.", 404

    return send_file(
        SUMMARY_FOLDER / f"{report_id}_summary.txt",
        as_attachment=True,
    )


@app.route("/download/transcript")
def download_transcript():

    report_id = get_report_id()

    if report_id is None:
        return "Report not found.", 404

    return send_file(
        TRANSCRIPT_FOLDER / f"{report_id}.txt",
        as_attachment=True,
    )


# -------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=False,
        threaded=True,
        port=5000,
    )
