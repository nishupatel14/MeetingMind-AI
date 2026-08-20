"""
MeetingMind AI Pipeline
"""

from ai_engine.speech.transcriber import MeetingTranscriber
from ai_engine.nlp.summary_generator import SummaryGenerator
from ai_engine.nlp.action_items import ActionItemExtractor
from ai_engine.nlp.topic_detector import TopicDetector
from ai_engine.nlp.decision_detector import DecisionDetector
from ai_engine.nlp.open_questions_detector import OpenQuestionsDetector
from ai_engine.nlp.key_insights_detector import KeyInsightsDetector
from ai_engine.nlp.key_discussion_detector import KeyDiscussionDetector
from ai_engine.report.report_generator import ReportGenerator
from ai_engine.nlp.report_validator import ReportValidator


class MeetingPipeline:

    def __init__(self):

        self.transcriber = MeetingTranscriber()
        self.summary = SummaryGenerator()
        self.actions = ActionItemExtractor()
        self.topics = TopicDetector()
        self.decisions = DecisionDetector()
        self.open_questions = OpenQuestionsDetector()
        self.key_insights = KeyInsightsDetector()
        self.key_discussion = KeyDiscussionDetector()
        self.report = ReportGenerator()

        try:
            from ai_engine.report.pdf_report_generator import PDFReportGenerator
            self.pdf = PDFReportGenerator()
        except Exception:
            self.pdf = None

    def run(
        self,
        audio_file,
        output_name="meeting_001",
        on_transcript_update=None,
        on_status_update=None,
    ):

        print("PIPELINE RUN CALLED")
        print("\n========== PIPELINE STARTED ==========\n")

        # -----------------------------------
        # 0. Speech To Text
        # -----------------------------------
        if on_status_update:
            on_status_update("🎤 Converting speech to text...")

        print("0. Speech To Text...")

        transcript, transcript_json = self.transcriber.transcribe(
            audio_file,
            on_segment=on_transcript_update,
        )

        self.transcriber.save_txt(transcript, output_name)
        self.transcriber.save_json(transcript_json, output_name)

        transcript_str = "\n".join(transcript)

        import torch

        def safe_empty_cache():
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # -----------------------------------
        # 1. Executive Summary
        # -----------------------------------
        if on_status_update:
            on_status_update("📝 Generating executive summary...")
        print("1. Generating Summary...")
        summary = self.summary.summarize(transcript_str)
        safe_empty_cache()

        # -----------------------------------
        # 2. Discussion Topics (Rich grouped format)
        # -----------------------------------
        if on_status_update:
            on_status_update("📌 Detecting detailed meeting topics...")
        print("2. Detecting Topics...")
        topics = self.topics.detect_topics(transcript_str)
        safe_empty_cache()

        # -----------------------------------
        # 3. Key Decisions (Structured format)
        # -----------------------------------
        if on_status_update:
            on_status_update("⚖ Detecting key decisions...")
        print("3. Detecting Decisions...")
        decisions = self.decisions.detect(transcript_str)
        safe_empty_cache()

        # -----------------------------------
        # 4. Action Items (Structured format)
        # -----------------------------------
        if on_status_update:
            on_status_update("✅ Extracting action items...")
        print("4. Extracting Action Items...")
        actions = self.actions.extract(transcript_str)
        safe_empty_cache()

        # Save output files
        self.summary.save(f"{output_name}_summary.txt", summary)
        self.topics.save(f"{output_name}_topics.txt", topics)
        self.decisions.save(f"{output_name}_decisions.txt", decisions)
        self.actions.save(f"{output_name}_action_items.txt", actions)

        # -----------------------------------
        # 6. Key Discussion Points (Timestamped facts/concerns/opinions)
        # -----------------------------------
        if on_status_update:
            on_status_update("🗣️ Extracting key discussion points & timestamps...")
        print("6. Extracting Key Discussion...")
        key_discussion = self.key_discussion.detect(transcript_json)
        self.key_discussion.save(f"{output_name}_key_discussion.json", key_discussion)
        safe_empty_cache()

        # -----------------------------------
        # 10. JSON Report
        # -----------------------------------
        if on_status_update:
            on_status_update("📂 Creating meeting report...")
        print("10. Creating JSON Report...")
        self.report.generate(output_name)

        # -----------------------------------
        # 11. PDF Report
        # -----------------------------------
        if on_status_update:
            on_status_update("📄 Creating PDF report...")
        print("11. Creating PDF Report...")

        if self.pdf is not None:
            self.pdf.generate(output_name)
        else:
            print("PDF generation skipped.")

        # -----------------------------------
        # Finished
        # -----------------------------------
        if on_status_update:
            on_status_update("✅ Meeting analysis completed successfully!")

        print("\n========== PIPELINE FINISHED ==========")


if __name__ == "__main__":
    print("Run app.py to use MeetingMind AI.")