"""
AI Decision Detector — Structured Output

Extracts confirmed decisions with context and responsible person.
"""

print("DECISION GENERATOR STARTED")

import re

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    DECISIONS_FOLDER,
)

from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.report_validator import ReportValidator


class DecisionDetector:

    def __init__(self):
        pass

    def read_transcript(self, filename):
        path = TRANSCRIPT_FOLDER / filename
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def split_text(self, text, chunk_size=900):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def detect(self, transcript):
        print("ENTERED detect()")
        top_keywords = ReportValidator.extract_keywords(transcript, top_n=15)
        keywords_str = ", ".join(top_keywords) if top_keywords else "N/A"

        chunks = self.split_text(transcript, chunk_size=500)
        chunk_decisions = []
        total = len(chunks)

        # ------ STEP 1: Extract from chunks ------
        for index, chunk in enumerate(chunks, start=1):
            print(f"Processing decision chunk {index}/{total}")

            prompt = f"""You are MeetingMind AI.

Read ONLY the meeting transcript below.

Key transcript concepts: {keywords_str}

Extract ONLY confirmed decisions where a speaker CLEARLY decided, approved, or committed to something.

STRICT RULES:
- A DECISION means someone explicitly concluded, approved, committed, or directed something concrete.
- Ideas, suggestions, explanations, opinions are NOT decisions.
- For each decision, write:
  Decision: [What was decided]
  Context: [Why it was decided or what led to it]
  Responsible: [Person or team name if mentioned, otherwise "Not specified"]
- Maximum 3 decisions per chunk.
- If no decisions exist, write ONLY:
  No decisions identified.
- Do NOT use markdown formatting. Plain text only.

Transcript:
{chunk}

Confirmed Decisions:
"""
            output = ActionModelLoader.generate(prompt)
            print(output)
            print("=" * 60)

            lower_out = output.strip().lower()
            if "no decisions" not in lower_out and "no final decision" not in lower_out and "discussion remained exploratory" not in lower_out:
                chunk_decisions.append(output.strip())

        # ------ STEP 2: Synthesize final list ------
        combined = "\n".join(chunk_decisions)

        if not combined.strip():
            return [{
                "decision": "No explicit decisions were identified in the meeting.",
                "context": "Discussion remained exploratory.",
                "responsible": "N/A"
            }]

        print("Creating final decision list...")

        final_prompt = f"""You are MeetingMind AI, acting as a Senior Business Analyst.

Key meeting concepts: {keywords_str}

The candidate decisions below were extracted from a meeting transcript.

Create a clean, deduplicated list of CONFIRMED decisions.

STRICT RULES:
- Only include decisions where someone CLEARLY decided something.
- Never invent decisions. Never convert suggestions into decisions.
- For each decision write:
  Decision: [What was decided]
  Context: [Why or what led to it]
  Responsible: [Person/team if mentioned, otherwise "Not specified"]
- Maximum 5 decisions.
- If no real decisions exist, write:
  Decision: No explicit decisions were identified in the meeting.
  Context: Discussion remained exploratory.
  Responsible: N/A
- Do NOT use markdown formatting. Plain text only.

Candidate Decisions:
{combined}

Final Decisions:
"""
        output = ActionModelLoader.generate(final_prompt)
        return self._parse_structured_decisions(output, transcript)

    def _parse_structured_decisions(self, output, transcript):
        """Parse Decision/Context/Responsible structured output."""
        decisions = []
        current = {}

        for line in output.splitlines():
            line = line.strip()
            if not line:
                if current.get("decision"):
                    decisions.append(current)
                    current = {}
                continue

            # Remove markdown
            line = re.sub(r'^#{1,4}\s*', '', line).strip()
            line = line.replace("**", "").strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()

            l_lower = line.lower()
            if l_lower.startswith("decision:"):
                if current.get("decision"):
                    decisions.append(current)
                current = {
                    "decision": line[9:].strip(),
                    "context": "",
                    "responsible": "Not specified"
                }
            elif l_lower.startswith("context:") and current:
                current["context"] = line[8:].strip()
            elif l_lower.startswith("responsible:") and current:
                current["responsible"] = line[12:].strip()
            elif l_lower.startswith("reason:") and current:
                current["context"] = line[7:].strip()

        if current.get("decision"):
            decisions.append(current)

        # Filter out non-decisions
        filtered = []
        no_decision_phrases = {
            "no decisions", "no explicit decisions", "no final decision",
            "discussion remained exploratory", "no confirmed decisions"
        }
        for d in decisions:
            dec_lower = d["decision"].lower()
            if any(phrase in dec_lower for phrase in no_decision_phrases):
                if not filtered:
                    filtered.append(d)
                continue
            if len(d["decision"].split()) >= 4:
                filtered.append(d)

        if not filtered:
            filtered = [{
                "decision": "No explicit decisions were identified in the meeting.",
                "context": "Discussion remained exploratory.",
                "responsible": "N/A"
            }]

        return filtered[:5]

    def detect_file(self, transcript_file):
        transcript = self.read_transcript(transcript_file)
        decisions = self.detect(transcript)
        output_name = transcript_file.replace(".txt", "_decisions.txt")
        self.save(output_name, decisions)
        return decisions

    def save(self, filename, decisions):
        DECISIONS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = DECISIONS_FOLDER / filename

        with open(output, "w", encoding="utf-8") as file:
            file.write("========== KEY DECISIONS ==========\n\n")
            if isinstance(decisions[0], dict) if decisions else False:
                for idx, d in enumerate(decisions, 1):
                    file.write(f"{idx}. Decision: {d['decision']}\n")
                    if d.get('context'):
                        file.write(f"   Context: {d['context']}\n")
                    if d.get('responsible'):
                        file.write(f"   Responsible: {d['responsible']}\n")
                    file.write("\n")
            else:
                for decision in decisions:
                    if isinstance(decision, str) and decision.strip():
                        file.write(f"• {decision.strip()}\n")

        print(f"\nDecisions saved to:\n{output}")


if __name__ == "__main__":
    detector = DecisionDetector()
    decisions = detector.detect_file("meeting_001.txt")
    print("\n========== DECISIONS ==========\n")
    for d in decisions:
        if isinstance(d, dict):
            print(f"Decision: {d['decision']}")
            print(f"Context: {d.get('context', '')}")
            print(f"Responsible: {d.get('responsible', 'Not specified')}")
            print()
        else:
            print(f"• {d}")
