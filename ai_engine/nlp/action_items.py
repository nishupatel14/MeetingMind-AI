"""
AI Action Item Extractor — Structured Output

Extracts action items with Owner, Task, Deadline, Priority, Context.
"""

from ai_engine.config import (
    TRANSCRIPT_FOLDER,
    ACTION_ITEMS_FOLDER,
)

from ai_engine.nlp.action_model import ActionModelLoader
from ai_engine.nlp.report_validator import ReportValidator

import re


class ActionItemExtractor:

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

    def extract(self, transcript):
        top_keywords = ReportValidator.extract_keywords(transcript, top_n=15)
        keywords_str = ", ".join(top_keywords) if top_keywords else "N/A"

        chunks = self.split_text(transcript, chunk_size=500)
        chunk_actions = []
        total = len(chunks)

        # ------ STEP 1: Extract from chunks ------
        for index, chunk in enumerate(chunks, start=1):
            print(f"Processing action chunk {index}/{total}...")

            prompt = f"""You are MeetingMind AI.

Read ONLY the meeting transcript below.

Key transcript concepts: {keywords_str}

Extract action items and follow-up tasks discussed in the transcript.

RULES:
- Include both formal and informal tasks (e.g. "look into", "develop", "follow up").
- For each action item write:
  Owner: [Exact name from transcript | "The team" | "Not specified"]
  Task: [Clear task description using words from the transcript]
  Deadline: [Exact deadline mentioned | "Not specified"]
  Priority: [High | Medium | Low — based on urgency/emphasis in discussion]
  Context: [Brief context of why this task was mentioned]
- Maximum 5 action items per chunk.
- If NO tasks mentioned, write ONLY: None
- Do NOT use markdown formatting. Plain text only.

Transcript:
{chunk}

Action Items:
"""
            output = ActionModelLoader.generate(prompt)
            print(output)
            print("=" * 60)

            if output.strip().lower() != "none" and "no action" not in output.strip().lower():
                chunk_actions.append(output.strip())

        # ------ STEP 2: Synthesize final list ------
        combined = "\n".join(chunk_actions)

        if not combined.strip():
            return [{
                "owner": "N/A",
                "task": "No action items were assigned during this meeting.",
                "deadline": "N/A",
                "priority": "N/A",
                "context": ""
            }]

        print("Creating final action item list...")

        final_prompt = f"""You are MeetingMind AI.

Key meeting concepts: {keywords_str}

These candidate action items were extracted from a real spoken meeting.

Create ONE clean, deduplicated action item list.

Rules:
- Keep ONLY confirmed follow-up tasks.
- Do NOT include vague discussion points as action items.
- Merge duplicates.
- For each action item write:
  Owner: [Exact name | "The team" | "Not specified"]
  Task: [Clear task description]
  Deadline: [If mentioned | "Not specified"]
  Priority: [High | Medium | Low]
  Context: [Brief context]
- Maximum 6 action items.
- Do NOT use markdown formatting. Plain text only.

Candidate Action Items:
{combined}

Final Action Items:
"""
        output = ActionModelLoader.generate(final_prompt)
        return self._parse_structured_actions(output, transcript)

    def _parse_structured_actions(self, output, transcript):
        """Parse Owner/Task/Deadline/Priority/Context structured output."""
        actions = []
        current = {}

        for line in output.splitlines():
            line = line.strip()
            if not line:
                if current.get("task"):
                    actions.append(current)
                    current = {}
                continue

            line = line.replace("**", "").strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            l_lower = line.lower()

            if l_lower.startswith("owner:"):
                if current.get("task"):
                    actions.append(current)
                val = line[6:].strip()
                current = {
                    "owner": val if val and val.lower() not in {"unknown", "none", "n/a", "unspecified"} else "Not specified",
                    "task": "",
                    "deadline": "Not specified",
                    "priority": "Medium",
                    "context": ""
                }
            elif l_lower.startswith("person:"):
                if current.get("task"):
                    actions.append(current)
                val = line[7:].strip()
                current = {
                    "owner": val if val and val.lower() not in {"unknown", "none", "n/a", "unspecified"} else "Not specified",
                    "task": "",
                    "deadline": "Not specified",
                    "priority": "Medium",
                    "context": ""
                }
            elif l_lower.startswith("task:") and current is not None:
                current["task"] = line[5:].strip()
            elif l_lower.startswith("deadline:") and current is not None:
                val = line[9:].strip()
                if val and val.lower() not in {"not mentioned", "none", "n/a", "not specified", "no deadline"}:
                    current["deadline"] = val
            elif l_lower.startswith("priority:") and current is not None:
                val = line[9:].strip().capitalize()
                if val in {"High", "Medium", "Low"}:
                    current["priority"] = val
            elif l_lower.startswith("context:") and current is not None:
                current["context"] = line[8:].strip()

        if current.get("task"):
            actions.append(current)

        # Filter out items with no real task
        filtered = []
        for a in actions:
            task = a.get("task", "")
            if task and len(task.split()) >= 3 and "no action" not in task.lower():
                filtered.append(a)

        if not filtered:
            filtered = [{
                "owner": "N/A",
                "task": "No action items were assigned during this meeting.",
                "deadline": "N/A",
                "priority": "N/A",
                "context": ""
            }]

        return filtered[:6]

    def extract_file(self, transcript_file):
        transcript = self.read_transcript(transcript_file)
        actions = self.extract(transcript)
        output_name = transcript_file.replace(".txt", "_action_items.txt")
        self.save(output_name, actions)
        return actions

    def save(self, filename, action_items):
        ACTION_ITEMS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = ACTION_ITEMS_FOLDER / filename

        with open(output, "w", encoding="utf-8") as file:
            file.write("========== ACTION ITEMS ==========\n\n")
            for index, item in enumerate(action_items, start=1):
                if isinstance(item, dict):
                    file.write(f"{index}.\n")
                    file.write(f"  Owner: {item.get('owner', 'Not specified')}\n")
                    file.write(f"  Task: {item.get('task', '')}\n")
                    file.write(f"  Deadline: {item.get('deadline', 'Not specified')}\n")
                    file.write(f"  Priority: {item.get('priority', 'Medium')}\n")
                    if item.get('context'):
                        file.write(f"  Context: {item['context']}\n")
                    file.write("\n")
                else:
                    file.write(f"{index}.\n{item.strip()}\n\n")

        print(f"\nAction items saved to:\n{output}")


if __name__ == "__main__":
    extractor = ActionItemExtractor()
    actions = extractor.extract_file("meeting_001.txt")
    print("\n========== ACTION ITEMS ==========\n")
    for a in actions:
        if isinstance(a, dict):
            print(f"Owner: {a['owner']}")
            print(f"Task: {a['task']}")
            print(f"Deadline: {a['deadline']}")
            print(f"Priority: {a['priority']}")
            print(f"Context: {a.get('context', '')}")
            print()
        else:
            print(a)
            print()
