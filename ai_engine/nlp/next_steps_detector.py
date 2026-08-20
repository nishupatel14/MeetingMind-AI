"""
Next Steps Detector

Synthesizes concrete next steps from decisions, action items,
and open questions already extracted from the meeting.
"""

import re
from ai_engine.config import PROCESSED_DATASET
from ai_engine.nlp.action_model import ActionModelLoader

NEXT_STEPS_FOLDER = PROCESSED_DATASET / "next_steps"


class NextStepsDetector:

    def detect(self, transcript, decisions=None, actions=None, open_questions=None):
        """Generate next steps based on meeting analysis results."""

        # Build context from already-extracted sections
        context_parts = []

        if decisions:
            dec_text = []
            for d in decisions:
                if isinstance(d, dict):
                    dec_text.append(d.get("decision", ""))
                else:
                    dec_text.append(str(d))
            if dec_text:
                context_parts.append("Decisions made:\n" + "\n".join(f"- {d}" for d in dec_text if d))

        if actions:
            act_text = []
            for a in actions:
                if isinstance(a, dict):
                    act_text.append(a.get("task", ""))
                else:
                    act_text.append(str(a))
            if act_text:
                context_parts.append("Action items:\n" + "\n".join(f"- {a}" for a in act_text if a))

        if open_questions:
            context_parts.append("Open questions:\n" + "\n".join(f"- {q}" for q in open_questions))

        context = "\n\n".join(context_parts)

        if not context.strip():
            return ["Review meeting outcomes and plan follow-up actions."]

        prompt = f"""You are MeetingMind AI, an executive meeting analyst.

Based on the meeting analysis below, identify 3 to 5 concrete NEXT STEPS
that need to happen after this meeting.

RULES:
1. Each next step must be a specific, actionable sentence.
2. Base next steps on the decisions, action items, and open questions provided.
3. Do NOT invent steps not connected to the meeting.
4. Do NOT use markdown formatting. Plain text only.
5. Format each step on a new line starting with a dash (-).

Meeting Analysis:
{context}

Next Steps:
"""
        output = ActionModelLoader.generate(prompt)

        items = []
        for line in output.splitlines():
            line = re.sub(r"^[\s•\-\*\d\.]+", "", line).strip()
            line = line.replace("**", "").strip()
            if line and len(line.split()) >= 4:
                items.append(line)

        if not items:
            items = ["Review meeting outcomes and plan follow-up actions."]

        return items[:5]

    def save(self, filename, items):
        NEXT_STEPS_FOLDER.mkdir(parents=True, exist_ok=True)
        output = NEXT_STEPS_FOLDER / filename
        with open(output, "w", encoding="utf-8") as f:
            f.write("========== NEXT STEPS ==========\n\n")
            for item in items:
                f.write(f"• {item}\n")
        print(f"\nNext steps saved to:\n{output}")
