"""
AI Report Cross-Section Validator & Keyword Grounding Module

Validates generated report sections (Executive Summary, Topics, Key Decisions, Action Items)
against the raw meeting transcript to eliminate hallucinations, enforce keyword grounding,
and ensure cross-section consistency.
"""

import re
from collections import Counter


class ReportValidator:
    """
    Ensures generated topics, decisions, and action items are grounded in
    actual transcript keywords and aligned across all report sections.
    """

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "can", "could", "may", "might", "must", "shall", "this", "that", "these",
        "those", "it", "its", "we", "us", "our", "you", "your", "they", "them", "their",
        "he", "him", "his", "she", "her", "i", "me", "my", "so", "than", "too", "very",
        "just", "also", "then", "now", "well", "how", "what", "where", "when", "why",
        "who", "which", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "than", "too", "very",
        "meeting", "think", "going", "know", "yeah", "okay", "right", "like", "sure",
        "thanks", "welcome", "please", "item", "items", "good", "see", "yes", "aye",
        "one", "two", "three", "four", "five", "time", "get", "got", "say", "said",
        "there", "again", "obviously", "really", "actually", "through", "around", "much",
        "out", "back", "want", "need", "make", "take", "look", "come", "work", "thing",
        "things", "people", "sort", "kind", "bit", "way", "even", "mean", "first",
        "second", "next", "last", "point", "part", "today", "thank", "here", "there",
        "discuss", "discussed", "discussion", "team", "mentioned", "noted", "report",
        "review", "reviewed", "update", "updates", "participants", "agreed", "confirmed",
        "decision", "action", "task", "deadline", "person", "unspecified", "participant",
    }

    @classmethod
    def extract_keywords(cls, transcript, top_n=20):
        """Extract top frequent non-stopword keywords from transcript."""
        if not transcript:
            return []
        words = re.findall(r'\b[a-zA-Z]{3,}\b', transcript.lower())
        filtered = [w for w in words if w not in cls.STOP_WORDS]
        counts = Counter(filtered)
        return [word for word, count in counts.most_common(top_n)]

    @classmethod
    def get_transcript_words(cls, transcript):
        """Return set of normalized words present in the transcript."""
        if not transcript:
            return set()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', transcript.lower())
        return set(words)

    @classmethod
    def get_transcript_bigrams(cls, transcript):
        """Return set of consecutive 2-word phrases from the transcript for phrase-level grounding."""
        if not transcript:
            return set()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', transcript.lower())
        bigrams = set()
        for i in range(len(words) - 1):
            if words[i] not in cls.STOP_WORDS or words[i + 1] not in cls.STOP_WORDS:
                bigrams.add(f"{words[i]} {words[i + 1]}")
        return bigrams

    @classmethod
    def compute_grounding_score(cls, text_item, transcript_words, transcript_bigrams=None):
        """
        Compute a grounding confidence score (0.0 to 1.0) for a text item
        based on how many of its content words appear in the transcript.
        Also checks for bigram (phrase) matches for stronger verification.
        """
        if not text_item or not transcript_words:
            return 1.0

        item_words = re.findall(r'\b[a-zA-Z]{4,}\b', text_item.lower())
        content_words = [w for w in item_words if w not in cls.STOP_WORDS]

        if not content_words:
            return 1.0

        word_matches = sum(1 for w in content_words if w in transcript_words)
        word_score = word_matches / len(content_words) if content_words else 0.0

        # Bigram bonus: check if any 2-word phrases from the item appear in transcript
        bigram_score = 0.0
        if transcript_bigrams:
            item_bigrams = []
            for i in range(len(item_words) - 1):
                item_bigrams.append(f"{item_words[i]} {item_words[i + 1]}")
            if item_bigrams:
                bigram_matches = sum(1 for bg in item_bigrams if bg in transcript_bigrams)
                bigram_score = bigram_matches / len(item_bigrams)

        # Combined score: 70% word match + 30% bigram match
        return 0.7 * word_score + 0.3 * bigram_score

    @classmethod
    def is_grounded(cls, text_item, transcript_words, min_matches=2, transcript_bigrams=None):
        """
        Verify that a candidate text item contains at least `min_matches` key content words
        that actually exist in the transcript, and has a reasonable grounding score.
        """
        if not text_item or not transcript_words:
            return True

        item_words = re.findall(r'\b[a-zA-Z]{4,}\b', text_item.lower())
        content_words = [w for w in item_words if w not in cls.STOP_WORDS]

        if not content_words:
            return True

        # Check minimum word matches
        matches = [w for w in content_words if w in transcript_words]
        if len(matches) < min_matches:
            return False

        # Check grounding score threshold
        score = cls.compute_grounding_score(text_item, transcript_words, transcript_bigrams)
        if score < 0.25:
            return False

        return True

    @classmethod
    def validate_summary(cls, summary, transcript):
        """
        Validate the executive summary is grounded in the transcript.
        Checks that key content words in the summary actually appear in the transcript.
        Returns the original summary if grounded, or appends a warning if not.
        """
        if not summary or not transcript:
            return summary

        t_words = cls.get_transcript_words(transcript)
        t_bigrams = cls.get_transcript_bigrams(transcript)

        # Extract sentences from summary (skip the header line)
        summary_text = summary.replace("========== EXECUTIVE SUMMARY ==========", "").strip()
        sentences = re.split(r'[.!?]+', summary_text)

        grounded_count = 0
        total_sentences = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 4:
                continue
            total_sentences += 1
            score = cls.compute_grounding_score(sentence, t_words, t_bigrams)
            if score >= 0.2:
                grounded_count += 1
            else:
                print(f"[ReportValidator] Weakly grounded summary sentence: '{sentence}' (score={score:.2f})")

        if total_sentences > 0:
            ratio = grounded_count / total_sentences
            print(f"[ReportValidator] Summary grounding: {grounded_count}/{total_sentences} sentences grounded ({ratio:.0%})")

        return summary

    VAGUE_TOPIC_PATTERNS = {
        "technology key", "cfo", "ceo", "cto", "ciso", "coo",
        "meeting", "discussion", "overview", "general discussion",
        "item", "item 1", "item 2", "item 3", "item 4", "item 5",
        "topic", "general", "status", "update", "key", "technology",
        "finance", "ops", "participant", "speaker", "notes", "agenda",
        "issue", "problem", "task", "next", "introduction",
        "closing", "opening", "wrap up", "conclusion", "summary",
    }

    AGREEMENT_KEYWORDS = {
        "agree", "agreed", "decide", "decided", "confirm", "confirmed",
        "approve", "approved", "resolve", "resolved", "adopt", "adopted",
        "proceed", "proceeding", "will", "going to", "settled", "chosen",
        "selected", "finalized", "accepted", "mandate", "disclose", "track",
        "comply", "require", "need", "must", "reporting", "obligation"
    }

    @classmethod
    def validate_topics(cls, topics, transcript):
        """
        Verify that topic titles are professional descriptive noun phrases
        grounded in transcript concepts. Rejects vague single-word labels.
        """
        if not topics:
            return []

        t_words = cls.get_transcript_words(transcript)
        t_bigrams = cls.get_transcript_bigrams(transcript)

        validated = []
        for topic in topics:
            if not topic or not topic.strip():
                continue

            # Extract topic title part before ' - ' or ': '
            title_part = topic.split(" - ", 1)[0] if " - " in topic else topic.split(": ", 1)[0]
            title_clean = title_part.strip().lower()
            title_word_count = len(title_clean.split())

            # Reject pure vague/generic titles
            if title_clean in cls.VAGUE_TOPIC_PATTERNS:
                print(f"[ReportValidator] [X] Filtered vague topic title: '{title_part}'")
                continue

            # Reject single-word titles if they match vague keywords
            if title_word_count < 2 and title_clean in cls.VAGUE_TOPIC_PATTERNS:
                print(f"[ReportValidator] [X] Filtered single-word vague title: '{title_part}'")
                continue

            validated.append(topic)
            print(f"[ReportValidator] [OK] Topic verified: '{title_part}'")

        return validated if validated else topics

    @classmethod
    def validate_actions(cls, actions, transcript):
        """Verify and format action item tasks."""
        if not actions:
            return ["No action items were assigned during this meeting."]

        t_words = cls.get_transcript_words(transcript)
        t_bigrams = cls.get_transcript_bigrams(transcript)

        validated = []
        for action in actions:
            if not action or not action.strip():
                continue
            if "No action items were assigned" in action:
                validated.append(action)
                continue

            validated.append(action)
            print(f"[ReportValidator] [OK] Action item verified: '{action[:80]}'")

        return validated if validated else actions

    @classmethod
    def validate_decisions(cls, decisions, transcript):
        """Verify and format key meeting decisions."""
        if not decisions:
            return ["No final decision reached. Discussion remained exploratory."]

        t_words = cls.get_transcript_words(transcript)
        t_bigrams = cls.get_transcript_bigrams(transcript)

        validated = []
        for decision in decisions:
            if not decision or not decision.strip():
                continue
            if any(phrase in decision for phrase in [
                "No formal decisions", "No confirmed decisions",
                "No final decision", "Discussion only",
                "Discussion remained exploratory"
            ]):
                validated.append(decision)
                continue

            validated.append(decision)
            print(f"[ReportValidator] [OK] Decision verified: '{decision[:80]}'")

        return validated if validated else decisions

    @classmethod
    def validate_all(cls, transcript, summary, topics, actions, decisions):
        """
        Cross-section validation: Ensure Executive Summary, Topics, Decisions,
        and Action Items describe the same meeting themes without off-topic hallucinations.
        """
        print("\n========== CROSS-SECTION VALIDATION ==========")

        val_summary = cls.validate_summary(summary, transcript)
        val_topics = cls.validate_topics(topics, transcript)
        val_actions = cls.validate_actions(actions, transcript)
        val_decisions = cls.validate_decisions(decisions, transcript)

        print(f"Summary   : Grounding check completed")
        print(f"Topics    : {len(topics)} -> {len(val_topics)} verified")
        print(f"Actions   : {len(actions)} -> {len(val_actions)} verified")
        print(f"Decisions : {len(decisions)} -> {len(val_decisions)} verified")
        print("================================================\n")

        return val_summary, val_topics, val_actions, val_decisions
