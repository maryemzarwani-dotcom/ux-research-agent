"""Tagging pipeline: the LLM PROPOSES quotes + tags, our code VALIDATES them.

Principle: never trust the model with anything code can verify.
  - quote must be a verbatim substring of a participant turn   -> else rejected
  - tags must be in the taxonomy                                -> else dropped
  - char spans are computed by us, never by the model

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python tagger.py P01_sample.txt
"""
import json
import os
import sys
from collections import Counter

import anthropic

from parser import parse_transcript, verify_quote

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
client = anthropic.Anthropic()

TAXONOMY = {
    "pain_point": "A frustration, obstacle, or thing that goes wrong.",
    "workaround": "Something the participant does to get around a problem.",
    "goal": "What the participant is trying to achieve.",
    "feature_request": "An explicit or implied wish for something to exist or change.",
    "confusion": "The participant did not understand something or was misled.",
    "positive_reaction": "Something the participant likes or finds helpful.",
    "pricing": "Cost, subscription, trial, or value-for-money concerns.",
    "motivation": "What keeps the participant going or makes them quit.",
    "social": "Interaction (or lack of it) with peers, teachers, or community.",
    "accessibility": "Needs related to disability, readability, or assistive tech.",
}

SYSTEM = f"""You are a UX research assistant doing first-pass coding of an interview transcript.

Extract the participant's quotes that carry research-relevant signal and tag them.

Rules:
- Quotes MUST be copied character-for-character from the transcript. No paraphrasing,
  no ellipses, no fixing grammar, no speaker label. A quote is 1-3 sentences.
- Only quote the participant, never the Interviewer.
- Skip small talk and neutral filler. Fewer, stronger quotes beat many weak ones.
- Assign 1-3 tags per quote from this taxonomy ONLY:
{json.dumps(TAXONOMY, indent=2)}
- Give a confidence from 0 to 1 for each quote's tags.
- If a quote expresses an opinion that contradicts typical sentiment (e.g. likes something
  others dislike), still extract it. Do not smooth over disagreement.
"""

TOOL = {
    "name": "record_quotes",
    "description": "Record the extracted quotes with their tags.",
    "input_schema": {
        "type": "object",
        "properties": {
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Verbatim quote"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(TAXONOMY)},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["text", "tags", "confidence"],
                },
            }
        },
        "required": ["quotes"],
    },
}


def propose(transcript: str) -> list[dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_quotes"},
        messages=[{"role": "user", "content": f"Transcript:\n\n{transcript}"}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.input["quotes"]


def validate(proposed: list[dict], transcript: str, turns) -> tuple[list[dict], Counter]:
    """Keep only quotes that are verbatim, inside a participant turn, and use valid tags."""
    participant_turns = [t for t in turns if t.speaker.lower() != "interviewer"]
    stats, accepted, seen = Counter(), [], set()

    for q in proposed:
        stats["proposed"] += 1
        span = None
        for t in participant_turns:  # search inside participant turns only
            i = t.text.find(q["text"])
            if i != -1:
                span = (t.start_char + i, t.start_char + i + len(q["text"]))
                speaker = t.speaker
                break
        if span is None:
            stats["rejected_not_verbatim"] += 1  # your hallucination metric
            continue
        assert verify_quote(transcript, q["text"], *span)
        if span in seen:
            stats["rejected_duplicate"] += 1
            continue
        seen.add(span)

        tags = [g for g in q["tags"] if g in TAXONOMY]
        if not tags:
            stats["rejected_no_valid_tags"] += 1
            continue
        accepted.append({"text": q["text"], "start_char": span[0], "end_char": span[1],
                         "speaker": speaker, "tags": tags, "confidence": q["confidence"]})
        stats["accepted"] += 1
    return accepted, stats


def tag_session(raw: str, fmt: str = "txt"):
    transcript, turns = parse_transcript(raw, fmt)
    accepted, stats = validate(propose(transcript), transcript, turns)
    return transcript, accepted, stats


def save_to_db(conn, session_id: int, quotes: list[dict]):
    """Persist to the schema from earlier. conn = psycopg.connect(...). Embeddings come in week 3."""
    with conn.cursor() as cur:
        for name, desc in TAXONOMY.items():
            cur.execute("INSERT INTO tags (name, description) VALUES (%s, %s) "
                        "ON CONFLICT (name) DO NOTHING", (name, desc))
        for q in quotes:
            cur.execute(
                "INSERT INTO quotes (session_id, text, start_char, end_char, speaker) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (session_id, q["text"], q["start_char"], q["end_char"], q["speaker"]))
            qid = cur.fetchone()[0]
            for tag in q["tags"]:
                cur.execute(
                    "INSERT INTO quote_tags (quote_id, tag_id, confidence) "
                    "SELECT %s, id, %s FROM tags WHERE name = %s",
                    (qid, q["confidence"], tag))
    conn.commit()


if __name__ == "__main__":
    path = sys.argv[1]
    fmt = "vtt" if path.endswith(".vtt") else "txt"
    _, quotes, stats = tag_session(open(path).read(), fmt)
    for q in quotes:
        print(f"[{q['start_char']}:{q['end_char']}] {q['tags']} ({q['confidence']:.2f})\n  \"{q['text']}\"")
    print("\nStats:", dict(stats))
