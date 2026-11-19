"""Transcript parser: turns .txt / .vtt into a normalized transcript plus speaker turns.

Key idea: we normalize every format to plain "Speaker: text" lines FIRST and store
that normalized string as sessions.raw_transcript. All char offsets refer to it,
so quote verification is always raw_transcript[start:end] == quote.
"""
import re
from dataclasses import dataclass

TURN_RE = re.compile(r"^(?P<speaker>[A-Za-z][\w .'-]{0,40}):\s*(?P<text>.+)$")
VTT_TIME_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?\.\d{3}\s+-->\s+")
VTT_VOICE_RE = re.compile(r"<v\s+([^>]+)>(.*?)(</v>)?$")


@dataclass
class Turn:
    speaker: str
    text: str
    start_char: int  # span of `text` (not the speaker label) in normalized transcript
    end_char: int


def normalize_vtt(raw: str) -> str:
    """Convert WebVTT to 'Speaker: text' lines, merging consecutive cues by the same speaker."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or VTT_TIME_RE.match(line) or line.isdigit():
            continue
        if line.startswith(("NOTE", "STYLE")):
            continue
        m = VTT_VOICE_RE.match(line)
        if m:
            line = f"{m.group(1).strip()}: {m.group(2).strip()}"
        lines.append(line)

    merged = []
    for line in lines:
        m = TURN_RE.match(line)
        if m:
            merged.append(line)
        elif merged:  # continuation of the previous cue
            merged[-1] += " " + line
    return "\n".join(merged)


def normalize(raw: str, fmt: str) -> str:
    if fmt == "vtt":
        return normalize_vtt(raw)
    return raw.replace("\r\n", "\n").strip()


def parse_turns(normalized: str) -> list[Turn]:
    turns, pos = [], 0
    for line in normalized.split("\n"):
        m = TURN_RE.match(line)
        if m:
            start = pos + m.start("text")
            turns.append(Turn(m.group("speaker"), m.group("text"), start, pos + m.end("text")))
        pos += len(line) + 1  # +1 for the newline
    return turns


def parse_transcript(raw: str, fmt: str = "txt") -> tuple[str, list[Turn]]:
    normalized = normalize(raw, fmt)
    return normalized, parse_turns(normalized)


def verify_quote(transcript: str, quote: str, start: int, end: int) -> bool:
    """The faithfulness check: the span must match the quote verbatim."""
    return 0 <= start < end <= len(transcript) and transcript[start:end] == quote


def locate_quote(transcript: str, quote: str) -> tuple[int, int] | None:
    """If the LLM returns only text, find its exact span (None if not verbatim -> reject)."""
    i = transcript.find(quote)
    return (i, i + len(quote)) if i != -1 else None


if __name__ == "__main__":
    sample = (
        "Interviewer: How do you track expenses today?\n"
        "P1: Honestly, a messy spreadsheet. I always forget to update it.\n"
        "Interviewer: What happens then?\n"
        "P1: I just give up for a month and start over."
    )
    text, turns = parse_transcript(sample)
    for t in turns:
        assert verify_quote(text, t.text, t.start_char, t.end_char)
        print(f"{t.speaker:12} [{t.start_char}:{t.end_char}] {t.text}")
    print("locate:", locate_quote(text, "a messy spreadsheet"))
    print("hallucinated quote ->", locate_quote(text, "I love spreadsheets"))
