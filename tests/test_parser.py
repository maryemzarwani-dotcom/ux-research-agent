from pathlib import Path

from parser import parse_transcript, verify_quote, locate_quote

SAMPLE = Path("data/P01_sample.txt")


def load():
    return parse_transcript(SAMPLE.read_text(encoding="utf-8"))


def test_every_turn_matches_its_position():
    transcript, turns = load()
    assert len(turns) > 0
    for t in turns:
        assert verify_quote(transcript, t.text, t.start_char, t.end_char)


def test_both_speakers_are_found():
    _, turns = load()
    assert {t.speaker for t in turns} == {"Interviewer", "P01"}


def test_fake_quote_is_rejected():
    transcript, _ = load()
    assert locate_quote(transcript, "I love spreadsheets") is None
