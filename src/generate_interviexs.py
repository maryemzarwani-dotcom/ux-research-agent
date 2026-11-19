"""Generate synthetic interviews from ground_truth.json.

For each participant the LLM writes a transcript AND lists the verbatim quotes
where it planted each assigned theme. We verify every planted quote exists in the
transcript (same faithfulness idea as your product), retrying on failure.
Result: quote-level ground truth for free, in data/planted_quotes.json.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python generate_interviews.py
"""
import json
import os
import pathlib
import re

import anthropic

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
MAX_RETRIES = 3
OUT = pathlib.Path("data")
OUT.mkdir(exist_ok=True)

gt = json.loads(pathlib.Path("ground_truth.json").read_text())
client = anthropic.Anthropic()

PROMPT = """You are writing a realistic, UNSCRIPTED-sounding user research interview transcript
for a synthetic dataset. Domain: {domain}.

Participant {pid}: {role}.
Themes this participant must raise (and ONLY these; do not touch other pain points):
{themes}

Rules:
- Format every line as "Interviewer: ..." or "{pid}: ..." (one turn per line, no blank lines).
- 16-22 turns. Natural speech: hesitations, tangents, specific anecdotes, no theme labels.
- Do not state the theme abstractly. Let it emerge through a concrete story or complaint.
- For themes ending in "+" the participant likes the thing; "-" means dislikes it.
- Include at least one off-topic or neutral exchange so not every line is signal.
- Return ONLY JSON: {{"transcript": "<full text, lines separated by \\n>",
  "planted": [{{"theme": "T1", "quote": "<verbatim substring of the transcript>"}}]}}
  Each assigned theme needs 1-2 planted quotes. Quotes must be copied EXACTLY.
"""


def describe(codes):
    lines = []
    for c in codes:
        base = c.rstrip("+-")
        suffix = {"+": " (participant LIKES this)", "-": " (participant DISLIKES this)"}.get(c[-1], "")
        lines.append(f"- {c}: {gt['themes'][base]['label']}{suffix}")
    return "\n".join(lines)


def generate(pid, info):
    for attempt in range(1, MAX_RETRIES + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": PROMPT.format(
                domain=gt["domain"], pid=pid, role=info["role"], themes=describe(info["themes"]))}],
        )
        text = resp.content[0].text
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            data = json.loads(text)
            transcript = data["transcript"].strip()
            bad = [p for p in data["planted"] if p["quote"] not in transcript]
            covered = {p["theme"] for p in data["planted"]}
            missing = set(info["themes"]) - covered
            if not bad and not missing:
                return transcript, data["planted"]
            print(f"  {pid} attempt {attempt}: {len(bad)} non-verbatim quotes, missing {missing}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  {pid} attempt {attempt}: bad JSON ({e})")
    raise RuntimeError(f"Could not generate a valid interview for {pid}")


if __name__ == "__main__":
    all_planted = {}
    for pid, info in gt["participants"].items():
        print(f"Generating {pid}...")
        transcript, planted = generate(pid, info)
        (OUT / f"{pid}.txt").write_text(transcript)
        all_planted[pid] = planted
    (OUT / "planted_quotes.json").write_text(json.dumps(all_planted, indent=2))
    print("Done. Review a few transcripts by hand before using them as your gold set.")
