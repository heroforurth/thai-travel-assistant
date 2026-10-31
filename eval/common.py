"""Shared helpers for the evaluation scripts: the test set and the relevance rule.

Relevance is defined by *evidence*, not by chunk id, so the same labels work for every
chunk size: a chunk is relevant to a question if it comes from the gold destination
and contains one of the question's evidence phrases (whitespace/case-insensitive).
"""
import json
import pathlib
import re

from app.chunking import Chunk

EVAL_DIR = pathlib.Path(__file__).parent
RESULTS = EVAL_DIR / "results"
TESTSET = EVAL_DIR / "testset.jsonl"

ABSTAIN_RE = re.compile(
    r"couldn'?t find|could not find|can'?t find|cannot find|not (?:mentioned|available|covered|included|listed)"
    r"|not in the (?:travel )?guides|no information|don'?t (?:have|know)|do not (?:have|know)|not sure"
    r"|(?:does|do)(?: not|n'?t) (?:mention|contain|provide|say)|unable to|no mention", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def load_testset() -> list[dict]:
    with open(TESTSET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(chunk: Chunk, ex: dict) -> bool:
    return chunk.destination == ex["destination"] and any(norm(e) in norm(chunk.text) for e in ex["evidence"])


def is_abstain(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer))


def keyword_correct(answer: str, ex: dict) -> bool:
    """Answerable: every gold keyword appears (commas ignored, so 3000 == 3,000) and the
    model did not abstain. Unanswerable: the model abstained."""
    if ex["type"] == "unanswerable":
        return is_abstain(answer)
    a = norm(answer).replace(",", "")
    return all(norm(k).replace(",", "") in a for k in ex["keywords"]) and not is_abstain(answer)


def write_md(path: pathlib.Path, title: str, rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text
