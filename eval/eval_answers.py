"""End-to-end answer evaluation on all 33 questions, comparing three systems:

  closed_book  - baseline: the model answers from memory, no retrieval
  single_shot  - baseline: one hybrid retrieval with the raw question, then answer
  agent        - our app: tool-calling loop (search + calculator) with validated steps

Metrics (per system, overall and by question type):
  accuracy      - keyword rule (all gold keywords present; unanswerable => must abstain)
  judge         - LLM-as-judge (gemma, temperature 0) vs. the gold answer
  cited_support - answer is keyword-correct AND cites a chunk containing the evidence

Run:  python -m eval.eval_answers                      # all systems
      python -m eval.eval_answers --systems agent      # just one
Writes eval/results/answers_<system>.jsonl and answers.{md,csv}
"""
import argparse
import csv
import json
import time

import numpy as np
from pydantic import BaseModel, ValidationError

from app import agent, llm, rag
from app.retrieve import Index
from app.schemas import extract_json
from eval.common import RESULTS, is_abstain, is_relevant, keyword_correct, load_testset, write_md

JUDGE = """You grade answers from a travel assistant against a gold answer.
Question: {q}
Gold answer: {gold}
Candidate answer: {cand}

If the gold answer is "Not in the guides", the candidate is correct only if it says it could not find
or does not know the answer. Otherwise the candidate is correct if it states the same key fact(s)
as the gold answer (wording and extra detail don't matter; wrong numbers do).
Reply with ONLY a JSON object: {{"correct": true}} or {{"correct": false}}"""


class Verdict(BaseModel):
    correct: bool


def judge(ex: dict, answer: str) -> bool | None:
    reply = llm.chat([{"role": "user", "content": JUDGE.format(q=ex["question"], gold=ex["answer"], cand=answer)}],
                     temperature=0, max_tokens=20)
    try:
        return Verdict(**extract_json(reply)).correct
    except (ValueError, ValidationError, TypeError):
        return None  # unparseable verdict; excluded from the judge mean


def cited_support(result, ex: dict, index: Index) -> bool:
    """Did the answer cite at least one chunk that actually contains the gold evidence?"""
    return any(c.id in index.by_id and is_relevant(index.by_id[c.id], ex) for c in result.citations)


def run_system(name: str, ex: dict, index: Index):
    if name == "closed_book":
        return rag.closed_book(ex["question"])
    if name == "single_shot":
        return rag.single_shot(ex["question"], index)
    return agent.run(ex["question"], index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["closed_book", "single_shot", "agent"])
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tests = load_testset()
    index = Index.build()

    summary = []
    for name in args.systems:
        records = []
        for ex in tests:
            t = time.time()
            res = run_system(name, ex, index)
            rec = {
                "id": ex["id"], "type": ex["type"], "question": ex["question"], "gold": ex["answer"],
                "answer": res.answer, "keyword_correct": keyword_correct(res.answer, ex),
                "judge_correct": judge(ex, res.answer), "abstained": is_abstain(res.answer),
                "cited_support": ex["type"] != "unanswerable" and cited_support(res, ex, index),
                "citations": [c.id for c in res.citations], "dropped_citations": res.dropped_citations,
                "steps": res.steps, "stopped_by": res.stopped_by, "parse_failures": res.parse_failures,
                "seconds": round(time.time() - t, 1), "trace": res.trace,
            }
            records.append(rec)
            print(f"[{name}] {ex['id']} kw={rec['keyword_correct']!s:5} judge={rec['judge_correct']!s:5} "
                  f"steps={rec['steps']} {res.answer[:90]!r}")
        with open(RESULTS / f"answers_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def mean(key, subset):
            vals = [r[key] for r in subset if r[key] is not None]
            return float(np.mean(vals)) if vals else float("nan")

        answerable = [r for r in records if r["type"] != "unanswerable"]
        row = {"system": name,
               "accuracy (all 33)": mean("keyword_correct", records),
               "judge (all 33)": mean("judge_correct", records)}
        for typ in ("lookup", "calc", "unanswerable"):
            row[f"acc {typ}"] = mean("keyword_correct", [r for r in records if r["type"] == typ])
        row["cited_support (27)"] = mean("cited_support", answerable) if name != "closed_book" else 0.0
        row["false_abstain (27)"] = mean("abstained", answerable)
        row["avg steps"] = float(np.mean([r["steps"] for r in records]))
        row["parse_failures"] = int(sum(r["parse_failures"] for r in records))
        row["avg sec"] = float(np.mean([r["seconds"] for r in records]))
        summary.append(row)

    # Merge with any previously evaluated systems so partial re-runs keep the table whole.
    path = RESULTS / "answers.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["system"] not in args.systems]
        summary = [{k: (float(v) if k != "system" else v) for k, v in r.items()} for r in old] + summary
    order = {"closed_book": 0, "single_shot": 1, "agent_v1": 2, "agent": 3}
    summary.sort(key=lambda r: order.get(r["system"], 9))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("\n" + write_md(RESULTS / "answers.md", "End-to-end answers (33 questions)", summary))


if __name__ == "__main__":
    main()
