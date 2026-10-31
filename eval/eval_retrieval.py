"""Retrieval evaluation: Hit@1, Recall@5, MRR@10 on the 27 answerable questions,
across retrievers (BM25 baseline / dense / hybrid) and chunk sizes, plus a min_score
sweep that uses the 6 unanswerable questions.

Run:  python -m eval.eval_retrieval              # everything (needs the proxy for embeddings)
      python -m eval.eval_retrieval --bm25-only  # lexical baseline only, no key needed
Writes eval/results/retrieval.{md,csv}, min_score.{md,csv}, min_score.png
"""
import argparse
import csv

import numpy as np

from app import config, llm
from app.retrieve import Index
from eval.common import RESULTS, is_relevant, load_testset, write_md

CHUNKINGS = [(400, 50), (800, 100), (1500, 150)]
K = 10


def first_relevant_rank(hits, ex) -> int | None:
    for rank, h in enumerate(hits, 1):
        if is_relevant(h.chunk, ex):
            return rank
    return None


def score(index: Index, questions: list[dict], mode: str, use_destination: bool = False) -> dict:
    ranks = [first_relevant_rank(index.search(ex["question"], k=K, mode=mode, min_score=0.0,
                                              destination=ex["destination"] if use_destination else None), ex)
             for ex in questions]
    return {
        "hit@1": float(np.mean([r == 1 for r in ranks])),
        "recall@5": float(np.mean([r is not None and r <= 5 for r in ranks])),
        "mrr@10": float(np.mean([1 / r if r else 0.0 for r in ranks])),
        "misses@10": [ex["id"] for ex, r in zip(questions, ranks) if r is None],
    }


def min_score_sweep(index: Index, answerable: list[dict], unanswerable: list[dict]) -> list[dict]:
    """For each floor t: does a relevant chunk survive in the top 5 (answerable), and does
    *nothing* survive (unanswerable, i.e. the app can say "not in the guides")?"""
    def cos_profile(ex):
        cos = index.matrix @ llm.embed([ex["question"]])[0]
        order = np.argsort(-cos)
        rel = [cos[i] for i in order[:5] if is_relevant(index.chunks[i], ex)]
        return (max(rel) if rel else None), float(cos[order[0]])

    ans = [cos_profile(ex) for ex in answerable]
    una = [cos_profile(ex)[1] for ex in unanswerable]
    rows = []
    for t in np.round(np.arange(0.30, 0.71, 0.05), 2):
        kept = np.mean([r is not None and r >= t for r, _ in ans])
        rejected = np.mean([top < t for top in una])
        rows.append({"min_score": float(t), "answerable_kept": float(kept), "unanswerable_rejected": float(rejected),
                     "balanced": float((kept + rejected) / 2)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bm25-only", action="store_true")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tests = load_testset()
    answerable = [ex for ex in tests if ex["type"] != "unanswerable"]
    unanswerable = [ex for ex in tests if ex["type"] == "unanswerable"]
    modes = ["bm25"] if args.bm25_only else ["bm25", "dense", "hybrid"]

    rows = []
    for size, overlap in CHUNKINGS:
        index = Index.build(size, overlap)
        for mode in modes:
            r = score(index, answerable, mode)
            rows.append({"chunk_size": size, "retriever": mode, **{k: v for k, v in r.items() if k != "misses@10"},
                         "misses@10": " ".join(r["misses@10"]) or "-"})
            print(rows[-1])
        if not args.bm25_only and (size, overlap) == (config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            # Query construction: the agent can pass a destination filter. Upper bound
            # when the filter is right (oracle destination from the label).
            r = score(index, answerable, "hybrid", use_destination=True)
            rows.append({"chunk_size": size, "retriever": "hybrid+destination filter",
                         **{k: v for k, v in r.items() if k != "misses@10"}, "misses@10": " ".join(r["misses@10"]) or "-"})
            print(rows[-1])
            sweep = min_score_sweep(index, answerable, unanswerable)

    with open(RESULTS / "retrieval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n" + write_md(RESULTS / "retrieval.md", f"Retrieval on {len(answerable)} answerable questions", rows))

    if args.bm25_only:
        return
    with open(RESULTS / "min_score.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    print(write_md(RESULTS / "min_score.md", f"min_score sweep (dense cosine, chunk {config.CHUNK_SIZE})", sweep))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["min_score"] for r in sweep]
    plt.figure(figsize=(6, 3.5))
    plt.plot(t, [r["answerable_kept"] for r in sweep], marker="o", label="answerable: relevant chunk kept")
    plt.plot(t, [r["unanswerable_rejected"] for r in sweep], marker="s", label="unanswerable: all chunks rejected")
    plt.xlabel("min_score (cosine floor)")
    plt.ylabel("fraction of questions")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "min_score.png", dpi=150)


if __name__ == "__main__":
    main()
