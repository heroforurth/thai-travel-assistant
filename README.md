# Thai Travel Assistant

A small, end-to-end LLM application that answers practical travel questions about 15
Thai destinations, such as "How much is the ferry to Ko Tao?", "How far is Pai Canyon
from town?" or "What would 3 people pay for a return longtail to Railay?". Every answer
is grounded in Wikivoyage guides and cites the passage it came from. When the guides
don't cover something, the assistant says so instead of guessing.

The model (`gemma-4-E4B-it` through the class proxy) runs as a **tool-calling agent**.
At each step it chooses one of these actions:

| Tool | What it does |
|---|---|
| `search` | Hybrid retrieval (bge-m3 dense + BM25, fused with RRF) over section-aware chunks, with an optional, allow-listed destination filter and a `min_score` floor. |
| `calculate` | A sandboxed AST calculator for totals and costs. It runs no `eval`; only numbers and `+ - * / // % **` are allowed. |
| `final` | The answer, plus citations. These are validated against the chunks actually retrieved during the run. |

## Course capability families covered

| # | Family | Where |
|---|---|---|
| 1 | Structured output + schema validation | [app/schemas.py](app/schemas.py): every agent step is parsed into a Pydantic discriminated union. Invalid steps are never executed and get one repair attempt. |
| 2 | RAG with citations | [app/chunking.py](app/chunking.py), [app/retrieve.py](app/retrieve.py), citation checks in [app/agent.py](app/agent.py) |
| 3 | Retrieval evaluation | [eval/eval_retrieval.py](eval/eval_retrieval.py): Hit@1, Recall@5, MRR on a labelled set, against a BM25 baseline |
| 4 | Agent loop with tools | [app/agent.py](app/agent.py): search, calculate and final, stopping on `final` or at `MAX_STEPS` |
| 5 | Sandboxed code execution | `calculate()` in [app/tools.py](app/tools.py) |
| 6 | Served interface | [app/api.py](app/api.py): `POST /api/ask` (JSON) and `POST /api/chat` (SSE), with the UI in [static/index.html](static/index.html) |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then put your course key in OPENAI_API_KEY

python -m app.ingest              # chunk + embed the corpus (cached in index/)
uvicorn app.api:app --port 8000   # open http://localhost:8000
```

From the command line:

```bash
python -m app.agent "How do I get from Chiang Mai to Pai and how much is it?"
```

The corpus is already committed in `data/raw/`. To re-download it, run `python data/fetch_data.py`.
Its text comes from Wikivoyage under CC BY-SA 4.0; the revision links are in [data/raw/SOURCES.md](data/raw/SOURCES.md).

## Reproduce the evaluation

```bash
pytest -q                                 # unit tests (no key needed; the model is stubbed)
python -m eval.eval_retrieval --bm25-only # lexical baseline (no key needed)
python -m eval.eval_retrieval             # BM25 vs dense vs hybrid x 3 chunk sizes + min_score sweep
python -m eval.eval_answers               # closed-book vs single-shot RAG vs agent, on 33 questions
```

Results are written to [eval/results/](eval/results/). Every model call uses
`temperature=0`, so reruns should match apart from proxy-side nondeterminism.

**Test set** ([eval/testset.jsonl](eval/testset.jsonl)) has 33 hand-written questions:

- 22 single-fact lookups
- 5 lookup-plus-arithmetic questions
- 6 unanswerable questions about things the guides don't cover

Each answerable question is labelled with its destination and a verbatim **evidence
phrase**. A chunk counts as relevant if it contains that phrase, so the same labels work
for every chunk size.

## Results

**Retrieval** (27 answerable questions, chunk size 400):

| Retriever | Hit@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 (baseline) | 0.56 | 0.85 | 0.67 |
| Dense (bge-m3) | 0.74 | 1.00 | 0.86 |
| **Hybrid (used)** | **0.78** | **1.00** | **0.85** |

**End-to-end** (33 questions, keyword accuracy):

| System | All | Lookup | Calc | Unanswerable | Cited support |
|---|---|---|---|---|---|
| Closed-book (baseline) | 0.27 | 0.18 | 0.20 | 0.67 | 0.00 |
| Single-shot RAG (baseline) | 0.97 | 1.00 | 0.80 | 1.00 | 0.96 |
| Agent v1 | 0.85 | 0.86 | 0.60 | 1.00 | 0.67 |
| **Agent (current)** | **0.97** | 0.95 | **1.00** | 1.00 | 0.96 |

**How the defaults were chosen (measured improvements):**

- **Chunk size:** 400 rather than 800 raises hybrid Hit@1 from 0.70 to 0.78.
- **min_score = 0.60:** with no floor, all 6 unanswerable questions get junk context. At 0.60, half of them get none, and every answerable question keeps its evidence.
- **Agent fixes:** agent v1 → current raised accuracy from 0.85 to 0.97. The fixes were a destination prefix on searches, recovering inline citations, and a stricter calculator rule.

These choices were tuned on the same 33 questions, so the numbers are optimistic. See the report's limitations section.

Details: [eval/results/retrieval.md](eval/results/retrieval.md), [eval/results/min_score.md](eval/results/min_score.md),
[eval/results/answers.md](eval/results/answers.md) and the report in [report/](report/).

## Layout

```
app/    config (env only) · llm (the only model caller) · chunking · bm25 · retrieve
        schemas (validation) · tools · agent · rag (baselines) · api (FastAPI + SSE)
eval/   testset.jsonl · common (relevance + scoring rules) · eval_retrieval · eval_answers · results/
data/   fetch_data.py · raw/ (corpus)
tests/  unit tests (calculator sandbox, schema allow-lists, chunking, agent loop with a stub model)
```

## Safety notes

- **Configuration:** the API key and base URL come only from the environment; `.env` is gitignored.
- **Tool arguments:** these are validated before anything runs. Destinations must be on the allow-list, calculator expressions must match `^[0-9+\-*/().%\s]+$` and then pass an AST whitelist, and unknown actions are rejected.
- **Search results:** these are untrusted web text. They are fenced in `<search_results>` tags, and the system prompt tells the model never to follow instructions inside them.
- **Citations:** any citation to a chunk that wasn't retrieved in the run is dropped and reported in `dropped_citations`.
