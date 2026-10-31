"""Build the project report (HTML -> PDF) from the committed evaluation results.

Run:  python report/build_report.py      (after the eval scripts; writes report/report.html + report.pdf)

All numbers, tables, the min_score figure and the failure table are read from
eval/results/, so the report can never disagree with the evaluation. Sections whose
eval has not been run yet are clearly marked instead of being filled with guesses.
"""
import base64
import collections
import csv
import html
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.retrieve import Index  # noqa: E402
from eval.common import RESULTS, is_relevant, load_testset  # noqa: E402

OUT = ROOT / "report"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def read_csv(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(v: str) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    return f"{x:.0f}" if x.is_integer() and abs(x) > 1 else f"{x:.2f}"


def table(rows: list[dict], cols: list[str] | None = None, bold_row=None) -> str:
    cols = cols or list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = ""
    for r in rows:
        cls = ' class="best"' if bold_row and bold_row(r) else ""
        body += f"<tr{cls}>" + "".join(f"<td>{fmt(r[c])}</td>" for c in cols) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def pending(cmd: str) -> str:
    return f'<p class="todo">Not run yet: run <code>{cmd}</code> and rebuild this report.</p>'


# --- sections ------------------------------------------------------------------------

def retrieval_section() -> str:
    rows = read_csv("retrieval.csv")
    if not rows:
        return pending("python -m eval.eval_retrieval")
    num = lambda r, k: float(r[k])  # noqa: E731
    plain = [r for r in rows if "filter" not in r["retriever"]]
    best = max(plain, key=lambda r: (num(r, "recall@5"), num(r, "mrr@10")))
    bm = [r for r in rows if r["retriever"] == "bm25" and r["chunk_size"] == best["chunk_size"]][0]
    out = table(rows, bold_row=lambda r: r is best)
    out += (f"<p>The best configuration without a filter is <b>{best['retriever']}</b> at chunk size "
            f"<b>{best['chunk_size']}</b>, with Recall@5 = {fmt(best['recall@5'])} and MRR = {fmt(best['mrr@10'])}. "
            f"The BM25 baseline at the same chunk size gets Recall@5 = {fmt(bm['recall@5'])} and MRR = {fmt(bm['mrr@10'])}.</p>")
    sizes = sorted({int(r["chunk_size"]) for r in plain})
    by = {(int(r["chunk_size"]), r["retriever"]): r for r in plain}
    line = ", ".join(f"{s}: {fmt(by[(s, best['retriever'])]['mrr@10'])}" for s in sizes if (s, best["retriever"]) in by)
    out += (f"<p><b>Chunk size</b> ({best['retriever']}, MRR by size): {line}. This is a trade-off: smaller chunks "
            "give sharper vectors but can cut a fact off from its context, while larger chunks keep the context "
            f"but mix several topics into one chunk. The app uses <code>CHUNK_SIZE={config.CHUNK_SIZE}</code>, "
            "the size with the best Recall@5 (ties broken by MRR).</p>")
    filt = [r for r in rows if "filter" in r["retriever"]]
    if filt:
        out += (f"<p><b>Query construction.</b> When the search is also filtered to the correct destination "
                f"(the agent's <code>destination</code> argument), Recall@5 is {fmt(filt[0]['recall@5'])} and MRR is "
                f"{fmt(filt[0]['mrr@10'])}. This is an upper bound for what the filter adds, since it assumes "
                "the agent picks the right destination.</p>")
    return out


def min_score_section() -> str:
    rows = read_csv("min_score.csv")
    if not rows:
        return pending("python -m eval.eval_retrieval")
    best = max(rows, key=lambda r: (float(r["balanced"]), -float(r["min_score"])))
    img = RESULTS / "min_score.png"
    fig = (f'<figure><img src="data:image/png;base64,{base64.b64encode(img.read_bytes()).decode()}">'
           "<figcaption>Figure 2: The min_score trade-off. A higher floor rejects more unanswerable questions "
           "but also drops real evidence.</figcaption></figure>") if img.exists() else ""
    zero = rows[0]
    return (fig + table(rows, bold_row=lambda r: r is best) +
            f"<p>The naive default is no floor. With no floor, retrieval always returns 5 chunks, so the model is "
            f"handed junk context for all 6 unanswerable questions. The sweep's lowest floor ({fmt(zero['min_score'])}) "
            f"already rejects {fmt(zero['unanswerable_rejected'])} of them. The best balanced value is "
            f"<b>min_score = {fmt(best['min_score'])}</b>: it keeps evidence for {fmt(best['answerable_kept'])} of "
            f"answerable questions and rejects {fmt(best['unanswerable_rejected'])} of unanswerable ones. "
            f"The app's floor is set to <code>MIN_SCORE={config.MIN_SCORE}</code>. With only 6 unanswerable "
            "questions, each one moves the rejection rate by 0.17, so this choice is coarse.</p>")


def answers_section() -> str:
    rows = read_csv("answers.csv")
    if not rows:
        return pending("python -m eval.eval_answers")
    out = table(rows, bold_row=lambda r: r["system"] == "agent")
    recs = {r["system"]: read_jsonl(f"answers_{r['system']}.jsonl") for r in rows}
    agree = [(x["keyword_correct"], x["judge_correct"]) for rs in recs.values() if rs for x in rs
             if x["judge_correct"] is not None]
    if agree:
        rate = sum(a == b for a, b in agree) / len(agree)
        out += (f"<p>The keyword metric and the LLM judge agree on {rate:.0%} of {len(agree)} graded answers. "
                "We report keyword accuracy as the primary metric because it is deterministic. The judge is "
                "the same small model being evaluated, so it is the less trustworthy of the two.</p>")
    return out


def failure_section() -> str:
    recs = read_jsonl("answers_agent.jsonl")
    if not recs:
        return pending("python -m eval.eval_answers --systems agent")
    index = Index.build()
    tests = {ex["id"]: ex for ex in load_testset()}

    def category(r: dict) -> str:
        ex = tests[r["id"]]
        if r["stopped_by"] in ("parse_error", "max_steps"):
            return f"loop ended by {r['stopped_by']}"
        if ex["type"] == "unanswerable":
            return "answered an unanswerable question"
        hits = [h for step in r["trace"] for h in step.get("hits", [])]
        found = any(h in index.by_id and is_relevant(index.by_id[h], ex) for h in hits)
        if r["abstained"]:
            return "abstained although evidence was retrieved" if found else "retrieval miss, then abstained"
        return "wrong/incomplete despite evidence" if found else "answered without retrieved evidence"

    fails = [r for r in recs if not r["keyword_correct"]]
    if not fails:
        return "<p>The agent answered every question correctly under the keyword metric.</p>"
    cats = collections.Counter(category(r) for r in fails)
    out = "<p>Agent failures by category: " + ", ".join(f"{c} ({n})" for c, n in cats.most_common()) + ".</p>"
    rows = []
    for r in fails:
        searches = [s["action"].get("query") for s in r["trace"] if s["action"]["action"] == "search"]
        rows.append({"id": r["id"], "question": r["question"], "gold": r["gold"], "agent answer": r["answer"][:220],
                     "searches": " | ".join(q for q in searches if q) or "-", "category": category(r)})
    return out + table(rows)


DIAGRAM = """<svg viewBox="0 0 760 270" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif" font-size="12">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#555"/></marker></defs>
<g stroke="#555" fill="none" marker-end="url(#a)"><path d="M95,135H140"/><path d="M290,120H340"/><path d="M340,150H290"/>
<path d="M480,90L535,55"/><path d="M480,135H535"/><path d="M480,180L535,215"/><path d="M640,55L700,55"/></g>
<rect x="10" y="110" width="85" height="50" rx="6" fill="#eef2ff" stroke="#6366f1"/><text x="52" y="132" text-anchor="middle">Browser UI</text><text x="52" y="148" text-anchor="middle" font-size="10">SSE stream</text>
<rect x="140" y="95" width="150" height="80" rx="6" fill="#ecfdf5" stroke="#10b981"/><text x="215" y="118" text-anchor="middle" font-weight="600">FastAPI</text><text x="215" y="136" text-anchor="middle" font-size="10">/api/chat (SSE)</text><text x="215" y="151" text-anchor="middle" font-size="10">/api/ask (JSON)</text>
<rect x="340" y="70" width="140" height="130" rx="6" fill="#fff7ed" stroke="#f97316"/><text x="410" y="92" text-anchor="middle" font-weight="600">Agent loop</text>
<text x="410" y="112" text-anchor="middle" font-size="10">gemma-4-E4B-it, T=0</text><text x="410" y="128" text-anchor="middle" font-size="10">JSON step → Pydantic</text><text x="410" y="144" text-anchor="middle" font-size="10">validate / repair once</text><text x="410" y="160" text-anchor="middle" font-size="10">≤ 5 steps + forced final</text><text x="410" y="176" text-anchor="middle" font-size="10">citation check</text>
<rect x="535" y="30" width="105" height="50" rx="6" fill="#f1f5f9" stroke="#64748b"/><text x="587" y="52" text-anchor="middle">search</text><text x="587" y="68" text-anchor="middle" font-size="10">dest. allow-list</text>
<rect x="700" y="20" width="55" height="70" rx="6" fill="#f1f5f9" stroke="#64748b"/><text x="727" y="45" text-anchor="middle" font-size="10">BM25</text><text x="727" y="60" text-anchor="middle" font-size="10">+ bge-m3</text><text x="727" y="75" text-anchor="middle" font-size="10">RRF, floor</text>
<rect x="535" y="110" width="105" height="50" rx="6" fill="#f1f5f9" stroke="#64748b"/><text x="587" y="132" text-anchor="middle">calculate</text><text x="587" y="148" text-anchor="middle" font-size="10">AST sandbox</text>
<rect x="535" y="190" width="105" height="50" rx="6" fill="#f1f5f9" stroke="#64748b"/><text x="587" y="212" text-anchor="middle">final</text><text x="587" y="228" text-anchor="middle" font-size="10">answer + cites</text>
<text x="215" y="215" text-anchor="middle" font-size="10" fill="#555">data/raw: 15 Wikivoyage guides</text><text x="215" y="230" text-anchor="middle" font-size="10" fill="#555">→ section-aware chunks (800/100)</text>
</svg>"""


def corpus_stats() -> str:
    index = Index.build()
    chars = sum(len((config.RAW_DIR / f"{d}.txt").read_text(encoding="utf-8")) for d in config.DESTINATIONS)
    return f"{len(config.DESTINATIONS)} guides, {chars:,} characters, {len(index.chunks):,} chunks at the default size"


def main() -> None:
    tests = load_testset()
    types = collections.Counter(ex["type"] for ex in tests)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Thai Travel Assistant: Project Report</title>
<style>
@page {{ size: A4; margin: 18mm 17mm; }}
body {{ font: 10.5pt/1.45 "Segoe UI", system-ui, sans-serif; color: #1a1a1a; }}
h1 {{ font-size: 19pt; margin: 0 0 2px; }} h2 {{ font-size: 13pt; margin: 18px 0 6px; border-bottom: 1px solid #ddd; padding-bottom: 2px; }}
h3 {{ font-size: 11pt; margin: 12px 0 4px; }}
.meta {{ color: #555; margin-bottom: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 8.8pt; page-break-inside: auto; }}
th, td {{ border: 1px solid #d4d4d4; padding: 3px 5px; text-align: left; vertical-align: top; }}
th {{ background: #f3f4f6; }} tr.best td {{ background: #ecfdf5; font-weight: 600; }}
code {{ font-size: 9pt; background: #f3f4f6; padding: 0 3px; }}
figure {{ margin: 8px 0; text-align: center; }} figure img {{ max-width: 78%; }} figure svg {{ width: 100%; }}
figcaption {{ font-size: 9pt; color: #555; }}
.todo {{ color: #b45309; font-weight: 600; }}
</style></head><body>
<h1>Thai Travel Assistant</h1>
<div class="meta">Capstone: Building LLM-Powered Applications (KMITL) · model <code>{config.CHAT_MODEL}</code> + <code>{config.EMBED_MODEL}</code> via the class proxy</div>

<h2>1. Problem and users</h2>
<p>Backpackers and students planning trips in Thailand ask concrete, checkable questions: what a ferry costs,
how long a bus takes, how far an attraction is from town, and what a group will pay in total. General chat models
answer these confidently from memory, with numbers that are often outdated or invented. Travel guides contain the
answers but are long. For example, the Chiang Mai guide alone is about 157k characters.</p>
<p>This app answers such questions from <b>Wikivoyage guides for 15 destinations</b>, with a citation to the
section each fact came from. It does arithmetic with a calculator rather than in the model's head, and it
<b>says so when the guides don't cover a question</b> instead of guessing.</p>
<p>The corpus ({corpus_stats()}) has nothing in common with the course reference app's corpus, which documents
the course's own software stack.</p>

<h2>2. Design</h2>
<figure>{DIAGRAM}<figcaption>Figure 1: Architecture. Every model output is validated before anything runs.</figcaption></figure>
<table><thead><tr><th>Technique (course family)</th><th>How it is used</th><th>Why</th></tr></thead><tbody>
<tr><td>Structured output + validation (1)</td><td>Each agent step is JSON parsed into a Pydantic discriminated union (search | calculate | final). An invalid step is sent back with the error once, then the agent gives up safely.</td><td>The proxy ignores <code>response_format</code>, and a 4B model often wraps JSON in prose. Nothing executes unless it validates.</td></tr>
<tr><td>RAG with citations (2)</td><td>Section-aware chunks with a "Destination — Section:" prefix, retrieved with hybrid BM25 + bge-m3 fused by RRF, and a cosine floor. Citations to chunks that were never retrieved are dropped.</td><td>Answers must be checkable. Hybrid retrieval catches both exact names and numbers (lexical) and paraphrases (dense).</td></tr>
<tr><td>Retrieval evaluation (3)</td><td>Hit@1, Recall@5 and MRR on 27 labelled questions, across 3 retrievers and 3 chunk sizes, plus a min_score sweep.</td><td>Every retrieval setting in the app is chosen from these numbers.</td></tr>
<tr><td>Agent loop with tools (4)</td><td>Tools: search (with an allow-listed destination filter) and calculate. The loop stops on <code>final</code> or after 5 steps, followed by one forced final turn. Repeated identical actions are refused.</td><td>Questions combining a fact with arithmetic need several steps. The destination filter is a form of query construction.</td></tr>
<tr><td>Sandboxed code execution (5)</td><td>The calculator parses expressions into an AST and evaluates only numeric literals and arithmetic operators. It never calls <code>eval</code>, and exponents are capped.</td><td>Small models make arithmetic mistakes, and running model-written code directly would be unsafe.</td></tr>
<tr><td>Served interface (6)</td><td>FastAPI with <code>/api/chat</code> streaming each agent step over SSE, plus <code>/api/ask</code> returning validated JSON.</td><td>Users can watch the agent search and calculate as it works.</td></tr>
</tbody></table>
<p><b>Deliberately left out:</b> text-to-SQL and MCP. The data is prose, not tables, and adding them would have
meant more moving parts without a question in the test set that needs them. All configuration (key, base URL,
models, retrieval settings) comes from environment variables, and every evaluated call uses <code>temperature=0</code>.</p>

<h2>3. Evaluation set and metrics</h2>
<p><b>Test set:</b> {len(tests)} hand-written questions (<code>eval/testset.jsonl</code>):
{types['lookup']} single-fact lookups, {types['calc']} lookup-plus-arithmetic questions, and {types['unanswerable']}
unanswerable questions about things absent from the guides (for example a "Sukhothai metro", Ko Lipe, or casinos).
A script confirmed that every answerable question's <b>evidence phrase</b> appears verbatim in its gold guide.</p>
<p><b>Relevance rule:</b> a chunk is relevant if it comes from the gold destination and contains the evidence phrase.
Because the labels don't depend on chunk IDs, the same test set scores every chunk size.</p>
<p><b>Metrics:</b></p>
<ul>
<li>Retrieval: Hit@1, Recall@5 and MRR@10.</li>
<li>Answers: keyword accuracy, meaning all gold keywords are present with commas normalised. For unanswerable
questions, the model must abstain to count as correct.</li>
<li>An LLM-as-judge score, as a second opinion.</li>
<li>Cited support: the answer is correct <i>and</i> cites a chunk containing the evidence.</li>
</ul>
<p><b>Baselines:</b></p>
<ul>
<li>For retrieval: BM25 only.</li>
<li>For answers: <i>closed-book</i> (no retrieval) and <i>single-shot RAG</i> (one retrieval with the raw question, no tools).</li>
</ul>

<h2>4. Results</h2>
<h3>4.1 Retrieval: retriever × chunk size (27 answerable questions)</h3>
{retrieval_section()}
<h3>4.2 Measured improvement: the min_score floor</h3>
{min_score_section()}
<h3>4.3 End-to-end answers ({len(tests)} questions)</h3>
{answers_section()}

<h2>5. Failure analysis</h2>
{failure_section()}

<h2>6. Limitations and next steps</h2>
<ul>
<li><b>Small evaluation set.</b> With {len(tests)} questions (only {types['unanswerable']} unanswerable), one question moves accuracy by about 3 points, so differences of one or two questions are within noise. Next step: grow the set to 100+ questions, including questions written by other students.</li>
<li><b>Stale facts.</b> Prices in the guides date from 2016–2025. The app reports what the guide says, not current prices. Next step: show each chunk's "updated" date next to its citation.</li>
<li><b>Single-hop bias.</b> Most questions need one passage. Questions comparing destinations, such as "cheaper to reach: Ko Tao or Ko Samet?", are untested.</li>
<li><b>Judge reliability.</b> The LLM judge is the same 4B model. A stronger judge, or a second human grader, would make that column more trustworthy.</li>
<li><b>English only.</b> The bge-m3 embeddings are multilingual, but Thai-language questions were not evaluated.</li>
</ul>
<p style="color:#666;font-size:9pt">Reproduce: <code>python -m eval.eval_retrieval &amp;&amp; python -m eval.eval_answers &amp;&amp; python report/build_report.py</code>.
Corpus text: Wikivoyage, CC BY-SA 4.0.</p>
</body></html>"""
    OUT.mkdir(exist_ok=True)
    (OUT / "report.html").write_text(page, encoding="utf-8")
    edge = next((e for e in EDGE if pathlib.Path(e).exists()), None)
    if edge:
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT / 'report.pdf'}", (OUT / "report.html").as_uri()],
                       check=True, capture_output=True, timeout=120)
        print(f"wrote {OUT / 'report.pdf'}")
    else:
        print(f"wrote {OUT / 'report.html'} (open it in a browser and Print -> Save as PDF)")


if __name__ == "__main__":
    main()
