# Demo script (3–6 min)

Start the app with `uvicorn app.api:app --port 8000` and open http://localhost:8000.

1. **Intro (30 s).** "This is a Thailand travel assistant. It answers from 15 Wikivoyage
   guides with citations, uses a calculator tool for arithmetic, and says when the guides
   don't know." Show `/api/health`, which reports the chunk count, retriever and min_score.

2. **Success: multi-step with arithmetic (60 s).** Ask:
   *"Two people take the Lomprayah ferry to Ko Tao and back. How much in total?"*
   Narrate the streamed steps: the search is filtered to Ko Tao and finds the 750 baht fare,
   `calculate 750*2*2` returns 3000, and the final answer cites `ko-tao/get-in/...`.
   Click through to show the cited snippet.

3. **Success: abstaining (30 s).** Ask *"How much is a ticket on the Sukhothai metro?"*
   The search returns nothing above the min_score floor, so the agent says it couldn't find
   it instead of inventing a price.

4. **Honest failure (60 s).** Pick a real failure from section 5 of the report
   (`eval/results/answers_agent.jsonl`, any row where `keyword_correct` is false). Ask it live
   and explain the category: was it a retrieval miss, or did the model have the evidence and
   still answer wrongly?

5. **Evaluation (60 s).** Show `eval/results/answers.md`: agent vs closed-book vs single-shot RAG.
   Then show `eval/results/min_score.png`: "a floor of X keeps Y% of the evidence and rejects
   Z% of unanswerable questions; with no floor, every unanswerable question gets junk context."

6. **Safety (30 s).** In `app/schemas.py`, show the destination allow-list and the calculator
   pattern. In `app/tools.py`, show the AST sandbox. Mention that `.env` is not committed.

**For the live defence**, be ready to explain these:

- why hybrid retrieval with RRF;
- why the relevance labels are evidence phrases rather than chunk IDs;
- what one row of the results table means.
