"""Part E - Evaluation, the way the Chapter 4 lab (Part 8) does it: RAGAS-style metrics
implemented directly so every number is explainable.

- context recall    : fraction of each question's gold guide(s) that retrieval surfaced.
                      Exact and reproducible - no LLM needed (gold sets are the answer key).
- faithfulness      : LLM judge - fraction of the answer's claims supported by the context.
- answer relevance  : LLM judge - how directly the answer addresses the question.
Plus two platform metrics:
- refusal accuracy  : out-of-scope questions must be refused / routed, never answered.
- search recall@5   : BM25 vs. vector vs. hybrid on the search benchmark queries.

This is the brief's sanctioned fallback ("recall against your ground truth plus an
LLM-judge score per answer") and is labeled as such in the app and README.
"""

import csv
import json
import time

from . import advanced, coach, llm
from .config import EVAL_RESULTS_CSV, EVAL_RESULTS_FILE

RAGAS_JUDGE_PROMPT = """You are evaluating a career-coach answer, RAGAS-style.

QUESTION: {question}

RETRIEVED CONTEXT (the only permitted evidence):
{context}

ANSWER TO EVALUATE:
{answer}

Return ONLY a JSON object:
{{"faithfulness": {{"score": <0.0-1.0, fraction of the answer's claims supported by the context>,
                    "explanation": "<one sentence>"}},
  "answer_relevance": {{"score": <0.0-1.0, how directly the answer addresses the question>,
                        "explanation": "<one sentence>"}}}}"""


def context_recall(retrieved_doc_ids, gold):
    gold = set(gold)
    return len(gold & set(retrieved_doc_ids)) / len(gold) if gold else 0.0


def ragas_messages(question, context, answer):
    return [{"role": "user", "content": RAGAS_JUDGE_PROMPT.format(
        question=question, context=context, answer=answer)}]


def score_answer(bb, result):
    """LLM-judged faithfulness + relevance of the answer the student would actually see."""
    if result["status"] != "answered":
        # A refusal of an in-scope question makes no claims (faithful) but helps no one.
        return {"faithfulness": {"score": 1.0, "explanation": "Refused/routed: no claims made."},
                "answer_relevance": {"score": 0.0, "explanation": "Refused/routed: question not answered."}}
    context = coach.format_excerpts(bb, result["hits"])
    return llm.chat_json(ragas_messages(result["question"], context, result["answer"]),
                         tag=f"ragas:{result['question'][:40]}", temperature=0)


def search_recall_at_k(bb, benchmarks, k=5):
    rows = []
    for b in benchmarks:
        gold = set(b["gold"])
        row = {"query": b["query"], "type": b["type"]}
        for method in ("bm25", "vector", "hybrid"):
            ids = [bb.postings[i]["posting_id"] for i, _ in bb.posting_search(b["query"], method)[:k]]
            row[method] = len(gold & set(ids)) / min(k, len(gold))
        rows.append(row)
    return rows


def run_evaluation(bb, scenarios, pipelines=None, progress=None):
    pipelines = pipelines or list(advanced.PIPELINES)
    questions = scenarios["coach_eval"]
    rows = []
    total = len(pipelines) * len(questions)
    n = 0
    for pipe in pipelines:
        for item in questions:
            res = coach.ask(bb, item["q"], pipeline=pipe)
            scores = score_answer(bb, res)
            rows.append({
                "pipeline": pipe,
                "question": item["q"],
                "gold": item["gold"],
                "retrieved": res["retrieved_doc_ids"],
                "status": res["status"],
                "judge_badge": res.get("badge"),
                "context_recall": context_recall(res["retrieved_doc_ids"], item["gold"]),
                "faithfulness": float(scores["faithfulness"]["score"]),
                "answer_relevance": float(scores["answer_relevance"]["score"]),
                "faithfulness_note": scores["faithfulness"]["explanation"],
                "relevance_note": scores["answer_relevance"]["explanation"],
                "answer": res["answer"],
            })
            n += 1
            if progress:
                progress(n / total, f"{pipe}: {item['q'][:50]}")

    summary = {}
    for pipe in pipelines:
        sub = [r for r in rows if r["pipeline"] == pipe]
        summary[pipe] = {m: round(sum(r[m] for r in sub) / len(sub), 3)
                         for m in ("context_recall", "faithfulness", "answer_relevance")}
        summary[pipe]["answered"] = sum(r["status"] == "answered" for r in sub)

    refusals = []
    for q in scenarios["coach_out_of_scope"]:
        res = coach.ask(bb, q, pipeline="rerank")
        refusals.append({"question": q, "status": res["status"],
                         "layer": res.get("refusal_layer", ""),
                         "top_similarity": round(res["top_similarity"], 3),
                         "correct": res["status"] in ("refused", "routed")})

    results = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Lab-style RAGAS metrics (Chapter 4 lab Part 8): exact context recall vs. gold "
                  "doc_ids + gpt-4o-mini judge for faithfulness and answer relevance. "
                  "This is the brief's sanctioned fallback to the ragas library.",
        "pipelines": {p: advanced.PIPELINES[p] for p in pipelines},
        "summary": summary,
        "rows": rows,
        "refusals": refusals,
        "refusal_accuracy": sum(r["correct"] for r in refusals) / len(refusals),
        "search_recall_at_5": search_recall_at_k(bb, scenarios["search_benchmarks"]),
    }
    EVAL_RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(EVAL_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["pipeline", "question", "gold", "retrieved", "status", "judge_badge",
                  "context_recall", "faithfulness", "answer_relevance"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gold": " ".join(r["gold"]), "retrieved": " ".join(r["retrieved"])})
    return results


def load_results():
    if EVAL_RESULTS_FILE.exists():
        return json.loads(EVAL_RESULTS_FILE.read_text(encoding="utf-8"))
    return None
