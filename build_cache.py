"""Build every cached asset the app needs to run with NO API key (cached demo mode).

1. Embeds all three corpora and writes the FAISS indexes to cache/index/ (free, local).
2. With OPENAI_API_KEY set, records every LLM output used by the demo presets
   (filter extraction, recommendation explanations v1/v2, coach answers + judge verdicts,
   stress tests, employer copilot steps) into cache/llm_cache.json.
3. Runs the full evaluation (5 retrieval pipelines x 15 questions + refusal tests) and writes
   cache/eval_results.json and cache/eval_results.csv.

Usage (PowerShell):
    $env:OPENAI_API_KEY = "sk-..."      # or set it once with setx
    python build_cache.py               # all steps
    python build_cache.py --index-only  # step 1 only (no key needed)

Re-running is cheap: anything already in the cache is replayed, not re-billed.
Estimated cost of a full first run with gpt-4o-mini: well under $0.50.
"""

import sys
import time

from careercompass import advanced, coach, data, employer, evaluation, llm, recommend, search
from careercompass.backbone import Backbone
from careercompass.config import REACH_MIN_COSINE


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def main():
    t0 = time.time()
    step("1. Embedding corpora and building FAISS indexes")
    bb = Backbone()
    print(f"postings={len(bb.postings)}  chunks={len(bb.chunks)}  candidates={len(bb.candidates)}")
    if "--index-only" in sys.argv:
        return
    if not llm.api_key_available():
        sys.exit("OPENAI_API_KEY is not set - cannot record LLM outputs. (Indexes were built.)")
    sc = data.load_scenarios()

    step("2. Search: few-shot filter extraction for presets and benchmarks")
    for q in dict.fromkeys(sc["search_presets"] + [b["query"] for b in sc["search_benchmarks"]]):
        f, src = search.extract_filters(q)
        print(f"  [{src}] {q[:60]} -> { {k: f[k] for k in search.FILTER_KEYS} }")

    step("3. For You: grounded explanations (v2) and the naive first draft (v1)")
    for prof in sc["profiles"]:
        for i, sim in recommend.recommend(bb, prof["text"], k=5):
            for v in ("v2", "v1"):
                ex = recommend.explain_fit(prof["text"], bb.postings[i], v)
                print(f"  {prof['id']} {bb.postings[i]['posting_id']} {v}: fit={ex.get('fit_score')} "
                      f"({ex.get('fit_label')}) quotes {ex['quotes_verified']}/{ex['quotes_total']}")

    step("4. Coach: default pipeline, archived-excluded variant, and judge stress tests")
    questions = [x["q"] for x in sc["coach_eval"]] + sc["coach_out_of_scope"]
    for q in questions:
        for excl in (False, True):
            r = coach.ask(bb, q, pipeline="rerank", exclude_archived=excl)
            print(f"  rerank excl={excl!s:5} {r['status']:9} {r.get('badge'):24} {q[:55]}")
    for x in sc["coach_eval"]:
        r = coach.ask(bb, x["q"], pipeline="rerank", stress_test=True)
        print(f"  stress  {r['status']:9} {r.get('badge'):24} {x['q'][:55]}")

    step("5. Employer Reach Copilot")
    vocab = sorted({s for p in bb.postings for s in p["skills"]} | {s for c in bb.candidates for s in c["skills"]})
    for d in sc["employer_drafts"]:
        cons, _ = employer.extract_constraints(d["text"], vocab)
        before = employer.reach(bb, d["text"], cons, REACH_MIN_COSINE)
        fl = employer.all_flags(d["text"], cons)
        reviews = {r.get("index"): r for r in employer.judge_flags(d["text"], fl)}
        kept = [f for i, f in enumerate(fl)
                if bool(reviews.get(i, {}).get("keep", True)) and employer.quote_in_draft(f.get("quote", ""), d["text"])]
        if fl:
            new_text = employer.rewrite(d["text"], kept)
            new_cons, _ = employer.extract_constraints(new_text, vocab)
            after = employer.reach(bb, d["text"], new_cons, REACH_MIN_COSINE)
            print(f"  {d['id']}: {len(fl)} flags ({len(kept)} kept); qualified reach "
                  f"{before['qualified']} -> {after['qualified']} of {before['relevant']} relevant")
        else:
            print(f"  {d['id']}: no flags; qualified reach {before['qualified']} of {before['relevant']}")

    step("6. Evaluation: 5 pipelines x 15 questions + refusal tests")
    res = evaluation.run_evaluation(bb, sc, progress=lambda f, t: print(f"  {f:5.0%} {t}", flush=True))
    for pipe, s in res["summary"].items():
        print(f"  {pipe:12} recall={s['context_recall']:.2f} faith={s['faithfulness']:.2f} "
              f"relevance={s['answer_relevance']:.2f} answered={s['answered']}")
    print(f"  refusal accuracy: {res['refusal_accuracy']:.0%}")

    n, tags = llm.cache_stats()
    print(f"\nDone in {time.time() - t0:.0f}s. Cache holds {n} LLM outputs: {tags}")


if __name__ == "__main__":
    main()
