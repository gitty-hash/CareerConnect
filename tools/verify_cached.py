"""Check that every preset in every tab works in cached demo mode (no API key, no network LLM).

Run:  python tools/verify_cached.py
Exits non-zero if any preset needs an LLM output that is missing from cache/llm_cache.json.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop("OPENAI_API_KEY", None)

from careercompass import advanced, coach, data, employer, evaluation, llm, recommend, search  # noqa: E402
from careercompass.backbone import Backbone  # noqa: E402
from careercompass.config import REACH_MIN_COSINE  # noqa: E402

llm.set_force_cached(True)
bb = Backbone()
sc = data.load_scenarios()
missing = []


def check(label, fn):
    try:
        fn()
        return True
    except llm.NotCached:
        missing.append(label)
        return False


for q in sc["search_presets"] + [b["query"] for b in sc["search_benchmarks"]]:
    f, src = search.extract_filters(q)
    if src == "rule-based fallback":
        missing.append(f"search filters: {q}")

for prof in sc["profiles"]:
    for i, _ in recommend.recommend(bb, prof["text"], k=5):
        for v in ("v1", "v2"):
            check(f"explain {v} {prof['id']} {bb.postings[i]['posting_id']}",
                  lambda: recommend.explain_fit(prof["text"], bb.postings[i], v))

questions = [x["q"] for x in sc["coach_eval"]] + sc["coach_out_of_scope"]
for q in questions:
    for excl in (False, True):
        check(f"coach rerank excl={excl}: {q}", lambda: coach.ask(bb, q, "rerank", exclude_archived=excl))
for x in sc["coach_eval"]:
    check(f"coach stress: {x['q']}", lambda: coach.ask(bb, x["q"], "rerank", stress_test=True))
    for p in advanced.PIPELINES:
        check(f"coach {p}: {x['q']}", lambda: coach.ask(bb, x["q"], p))

vocab = sorted({s for p in bb.postings for s in p["skills"]} | {s for c in bb.candidates for s in c["skills"]})
for d in sc["employer_drafts"]:
    def run_employer():
        cons, src = employer.extract_constraints(d["text"], vocab)
        assert src != "rule-based fallback"
        fl = employer.all_flags(d["text"], cons)
        rv = {r.get("index"): r for r in employer.judge_flags(d["text"], fl)}
        kept = [f for i, f in enumerate(fl) if bool(rv.get(i, {}).get("keep", True))
                and employer.quote_in_draft(f.get("quote", ""), d["text"])]
        if fl:
            new_text = employer.rewrite(d["text"], kept)
            employer.extract_constraints(new_text, vocab)
            employer.reach(bb, d["text"], cons, REACH_MIN_COSINE)
    check(f"employer {d['id']}", run_employer)

res = evaluation.load_results()
if not res:
    missing.append("cache/eval_results.json")

if missing:
    print(f"FAIL - {len(missing)} preset(s) not cached:")
    for m in missing:
        print("  -", m)
    sys.exit(1)
print("OK - every preset in every tab replays from the cache with no API key.")
