"""Regenerate prompts.md from the prompt constants in the code, so the documented prompts
can never drift from what the app actually sends.

Run:  python tools/export_prompts.py
The "Observed results" section at the end is preserved between runs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from careercompass import advanced, coach, employer, evaluation, recommend, search  # noqa: E402

FENCE = "`" * 3
MARKER = "## Observed results of the prompt iteration"


def block(text):
    return f"{FENCE}text\n{text.strip()}\n{FENCE}\n"


DATA_SPEC = """Create a synthetic corpus for CareerCompass, a fictional internship marketplace for university students.

Corpus A - 44 internship postings as JSON with: posting_id, title, company (fictional), company_stage, category,
location, remote (remote|hybrid|onsite), paid, pay, visa_sponsorship, skills (list), description (>= 120 words).
- 8 categories (Data Analytics, Software Engineering, Product Management, Marketing, UX/UI Design, Finance,
  Cybersecurity, Operations & Supply Chain), a mix of remote / hybrid / onsite, some unpaid academic-credit roles.
- The description text must agree with the flags (state pay, and either "supports CPT/OPT" or "must be authorized
  without sponsorship"). Include two roles where a citizenship restriction is legally required (export control).
- Plant retrieval challenges: near-duplicate pairs that differ only in remote/pay/sponsorship (same company, or the
  same text at two companies); rare exact terms that appear in only one or two postings (PostgreSQL, SAP Ariba,
  OR-Tools); and two paraphrase targets whose text shares NO keywords with these queries:
  "help a hospital understand why patients miss appointments" and "make websites usable for people who are blind".

Corpus B - 14 career-guide articles (doc_id, title, topic, last_updated, status, disclaimer, body >= 300 words)
on resumes, graduate CVs, behavioral and technical interviews, networking, evaluating and negotiating offers,
etiquette, cover letters, work authorization, career fairs, return offers and portfolios.
- Plant conflicts: an undergraduate one-page rule vs. a graduate two-page exception; and an ARCHIVED 2021
  virtual-only career-fair guide that contradicts the current 2026 in-person playbook.
- Label the work-authorization article "SIMPLIFIED COURSE MATERIAL - NOT LEGAL GUIDANCE"."""

TABLE = """| # | Feature | File | Technique (chapter) | Temp. | Output |
|---|---|---|---|---|---|
| 1 | Search filter extraction | `search.py` | Few-shot prompting, structured JSON (Ch3 Step 2, Zillow pattern) | 0 | JSON |
| 2 | Recommendation explanation v1 -> v2 | `recommend.py` | Grounded generation, chain-of-thought scoring, honesty rule (Ch3 Steps 3+5, Ch4 lab Part 5) | 0 | JSON |
| 3 | Coach persona "Juno" | `coach.py` | System prompt + role-playing, RAG with citations, refusal (Ch3 Step 4, Ch4) | 0.3 | text |
| 4 | Coach guardrail judge | `coach.py` | LLM-as-judge (Ch3 Step 6, DoorDash pattern) | 0 | JSON |
| 5 | HyDE hypothetical answer | `advanced.py` | Hypothetical Document Embeddings (Ch4 lab Part 7) | 0.7 | text |
| 6 | RAG-fusion query rewrites | `advanced.py` | Multi-query rewriting + reciprocal rank fusion (Ch4 advanced RAG) | 0 | JSON |
| 7 | RAGAS-style evaluation judge | `evaluation.py` | Faithfulness + answer relevance (Ch4 lab Part 8) | 0 | JSON |
| 8 | Employer constraint extraction | `employer.py` | Few-shot structured extraction (Ch3 Step 2) | 0 | JSON |
| 9 | Employer language flags | `employer.py` | Zero-shot classification with a pinned label set (Ch3 Step 1) | 0 | JSON |
| 10 | Employer flag judge | `employer.py` | LLM-as-judge (Ch3 Step 6) | 0 | JSON |
| 11 | Employer posting rewrite | `employer.py` | Constrained generation (Ch3) | 0.2 | text |
"""


def build():
    md = ["# CareerCompass - Prompt Library\n",
          "Every prompt the app sends to `gpt-4o-mini`, generated from the code (`careercompass/*.py`) by "
          "`tools/export_prompts.py`. All calls go through one gateway, `careercompass/llm.py::chat()` (the "
          "Chapter 3 tutorial's single-helper pattern), which records each response in `cache/llm_cache.json`.\n",
          TABLE]
    md.append("## 1. Search filter extraction (few-shot)\n\nSystem message:\n" + block(search.FILTER_INSTRUCTION))
    md.append("User message = these shots + the new query (`Query: \"...\"` / `Output:`). Shot 3 mentions no "
              "filters, which teaches the model to return `null` instead of guessing (a homework requirement):\n"
              + block(search.FILTER_FEW_SHOT))
    md.append("## 2. Recommendation explanation - the documented prompt iteration\n\n"
              "### v1 (first draft, kept in the app for the honesty test)\n" + block(recommend.EXPLAIN_V1))
    md.append("**Problems with v1:** it asks the model to explain *why the posting is a great fit*, which "
              "presupposes a good match. It has no quote requirement, so explanations drift into generic "
              "praise, and no rule for weak matches.\n\n"
              "**Changes in v2:** (a) a stated stake (\"overselling wastes their time\"); (b) chain-of-thought "
              "steps *before* scoring (requirements -> evidence -> dealbreakers -> score); (c) an explicit scoring "
              "rubric; (d) a requirement to copy short **verbatim** quotes from both texts, verified afterwards in "
              "code (`recommend.verify_quotes`); (e) \"if the match is weak, say so in the first sentence\"; and "
              "(f) a `gaps` field.\n\n### v2 (production)\n" + block(recommend.EXPLAIN_V2))
    md.append("## 3. Coach persona - Juno (system prompt)\n" + block(coach.COACH_SYSTEM_PROMPT))
    md.append("User message template: `STUDENT QUESTION: ...` + `GUIDE EXCERPTS:` (each excerpt headed `[Gxx] title "
              "(status=..., updated ...)` plus its disclaimer when present). On a retry the judge's reasons are "
              "appended as `[INTERNAL reviewer feedback ...]` (the Ch3 `guarded_support_reply` pattern).\n\n"
              "Refusal has two layers: a cosine **scope gate** in code (no LLM call when nothing in the corpus is "
              "even loosely related), then rule 3 above (the model must reply `OUT_OF_SCOPE`). Both lead to the "
              "same route-to-a-human message:\n" + block(coach.ROUTE_TO_HUMAN))
    md.append("## 4. Coach guardrail judge\n" + block(coach.JUDGE_SYSTEM_PROMPT))
    md.append("A code check runs alongside the judge: every `[Gxx]` citation must exist and point at a retrieved "
              "guide.\n\nStress-test draft injected by the UI to exercise the judge and fallback path (the Ch3 "
              "Step 6 \"bad reply\" idea):\n" + block(coach.HALLUCINATED_DRAFT))
    md.append("## 5. HyDE\n" + block(advanced.hyde_messages("{question}")[0]["content"]))
    md.append("## 6. RAG-fusion rewrites\n" + block(advanced.fusion_messages("{question}")[0]["content"]))
    md.append("## 7. RAGAS-style evaluation judge\n" + block(evaluation.RAGAS_JUDGE_PROMPT))
    md.append("## 8. Employer constraint extraction (few-shot)\n" + block(employer.EXTRACT_INSTRUCTION)
              + "\nShots:\n" + block(employer.EXTRACT_FEW_SHOT))
    md.append("## 9. Employer language flags\n" + block(employer.FLAG_PROMPT))
    md.append("## 10. Employer flag judge\n" + block(employer.FLAG_JUDGE_PROMPT))
    md.append("## 11. Employer posting rewrite\n" + block(employer.REWRITE_PROMPT))
    md.append("## Data-generation prompts\n\nThe postings and guides were drafted with an AI coding assistant "
              "(Claude Code, model Claude Opus 5.5) from the specification below, then checked by script "
              "(word counts, consistency between flags and text, no keyword overlap for the paraphrase targets).\n\n"
              + block(DATA_SPEC))
    md.append("The 80 candidate profiles for the Employer Reach Copilot come from a seeded script, "
              "`data/generate_candidates.py`, not from an LLM. Following the Ch4 credit-scoring lab, they "
              "contain no name, age, gender, race or address.\n")
    return "\n".join(md)


if __name__ == "__main__":
    out = ROOT / "prompts.md"
    tail = MARKER + "\n\n_Run `python build_cache.py`, then record the v1 vs. v2 results here._\n"
    if out.exists() and MARKER in out.read_text(encoding="utf-8"):
        tail = MARKER + out.read_text(encoding="utf-8").split(MARKER, 1)[1]
    out.write_text(build() + "\n" + tail, encoding="utf-8")
    print(f"wrote {out}")
