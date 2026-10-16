"""Part C - "For You" recommendations: the profile IS the query.

Chapter 4 ("RAG meets the platform": in recommendation the user profile is the query)
- the student's profile text is embedded with the same model and used as the query
  vector against the posting index; top-5 postings are returned.

Chapter 3 + Chapter 4 lab Part 5 (grounded generation + code-side guardrail)
- an LLM writes a why-this-fits-you explanation that must quote phrases from BOTH the
  profile and the posting; code then verifies every quote really appears in the source
  text ("prompts ask nicely; validation checks").
- the fit score comes with step-by-step reasoning shown on demand (Ch3 chain-of-thought).

Two prompt versions are kept on purpose (documented iteration in prompts.md):
  v1: the naive first draft - no honesty rule, no quote requirement (it oversells).
  v2: grounded + honest - quotes required, weak matches must be called weak.
"""

import re

from . import data, llm

EXPLAIN_V1 = """You are CareerCompass's recommendation assistant.
Student profile:
{profile}

Recommended internship:
{posting}

Explain why this internship is a great fit for the student and give a fit score from 0 to 100.
Return ONLY JSON: {{"fit_score": <0-100>, "fit_label": "strong|moderate|weak", "explanation": "<2-3 sentences>",
"profile_quotes": [], "posting_quotes": [], "gaps": [], "reasoning": "<your reasoning>"}}"""

EXPLAIN_V2 = """You are CareerCompass's recommendation assistant. Students rely on you to be honest:
overselling a poor match wastes their time and erodes trust in the platform.

STUDENT PROFILE:
{profile}

RETRIEVED INTERNSHIP POSTING:
{posting}

Think step by step before scoring:
1. List what the posting requires (skills, experience, work mode, pay, work authorization).
2. For each requirement, find concrete evidence in the profile - or note that it is missing.
3. Check dealbreakers: work authorization, paid vs. unpaid, remote/onsite, career direction.
4. Only then assign a fit score.

Scoring guide: 80-100 strong (most requirements evidenced, no dealbreaker); 50-79 moderate
(some evidence, real gaps); 0-49 weak (little evidence, different career direction, or a dealbreaker).

Rules:
- "profile_quotes" and "posting_quotes": copy 1-3 SHORT phrases (3-12 words) EXACTLY, word for word,
  from the profile and from the posting. Never paraphrase inside a quote.
- The explanation must reference those specific phrases. Generic praise ("great opportunity",
  "perfect fit") without evidence is not allowed.
- If the match is weak, say so plainly in the first sentence and explain why. Do not oversell.
- List concrete gaps or dealbreakers in "gaps" (empty list only if there truly are none).

Return ONLY JSON:
{{"reasoning": "<steps 1-3, concise>", "fit_score": <0-100>, "fit_label": "strong|moderate|weak",
"explanation": "<2-3 sentences addressed to the student>", "profile_quotes": ["..."],
"posting_quotes": ["..."], "gaps": ["..."]}}"""

PROMPTS = {"v1": EXPLAIN_V1, "v2": EXPLAIN_V2}


def recommend(bb, profile_text, k=5):
    """Embed the profile and use it as the query vector (Ch4). Returns [(idx, cosine)]."""
    return bb.posting_vector_search(profile_text, k=k)


def must_have_conflicts(posting, must_haves):
    """Dealbreaker check the student controls - shown as badges, never silently hidden."""
    issues = []
    if must_haves.get("needs_sponsorship") and not posting["visa_sponsorship"]:
        issues.append("Does not support CPT/OPT")
    if must_haves.get("paid_only") and not posting["paid"]:
        issues.append("Unpaid")
    if must_haves.get("remote_only") and posting["remote"] != "remote":
        issues.append(f"{posting['remote'].capitalize()}, not remote")
    return issues


def explain_messages(profile_text, posting, version="v2"):
    prompt = PROMPTS[version].format(profile=profile_text, posting=data.posting_text(posting))
    return [{"role": "user", "content": prompt}]


def _norm(s):
    s = s.lower().replace("’", "'").replace("—", "-").replace("–", "-")
    return re.sub(r"[^a-z0-9%$/+#.' -]+", " ", re.sub(r"\s+", " ", s)).strip(" .")


def verify_quotes(quotes, source):
    """Code-side guardrail: is every quoted phrase really in the source text?"""
    src = _norm(source)
    return [(q, bool(q.strip()) and _norm(q) in src) for q in quotes]


def explain_fit(profile_text, posting, version="v2"):
    """Returns the parsed explanation plus quote verification, or raises llm.NotCached."""
    result = llm.chat_json(explain_messages(profile_text, posting, version),
                           tag=f"explain_{version}:{posting['posting_id']}", temperature=0)
    posting_source = data.posting_text(posting)
    result["profile_quote_check"] = verify_quotes(result.get("profile_quotes", []), profile_text)
    result["posting_quote_check"] = verify_quotes(result.get("posting_quotes", []), posting_source)
    checks = result["profile_quote_check"] + result["posting_quote_check"]
    result["quotes_verified"] = sum(ok for _, ok in checks)
    result["quotes_total"] = len(checks)
    has_both = bool(result["profile_quote_check"]) and bool(result["posting_quote_check"])
    result["grounded"] = has_both and result["quotes_verified"] == result["quotes_total"]
    return result
