"""Part F - Innovation feature: the Employer Reach Copilot (stakeholder: employers).

Business problem: a marketplace only works when postings reach qualified students. Employers
often copy requirements from full-time job descriptions ("3+ years of experience", "US
citizens only", "recent graduates only") into internship postings, silently excluding most
of the student pool - and sometimes excluding people unlawfully or unfairly. The employer
never sees who they lost. The copilot shows them, BEFORE the posting goes live.

Decision supported: "Which requirements in my draft are worth their cost in reach?"

Pipeline (course techniques only):
1. Ch3 few-shot structured extraction : draft posting -> JSON constraints.
2. Ch4 retrieval, reversed            : the posting is the query; cosine against every
                                         candidate profile gives the semantically relevant pool.
3. Deterministic reach funnel          : each constraint applied to candidate metadata in code
                                         (exact counts, no LLM guessing), plus a counterfactual:
                                         how many students each single constraint excludes.
4. Ch3 prompting + LLM-as-judge        : flag exclusionary or inflated language; a judge scores
                                         each flag's justification and drops weak ones.
5. Ch3 generation                      : rewrite suggestion; its reach is re-measured (before/after).

Privacy: only aggregate counts are shown - never an individual candidate profile.
"""

import math
import re

from . import llm, search

EXTRACT_INSTRUCTION = """You extract the hiring constraints from an internship posting draft for CareerCompass.
Return a JSON object with exactly these keys:
- "category": one of %s or null
- "required_skills": list of skills the draft REQUIRES (not merely prefers); [] if none
- "min_experience_years": number of years of experience required, or null
- "work_authorization": "citizens_only", "no_sponsorship", "open", or null if not mentioned
- "authorization_legally_required": true if the draft states a legal reason (export control, ITAR,
  security clearance, government contract); otherwise false
- "work_mode": "remote", "hybrid", "onsite", or null
- "paid": true, false, or null
- "graduation_years": list of graduation years the draft restricts to, or null
Set fields to null when the draft does not mention them. Return ONLY the JSON object.""" % search.CATEGORIES

EXTRACT_FEW_SHOT = """Draft: "Marketing Intern (remote). Paid $20/hr. Must have 1+ year of social media experience and know Canva. Open to international students on CPT/OPT."
Output: {"category": "Marketing", "required_skills": ["social media", "Canva"], "min_experience_years": 1, "work_authorization": "open", "authorization_legally_required": false, "work_mode": "remote", "paid": true, "graduation_years": null}

Draft: "Satellite ground-systems security intern, onsite in Huntsville. Paid. Applicants must be U.S. citizens because the work requires a security clearance. Linux required."
Output: {"category": "Cybersecurity", "required_skills": ["Linux"], "min_experience_years": null, "work_authorization": "citizens_only", "authorization_legally_required": true, "work_mode": "onsite", "paid": true, "graduation_years": null}

Draft: "Help our team analyze survey results this summer."
Output: {"category": null, "required_skills": [], "min_experience_years": null, "work_authorization": null, "authorization_legally_required": false, "work_mode": null, "paid": null, "graduation_years": null}"""

FLAG_PROMPT = """You review internship postings for a university career marketplace before they go live.
Find language or requirements in the DRAFT that would unnecessarily shrink or bias the pool of
qualified STUDENT applicants. Look for:
- inflated experience (multi-year professional experience for an internship)
- work-authorization restrictions with no stated legal reason
- age-coded wording (e.g. "digital native", "young and energetic", "recent graduates only")
- gender-coded or exclusionary jargon (e.g. "rockstar", "ninja", "aggressive")
- unpaid work that looks like full-time staff work
- graduation-date restrictions that exclude current students
Do NOT flag requirements that are genuinely needed for the job or legally required.

DRAFT:
{draft}

Return ONLY JSON: {{"flags": [{{"quote": "<exact phrase from the draft>", "issue": "<category from the list>",
"why": "<one sentence on who it excludes and why that matters>", "suggestion": "<a better phrasing>"}}]}}
Return {{"flags": []}} if the draft has no such problems."""

FLAG_JUDGE_PROMPT = """You are a strict reviewer checking another model's flags on an internship posting draft.
For EACH flag decide whether it is justified: is the quoted phrase really in the draft, and does it
really exclude or bias qualified student applicants without a legitimate job or legal need?

DRAFT:
{draft}

FLAGS:
{flags}

Return ONLY JSON: {{"reviews": [{{"index": <flag index>, "keep": true or false,
"justification_score": <1-5, 5 = clearly justified>, "rationale": "<one sentence>"}}]}}"""

REWRITE_PROMPT = """Rewrite this internship posting draft so it keeps every genuine job duty and every
LEGALLY REQUIRED restriction, but fixes the issues listed below. Keep the same company, role, location,
and tools. Use inclusive, plain language suitable for university students. Do not invent pay, benefits,
or sponsorship that the draft does not offer, EXCEPT where an issue below explicitly recommends paying
interns, in which case write "Paid (rate to be confirmed)".

DRAFT:
{draft}

ISSUES TO FIX:
{issues}

Return only the rewritten posting text."""


def extract_messages(draft):
    return [{"role": "system", "content": EXTRACT_INSTRUCTION},
            {"role": "user", "content": f"{EXTRACT_FEW_SHOT}\n\nDraft: \"{draft}\"\nOutput:"}]


def rule_based_constraints(draft, skill_vocab):
    """Labeled fallback when a typed draft was never sent to the LLM (cached demo mode)."""
    d = draft.lower()
    years = re.search(r"(\d+)\s*\+?\s*(?:years|yrs)", d)
    auth = None
    if re.search(r"citizen", d):
        auth = "citizens_only"
    elif re.search(r"without (?:employer )?sponsorship|no (?:visa )?sponsorship|cannot sponsor", d):
        auth = "no_sponsorship"
    elif re.search(r"cpt|opt\b|international students are welcome|sponsor", d):
        auth = "open"
    mode = next((m for m in ("remote", "hybrid", "onsite") if m in d or (m == "onsite" and "on-site" in d)), None)
    paid = False if "unpaid" in d else (True if re.search(r"\$\d|\bpaid\b", d) else None)
    grad = [int(y) for y in re.findall(r"class of (\d{4})", d)] or None
    skills = [s for s in skill_vocab if re.search(r"(?<![a-z])" + re.escape(s.lower()) + r"(?![a-z])", d)]
    return {"category": search.rule_based_filters(draft)["category"], "required_skills": skills,
            "min_experience_years": float(years.group(1)) if years else None,
            "work_authorization": auth,
            "authorization_legally_required": bool(re.search(r"itar|export|clearance|government contract", d)),
            "work_mode": mode, "paid": paid, "graduation_years": grad}


def extract_constraints(draft, skill_vocab):
    try:
        return llm.chat_json(extract_messages(draft), tag="employer_extract", temperature=0), "llm/cache"
    except llm.NotCached:
        return rule_based_constraints(draft, skill_vocab), "rule-based fallback"


def _has_skill(cand, skill):
    s = skill.lower()
    return any(s in cs.lower() or cs.lower() in s for cs in cand["skills"])


def constraint_checks(cons):
    """Turn extracted constraints into (name, description, predicate) steps for the funnel."""
    steps = []
    skills = cons.get("required_skills") or []
    if skills:
        need = math.ceil(len(skills) / 2)
        steps.append(("skills", f"Has at least {need} of: {', '.join(skills)}",
                      lambda c, sk=skills, n=need: sum(_has_skill(c, s) for s in sk) >= n))
    yrs = cons.get("min_experience_years")
    if yrs:
        steps.append(("experience", f"{yrs:g}+ years of experience",
                      lambda c, y=yrs: c["experience_years"] >= y))
    auth = cons.get("work_authorization")
    if auth in ("citizens_only", "no_sponsorship"):
        label = "U.S. citizens only" if auth == "citizens_only" else "No CPT/OPT support"
        steps.append(("work authorization", label, lambda c: not c["needs_sponsorship"]))
    mode = cons.get("work_mode")
    if mode:
        accepted = {"remote": {"remote"}, "hybrid": {"hybrid", "onsite"}, "onsite": {"onsite"}}[mode]
        steps.append(("work mode", f"Willing to work {mode}",
                      lambda c, a=accepted: bool(a & set(c["work_modes"]))))
    if cons.get("paid") is False:
        steps.append(("pay", "Can accept an unpaid role", lambda c: not c["paid_only"]))
    grads = cons.get("graduation_years")
    if grads:
        steps.append(("graduation year", f"Graduating in {', '.join(map(str, grads))}",
                      lambda c, g=tuple(grads): c["grad_year"] in g))
    return steps


def reach(bb, draft_text, cons, min_cosine):
    """Deterministic funnel over the candidate pool. Returns counts only (privacy)."""
    sims = bb.candidate_similarities(draft_text)
    relevant = [c for c, s in zip(bb.candidates, sims) if s >= min_cosine]
    steps = constraint_checks(cons)
    funnel = [{"step": "Semantically relevant students", "rule": f"cosine >= {min_cosine}",
               "remaining": len(relevant)}]
    pool = relevant
    for name, desc, pred in steps:
        pool = [c for c in pool if pred(c)]
        funnel.append({"step": name, "rule": desc, "remaining": len(pool)})

    # Two views of each requirement's cost:
    # - on its own: how many relevant students this requirement alone rules out;
    # - if dropped: how many come back when ONLY this one is removed (0 when another
    #   requirement already excludes the same students - redundant blockers).
    costs = []
    for j, (name, desc, pred) in enumerate(steps):
        others = [p for i, (_, _, p) in enumerate(steps) if i != j]
        without = [c for c in relevant if all(p(c) for p in others)]
        costs.append({"constraint": name, "rule": desc,
                      "excluded_on_its_own": sum(not pred(c) for c in relevant),
                      "regained_if_dropped": len(without) - len(pool)})

    programs = {}
    for c in pool:
        programs[c["program"]] = programs.get(c["program"], 0) + 1
    return {"relevant": len(relevant), "qualified": len(pool), "funnel": funnel,
            "costs": costs, "programs": programs, "pool_size": len(bb.candidates)}


def flags(draft):
    return llm.chat_json([{"role": "user", "content": FLAG_PROMPT.format(draft=draft)}],
                         tag="employer_flags", temperature=0).get("flags", [])


_AUTH_PHRASE = re.compile(r"(must be (?:a )?U\.?S\.? citizens?|U\.?S\.? citizens? only|"
                          r"without (?:employer )?sponsorship|no (?:visa )?sponsorship|cannot sponsor)", re.I)


def code_flags(draft, cons):
    """Code-side check (Ch4 lab Part 5: prompts ask nicely, code verifies). The LLM flagger can
    miss the most expensive restriction, so any work-authorization restriction without a stated
    legal basis is always flagged, straight from the extracted constraints."""
    if cons.get("work_authorization") not in ("citizens_only", "no_sponsorship"):
        return []
    if cons.get("authorization_legally_required"):
        return []
    m = _AUTH_PHRASE.search(draft)
    return [{"quote": m.group(0) if m else "work-authorization restriction",
             "issue": "work-authorization restriction with no stated legal reason",
             "why": "It excludes every international student on CPT/OPT, and the draft gives no legal "
                    "basis (such as export control or a security clearance) that would require it.",
             "suggestion": "Remove the restriction, or state the specific legal requirement if one applies.",
             "source": "code check"}]


def all_flags(draft, cons):
    """LLM flags plus the code-side authorization check (deduplicated)."""
    out = [dict(f, source="LLM") for f in flags(draft)]
    already = any(re.search(r"citizen|sponsor|authoriz", f.get("quote", "") + f.get("issue", ""), re.I) for f in out)
    if not already:
        out += code_flags(draft, cons)
    return out


def judge_flags(draft, flag_list):
    if not flag_list:
        return []
    listing = "\n".join(f"{i}. \"{f.get('quote', '')}\" - {f.get('issue', '')}: {f.get('why', '')}"
                        for i, f in enumerate(flag_list))
    out = llm.chat_json([{"role": "user", "content": FLAG_JUDGE_PROMPT.format(draft=draft, flags=listing)}],
                        tag="employer_flag_judge", temperature=0)
    return out.get("reviews", [])


def rewrite(draft, kept_flags):
    issues = "\n".join(f"- \"{f.get('quote', '')}\" ({f.get('issue', '')}): {f.get('suggestion', '')}"
                       for f in kept_flags) or "- (no issues)"
    return llm.chat([{"role": "user", "content": REWRITE_PROMPT.format(draft=draft, issues=issues)}],
                    tag="employer_rewrite", temperature=0.2).strip()


def quote_in_draft(quote, draft):
    return bool(quote) and quote.lower().strip(" .\"'") in draft.lower()
