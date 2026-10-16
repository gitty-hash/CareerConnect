"""Part B - Semantic job search with filter extraction.

Chapter 3, Step 2 (few-shot prompting, Zillow pattern): worked examples inside the prompt
teach the model to turn a free-text query into structured JSON filters. One example
deliberately mentions NO filters, so the model learns to return null for absent fields
instead of guessing.

Chapter 4: the filters are applied to posting metadata alongside vector / BM25 / hybrid
retrieval, and the remaining "semantic_query" (the need, without the filter words) is
what gets embedded.
"""

import re

from . import llm

CATEGORIES = ["Data Analytics", "Software Engineering", "Product Management", "Marketing",
              "UX/UI Design", "Finance", "Cybersecurity", "Operations & Supply Chain"]

FILTER_INSTRUCTION = f"""You convert internship search queries for CareerCompass, a student job marketplace, into search filters.
Return a JSON object with exactly these keys:
- "category": one of {CATEGORIES} or null
- "remote": one of "remote", "hybrid", "onsite" or null
- "paid": true, false, or null
- "visa_sponsorship": true, false, or null
- "semantic_query": the student's need rewritten as a short phrase WITHOUT the filter words
Set a field to null whenever the query does not clearly mention it. Never guess.
Follow the format of the examples exactly. Return ONLY the JSON object."""

# The "shots". Example 3 mentions no filter at all -> every filter is null.
FILTER_FEW_SHOT = """Query: "paid remote UX design internship, I'm an international student on F-1"
Output: {"category": "UX/UI Design", "remote": "remote", "paid": true, "visa_sponsorship": true, "semantic_query": "UX design internship"}

Query: "onsite finance internship in Phoenix, unpaid is fine, I'm a US citizen"
Output: {"category": "Finance", "remote": "onsite", "paid": null, "visa_sponsorship": null, "semantic_query": "finance internship in Phoenix"}

Query: "something where I can help a nonprofit with its website"
Output: {"category": null, "remote": null, "paid": null, "visa_sponsorship": null, "semantic_query": "help a nonprofit with its website"}

Query: "hybrid cybersecurity role that pays and sponsors visas"
Output: {"category": "Cybersecurity", "remote": "hybrid", "paid": true, "visa_sponsorship": true, "semantic_query": "cybersecurity role"}"""

FILTER_KEYS = ["category", "remote", "paid", "visa_sponsorship"]


def filter_messages(query):
    return [
        {"role": "system", "content": FILTER_INSTRUCTION},
        {"role": "user", "content": f"{FILTER_FEW_SHOT}\n\nQuery: \"{query}\"\nOutput:"},
    ]


def _clean(raw, query):
    out = {k: raw.get(k) for k in FILTER_KEYS}
    if out["category"] not in CATEGORIES:
        out["category"] = None
    if out["remote"] not in ("remote", "hybrid", "onsite"):
        out["remote"] = None
    for k in ("paid", "visa_sponsorship"):
        if not isinstance(out[k], bool):
            out[k] = None
    out["semantic_query"] = (raw.get("semantic_query") or query).strip() or query
    return out


def rule_based_filters(query):
    """Fallback for cached demo mode when a typed query was never sent to the LLM.
    Clearly labeled in the UI; it is a keyword heuristic, not the few-shot model."""
    q = query.lower()
    f = {"category": None, "remote": None, "paid": None, "visa_sponsorship": None}
    cat_words = {
        "Data Analytics": ["data anal", "analytics", "data science", "business intelligence", "data engineer"],
        "Software Engineering": ["software", "developer", "engineering intern", "backend", "frontend", "full-stack"],
        "Product Management": ["product manag", "product analyst", "program manag"],
        "Marketing": ["marketing", "social media", "content"],
        "UX/UI Design": ["ux", "ui ", "design", "user research"],
        "Finance": ["finance", "financial", "banking", "fp&a"],
        "Cybersecurity": ["cyber", "security", "soc "],
        "Operations & Supply Chain": ["supply chain", "operations", "logistics", "procurement"],
    }
    for cat, words in cat_words.items():
        if any(w in q for w in words):
            f["category"] = cat
            break
    for mode in ("remote", "hybrid", "onsite"):
        if mode in q or (mode == "onsite" and "on-site" in q):
            f["remote"] = mode
            break
    if re.search(r"\bunpaid\b|academic credit", q):
        f["paid"] = False
    elif re.search(r"\bpaid\b|\bpays\b", q):
        f["paid"] = True
    if re.search(r"sponsor|international|f-1|cpt|opt\b|visa", q):
        f["visa_sponsorship"] = True
    f["semantic_query"] = query
    return f


def extract_filters(query):
    """Returns (filters, source) where source is 'llm', 'cache' or 'rule-based fallback'."""
    msgs = filter_messages(query)
    was_cached = llm.is_cached(msgs, temperature=0, json_mode=True)
    try:
        raw = llm.chat_json(msgs, tag=f"filters:{query[:40]}", temperature=0)
        return _clean(raw, query), ("cache" if was_cached else "llm")
    except llm.NotCached:
        return rule_based_filters(query), "rule-based fallback"


def passes(posting, filters):
    if filters.get("category") and posting["category"] != filters["category"]:
        return False
    if filters.get("remote") and posting["remote"] != filters["remote"]:
        return False
    if filters.get("paid") is not None and posting["paid"] != filters["paid"]:
        return False
    if filters.get("visa_sponsorship") is True and not posting["visa_sponsorship"]:
        return False
    if filters.get("visa_sponsorship") is False and posting["visa_sponsorship"]:
        return False
    return True


def filtered_search(bb, filters, method="hybrid", k=5, alpha=0.5, rerank=False):
    """Score every posting with the chosen retriever, keep those passing the filters.
    (At platform scale you would push the filter into the ANN index; at 44 postings
    scoring everything is instant.)"""
    q = filters.get("semantic_query") or ""
    hits = bb.posting_search(q, method=method, alpha=alpha)
    kept = [(i, s) for i, s in hits if passes(bb.postings[i], filters)]
    if rerank:
        kept = bb.rerank_postings(q, kept, pool=10)
    return kept[:k], len([p for p in bb.postings if passes(p, filters)])
