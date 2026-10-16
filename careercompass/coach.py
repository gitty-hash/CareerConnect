"""Part D - "Ask the Coach": a RAG advisor with a persona, citations, refusal, and a judge.

Chapter 3 Step 4  system prompt + role-playing: "Juno" has a voice AND a policy.
Chapter 4         RAG: retrieve top-k guide chunks, generate an answer grounded in them,
                  cite each claim inline by doc_id.
Refusal           two layers: (1) a retrieval scope gate - if no chunk is even moderately
                  similar to the question, refuse before spending an LLM call; (2) the
                  prompt tells Juno to answer OUT_OF_SCOPE when the excerpts do not cover it.
Chapter 3 Step 6  LLM-as-judge guardrail (DoorDash pattern): a second model call checks
                  faithfulness to the retrieved chunks BEFORE display; a code-side check
                  verifies every citation was actually retrieved. On a fail: regenerate once
                  with the judge's feedback, then route to a human advisor.
"""

import re

from . import advanced, llm
from .config import SCOPE_GATE_MIN_COSINE, TOP_K_CHUNKS

COACH_SYSTEM_PROMPT = """You are Juno, the CareerCompass career coach for university students.

PERSONA
- Warm, direct, and practical, like a senior student who has already been through recruiting.
- 90-170 words. Start with the direct answer, then 2-4 concrete steps or tips.
- Plain language; no corporate filler.

SCOPE AND GROUNDING (never break these)
1. Answer ONLY from the GUIDE EXCERPTS provided in the user message. Do not use outside
   knowledge, even if you are confident.
2. Cite every factual sentence inline with its guide id in square brackets, e.g. [G03].
   Only cite ids that appear in the excerpts.
3. If the excerpts do not contain the answer, reply with exactly the single word OUT_OF_SCOPE
   and nothing else. Do not attempt a partial answer from memory.
4. If two excerpts conflict, prefer the one marked status=current and mention the difference.
5. If an excerpt carries a disclaimer (for example work-authorization material), end your
   answer by repeating that it is simplified course material and that the student must confirm
   with the international student office.
6. Never promise outcomes (offers, visas, pay) and never invent policies, deadlines, or numbers.
"""

JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness reviewer for CareerCompass's career coach.
You receive a student's question, the GUIDE EXCERPTS that were retrieved, and a DRAFT answer.
Evaluate the DRAFT ONLY, against the excerpts ONLY (ignore your own knowledge).

Check:
1. faithfulness - every factual claim in the draft is supported by the excerpts. Advice that
   goes beyond the excerpts, invented numbers, or invented policies are failures.
2. citations - claims cite the guide ids [Gxx] whose excerpt actually supports them.
3. safety - no promised outcomes; work-authorization answers include the simplified-material caveat
   when the excerpt has one.

Return ONLY a JSON object:
{"verdict": "pass" or "fail",
 "faithfulness": "pass" or "fail",
 "citations": "pass" or "fail",
 "safety": "pass" or "fail",
 "unsupported_claims": ["each claim not supported by the excerpts"],
 "reasons": ["short explanation for every fail"]}"""

ROUTE_TO_HUMAN = ("I can't answer this reliably from the CareerCompass guides, so I'm handing it to a "
                  "human. Please book a 20-minute appointment with a Career Center advisor "
                  "(Career Center > Appointments) and mention your question; they can help with "
                  "topics our guides don't cover.")

# A deliberately unfaithful draft, used by the UI's "stress test" button to show the judge
# and fallback path (same idea as the rule-breaking reply in Ch3 Step 6, Test B).
HALLUCINATED_DRAFT = ("Absolutely - most large companies will raise intern pay by 15-20% if you ask, "
                      "and the career center guarantees a matching offer if they refuse [G07]. "
                      "Always mention that you have a competing offer, even if you don't, because "
                      "recruiters never check [G07].")


def format_excerpts(bb, hits):
    blocks = []
    for idx, _ in hits:
        c = bb.chunks[idx]
        header = f"[{c['doc_id']}] {c['title']} (status={c['status']}, updated {c['last_updated']})"
        if c["disclaimer"]:
            header += f"\nDISCLAIMER: {c['disclaimer']}"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_messages(question, excerpts, feedback=""):
    user = f"STUDENT QUESTION:\n{question}\n\nGUIDE EXCERPTS:\n{excerpts}"
    if feedback:
        user += ("\n\n[INTERNAL reviewer feedback on your previous draft - fix these issues, "
                 f"do not mention them]: {feedback}")
    return [{"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def judge_messages(question, excerpts, draft):
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nGUIDE EXCERPTS:\n{excerpts}"
                                        f"\n\nDRAFT ANSWER:\n{draft}"}]


def cited_ids(text):
    return sorted(set(re.findall(r"\[(G\d{2})\]", text)))


def citation_check(draft, retrieved_doc_ids):
    """Code-side guardrail: citations must exist and point at retrieved guides."""
    cites = cited_ids(draft)
    bogus = [c for c in cites if c not in retrieved_doc_ids]
    return {"cited": cites, "not_retrieved": bogus, "ok": bool(cites) and not bogus}


def judge(question, excerpts, draft, retrieved_doc_ids):
    verdict = llm.chat_json(judge_messages(question, excerpts, draft),
                            tag=f"coach_judge:{question[:40]}", temperature=0)
    cc = citation_check(draft, retrieved_doc_ids)
    verdict["citation_code_check"] = cc
    if not cc["ok"]:
        verdict["verdict"] = "fail"
        verdict.setdefault("reasons", []).append(
            "Code check: answer has no citations." if not cc["cited"]
            else f"Code check: cites guides that were not retrieved: {cc['not_retrieved']}")
    return verdict


def ask(bb, question, pipeline="rerank", exclude_archived=False, k=TOP_K_CHUNKS,
        stress_test=False):
    """Full guarded RAG loop. Returns a dict the UI renders; raises llm.NotCached if a
    needed LLM call was never recorded in cached demo mode."""
    out = {"question": question, "pipeline": pipeline, "attempts": []}

    # Layer 1 - scope gate on raw similarity (no LLM call needed to refuse).
    top_sim = bb.chunk_vector_search(question, k=1, exclude_archived=exclude_archived)[0][1]
    out["top_similarity"] = top_sim
    hits, trace = advanced.retrieve(bb, question, pipeline, k=k, exclude_archived=exclude_archived)
    out["hits"], out["trace"] = hits, trace
    retrieved_ids = sorted({bb.chunks[i]["doc_id"] for i, _ in hits})
    out["retrieved_doc_ids"] = retrieved_ids
    if top_sim < SCOPE_GATE_MIN_COSINE:
        out.update(status="refused", refusal_layer="scope gate", answer=ROUTE_TO_HUMAN,
                   badge="refused")
        return out

    excerpts = format_excerpts(bb, hits)
    feedback = ""
    for attempt in range(2):                      # first draft + one regeneration
        if stress_test and attempt == 0:
            draft = HALLUCINATED_DRAFT
        else:
            draft = llm.chat(answer_messages(question, excerpts, feedback),
                             tag=f"coach_answer:{question[:40]}", temperature=0.3).strip()

        # Layer 2 - the model itself declares the question out of scope.
        if draft.upper().startswith("OUT_OF_SCOPE"):
            out["attempts"].append({"draft": draft, "verdict": None})
            out.update(status="refused", refusal_layer="coach prompt", answer=ROUTE_TO_HUMAN,
                       badge="refused")
            return out

        verdict = judge(question, excerpts, draft, retrieved_ids)
        out["attempts"].append({"draft": draft, "verdict": verdict})
        if verdict.get("verdict") == "pass":
            out.update(status="answered", answer=draft,
                       badge="pass" if attempt == 0 else "pass after retry")
            return out
        feedback = "; ".join(verdict.get("reasons", []) + verdict.get("unsupported_claims", []))

    out.update(status="routed", answer=ROUTE_TO_HUMAN, badge="fail -> routed to human")
    return out
