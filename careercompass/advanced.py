"""Part E - Advanced RAG techniques for the Coach's retrieval step (Chapter 4, section 06).

  baseline     : embed the question, FAISS top-3 chunks                 (Ch4 lab Part 3)
  rerank       : retrieve 10, cross-encoder re-ranks, keep 3            (Ch4 lab Part 6)
  hyde         : embed an LLM-written hypothetical answer, search with it (Ch4 lab Part 7)
  fusion       : RAG fusion - LLM writes 3 query rewrites, each is searched,
                 results merged with Reciprocal Rank Fusion (Ch4 slide "RAG Fusion";
                 the lab notes RRF merges rank positions instead of scores)
  hyde_rerank  : union of question + HyDE candidates, cross-encoder picks the best 3
                 (the "upgraded" pipeline: two techniques stacked)
"""

from . import llm
from .config import RERANK_POOL, RRF_K, TOP_K_CHUNKS

PIPELINES = {
    "baseline": "Baseline - vector top-3",
    "rerank": "Re-rank - vector top-10 -> cross-encoder top-3",
    "hyde": "HyDE - search with a hypothetical answer",
    "fusion": "RAG fusion - 3 rewrites + reciprocal rank fusion",
    "hyde_rerank": "Upgraded - HyDE + question pool -> cross-encoder top-3",
}


def hyde_messages(question):
    return [{"role": "user",
             "content": "In 3-4 sentences, answer this student's career question the way a "
                        f"university career-center guide article would: {question}"}]


def hypothetical_answer(question):
    """Step 1 of HyDE (Ch4 lab Part 7): the LLM imagines what a good answer would say.
    temperature=0.7 as in the lab; the shopper (here: the student) never sees it."""
    return llm.chat(hyde_messages(question), tag=f"hyde:{question[:40]}", temperature=0.7)


def fusion_messages(question):
    return [{"role": "user",
             "content": "Rewrite the student's question below into 3 different search queries "
                        "that a career-guide search engine could match. Use different vocabulary "
                        "in each (synonyms, the formal term, the underlying topic). "
                        'Return ONLY JSON: {"queries": ["...", "...", "..."]}\n\n'
                        f"Question: {question}"}]


def multi_queries(question):
    out = llm.chat_json(fusion_messages(question), tag=f"fusion:{question[:40]}", temperature=0)
    return [q for q in out.get("queries", []) if isinstance(q, str) and q.strip()][:3]


def reciprocal_rank_fusion(rank_lists, k=RRF_K):
    """RRF: score(d) = sum over lists of 1 / (k + rank). Rank-based, so BM25-scale vs.
    cosine-scale differences do not matter."""
    scores = {}
    for hits in rank_lists:
        for rank, (idx, _) in enumerate(hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(bb, question, pipeline="rerank", k=TOP_K_CHUNKS, exclude_archived=False):
    """Run one retrieval pipeline. Returns hits [(chunk_idx, score)] plus a trace dict
    describing what the technique did (shown in the UI's before/after table)."""
    trace = {"pipeline": pipeline}
    base_pool = bb.chunk_vector_search(question, k=RERANK_POOL, exclude_archived=exclude_archived)
    trace["baseline_pool"] = base_pool

    if pipeline == "baseline":
        hits = base_pool[:k]
    elif pipeline == "rerank":
        hits = bb.rerank_chunks(question, base_pool)[:k]
    elif pipeline == "hyde":
        hypo = hypothetical_answer(question)
        trace["hypothetical_answer"] = hypo
        hits = bb.chunk_vector_search(hypo, k=k, exclude_archived=exclude_archived)
    elif pipeline == "fusion":
        rewrites = multi_queries(question)
        trace["rewrites"] = rewrites
        lists = [base_pool] + [bb.chunk_vector_search(q, k=RERANK_POOL, exclude_archived=exclude_archived)
                               for q in rewrites]
        hits = reciprocal_rank_fusion(lists)[:k]
    elif pipeline == "hyde_rerank":
        hypo = hypothetical_answer(question)
        trace["hypothetical_answer"] = hypo
        hyde_pool = bb.chunk_vector_search(hypo, k=RERANK_POOL, exclude_archived=exclude_archived)
        pool, seen = [], set()
        for idx, s in base_pool + hyde_pool:
            if idx not in seen:
                seen.add(idx)
                pool.append((idx, s))
        trace["pool_size"] = len(pool)
        hits = bb.rerank_chunks(question, pool)[:k]
    else:
        raise ValueError(f"unknown pipeline {pipeline}")
    return hits, trace


def rank_table(bb, hits, trace, gold=None):
    """Before/after rows: where each final chunk sat in the plain vector ranking."""
    base_rank = {idx: r for r, (idx, _) in enumerate(trace["baseline_pool"], start=1)}
    rows = []
    for new_rank, (idx, score) in enumerate(hits, start=1):
        c = bb.chunks[idx]
        rows.append({
            "new rank": new_rank,
            "vector-only rank": base_rank.get(idx, ">10"),
            "chunk": c["chunk_id"],
            "guide": f"{c['doc_id']} - {c['title']}",
            "score": round(score, 3),
            "gold": ("yes" if gold and c["doc_id"] in gold else ("no" if gold else "")),
        })
    return rows
