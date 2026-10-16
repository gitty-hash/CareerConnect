"""Load the two corpora (plus the candidate pool) and turn them into retrievable text.

Chapter 4 concepts: the knowledge base, and chunking (the "How RAG can go wrong" slide:
large chunks give imprecise search, small chunks lose context, overlap preserves
context across boundaries).
"""

import json
import re

from .config import CHUNK_OVERLAP, CHUNK_WORDS, DATA_DIR


def load_json(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def load_postings():
    return load_json("postings.json")


def load_guides():
    return load_json("guides.json")


def load_candidates():
    return load_json("candidates.json")


def load_scenarios():
    return load_json("scenarios.json")


# --- Corpus A: one posting = one chunk ---------------------------------------
def posting_text(p):
    """The text we embed and search for a posting (Ch4 lab doc_text pattern).

    Structured facts go FIRST: all-MiniLM-L6-v2 truncates after 256 tokens, so if
    anything is cut it is the tail of the description, never the title, category,
    work mode, pay, sponsorship, or skills.
    """
    sponsorship = ("Open to international students (CPT/OPT supported)."
                   if p["visa_sponsorship"] else "No visa sponsorship.")
    return (f"{p['title']} at {p['company']} ({p['company_stage']}). "
            f"Category: {p['category']}. {p['remote'].capitalize()}, {p['location']}. "
            f"{'Paid' if p['paid'] else 'Unpaid'}: {p['pay']}. {sponsorship} "
            f"Skills: {', '.join(p['skills'])}. {p['description']}")


# --- Corpus B: guide articles are chunked ------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")


def _sentences(text):
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return out


def chunk_article(body, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Sentence-aware sliding window: pack whole sentences up to ~`size` words,
    then start the next chunk with the last ~`overlap` words of sentences so an
    idea that straddles a boundary appears complete in at least one chunk."""
    sents = _sentences(body)
    chunks, current = [], []
    for s in sents:
        current.append(s)
        if sum(len(x.split()) for x in current) >= size:
            chunks.append(" ".join(current))
            carry, words = [], 0
            for prev in reversed(current):          # overlap: carry trailing sentences
                if words >= overlap:
                    break
                carry.insert(0, prev)
                words += len(prev.split())
            current = carry if len(carry) < len(current) else []
    if current:
        tail = " ".join(current)
        if not chunks or tail not in chunks[-1]:
            chunks.append(tail)
    return chunks


def build_chunks(guides):
    """Chunk every guide; each chunk carries its parent doc's id, title and status."""
    rows = []
    for g in guides:
        for i, text in enumerate(chunk_article(g["body"]), start=1):
            rows.append({
                "chunk_id": f"{g['doc_id']}-c{i}",
                "doc_id": g["doc_id"],
                "title": g["title"],
                "status": g["status"],
                "last_updated": g["last_updated"],
                "disclaimer": g["disclaimer"],
                "text": text,
            })
    return rows


def chunk_embed_text(c):
    """Contextual header: prefix each chunk with its article title so a chunk like
    'Bring printed copies...' still knows it is about career fairs."""
    return f"{c['title']}. {c['text']}"


def candidate_text(c):
    return c["summary"]
