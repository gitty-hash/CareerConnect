"""Paths, model names, and tunable constants in one place."""

import os
from pathlib import Path

# Same quality-of-life settings as the Chapter 4 lab setup cell.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
INDEX_DIR = CACHE_DIR / "index"
LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"
EVAL_RESULTS_FILE = CACHE_DIR / "eval_results.json"
EVAL_RESULTS_CSV = CACHE_DIR / "eval_results.csv"

# --- Models (Chapter 4 lab stack) -------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"                     # 384-dim, free, local (Ch4 Part 2)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2" # cross-encoder re-ranker (Ch4 Part 6)
LLM_MODEL = "gpt-4o-mini"                            # generation + judging (Ch3/Ch4)

# --- Chunking (Ch4 "How RAG can go wrong: chunking") -------------------------
# all-MiniLM-L6-v2 silently truncates input after 256 word-piece tokens (~190 words),
# so a chunk must stay well under that or its tail is never embedded.
CHUNK_WORDS = 110      # target chunk size in words  (~150 tokens incl. title header)
CHUNK_OVERLAP = 30     # words of overlap between consecutive chunks

# --- Retrieval ---------------------------------------------------------------
TOP_K_POSTINGS = 5
TOP_K_CHUNKS = 3
RERANK_POOL = 10       # retrieve wide, then re-rank (Ch4 lab Part 6)
HYBRID_ALPHA = 0.5     # 50/50 blend of BM25 and embedding scores (Ch4 lab Part 4)
RRF_K = 60             # reciprocal-rank-fusion constant (RAG fusion)

# Coach scope gate: if no guide chunk is at least this similar to the question,
# refuse before calling the LLM (cheap, deterministic first line of refusal).
SCOPE_GATE_MIN_COSINE = 0.22   # calibrated: lowest in-scope question scored 0.278

# Employer Reach Copilot: a candidate is "semantically relevant" to a posting
# when cosine(posting, candidate profile) >= this threshold.
REACH_MIN_COSINE = 0.45        # calibrated on the draft postings (see README)
