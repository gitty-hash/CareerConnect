"""The retrieval backbone shared by Search, For You, the Coach, and the Employer Copilot.

Chapter 4 end to end, adapted from the 04_rag_lab notebook:
  Part 2  text embeddings      -> sentence-transformers all-MiniLM-L6-v2, normalized
  Part 3  vector database      -> FAISS IndexFlatIP (inner product = cosine)
  Part 4  keyword vs. vector   -> BM25 (rank-bm25) and a 50/50 min-max hybrid blend
  Part 6  re-ranking           -> cross-encoder ms-marco-MiniLM-L6-v2
plus the "semantic space" view: PCA projects the posting vectors to 2-D.

"One pattern, three products" (Ch4 closing slide): the same embed -> index -> retrieve
step serves search (query = typed text), recommendation (query = profile) and the RAG
coach (query = question). Only the query changes.
"""

import hashlib
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import data
from .config import EMBED_MODEL, HYBRID_ALPHA, INDEX_DIR, RERANK_MODEL


def tokenize(text):
    """BM25 tokenizer (Ch4 lab lowercases and splits; we also keep 'c++', 'c#', 'a/b')."""
    return re.findall(r"[a-z0-9+#/]+", text.lower())


def normalize_scores(hits):
    """Min-max rescale a [(item, score)] list to 0-1, keyed by position id (Ch4 lab)."""
    if not hits:
        return {}
    scores = [s for _, s in hits]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    return {i: (1.0 if span == 0 else (s - lo) / span) for i, s in hits}


class Backbone:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.postings = data.load_postings()
        self.guides = data.load_guides()
        self.chunks = data.build_chunks(self.guides)
        self.candidates = data.load_candidates()
        self.by_pid = {p["posting_id"]: p for p in self.postings}
        self.guide_by_id = {g["doc_id"]: g for g in self.guides}

        self.embedder = SentenceTransformer(EMBED_MODEL)   # ~90 MB on first run, then cached
        self._reranker = None

        self.posting_texts = [data.posting_text(p) for p in self.postings]
        self.chunk_texts = [data.chunk_embed_text(c) for c in self.chunks]
        self.candidate_texts = [data.candidate_text(c) for c in self.candidates]

        # Prebuilt index files ship with the project (cached demo mode); rebuilt only
        # when a corpus changes.
        self.posting_emb, self.posting_index = self._load_or_build("postings", self.posting_texts)
        self.chunk_emb, self.chunk_index = self._load_or_build("chunks", self.chunk_texts)
        self.candidate_emb, self.candidate_index = self._load_or_build("candidates", self.candidate_texts)

        self.posting_bm25 = BM25Okapi([tokenize(t) for t in self.posting_texts])
        self.chunk_bm25 = BM25Okapi([tokenize(t) for t in self.chunk_texts])

        self._fit_semantic_map()

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts):
        return self.embedder.encode(list(texts), normalize_embeddings=True,
                                    show_progress_bar=False).astype(np.float32)

    def _load_or_build(self, name, texts):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((EMBED_MODEL + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        meta_f = INDEX_DIR / f"{name}.meta.json"
        emb_f = INDEX_DIR / f"{name}.npy"
        idx_f = INDEX_DIR / f"{name}.faiss"
        if meta_f.exists() and emb_f.exists() and idx_f.exists():
            meta = json.loads(meta_f.read_text())
            if meta.get("digest") == digest:
                return np.load(emb_f), faiss.read_index(str(idx_f))
        emb = self.embed(texts)
        index = faiss.IndexFlatIP(emb.shape[1])          # IP = cosine on normalized vectors
        index.add(emb)
        np.save(emb_f, emb)
        faiss.write_index(index, str(idx_f))
        meta_f.write_text(json.dumps({"digest": digest, "model": EMBED_MODEL,
                                      "count": len(texts), "dim": int(emb.shape[1])}))
        return emb, index

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)  # ~90 MB on first use
        return self._reranker

    # ------------------------------------------------------------------ postings
    def posting_vector_search(self, query, k=None, qvec=None):
        """Embedding search over postings -> [(posting_index, cosine)] (Ch4 lab Part 3)."""
        k = k or len(self.postings)
        if qvec is None:
            qvec = self.embed([query])
        scores, idxs = self.posting_index.search(qvec.reshape(1, -1), k)
        return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0])]

    def posting_bm25_search(self, query, k=None):
        """Keyword search; documents with no shared word are non-matches (Ch4 lab Part 4)."""
        k = k or len(self.postings)
        scores = self.posting_bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def posting_hybrid_search(self, query, k=None, alpha=HYBRID_ALPHA):
        """Blend normalized BM25 and embedding scores: alpha*BM25 + (1-alpha)*vector."""
        k = k or len(self.postings)
        bm = normalize_scores(self.posting_bm25_search(query))
        vec = normalize_scores(self.posting_vector_search(query))
        blended = {i: alpha * bm.get(i, 0.0) + (1 - alpha) * vec.get(i, 0.0)
                   for i in set(bm) | set(vec)}
        return sorted(blended.items(), key=lambda x: -x[1])[:k]

    def posting_search(self, query, method="hybrid", alpha=HYBRID_ALPHA):
        if method == "bm25":
            return self.posting_bm25_search(query)
        if method == "vector":
            return self.posting_vector_search(query)
        return self.posting_hybrid_search(query, alpha=alpha)

    def rerank_postings(self, query, hits, pool=10):
        """Ch4 lab Part 6: cross-encoder reads (query, posting) together; re-order the pool."""
        pool_hits = hits[:pool]
        if not pool_hits:
            return []
        pairs = [(query, self.posting_texts[i]) for i, _ in pool_hits]
        ce = self.reranker.predict(pairs)
        order = np.argsort(ce)[::-1]
        return [(pool_hits[j][0], float(ce[j])) for j in order]

    # ------------------------------------------------------------------ guide chunks
    def chunk_vector_search(self, query=None, k=10, qvec=None, exclude_archived=False):
        if qvec is None:
            qvec = self.embed([query])
        n = len(self.chunks)
        scores, idxs = self.chunk_index.search(qvec.reshape(1, -1), n)
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if exclude_archived and self.chunks[i]["status"] == "archived":
                continue
            out.append((int(i), float(s)))
            if len(out) >= k:
                break
        return out

    def rerank_chunks(self, query, hits):
        pairs = [(query, self.chunk_texts[i]) for i, _ in hits]
        ce = self.reranker.predict(pairs)
        order = np.argsort(ce)[::-1]
        return [(hits[j][0], float(ce[j])) for j in order]

    # ------------------------------------------------------------------ candidates
    def candidate_similarities(self, text=None, qvec=None):
        if qvec is None:
            qvec = self.embed([text])
        return (self.candidate_emb @ qvec.reshape(-1)).astype(float)

    # ------------------------------------------------------------------ semantic map
    def _fit_semantic_map(self):
        """PCA (not t-SNE): PCA is a fixed linear map, so a NEW query vector can be
        projected into the same 2-D space - exactly what the query marker needs."""
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=2, random_state=0).fit(self.posting_emb)
        self.posting_xy = self.pca.transform(self.posting_emb)
        self.pca_explained = float(self.pca.explained_variance_ratio_.sum())

    def project(self, vecs):
        return self.pca.transform(np.asarray(vecs).reshape(-1, self.posting_emb.shape[1]))
