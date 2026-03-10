"""
Context Retriever - Semantic search over indexed project files.
Uses FAISS/cosine similarity to find relevant chunks.

Context-aware retrieval improvements:
  - build_retrieval_query(): blends recent conversation turns + extracts explicit
    file/symbol mentions so follow-up questions ("add a comment to it") still
    retrieve the correct chunks.
  - retrieve_with_history(): drop-in replacement for retrieve() that accepts
    conversation history and boosts scores for explicitly mentioned files.
"""
from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragebot.search.embedder import Embedder
    from ragebot.storage.db import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Regex that matches anything that looks like a file path in a message,
# e.g.  config.py  |  core/engine.py  |  src\utils\tokens.py
_FILE_MENTION_RE = re.compile(
    r"\b([\w/\\.-]+\.(?:py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|"
    r"swift|kt|cs|md|txt|json|yaml|yml|toml|cfg|ini|env))\b",
    re.IGNORECASE,
)


def extract_file_mentions(text: str) -> list[str]:
    """Return all file-path-like tokens found in *text*, lowercased."""
    return [m.lower() for m in _FILE_MENTION_RE.findall(text)]


def build_retrieval_query(
    current_query: str,
    messages: list[dict],
    context_window_turns: int = 3,
) -> tuple[str, list[str]]:
    """
    Build an enriched retrieval query from the current user message plus
    recent conversation history.

    Returns
    -------
    enriched_query : str
        A single string that blends the current query with recent context,
        used as the embedding lookup.
    mentioned_files : list[str]
        Lowercased file names / paths explicitly referenced across the
        selected turns.  These are used to boost retrieval scores later.
    """
    # Collect the last N *user* turns (excluding the current one)
    user_turns = [
        m["content"] for m in messages
        if m.get("role") == "user" and m.get("content") != current_query
    ]
    recent_turns = user_turns[-(context_window_turns - 1):]  # leave room for current

    # Extract file mentions from ALL selected turns + current query
    all_text = " ".join(recent_turns + [current_query])
    mentioned_files = list(dict.fromkeys(extract_file_mentions(all_text)))  # dedup, order-preserving

    # Build blended query: current query is most important, then recent turns
    # Use a simple weighted concatenation — the embedding model handles the rest
    blended_parts = [current_query]
    for turn in reversed(recent_turns):  # most recent first
        blended_parts.append(turn)

    enriched_query = " | ".join(blended_parts)
    return enriched_query, mentioned_files


# ---------------------------------------------------------------------------
# ContextRetriever
# ---------------------------------------------------------------------------

class ContextRetriever:
    def __init__(self, embedder: "Embedder", db: "Database", top_k: int = 5):
        self.embedder = embedder
        self.db = db
        self.top_k = top_k
        self._faiss_index = None
        self._faiss_metadata: list[dict] = []

    # ── Primary public API ────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Basic retrieval — uses *query* only, no conversation history.
        Kept for backward compatibility with non-chat code paths.
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed(query)

        try:
            results = self._faiss_retrieve(query_embedding, k)
            if results:
                return results
        except Exception:
            pass

        return self._brute_force_retrieve(query_embedding, k)

    def retrieve_with_history(
        self,
        query: str,
        messages: list[dict],
        top_k: int | None = None,
        context_window_turns: int = 3,
        cached_chunks: list[dict] | None = None,
    ) -> list[dict]:
        """
        Context-aware retrieval.

        Steps
        -----
        1. Build an enriched query from current message + recent history.
        2. Run semantic retrieval on the enriched query.
        3. Boost scores for chunks whose file_path matches any explicitly
           mentioned file in the conversation.
        4. Merge with *cached_chunks* from prior turns (deduped by file_path),
           re-ranking the combined list so the freshest/most relevant appear first.
        5. Return top-k of the merged, re-ranked list.
        """
        k = top_k or self.top_k

        # Step 1 — build enriched query + collect file mentions
        enriched_query, mentioned_files = build_retrieval_query(
            current_query=query,
            messages=messages,
            context_window_turns=context_window_turns,
        )

        # Step 2 — semantic retrieval on enriched query
        query_embedding = self.embedder.embed(enriched_query)
        try:
            fresh_results = self._faiss_retrieve(query_embedding, k * 2) or []
        except Exception:
            fresh_results = []
        if not fresh_results:
            fresh_results = self._brute_force_retrieve(query_embedding, k * 2)

        # Step 3 — boost scores for explicitly mentioned files
        if mentioned_files:
            fresh_results = self._boost_mentioned_files(fresh_results, mentioned_files)

        # Step 4 — merge with cached chunks from prior turns
        if cached_chunks:
            fresh_results = self._merge_with_cache(fresh_results, cached_chunks, mentioned_files)

        # Step 5 — return top-k
        fresh_results.sort(key=lambda r: r["score"], reverse=True)
        return fresh_results[:k]

    # ── Score boosting ────────────────────────────────────────────────────────

    def _boost_mentioned_files(
        self,
        results: list[dict],
        mentioned_files: list[str],
        boost: float = 0.25,
    ) -> list[dict]:
        """
        Add *boost* to the score of any chunk whose file_path partially matches
        one of the explicitly mentioned file names.  The boost is additive so
        that a strong semantic match always wins over a weak name match.
        """
        boosted = []
        for r in results:
            fp = r.get("file_path", "").lower().replace("\\", "/")
            extra = 0.0
            for mention in mentioned_files:
                mention_norm = mention.replace("\\", "/")
                # Match on filename alone (e.g. "config.py") or full sub-path
                if fp.endswith(mention_norm) or mention_norm in fp:
                    extra = boost
                    break
            boosted.append({**r, "score": r["score"] + extra})
        return boosted

    # ── Cache merging ─────────────────────────────────────────────────────────

    def _merge_with_cache(
        self,
        fresh: list[dict],
        cached: list[dict],
        mentioned_files: list[str],
        cache_decay: float = 0.15,
    ) -> list[dict]:
        """
        Merge fresh retrieval results with cached chunks from previous turns.

        Cached chunks receive a small score decay so that a strong fresh result
        always ranks above a stale cached one.  Chunks that are also explicitly
        mentioned in the current query do NOT get the decay applied (they stay
        relevant regardless of age).
        """
        # Index fresh results by file_path for dedup
        seen: dict[str, dict] = {r["file_path"]: r for r in fresh}

        for cached_chunk in cached:
            fp = cached_chunk.get("file_path", "")
            if fp in seen:
                # Already retrieved fresh — keep the fresher copy
                continue
            fp_lower = fp.lower().replace("\\", "/")
            is_mentioned = any(
                fp_lower.endswith(m.replace("\\", "/")) or m.replace("\\", "/") in fp_lower
                for m in mentioned_files
            )
            decayed_score = cached_chunk["score"] * (1.0 - (0.0 if is_mentioned else cache_decay))
            seen[fp] = {**cached_chunk, "score": decayed_score, "from_cache": True}

        return list(seen.values())

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def build_faiss_index(self):
        try:
            import faiss
            import numpy as np

            chunks = self.db.get_all_chunks()
            if not chunks:
                return

            embeddings = []
            self._faiss_metadata = []
            for chunk in chunks:
                emb = json.loads(chunk["embedding"])
                if emb:
                    embeddings.append(emb)
                    self._faiss_metadata.append(chunk)

            if not embeddings:
                return

            matrix = np.array(embeddings, dtype="float32")
            dim = matrix.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(matrix)
            self._faiss_index.add(matrix)

        except ImportError:
            self._faiss_index = None

    def _faiss_retrieve(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self._faiss_index is None:
            self.build_faiss_index()
        if self._faiss_index is None:
            return []

        import numpy as np
        import faiss

        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self._faiss_index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_metadata):
                continue
            chunk = self._faiss_metadata[idx]
            meta = json.loads(chunk.get("metadata", "{}")) if chunk.get("metadata") else {}
            results.append({
                "file_path": chunk["file_path"],
                "content":   chunk["content"],
                "score":     float(score),
                "file_type": meta.get("type", "unknown"),
                "summary":   meta.get("summary", ""),
                "functions": meta.get("functions", []),
                "classes":   meta.get("classes", []),
            })
        return results

    def _brute_force_retrieve(self, query_embedding: list[float], top_k: int) -> list[dict]:
        chunks = self.db.get_all_chunks()
        if not chunks:
            return []

        scored = []
        for chunk in chunks:
            try:
                emb = json.loads(chunk.get("embedding", "[]"))
                if not emb:
                    continue
                score = cosine_similarity(query_embedding, emb)
                meta  = json.loads(chunk.get("metadata", "{}")) if chunk.get("metadata") else {}
                scored.append({
                    "file_path": chunk["file_path"],
                    "content":   chunk["content"],
                    "score":     score,
                    "file_type": meta.get("type", "unknown"),
                    "summary":   meta.get("summary", ""),
                    "functions": meta.get("functions", []),
                    "classes":   meta.get("classes", []),
                })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]