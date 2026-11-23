"""Lightweight Hybrid Retrieval Engine: SQLite FTS5 BM25 + Character/Word TF-IDF + RRF Fusion.

Designed specifically for Persian + English on 8GB RAM / CPU-only PCs:
- Zero heavy neural embeddings or GPU dependencies
- Handles Persian morphological suffixes (`پروژه‌ها`, `سایه`, ZWNJ variations) via char_wb(3,5) TF-IDF
- Combines FTS5 exact/token BM25 ranks with TF-IDF cosine ranks using Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.language import normalize_persian
from memory.database import DatabaseManager, db_manager
from memory.long_memory import LongTermMemory, long_memory

PERSIAN_STOPWORDS = {
    "است", "هست", "بود", "شد", "چه", "چی", "چیست", "کیست", "کجا", "چطور", "چگونه",
    "این", "آن", "را", "به", "از", "در", "با", "برای", "که", "من", "تو", "ما", "شما",
    "آیا", "لطفا", "لطفاً", "بگو", "درباره", "مورد",
}


def _build_fts_query(norm_query: str) -> str:
    tokens = [
        t for t in re.findall(r"[\w\u0600-\u06FF]{2,}", norm_query)
        if t not in PERSIAN_STOPWORDS
    ]
    if not tokens:
        tokens = re.findall(r"[\w\u0600-\u06FF]{2,}", norm_query)
    if not tokens:
        return ""
    # Quote each token safely for SQLite FTS5 OR matching
    return " OR ".join(f'"{t}"' for t in tokens[:10])


class HybridRetriever:
    """Fast local RAG retriever over Long-Term Memories and Distilled Teacher Exemplars."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        ltm: Optional[LongTermMemory] = None,
    ) -> None:
        self.db = db or db_manager
        self.ltm = ltm or long_memory

    def search_memories(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.12,
        mark_used: bool = True,
    ) -> List[Dict[str, Any]]:
        norm_q = normalize_persian(query)
        if not norm_q:
            return []

        active_memories = self.ltm.list_memories(include_inactive=False, limit=300)
        if not active_memories:
            return []

        by_id: Dict[int, Dict[str, Any]] = {int(m["id"]): m for m in active_memories}

        # 1. SQLite FTS5 BM25 Rank
        fts_ranks: Dict[int, int] = {}
        fts_q = _build_fts_query(norm_q)
        if self.db.fts5_available and fts_q:
            try:
                with self.db.connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT memory_id, bm25(memories_fts) AS rank_score
                        FROM memories_fts
                        WHERE memories_fts MATCH ?
                        ORDER BY rank_score ASC
                        LIMIT 20
                        """,
                        (fts_q,),
                    ).fetchall()
                for idx, r in enumerate(rows):
                    mid = int(r["memory_id"])
                    if mid in by_id:
                        fts_ranks[mid] = idx + 1
            except Exception:
                pass

        # 2. Subword Character n-gram TF-IDF Cosine Similarity (handles Persian suffixes & typos)
        corpus = [
            f"{normalize_persian(m['title'])} {m['normalized_content']} {m['category']}"
            for m in active_memories
        ]
        tfidf_scores: Dict[int, float] = {}
        tfidf_ranks: Dict[int, int] = {}
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            mat = vec.fit_transform(corpus + [norm_q])
            sims = cosine_similarity(mat[-1:], mat[:-1])[0]
            ranked_indices = sims.argsort()[::-1]
            for rank_pos, doc_idx in enumerate(ranked_indices[:20]):
                sim_val = float(sims[doc_idx])
                if sim_val >= 0.08:
                    mid = int(active_memories[doc_idx]["id"])
                    tfidf_scores[mid] = sim_val
                    tfidf_ranks[mid] = rank_pos + 1
        except Exception:
            pass

        # 3. Reciprocal Rank Fusion (RRF) + Importance/Confidence Weighting
        scored: List[Dict[str, Any]] = []
        candidate_ids = set(fts_ranks.keys()) | set(tfidf_ranks.keys())
        for mid in candidate_ids:
            mem = by_id[mid]
            rrf = 0.0
            if mid in fts_ranks:
                rrf += 0.55 * (1.0 / (10 + fts_ranks[mid]))
            if mid in tfidf_ranks:
                rrf += 0.45 * (1.0 / (10 + tfidf_ranks[mid]))
            cos_val = tfidf_scores.get(mid, 0.0)
            # Combine RRF + raw cosine + importance bonus
            quality_bonus = (float(mem["confidence"]) * (int(mem["importance"]) / 5.0)) * 0.08
            final_score = round((rrf * 10.0) + (cos_val * 0.55) + quality_bonus, 4)
            if final_score >= min_score or mid in fts_ranks:
                enriched = dict(mem)
                enriched["retrieval_score"] = final_score
                scored.append(enriched)

        scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
        top_results = scored[:top_k]

        if mark_used and top_results:
            self.ltm.mark_used([int(r["id"]) for r in top_results])

        return top_results

    def search_exemplars(
        self,
        query: str,
        language: str = "fa",
        top_k: int = 2,
        min_cosine: float = 0.22,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant Teacher-distilled exemplars so `qwen3:0.6b` learns in-context offline!"""
        norm_q = normalize_persian(query)
        if not norm_q:
            return []
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM teacher_exemplars WHERE language = ? ORDER BY quality_score DESC, id DESC LIMIT 200",
                (language,),
            ).fetchall()
        items = [dict(r) for r in rows]
        if not items:
            return []

        corpus = [it["normalized_prompt"] for it in items]
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            mat = vec.fit_transform(corpus + [norm_q])
            sims = cosine_similarity(mat[-1:], mat[:-1])[0]
            out: List[Dict[str, Any]] = []
            for idx in sims.argsort()[::-1][:top_k]:
                sim = float(sims[idx])
                if sim >= min_cosine:
                    d = dict(items[idx])
                    d["similarity"] = round(sim, 4)
                    out.append(d)
            return out
        except Exception:
            return []


retriever = HybridRetriever()
