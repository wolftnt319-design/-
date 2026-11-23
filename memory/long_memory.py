"""Long-Term SQLite Memory with Deduplication, Secret Blocking, and Interactive Learning."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.language import normalize_persian
from memory.database import DatabaseManager, db_manager
from security.validator import SecretRedactor

VALID_CATEGORIES = {"project", "preference", "fact", "instruction", "web_note", "personal"}

# Interactive declarative patterns that suggest saving to long-term memory
MEMORY_TEACH_PATTERNS: List[Tuple[str, str, re.Pattern[str]]] = [
    ("project", "نام یا مشخصات پروژه", re.compile(r"(?:اسم|نام)\s+(?:این\s+)?پروژه\s+(.+?)\s+(?:است|هست)\.?$")),
    ("personal", "مشخصات کاربر", re.compile(r"(?:اسم|نام)\s+من\s+(.+?)\s+(?:است|هست)\.?$")),
    ("preference", "ترجیح کاربر", re.compile(r"من\s+(.+?)\s+را\s+(?:بیشتر\s+)?(?:دوست\s+دارم|ترجیح\s+می‌?دهم)\.?$")),
    ("fact", "نکته آموزشی کاربر", re.compile(r"(?:یادت\s+بمونه|یادت\s+باشه|به\s+خاطر\s+بسپار|در\s+حافظه\s+ذخیره\s+کن)\s*(?:که\s*)?[:：]?\s*(.+)$")),
    ("project", "Project Fact", re.compile(r"(?i)^(?:the\s+)?project\s+name\s+is\s+(.+?)\.?$")),
    ("fact", "User Memory Note", re.compile(r"(?i)^(?:remember\s+that|save\s+to\s+memory[:\s]+)\s*(.+)$")),
]


def _token_set(norm_text: str) -> set[str]:
    return {t for t in re.findall(r"[\w\u0600-\u06FF]{2,}", norm_text)}


def _jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class LongTermMemory:
    """Persistent, deduplicated, secret-safe SQLite memory store."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or db_manager

    def add_or_merge_memory(
        self,
        content: str,
        title: Optional[str] = None,
        category: str = "fact",
        source: str = "user",
        confidence: float = 0.9,
        importance: int = 3,
    ) -> Dict[str, Any]:
        """Store meaningful knowledge, blocking secrets and merging duplicates."""
        raw = (content or "").strip()
        if len(raw) < 4:
            return {"ok": False, "reason": "too_short", "message": "متن برای ذخیره در حافظه بسیار کوتاه یا بی‌معناست."}

        has_secret, secret_kinds = SecretRedactor.contains_secret(raw)
        if has_secret:
            return {
                "ok": False,
                "reason": "sensitive_secret_blocked",
                "secret_types": secret_kinds,
                "message": "ذخیره اطلاعات حساس (رمز عبور، کارت بانکی یا توکن) در حافظه بلندمدت ممنوع است و مسدود شد.",
            }

        cat = category if category in VALID_CATEGORIES else "fact"
        norm = normalize_persian(raw)
        clean_title = (title or raw[:48]).strip()[:80]
        now = datetime.now(timezone.utc).isoformat()
        imp = max(1, min(5, int(importance)))
        conf = max(0.1, min(1.0, float(confidence)))

        with self.db.connect() as conn:
            # 1. Exact normalized duplicate check
            exact = conn.execute(
                "SELECT * FROM memories WHERE normalized_content = ?", (norm,)
            ).fetchone()
            if exact:
                new_conf = min(1.0, round(max(float(exact["confidence"]), conf) + 0.03, 3))
                new_imp = max(int(exact["importance"]), imp)
                conn.execute(
                    """
                    UPDATE memories
                    SET is_active = 1, confidence = ?, importance = ?, updated_at = ?, use_count = use_count + 1
                    WHERE id = ?
                    """,
                    (new_conf, new_imp, now, exact["id"]),
                )
                self._sync_fts(conn, int(exact["id"]), clean_title, raw, norm, cat, active=True)
                updated = dict(conn.execute("SELECT * FROM memories WHERE id = ?", (exact["id"],)).fetchone())
                return {"ok": True, "action": "merged_exact", "memory": updated}

            # 2. High Jaccard similarity merge within same category
            candidates = conn.execute(
                "SELECT * FROM memories WHERE category = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 50",
                (cat,),
            ).fetchall()
            for cand in candidates:
                sim = _jaccard_similarity(norm, cand["normalized_content"])
                if sim >= 0.78:
                    merged_content = raw if len(raw) >= len(cand["content"]) else cand["content"]
                    merged_norm = normalize_persian(merged_content)
                    new_conf = min(1.0, round(max(float(cand["confidence"]), conf) + 0.05, 3))
                    conn.execute(
                        """
                        UPDATE memories
                        SET title = ?, content = ?, normalized_content = ?, confidence = ?, importance = ?, updated_at = ?, use_count = use_count + 1
                        WHERE id = ?
                        """,
                        (clean_title, merged_content, merged_norm, new_conf, max(int(cand["importance"]), imp), now, cand["id"]),
                    )
                    self._sync_fts(conn, int(cand["id"]), clean_title, merged_content, merged_norm, cat, active=True)
                    updated = dict(conn.execute("SELECT * FROM memories WHERE id = ?", (cand["id"],)).fetchone())
                    return {"ok": True, "action": "merged_similar", "memory": updated}

            # 3. Insert fresh memory
            cur = conn.execute(
                """
                INSERT INTO memories (
                    category, title, content, normalized_content, source,
                    confidence, importance, created_at, updated_at, last_used_at, use_count, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 1)
                """,
                (cat, clean_title, raw, norm, source, conf, imp, now, now),
            )
            mem_id = int(cur.lastrowid)
            self._sync_fts(conn, mem_id, clean_title, raw, norm, cat, active=True)
            created = dict(conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone())
            return {"ok": True, "action": "created", "memory": created}

    def _sync_fts(
        self,
        conn: Any,
        memory_id: int,
        title: str,
        content: str,
        normalized_content: str,
        category: str,
        active: bool,
    ) -> None:
        if not self.db.fts5_available:
            return
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (str(memory_id),))
        if active:
            conn.execute(
                "INSERT INTO memories_fts (memory_id, title, content, normalized_content, category) VALUES (?, ?, ?, ?, ?)",
                (str(memory_id), title, content, normalized_content, category),
            )

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        title: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        with self.db.connect() as conn:
            existing = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not existing:
                return {"ok": False, "reason": "not_found", "message": "حافظه مورد نظر یافت نشد."}

            new_content = (content.strip() if content is not None else existing["content"])
            has_secret, secret_kinds = SecretRedactor.contains_secret(new_content)
            if has_secret:
                return {
                    "ok": False,
                    "reason": "sensitive_secret_blocked",
                    "secret_types": secret_kinds,
                    "message": "ذخیره داده حساس در حافظه مسدود شد.",
                }

            new_title = (title.strip()[:80] if title is not None else existing["title"])
            new_cat = category if (category in VALID_CATEGORIES) else existing["category"]
            new_imp = max(1, min(5, int(importance))) if importance is not None else int(existing["importance"])
            new_active = (1 if is_active else 0) if is_active is not None else int(existing["is_active"])
            new_norm = normalize_persian(new_content)
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                UPDATE memories
                SET title = ?, content = ?, normalized_content = ?, category = ?, importance = ?, is_active = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, new_content, new_norm, new_cat, new_imp, new_active, now, memory_id),
            )
            self._sync_fts(conn, memory_id, new_title, new_content, new_norm, new_cat, active=bool(new_active))
            updated = dict(conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone())
            return {"ok": True, "memory": updated}

    def deactivate_memory(self, memory_id: int) -> Dict[str, Any]:
        return self.update_memory(memory_id, is_active=False)

    def forget_memory(self, memory_id: int) -> Dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not row:
                return {"ok": False, "reason": "not_found"}
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self.db.fts5_available:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (str(memory_id),))
            return {"ok": True, "deleted_id": memory_id}

    def list_memories(self, include_inactive: bool = False, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memories WHERE 1=1"
        params: List[Any] = []
        if not include_inactive:
            query += " AND is_active = 1"
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def mark_used(self, memory_ids: List[int]) -> None:
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in memory_ids)
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE memories SET last_used_at = ?, use_count = use_count + 1 WHERE id IN ({placeholders})",
                [now, *memory_ids],
            )

    @staticmethod
    def detect_interactive_candidate(user_text: str) -> Optional[Dict[str, Any]]:
        """Detect if the user shared an important fact/project/preference worth asking confirmation to store."""
        cleaned = user_text.strip()
        if "?" in cleaned or "؟" in cleaned or len(cleaned) > 260:
            return None
        has_secret, _ = SecretRedactor.contains_secret(cleaned)
        if has_secret:
            return None

        for category, title_hint, pat in MEMORY_TEACH_PATTERNS:
            m = pat.search(cleaned)
            if m:
                extracted = m.group(1).strip()
                if len(extracted) >= 2:
                    return {
                        "category": category,
                        "title": f"{title_hint}: {extracted[:36]}",
                        "content": cleaned,
                        "extracted_value": extracted,
                        "importance": 4 if category in ("project", "personal") else 3,
                    }
        return None

    # --- Distilled Teacher Exemplar Store (Local Few-Shot Learning for qwen3:0.6b) ---
    def save_teacher_exemplar(
        self,
        user_prompt: str,
        ideal_response: str,
        language: str = "fa",
        intent: str = "chat",
        teacher_source: str = "pollinations_free",
        quality_score: float = 0.95,
    ) -> Dict[str, Any]:
        has_s1, _ = SecretRedactor.contains_secret(user_prompt)
        has_s2, _ = SecretRedactor.contains_secret(ideal_response)
        if has_s1 or has_s2:
            return {"ok": False, "reason": "sensitive_secret_blocked"}
        norm_p = normalize_persian(user_prompt)
        if len(norm_p) < 4 or len(ideal_response.strip()) < 4:
            return {"ok": False, "reason": "too_short"}

        now = datetime.now(timezone.utc).isoformat()
        with self.db.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM teacher_exemplars WHERE normalized_prompt = ?", (norm_p,)
            ).fetchone()
            if existing:
                ex_id = int(existing["id"])
                conn.execute(
                    "UPDATE teacher_exemplars SET ideal_response = ?, teacher_source = ?, quality_score = ?, created_at = ? WHERE id = ?",
                    (ideal_response.strip(), teacher_source, quality_score, now, ex_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO teacher_exemplars (language, intent, user_prompt, normalized_prompt, ideal_response, teacher_source, quality_score, use_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (language, intent, user_prompt.strip(), norm_p, ideal_response.strip(), teacher_source, quality_score, now),
                )
                ex_id = int(cur.lastrowid)

            if self.db.fts5_available:
                conn.execute("DELETE FROM exemplars_fts WHERE exemplar_id = ?", (str(ex_id),))
                conn.execute(
                    "INSERT INTO exemplars_fts (exemplar_id, user_prompt, normalized_prompt, ideal_response) VALUES (?, ?, ?, ?)",
                    (str(ex_id), user_prompt.strip(), norm_p, ideal_response.strip()),
                )
        return {"ok": True, "exemplar_id": ex_id}

    def list_teacher_exemplars(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM teacher_exemplars ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


long_memory = LongTermMemory()
