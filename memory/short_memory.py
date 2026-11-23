"""Short-Term Conversation Memory with Progressive Summarization & Context Budgeting."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import config_manager
from core.model_manager import strip_thinking_blocks
from memory.database import DatabaseManager, db_manager
from security.validator import SecretRedactor


def estimate_tokens(text: str) -> int:
    """Fast CPU token estimator suitable for Persian (UTF-8 multi-byte) and English."""
    if not text:
        return 0
    # Roughly 1 token per 2.8 chars in mixed Persian/English BPE
    return max(1, int(len(text) / 2.8))


class ShortTermMemory:
    """Manages active session history, topic tracking, progressive summarization, and clearing."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or db_manager

    def ensure_session(self, session_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                return dict(row)
            initial_title = (title or "گفتگوی جدید")[:60]
            conn.execute(
                """
                INSERT INTO conversations (session_id, title, active_topic, rolling_summary, language, created_at, updated_at)
                VALUES (?, ?, '', '', 'auto', ?, ?)
                """,
                (session_id, initial_title, now, now),
            )
            row = conn.execute(
                "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        provenance: str = "general",
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        clean_content = SecretRedactor.redact(strip_thinking_blocks(content))
        session = self.ensure_session(session_id)
        now = datetime.now(timezone.utc).isoformat()

        with self.db.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, provenance, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    clean_content,
                    provenance,
                    json.dumps(SecretRedactor.redact_dict(meta or {}), ensure_ascii=False),
                    now,
                ),
            )
            msg_id = int(cur.lastrowid)

            # Update title & active_topic automatically from user messages
            new_title = session["title"]
            active_topic = session["active_topic"]
            if role == "user" and clean_content.strip():
                snippet = re.sub(r"\s+", " ", clean_content).strip()
                if new_title in ("گفتگوی جدید", "New Chat"):
                    new_title = snippet[:48]
                if len(snippet) > 6 and not snippet.startswith("/"):
                    active_topic = snippet[:90]

            conn.execute(
                """
                UPDATE conversations
                SET title = ?, active_topic = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (new_title, active_topic, now, session_id),
            )

        self._compact_if_needed(session_id)
        return msg_id

    def _compact_if_needed(self, session_id: str) -> None:
        """Progressively summarize older turns when turn count or char budget overflows."""
        max_turns = config_manager.settings.short_memory_max_turns
        max_chars = config_manager.settings.short_memory_max_chars

        with self.db.connect() as conn:
            conv = conn.execute(
                "SELECT rolling_summary, active_topic FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not conv:
                return
            rows = conn.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

            total_chars = sum(len(r["content"]) for r in rows)
            max_messages = max_turns * 2
            if len(rows) <= max_messages and total_chars <= max_chars:
                return

            # Summarize older half deterministically (zero LLM overhead, instant & faithful!)
            cutoff = max(2, len(rows) - max_messages)
            older = rows[:cutoff]
            bullet_points: List[str] = []
            for r in older[-6:]:
                speaker = "کاربر" if r["role"] == "user" else "لونا"
                compact_line = re.sub(r"\s+", " ", r["content"]).strip()[:110]
                if compact_line:
                    bullet_points.append(f"{speaker}: {compact_line}")

            prev_summary = (conv["rolling_summary"] or "").strip()
            merged_parts = ([prev_summary] if prev_summary else []) + bullet_points
            # Bound rolling summary to ~700 chars so it never bloats context
            new_summary = " | ".join(merged_parts)[-700:]
            conn.execute(
                "UPDATE conversations SET rolling_summary = ? WHERE session_id = ?",
                (new_summary, session_id),
            )

    def get_context_window(self, session_id: str) -> Dict[str, Any]:
        """Return bounded recent history + rolling summary + active topic within token budget."""
        self.ensure_session(session_id)
        max_turns = config_manager.settings.short_memory_max_turns
        max_chars = config_manager.settings.short_memory_max_chars

        with self.db.connect() as conn:
            conv = dict(
                conn.execute(
                    "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
                ).fetchone()
            )
            rows = conn.execute(
                "SELECT id, role, content, provenance, meta_json, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, max_turns * 2),
            ).fetchall()

        ordered = [dict(r) for r in reversed(rows)]
        selected: List[Dict[str, str]] = []
        used_chars = 0
        for item in reversed(ordered):
            c = strip_thinking_blocks(item["content"])
            if used_chars + len(c) > max_chars and selected:
                break
            selected.append({"role": item["role"], "content": c})
            used_chars += len(c)
        selected.reverse()

        return {
            "session_id": session_id,
            "title": conv["title"],
            "active_topic": conv["active_topic"],
            "rolling_summary": conv["rolling_summary"],
            "language": conv["language"],
            "messages": selected,
            "estimated_tokens": estimate_tokens(conv["rolling_summary"]) + sum(
                estimate_tokens(m["content"]) for m in selected
            ),
        }

    def get_full_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.ensure_session(session_id)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, provenance, meta_json, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["meta"] = json.loads(d.pop("meta_json", "{}"))
            except Exception:
                d["meta"] = {}
            out.append(d)
        return out

    def set_session_language(self, session_id: str, language: str) -> None:
        self.ensure_session(session_id)
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE conversations SET language = ?, updated_at = ? WHERE session_id = ?",
                (language, datetime.now(timezone.utc).isoformat(), session_id),
            )

    def clear_session(self, session_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE conversations SET active_topic = '', rolling_summary = '', updated_at = ? WHERE session_id = ?",
                (datetime.now(timezone.utc).isoformat(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))

    def list_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON c.session_id = m.session_id
                GROUP BY c.session_id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


short_memory = ShortTermMemory()
