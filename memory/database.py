"""SQLite WAL Database & FTS5 Schema Manager for LUNA AI."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from config import DB_PATH


class DatabaseManager:
    """Lightweight SQLite WAL connection manager with FTS5 full-text search support."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = (db_path or DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fts5_available: bool = False
        self._init_schema()

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL DEFAULT 'fact',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL NOT NULL DEFAULT 0.9,
                    importance INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_memories_norm ON memories(normalized_content);
                CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active, importance DESC);

                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'گفتگوی جدید',
                    active_topic TEXT NOT NULL DEFAULT '',
                    rolling_summary TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT 'auto',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    provenance TEXT NOT NULL DEFAULT 'general',
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES conversations(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id ASC);

                CREATE TABLE IF NOT EXISTS teacher_exemplars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    language TEXT NOT NULL DEFAULT 'fa',
                    intent TEXT NOT NULL DEFAULT 'chat',
                    user_prompt TEXT NOT NULL,
                    normalized_prompt TEXT NOT NULL,
                    ideal_response TEXT NOT NULL,
                    teacher_source TEXT NOT NULL DEFAULT 'pollinations_free',
                    quality_score REAL NOT NULL DEFAULT 0.95,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_exemplars_norm ON teacher_exemplars(normalized_prompt);
                """
            )

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        memory_id UNINDEXED,
                        title,
                        content,
                        normalized_content,
                        category
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS exemplars_fts USING fts5(
                        exemplar_id UNINDEXED,
                        user_prompt,
                        normalized_prompt,
                        ideal_response
                    );
                    """
                )
                self.fts5_available = True
            except sqlite3.OperationalError:
                self.fts5_available = False


db_manager = DatabaseManager()
