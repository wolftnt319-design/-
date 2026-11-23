"""Shared Pytest Fixtures for LUNA AI."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.database import DatabaseManager
from memory.long_memory import LongTermMemory
from memory.retrieval import HybridRetriever
from memory.short_memory import ShortTermMemory
from security.sandbox import SandboxManager


@pytest.fixture()
def temp_sandbox(tmp_path: Path) -> SandboxManager:
    ws = tmp_path / "workspace"
    bak = tmp_path / "backups"
    ws.mkdir()
    bak.mkdir()
    return SandboxManager(allowed_roots=[ws], backup_dir=bak, max_file_size_bytes=2 * 1024 * 1024)


@pytest.fixture()
def temp_memory_stack(tmp_path: Path):
    db = DatabaseManager(db_path=tmp_path / "test_luna.db")
    stm = ShortTermMemory(db=db)
    ltm = LongTermMemory(db=db)
    ret = HybridRetriever(db=db, ltm=ltm)
    return db, stm, ltm, ret
