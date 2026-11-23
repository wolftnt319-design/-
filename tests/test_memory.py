"""Stage 4 Tests: Short-Term Summarization, Long-Term SQLite CRUD, Deduplication, Secret Block, & Hybrid Retrieval."""
from __future__ import annotations


def test_short_term_memory_summarization_and_clear(temp_memory_stack):
    _, stm, _, _ = temp_memory_stack
    sid = "sess_summary_test"

    for i in range(12):
        stm.add_message(sid, "user", f"این پیام آزمایشی شماره {i} درباره بهینه‌سازی حافظه در ویندوز ده است.")
        stm.add_message(sid, "assistant", f"پاسخ شماره {i}: مدیریت حافظه و کش بررسی شد. <think>internal thought</think>")

    ctx = stm.get_context_window(sid)
    assert ctx["active_topic"] != ""
    assert ctx["rolling_summary"] != ""
    assert all("<think>" not in m["content"] for m in ctx["messages"])

    stm.clear_session(sid)
    cleared = stm.get_context_window(sid)
    assert len(cleared["messages"]) == 0
    assert cleared["rolling_summary"] == ""


def test_long_term_memory_dedup_crud_and_retrieval(temp_memory_stack):
    _, _, ltm, ret = temp_memory_stack

    # 1. Create
    r1 = ltm.add_or_merge_memory("اسم این پروژه سایه است و برای اتوماسیون ویندوز ساخته شده.", title="نام پروژه", category="project", importance=5)
    assert r1["ok"] is True
    assert r1["action"] == "created"
    mem_id = r1["memory"]["id"]

    # 2. Exact Duplicate Merge (normalized ي/ی & digits)
    r2 = ltm.add_or_merge_memory("اسم اين پروژه سايه است و برای اتوماسیون ویندوز ساخته شده.", title="نام پروژه", category="project")
    assert r2["ok"] is True
    assert r2["action"] == "merged_exact"
    assert r2["memory"]["id"] == mem_id

    # 3. Similar Duplicate Merge
    r3 = ltm.add_or_merge_memory("اسم این پروژه سایه است و جهت اتوماسیون ویندوز ساخته شده است.", category="project")
    assert r3["ok"] is True
    assert r3["action"] in ("merged_similar", "merged_exact")
    assert len(ltm.list_memories()) == 1

    # 4. Hybrid FTS5 + char_wb TF-IDF Retrieval (even with suffix variation `پروژه‌ی سایه`)
    hits = ret.search_memories("هدف پروژه‌ی سایه چیست؟", top_k=3)
    assert len(hits) == 1
    assert "سایه" in hits[0]["content"]

    # 5. Update, Deactivate, & Forget
    upd = ltm.update_memory(mem_id, content="نام این پروژه سایه نسخه ۲ است.")
    assert upd["ok"] is True
    assert "نسخه ۲" in upd["memory"]["content"]

    ltm.deactivate_memory(mem_id)
    assert len(ret.search_memories("سایه")) == 0

    deleted = ltm.forget_memory(mem_id)
    assert deleted["ok"] is True
    assert len(ltm.list_memories(include_inactive=True)) == 0


def test_sensitive_secrets_never_stored_in_memory(temp_memory_stack):
    _, _, ltm, _ = temp_memory_stack

    pw = ltm.add_or_merge_memory("رمز عبور ایمیل من MySuperSecret123! است")
    assert pw["ok"] is False
    assert pw["reason"] == "sensitive_secret_blocked"

    card = ltm.add_or_merge_memory("شماره کارت بانکی: 6037-9975-1234-5678")
    assert card["ok"] is False

    tok = ltm.add_or_merge_memory("api_key=sk-abcdefghijklmnopqrstuvwxyz123456")
    assert tok["ok"] is False
    assert len(ltm.list_memories(include_inactive=True)) == 0
