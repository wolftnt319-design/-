"""Stage 5 & 13 Agent Tests: Routing, Confirmation Broker, Multi-Tool Execution, & Honesty on Failed Operations."""
from __future__ import annotations

import pytest

from core.agent import LunaAgent
from core.response_validator import ResponseValidator
from security.sandbox import SandboxManager
from tools.file_manager import FileManagerTool
from tools.registry import ToolRegistry


@pytest.fixture()
def isolated_agent(temp_sandbox: SandboxManager) -> LunaAgent:
    reg = ToolRegistry()
    reg.file_tool = FileManagerTool(sandbox=temp_sandbox)
    # Rebind file handlers to isolated temp_sandbox
    reg._tools["file_list"].handler = reg.file_tool.list_files
    reg._tools["file_read"].handler = reg.file_tool.read_file
    reg._tools["file_write"].handler = reg.file_tool.write_file
    reg._tools["file_delete"].handler = reg.file_tool.delete_file
    return LunaAgent(registry=reg)


@pytest.mark.asyncio
async def test_interactive_memory_requires_confirmation_then_saves(isolated_agent: LunaAgent):
    r1 = await isolated_agent.process_message("اسم این پروژه ققنوس است.", session_id="agent_mem_sess")
    assert r1["ok"] is True
    assert r1["provenance"] == "confirmation"
    assert r1["pending_confirmation"] is not None

    # User replies "بله" in chat -> executes pending memory save!
    r2 = await isolated_agent.process_message("بله", session_id="agent_mem_sess")
    assert r2["ok"] is True
    assert r2["approved"] is True
    assert r2["provenance"] == "memory"


@pytest.mark.asyncio
async def test_multi_tool_calculation_and_file_save(isolated_agent: LunaAgent, temp_sandbox: SandboxManager):
    res = await isolated_agent.process_message(
        "حساب کن ۲۵ درصد ۸۰۰ چند می‌شود در فایل tax_calc.txt",
        session_id="multi_step_sess",
    )
    assert res["ok"] is True
    assert len(res["executed_tools"]) == 2
    assert res["executed_tools"][0]["tool"] == "calc_evaluate"
    assert res["executed_tools"][0]["result"] == 200
    assert res["executed_tools"][1]["tool"] == "file_write"
    saved = (temp_sandbox.default_root / "tax_calc.txt").read_text(encoding="utf-8")
    assert "200" in saved


@pytest.mark.asyncio
async def test_sensitive_delete_blocks_until_approved(isolated_agent: LunaAgent, temp_sandbox: SandboxManager):
    target = temp_sandbox.default_root / "important.txt"
    target.write_text("محتوای مهم", encoding="utf-8")

    r1 = await isolated_agent.process_message("فایل important.txt را حذف کن", session_id="del_sess")
    assert r1["pending_confirmation"] is not None
    assert target.exists()  # NOT deleted before approval!

    # Reject first
    r_no = await isolated_agent.process_message("نه", session_id="del_sess")
    assert r_no["approved"] is False
    assert target.exists()

    # Request again and approve
    r2 = await isolated_agent.process_message("فایل important.txt را حذف کن", session_id="del_sess")
    tok = r2["pending_confirmation"]["token"]
    r_yes = await isolated_agent.resolve_confirmation(tok, approved=True, language="fa")
    assert r_yes["ok"] is True
    assert not target.exists()


@pytest.mark.asyncio
async def test_failed_tool_never_hallucinates_success(isolated_agent: LunaAgent):
    res = await isolated_agent.process_message("فایل nonexistent_missing_doc.docx را بخوان", session_id="fail_sess")
    assert res["ok"] is True
    assert len(res["executed_tools"]) == 1
    assert res["executed_tools"][0]["ok"] is False
    assert "ناموفق" in res["reply"] or "یافت نشد" in res["reply"]
    val = ResponseValidator.validate_chat_response(
        res["reply"], "فایل nonexistent_missing_doc.docx را بخوان", language="fa", executed_tools=res["executed_tools"]
    )
    assert val.hallucinated_action is False


def test_validator_catches_unverified_action_hallucination():
    fake_claim = "فایل گزارش با موفقیت ذخیره شد و در پوشه قرار گرفت."
    rep = ResponseValidator.validate_chat_response(fake_claim, "یک گزارش بساز", language="fa", executed_tools=[])
    assert rep.hallucinated_action is True
    assert rep.valid is False
