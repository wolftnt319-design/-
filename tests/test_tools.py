"""Stage 7, 8, & 9 Tool Tests: Word, Excel, Automatic Backup, Notepad, Offline Search, & Browser CAPTCHA."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import httpx
import pytest

from security.sandbox import SandboxManager
from tools.browser import BrowserTool
from tools.excel_editor import ExcelEditorTool
from tools.file_manager import FileManagerTool
from tools.google_search import WebSearchTool
from tools.notepad import NotepadTool
from tools.word_editor import WordEditorTool


def test_word_create_read_edit_and_backup(temp_sandbox: SandboxManager):
    word = WordEditorTool(sandbox=temp_sandbox)
    c_res = word.create_docx(
        path="project_spec.docx",
        title="مستندات پروژه سایه",
        paragraphs=["بند اول مستندات", "وضعیت: پیش‌نویس"],
        table_rows=[["ماژول", "وضعیت"], ["حافظه", "فعال"]],
    )
    assert c_res["ok"] is True
    assert c_res["backup_created"] is None

    e_res = word.edit_docx(
        path="project_spec.docx",
        operation="replace_text",
        find_text="پیش‌نویس",
        replace_with="نهایی‌شده",
    )
    assert e_res["ok"] is True
    assert e_res["backup_created"] is not None
    assert (temp_sandbox.backup_dir / e_res["backup_created"]).exists()

    r_res = word.read_docx("project_spec.docx")
    assert r_res["ok"] is True
    assert "نهایی‌شده" in r_res["text"]
    assert r_res["table_count"] == 1


def test_excel_create_edit_formula_sort_and_backup(temp_sandbox: SandboxManager):
    excel = ExcelEditorTool(sandbox=temp_sandbox)
    c_res = excel.create_xlsx(
        path="finance.xlsx",
        headers=["کالا", "قیمت"],
        rows=[["کیبورد", 900], ["ماوس", 350], ["مانیتور", 4200]],
    )
    assert c_res["ok"] is True

    # Sort ascending by column B (2)
    s_res = excel.edit_xlsx(path="finance.xlsx", operation="sort_sheet", sort_by_col="B", descending=False)
    assert s_res["ok"] is True
    assert s_res["backup_created"] is not None

    # Safe formula works
    f_ok = excel.edit_xlsx(path="finance.xlsx", operation="set_formula", cell="B5", value="=SUM(B2:B4)")
    assert f_ok["ok"] is True

    # Dangerous DDE/cmd formula is blocked
    f_bad = excel.edit_xlsx(path="finance.xlsx", operation="set_formula", cell="B6", value="=CMD|'/C calc'!A0")
    assert f_bad["ok"] is False

    r_res = excel.read_xlsx("finance.xlsx")
    assert r_res["ok"] is True
    assert r_res["rows"][1][0] == "ماوس"  # Lowest price sorted first!
    assert r_res["rows"][4][1] == "=SUM(B2:B4)"


def test_file_manager_backup_and_delete(temp_sandbox: SandboxManager):
    fm = FileManagerTool(sandbox=temp_sandbox)
    w1 = fm.write_file("notes.txt", "خط اول")
    assert w1["ok"] is True and w1["backup_created"] is None

    w2 = fm.write_file("notes.txt", "خط جدید جایگزین")
    assert w2["ok"] is True and w2["backup_created"] is not None

    s_res = fm.search_files("جدید")
    assert s_res["ok"] is True and s_res["count"] == 1

    d_res = fm.delete_file("notes.txt")
    assert d_res["ok"] is True and d_res["backup_created"] is not None


@pytest.mark.notepad
def test_notepad_safe_subprocess_and_cross_platform(temp_sandbox: SandboxManager):
    np_tool = NotepadTool(sandbox=temp_sandbox)

    # 1. Reject traversal/bad filename
    bad = np_tool.write_and_open("../evil.txt", "bad")
    assert bad["path"] == "evil.txt" if bad.get("ok") else True  # basename stripped safely inside sandbox!

    # 2. Simulate Windows 10 `notepad.exe` launch and verify `shell=False` + argument list
    fake_proc = MagicMock()
    fake_proc.pid = 4321
    with patch("tools.notepad._locate_allowed_notepad", return_value=r"C:\Windows\System32\notepad.exe"), patch(
        "subprocess.Popen", return_value=fake_proc
    ) as mock_popen:
        res = np_tool.write_and_open("story_fa.txt", "روزی روزگاری در شهر شیشه‌ای...", open_window=True)
        assert res["ok"] is True
        assert res["notepad_launched"] is True
        assert res["pid"] == 4321
        called_args, called_kwargs = mock_popen.call_args
        assert isinstance(called_args[0], list)
        assert called_args[0][0] == r"C:\Windows\System32\notepad.exe"
        assert called_kwargs["shell"] is False


@pytest.mark.browser
@pytest.mark.asyncio
async def test_web_search_offline_graceful_handling():
    search_tool = WebSearchTool()
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Network unreachable")):
        res = await search_tool.search("تست قطع اینترنت")
        assert res["ok"] is False
        assert res["offline"] is True
        assert "اینترنت" in res["error"]


@pytest.mark.browser
@pytest.mark.asyncio
async def test_browser_blocks_localhost_ssrf_and_unconfirmed_download():
    bt = BrowserTool()
    ssrf = await bt.extract_page("http://127.0.0.1:8000/api/env")
    assert ssrf["ok"] is False
    assert ssrf.get("blocked") is True

    file_scheme = await bt.extract_page("file:///etc/passwd")
    assert file_scheme["ok"] is False

    dl_block = await bt.extract_page("https://example.com/file.zip", allow_download=True)
    assert dl_block["ok"] is False
    assert dl_block.get("requires_confirmation") is True
