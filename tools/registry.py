"""Whitelisted Tool Registry with Strict JSON Schemas, Timeouts, and Risk Levels."""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from config import config_manager
from memory.long_memory import long_memory
from memory.retrieval import retriever
from security.permissions import RiskLevel, audit_logger
from security.validator import ArgumentSchemaValidator, SecretRedactor
from tools.browser import browser_tool
from tools.calculator import calculate
from tools.excel_editor import ExcelEditorTool
from tools.file_manager import FileManagerTool
from tools.google_search import web_search_tool
from tools.notepad import NotepadTool
from tools.word_editor import WordEditorTool


@dataclass
class ToolSpec:
    name: str
    intent: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[..., Any]
    risk_level: RiskLevel = RiskLevel.L0_READ_PURE
    requires_confirmation: bool = False


class ToolRegistry:
    """Executes ONLY allowlisted tools after strict JSON Schema validation and timeout wrapping."""

    def __init__(self) -> None:
        self.file_tool = FileManagerTool()
        self.word_tool = WordEditorTool()
        self.excel_tool = ExcelEditorTool()
        self.notepad_tool = NotepadTool()
        self._tools: Dict[str, ToolSpec] = {}
        self._register_defaults()

    def _register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def _register_defaults(self) -> None:
        # 1. Calculator
        self._register(ToolSpec(
            name="calc_evaluate",
            intent="calculator",
            description="محاسبه دقیق عبارت‌های ریاضی با AST امن",
            schema={
                "type": "object",
                "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 300}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=lambda expression: calculate(expression),
            risk_level=RiskLevel.L0_READ_PURE,
        ))

        # 2. Web Search
        self._register(ToolSpec(
            name="web_search",
            intent="search",
            description="جستجوی کنترل‌شده وب با جداسازی محتوای غیرمطمئن",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 250},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    "fetch_top_page": {"type": "boolean"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=web_search_tool.search,
            risk_level=RiskLevel.L2_EXTERNAL_READ,
        ))

        # 3. Browser Page Extract
        self._register(ToolSpec(
            name="browser_extract",
            intent="browser",
            description="باز کردن و استخراج متن تمیز صفحه وب با Playwright",
            schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 8, "maxLength": 500},
                    "allow_download": {"type": "boolean"},
                    "submit_form": {"type": "boolean"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=browser_tool.extract_page,
            risk_level=RiskLevel.L2_EXTERNAL_READ,
        ))

        # 4. Word Tools
        self._register(ToolSpec(
            name="word_read",
            intent="word",
            description="خواندن متن و جدول‌های فایل Word (.docx) در فضای مجاز",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 200}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=self.word_tool.read_docx,
            risk_level=RiskLevel.L0_READ_PURE,
        ))
        self._register(ToolSpec(
            name="word_create",
            intent="word",
            description="ایجاد فایل Word (.docx) جدید با پشتیبان‌گیری خودکار",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 200},
                    "title": {"type": "string", "maxLength": 200},
                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                    "table_rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=self.word_tool.create_docx,
            risk_level=RiskLevel.L1_WORKSPACE_WRITE,
        ))
        self._register(ToolSpec(
            name="word_edit",
            intent="word",
            description="ویرایش فایل Word موجود با پشتیبان‌گیری خودکار (.bak)",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 200},
                    "operation": {"type": "string", "enum": ["append_paragraph", "add_heading", "replace_text"]},
                    "text": {"type": "string"},
                    "find_text": {"type": "string"},
                    "replace_with": {"type": "string"},
                    "heading_level": {"type": "integer", "minimum": 1, "maximum": 4},
                },
                "required": ["path", "operation"],
                "additionalProperties": False,
            },
            handler=self.word_tool.edit_docx,
            risk_level=RiskLevel.L1_WORKSPACE_WRITE,
        ))

        # 5. Excel Tools
        self._register(ToolSpec(
            name="excel_read",
            intent="excel",
            description="خواندن شیت و سلول‌های فایل Excel (.xlsx)",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 200},
                    "sheet_name": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=self.excel_tool.read_xlsx,
            risk_level=RiskLevel.L0_READ_PURE,
        ))
        self._register(ToolSpec(
            name="excel_create",
            intent="excel",
            description="ایجاد فایل Excel (.xlsx) جدید",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 200},
                    "headers": {"type": "array"},
                    "rows": {"type": "array", "items": {"type": "array"}},
                    "sheet_name": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=self.excel_tool.create_xlsx,
            risk_level=RiskLevel.L1_WORKSPACE_WRITE,
        ))
        self._register(ToolSpec(
            name="excel_edit",
            intent="excel",
            description="ویرایش سلول، ردیف، ستون، فرمول و مرتب‌سازی Excel با پشتیبان‌گیری خودکار",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 200},
                    "operation": {"type": "string", "enum": ["write_cell", "append_row", "add_column", "set_formula", "sort_sheet"]},
                    "sheet_name": {"type": "string"},
                    "cell": {"type": "string"},
                    "value": {},
                    "row_values": {"type": "array"},
                    "column_header": {"type": "string"},
                    "column_values": {"type": "array"},
                    "sort_by_col": {"type": ["string", "integer"]},
                    "descending": {"type": "boolean"},
                    "has_header": {"type": "boolean"},
                },
                "required": ["path", "operation"],
                "additionalProperties": False,
            },
            handler=self.excel_tool.edit_xlsx,
            risk_level=RiskLevel.L1_WORKSPACE_WRITE,
        ))

        # 6. Notepad Tool
        self._register(ToolSpec(
            name="notepad_write",
            intent="notepad",
            description="ذخیره امن متن/داستان و باز کردن در Notepad ویندوز",
            schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "minLength": 1, "maxLength": 80},
                    "content": {"type": "string", "minLength": 1},
                    "open_window": {"type": "boolean"},
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
            handler=self.notepad_tool.write_and_open,
            risk_level=RiskLevel.L1_WORKSPACE_WRITE,
        ))

        # 7. Workspace File Tools
        self._register(ToolSpec(
            name="file_list",
            intent="file",
            description="فهرست فایل‌های داخل محیط کاری مجاز",
            schema={
                "type": "object",
                "properties": {"subdir": {"type": "string", "maxLength": 120}},
                "additionalProperties": False,
            },
            handler=self.file_tool.list_files,
            risk_level=RiskLevel.L0_READ_PURE,
        ))
        self._register(ToolSpec(
            name="file_read",
            intent="file",
            description="خواندن محتوای فایل متنی در محیط مجاز",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 200}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=self.file_tool.read_file,
            risk_level=RiskLevel.L0_READ_PURE,
        ))
        self._register(ToolSpec(
            name="file_write",
            intent="file",
            description="نوشتن فایل متنی در محیط مجاز همراه با پشتیبان‌گیری خودکار",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 200},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=self.file_tool.write_file,
            risk_level=RiskLevel.L1_WORKSPACE_WRITE,
        ))
        self._register(ToolSpec(
            name="file_search",
            intent="file",
            description="جستجوی نام و محتوای فایل‌ها در فضای مجاز",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 120}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=self.file_tool.search_files,
            risk_level=RiskLevel.L0_READ_PURE,
        ))
        self._register(ToolSpec(
            name="file_delete",
            intent="file",
            description="حذف فایل در فضای مجاز (حساس — نیازمند تأیید کاربر)",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 200}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=self.file_tool.delete_file,
            risk_level=RiskLevel.L3_SENSITIVE_MUTATE,
            requires_confirmation=True,
        ))

        # 8. Memory Tools
        self._register(ToolSpec(
            name="memory_save",
            intent="memory",
            description="ذخیره نکته مهم در حافظه بلندمدت SQLite (با تأیید کاربر برای نکات مهم)",
            schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "minLength": 4, "maxLength": 800},
                    "title": {"type": "string", "maxLength": 80},
                    "category": {"type": "string", "enum": ["project", "preference", "fact", "instruction", "web_note", "personal"]},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            handler=lambda content, title=None, category="fact", importance=3: long_memory.add_or_merge_memory(
                content=content, title=title, category=category, source="interactive_confirmed", importance=importance
            ),
            risk_level=RiskLevel.L3_SENSITIVE_MUTATE,
            requires_confirmation=True,
        ))
        self._register(ToolSpec(
            name="memory_search",
            intent="memory",
            description="جستجو در حافظه بلندمدت کاربر با FTS5 + TF-IDF",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda query: {
                "ok": True,
                "results": retriever.search_memories(query, top_k=5),
                "summary": f"جستجو در حافظه برای «{query}» انجام شد.",
            },
            risk_level=RiskLevel.L0_READ_PURE,
        ))

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get_spec(self, tool_name: str) -> Optional[ToolSpec]:
        return self._tools.get(tool_name)

    def list_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name,
                "intent": s.intent,
                "description": s.description,
                "risk_level": int(s.risk_level),
                "requires_confirmation": s.requires_confirmation,
                "schema": s.schema,
            }
            for s in self._tools.values()
        ]

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Validate schema, enforce allowlist & timeout, log redacted audit entry, and execute."""
        spec = self._tools.get(tool_name)
        if spec is None:
            audit_logger.log(
                event_type="tool_blocked_not_allowlisted",
                actor="agent",
                status="blocked",
                details={"tool_name": tool_name},
                session_id=session_id,
            )
            return {
                "ok": False,
                "tool": tool_name,
                "error": f"ابزار «{tool_name}» در فهرست مجاز (Allowlist) وجود ندارد و مسدود شد.",
            }

        valid, err_msg = ArgumentSchemaValidator.validate(arguments, spec.schema)
        if not valid:
            audit_logger.log(
                event_type="tool_schema_invalid",
                actor="agent",
                status="rejected",
                details={"tool_name": tool_name, "error": err_msg, "arguments": arguments},
                session_id=session_id,
            )
            return {
                "ok": False,
                "tool": tool_name,
                "error": f"اعتبارسنجی آرگومان‌های ابزار «{tool_name}» ناموفق بود: {err_msg}",
            }

        timeout_s = config_manager.settings.security.tool_timeout_seconds
        t0 = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(spec.handler):
                raw_res = await asyncio.wait_for(spec.handler(**arguments), timeout=timeout_s)
            else:
                raw_res = await asyncio.wait_for(
                    asyncio.to_thread(spec.handler, **arguments),
                    timeout=timeout_s,
                )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            result = dict(raw_res) if isinstance(raw_res, dict) else {"ok": True, "result": raw_res}
            result.setdefault("tool", tool_name)
            result["latency_ms"] = elapsed_ms

            audit_logger.log(
                event_type="tool_executed",
                actor="agent",
                status="success" if result.get("ok") else "failed",
                details={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "latency_ms": elapsed_ms,
                    "ok": bool(result.get("ok")),
                    "error": result.get("error"),
                },
                session_id=session_id,
            )
            return SecretRedactor.redact_dict(result)
        except asyncio.TimeoutError:
            audit_logger.log(
                event_type="tool_timeout",
                actor="agent",
                status="timeout",
                details={"tool_name": tool_name, "timeout_seconds": timeout_s},
                session_id=session_id,
            )
            return {
                "ok": False,
                "tool": tool_name,
                "error": f"اجرای ابزار «{tool_name}» پس از {timeout_s} ثانیه به دلیل اتمام مهلت متوقف شد.",
            }
        except Exception as exc:
            audit_logger.log(
                event_type="tool_exception",
                actor="agent",
                status="error",
                details={"tool_name": tool_name, "error": str(exc)},
                session_id=session_id,
            )
            return {"ok": False, "tool": tool_name, "error": str(exc)}
