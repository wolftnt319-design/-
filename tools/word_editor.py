"""Sandboxed Word (.docx) Editor with Automatic `.bak` Backup (`python-docx`)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from docx import Document

from security.sandbox import SandboxManager


class WordEditorTool:
    """Read, create, and edit `.docx` documents safely inside LUNA's sandbox."""

    def __init__(self, sandbox: Optional[SandboxManager] = None) -> None:
        self.sandbox = sandbox or SandboxManager()

    def read_docx(self, path: str, max_chars: int = 8000) -> Dict[str, Any]:
        try:
            resolved = self.sandbox.resolve_safe_path(path, must_exist=True, allowed_extensions=[".docx"])
            doc = Document(str(resolved))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            tables_data: List[List[List[str]]] = []
            for tbl in doc.tables[:10]:
                rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
                tables_data.append(rows)

            full_text = "\n".join(paragraphs)
            return {
                "ok": True,
                "path": resolved.name,
                "paragraph_count": len(paragraphs),
                "table_count": len(doc.tables),
                "text": full_text[:max_chars],
                "tables": tables_data,
                "truncated": len(full_text) > max_chars,
                "summary": f"فایل Word «{resolved.name}» خوانده شد ({len(paragraphs)} پاراگراف، {len(doc.tables)} جدول).",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def create_docx(
        self,
        path: str,
        title: str = "",
        paragraphs: Optional[List[str]] = None,
        table_rows: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        try:
            if not path.lower().endswith(".docx"):
                path = f"{path}.docx"
            resolved = self.sandbox.resolve_safe_path(path, must_exist=False, allowed_extensions=[".docx"])
            resolved.parent.mkdir(parents=True, exist_ok=True)
            backup = self.sandbox.create_backup_if_exists(resolved)

            doc = Document()
            if title:
                doc.add_heading(title, level=1)
            for p in (paragraphs or []):
                doc.add_paragraph(str(p))
            if table_rows and len(table_rows) > 0:
                cols = max(len(r) for r in table_rows)
                tbl = doc.add_table(rows=len(table_rows), cols=cols)
                for r_idx, row in enumerate(table_rows):
                    for c_idx, val in enumerate(row):
                        tbl.cell(r_idx, c_idx).text = str(val)

            doc.save(str(resolved))
            return {
                "ok": True,
                "path": resolved.name,
                "backup_created": backup.name if backup else None,
                "summary": (
                    f"سند Word «{resolved.name}» ساخته و ذخیره شد"
                    + (f" (پشتیبان قبلی: {backup.name})." if backup else ".")
                ),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def edit_docx(
        self,
        path: str,
        operation: str = "append_paragraph",
        text: str = "",
        find_text: str = "",
        replace_with: str = "",
        heading_level: int = 2,
    ) -> Dict[str, Any]:
        try:
            resolved = self.sandbox.resolve_safe_path(path, must_exist=True, allowed_extensions=[".docx"])
            backup = self.sandbox.create_backup_if_exists(resolved)
            doc = Document(str(resolved))

            if operation == "append_paragraph":
                doc.add_paragraph(text)
                detail = "پاراگراف جدید اضافه شد"
            elif operation == "add_heading":
                lvl = max(1, min(4, int(heading_level)))
                doc.add_heading(text, level=lvl)
                detail = f"تیتر سطح {lvl} اضافه شد"
            elif operation == "replace_text":
                if not find_text:
                    return {"ok": False, "error": "متن جستجو (`find_text`) برای جایگزینی خالی است."}
                count = 0
                for p in doc.paragraphs:
                    if find_text in p.text:
                        count += p.text.count(find_text)
                        p.text = p.text.replace(find_text, replace_with)
                for tbl in doc.tables:
                    for row in tbl.rows:
                        for cell in row.cells:
                            if find_text in cell.text:
                                count += cell.text.count(find_text)
                                cell.text = cell.text.replace(find_text, replace_with)
                detail = f"{count} مورد جایگزین شد"
            else:
                return {"ok": False, "error": f"عملیات ناشناخته Word: {operation}"}

            doc.save(str(resolved))
            return {
                "ok": True,
                "path": resolved.name,
                "operation": operation,
                "backup_created": backup.name if backup else None,
                "summary": f"سند «{resolved.name}» ویرایش شد ({detail}؛ نسخه پشتیبان: {backup.name if backup else 'ایجاد نشد'}).",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
