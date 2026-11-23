"""Sandboxed Excel (.xlsx) Editor with Automatic `.bak` Backup (`openpyxl`)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from openpyxl import Workbook, load_workbook
from openpyxl.utils import column_index_from_string

from security.sandbox import SandboxManager

# Allow standard mathematical/statistical Excel formulas while blocking DDE/cmd injection
DANGEROUS_EXCEL_FORMULA_RE = re.compile(r"(?i)(cmd|powershell|msexe|!\s*['\"]|/c\s|\|)")
SAFE_FORMULA_RE = re.compile(r"^=[A-Za-z0-9_.:(),+\-*/^%\"'\s<>=$]+$")


class ExcelEditorTool:
    """Read, create, edit cells/rows/columns, set safe formulas, and sort `.xlsx` workbooks."""

    def __init__(self, sandbox: Optional[SandboxManager] = None) -> None:
        self.sandbox = sandbox or SandboxManager()

    def read_xlsx(self, path: str, sheet_name: Optional[str] = None, max_rows: int = 60) -> Dict[str, Any]:
        try:
            resolved = self.sandbox.resolve_safe_path(path, must_exist=True, allowed_extensions=[".xlsx"])
            wb = load_workbook(str(resolved), data_only=False)
            ws = wb[sheet_name] if (sheet_name and sheet_name in wb.sheetnames) else wb.active
            rows_out: List[List[Any]] = []
            for idx, row in enumerate(ws.iter_rows(values_only=True)):
                if idx >= max_rows:
                    break
                rows_out.append(["" if v is None else v for v in row])
            total_rows = ws.max_row or 0
            total_cols = ws.max_column or 0
            wb.close()
            return {
                "ok": True,
                "path": resolved.name,
                "sheets": wb.sheetnames,
                "active_sheet": ws.title,
                "total_rows": total_rows,
                "total_cols": total_cols,
                "rows": rows_out,
                "summary": f"فایل Excel «{resolved.name}» (شیت {ws.title}: {total_rows} ردیف و {total_cols} ستون) خوانده شد.",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def create_xlsx(
        self,
        path: str,
        headers: Optional[List[Any]] = None,
        rows: Optional[List[List[Any]]] = None,
        sheet_name: str = "Sheet1",
    ) -> Dict[str, Any]:
        try:
            if not path.lower().endswith(".xlsx"):
                path = f"{path}.xlsx"
            resolved = self.sandbox.resolve_safe_path(path, must_exist=False, allowed_extensions=[".xlsx"])
            resolved.parent.mkdir(parents=True, exist_ok=True)
            backup = self.sandbox.create_backup_if_exists(resolved)

            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name[:31] or "Sheet1"
            if headers:
                ws.append(list(headers))
            for r in (rows or []):
                ws.append(list(r))
            wb.save(str(resolved))
            wb.close()
            return {
                "ok": True,
                "path": resolved.name,
                "sheet": ws.title,
                "backup_created": backup.name if backup else None,
                "summary": f"فایل Excel «{resolved.name}» با موفقیت ساخته شد.",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def edit_xlsx(
        self,
        path: str,
        operation: str,
        sheet_name: Optional[str] = None,
        cell: str = "",
        value: Any = None,
        row_values: Optional[List[Any]] = None,
        column_header: str = "",
        column_values: Optional[List[Any]] = None,
        sort_by_col: str | int = 1,
        descending: bool = False,
        has_header: bool = True,
    ) -> Dict[str, Any]:
        """Supported operations: `write_cell`, `append_row`, `add_column`, `set_formula`, `sort_sheet`."""
        try:
            resolved = self.sandbox.resolve_safe_path(path, must_exist=True, allowed_extensions=[".xlsx"])
            backup = self.sandbox.create_backup_if_exists(resolved)
            wb = load_workbook(str(resolved))
            ws = wb[sheet_name] if (sheet_name and sheet_name in wb.sheetnames) else wb.active

            if operation == "write_cell":
                if not re.match(r"^[A-Za-z]{1,3}[1-9][0-9]{0,5}$", cell.strip()):
                    wb.close()
                    return {"ok": False, "error": f"آدرس سلول نامعتبر است: {cell}"}
                if isinstance(value, str) and value.startswith("="):
                    wb.close()
                    return {"ok": False, "error": "برای درج فرمول از عملیات `set_formula` استفاده کنید."}
                ws[cell.strip().upper()] = value
                detail = f"سلول {cell.upper()} مقداردهی شد"

            elif operation == "set_formula":
                if not re.match(r"^[A-Za-z]{1,3}[1-9][0-9]{0,5}$", cell.strip()):
                    wb.close()
                    return {"ok": False, "error": f"آدرس سلول نامعتبر است: {cell}"}
                formula = str(value or "").strip()
                if not formula.startswith("="):
                    formula = "=" + formula
                if DANGEROUS_EXCEL_FORMULA_RE.search(formula) or not SAFE_FORMULA_RE.match(formula):
                    wb.close()
                    return {"ok": False, "error": "فرمول ارسالی به دلایل امنیتی مسدود شد."}
                ws[cell.strip().upper()] = formula
                detail = f"فرمول {formula} در سلول {cell.upper()} ثبت شد"

            elif operation == "append_row":
                if not row_values:
                    wb.close()
                    return {"ok": False, "error": "`row_values` خالی است."}
                ws.append(list(row_values))
                detail = f"یک ردیف جدید ({len(row_values)} ستون) اضافه شد"

            elif operation == "add_column":
                col_idx = (ws.max_column or 0) + 1
                start_row = 1
                if column_header:
                    ws.cell(row=1, column=col_idx, value=column_header)
                    start_row = 2
                for offset, val in enumerate(column_values or []):
                    ws.cell(row=start_row + offset, column=col_idx, value=val)
                detail = f"ستون جدید شماره {col_idx} اضافه شد"

            elif operation == "sort_sheet":
                all_rows = [list(r) for r in ws.iter_rows(values_only=True)]
                if len(all_rows) <= (1 if has_header else 0):
                    wb.close()
                    return {"ok": True, "path": resolved.name, "summary": "ردیفی برای مرتب‌سازی وجود نداشت."}
                header_row = all_rows[0] if has_header else None
                body_rows = all_rows[1:] if has_header else all_rows

                if isinstance(sort_by_col, str) and sort_by_col.strip().isalpha():
                    col_i = column_index_from_string(sort_by_col.strip().upper()) - 1
                else:
                    col_i = max(0, int(sort_by_col) - 1)

                def _sort_key(r: List[Any]) -> Any:
                    v = r[col_i] if col_i < len(r) else None
                    if v is None:
                        return (1, "")
                    if isinstance(v, (int, float)):
                        return (0, v)
                    return (0, str(v).lower())

                body_rows.sort(key=_sort_key, reverse=bool(descending))
                ws.delete_rows(1, ws.max_row)
                if header_row is not None:
                    ws.append(header_row)
                for r in body_rows:
                    ws.append(r)
                detail = f"ردیف‌ها بر اساس ستون {col_i + 1} مرتب شدند"

            else:
                wb.close()
                return {"ok": False, "error": f"عملیات ناشناخته Excel: {operation}"}

            wb.save(str(resolved))
            wb.close()
            return {
                "ok": True,
                "path": resolved.name,
                "operation": operation,
                "backup_created": backup.name if backup else None,
                "summary": f"فایل Excel «{resolved.name}» ویرایش شد ({detail}؛ پشتیبان: {backup.name if backup else 'ندارد'}).",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
