"""Sandboxed Workspace File Manager (Read, Write with Backup, List, Search, Delete)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from security.sandbox import SandboxManager, SandboxViolationError

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".py", ".yaml", ".yml", ".ini", ".html", ".css", ".js"}


class FileManagerTool:
    """Performs canonicalized, sandboxed file operations inside allowed workspace roots."""

    def __init__(self, sandbox: Optional[SandboxManager] = None) -> None:
        self.sandbox = sandbox or SandboxManager()

    def list_files(self, subdir: str = "") -> Dict[str, Any]:
        try:
            target_dir = self.sandbox.resolve_safe_path(subdir or ".", must_exist=True)
            if not target_dir.is_dir():
                return {"ok": False, "error": "مسیر مشخص‌شده یک پوشه نیست."}
            items: List[Dict[str, Any]] = []
            for p in sorted(target_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:200]:
                if p.is_symlink():
                    continue
                rel = str(p.relative_to(self.sandbox.default_root)) if p.is_relative_to(self.sandbox.default_root) else p.name
                items.append({
                    "name": p.name,
                    "path": rel,
                    "type": "dir" if p.is_dir() else "file",
                    "size_bytes": p.stat().st_size if p.is_file() else 0,
                })
            return {
                "ok": True,
                "directory": str(target_dir.name),
                "count": len(items),
                "files": items,
                "summary": f"{len(items)} مورد در پوشه «{target_dir.name}» یافت شد.",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def read_file(self, path: str, max_chars: int = 8000) -> Dict[str, Any]:
        try:
            resolved = self.sandbox.resolve_safe_path(path, must_exist=True, allowed_extensions=TEXT_EXTENSIONS)
            content = resolved.read_text(encoding="utf-8", errors="replace")
            truncated = len(content) > max_chars
            return {
                "ok": True,
                "path": resolved.name,
                "content": content[:max_chars],
                "truncated": truncated,
                "size_bytes": resolved.stat().st_size,
                "summary": f"فایل «{resolved.name}» خوانده شد ({len(content)} کاراکتر).",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        try:
            self.sandbox.check_payload_size(content)
            resolved = self.sandbox.resolve_safe_path(path, must_exist=False, allowed_extensions=TEXT_EXTENSIONS)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            backup = self.sandbox.create_backup_if_exists(resolved)
            if append and resolved.exists():
                existing = resolved.read_text(encoding="utf-8", errors="replace")
                new_body = existing + ("\n" if existing and not existing.endswith("\n") else "") + content
                self.sandbox.check_payload_size(new_body)
                resolved.write_text(new_body, encoding="utf-8")
            else:
                resolved.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "path": resolved.name,
                "backup_created": backup.name if backup else None,
                "bytes_written": resolved.stat().st_size,
                "summary": (
                    f"فایل «{resolved.name}» با موفقیت ذخیره شد"
                    + (f" (نسخه پشتیبان: {backup.name})." if backup else ".")
                ),
            }
        except SandboxViolationError as sve:
            return {"ok": False, "error": str(sve)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def search_files(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip().lower()
        if not q:
            return {"ok": False, "error": "عبارت جستجو خالی است."}
        matches: List[Dict[str, Any]] = []
        for root in self.sandbox.allowed_roots:
            for p in root.rglob("*"):
                if len(matches) >= 30:
                    break
                if not p.is_file() or p.is_symlink():
                    continue
                name_hit = q in p.name.lower()
                snippet = ""
                content_hit = False
                if p.suffix.lower() in TEXT_EXTENSIONS and p.stat().st_size <= self.sandbox.max_file_size_bytes:
                    try:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        idx = txt.lower().find(q)
                        if idx != -1:
                            content_hit = True
                            snippet = txt[max(0, idx - 40) : idx + 100].replace("\n", " ")
                    except Exception:
                        pass
                if name_hit or content_hit:
                    matches.append({
                        "name": p.name,
                        "matched_by": "name+content" if (name_hit and content_hit) else ("name" if name_hit else "content"),
                        "snippet": snippet,
                    })
        return {
            "ok": True,
            "query": query,
            "count": len(matches),
            "matches": matches,
            "summary": f"{len(matches)} فایل منطبق با «{query}» پیدا شد.",
        }

    def delete_file(self, path: str) -> Dict[str, Any]:
        """Sensitive operation (guarded by ConfirmationManager in ToolRegistry)."""
        try:
            resolved = self.sandbox.resolve_safe_path(path, must_exist=True)
            if resolved.is_dir():
                return {"ok": False, "error": "حذف پوشه مجاز نیست؛ فقط حذف تک‌فایل با تأیید مجاز است."}
            backup = self.sandbox.create_backup_if_exists(resolved)
            resolved.unlink()
            return {
                "ok": True,
                "deleted": resolved.name,
                "backup_created": backup.name if backup else None,
                "summary": f"فایل «{resolved.name}» حذف شد (نسخه پشتیبان امن: {backup.name if backup else 'ندارد'}).",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
