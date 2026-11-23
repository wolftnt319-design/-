"""Windows 10 Safe Notepad Automation Tool (`tools/notepad.py`).

Security & Honesty Rules:
- Strict filename and canonical path validation via `SandboxManager` (`.txt`, `.md`, `.log`)
- Automatic `.bak` backup before overwriting any existing text file
- NEVER uses shell string construction (`shell=False` strictly enforced)
- Uses explicit argument vector `[notepad_executable, str(resolved_path)]` from an allowlisted Windows path
- On non-Windows hosts (e.g., Linux test containers), writes the text file safely and truthfully reports
  that launching `notepad.exe` GUI requires Windows!
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from security.sandbox import SandboxManager

SAFE_FILENAME_RE = re.compile(r"^[\w\u0600-\u06FF\-. ]{1,80}$")


def _locate_allowed_notepad() -> Optional[str]:
    """Allowlist resolution for Windows `notepad.exe`."""
    if sys.platform != "win32":
        return None
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidates = [
        Path(sys_root) / "System32" / "notepad.exe",
        Path(sys_root) / "notepad.exe",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c.resolve())
    found = shutil.which("notepad.exe")
    return str(Path(found).resolve()) if found else None


class NotepadTool:
    """Safe story/note writer and Windows Notepad launcher."""

    def __init__(self, sandbox: Optional[SandboxManager] = None) -> None:
        self.sandbox = sandbox or SandboxManager()

    def write_and_open(
        self,
        filename: str,
        content: str,
        open_window: bool = True,
    ) -> Dict[str, Any]:
        try:
            clean_name = Path(filename.strip()).name
            if not clean_name.lower().endswith((".txt", ".md", ".log")):
                clean_name = f"{clean_name}.txt"
            if not SAFE_FILENAME_RE.match(clean_name):
                return {"ok": False, "error": f"نام فایل نامعتبر است: {clean_name}"}

            self.sandbox.check_payload_size(content)
            resolved = self.sandbox.resolve_safe_path(
                clean_name, must_exist=False, allowed_extensions=[".txt", ".md", ".log"]
            )
            backup = self.sandbox.create_backup_if_exists(resolved)
            resolved.write_text(content, encoding="utf-8")

            launched = False
            pid: Optional[int] = None
            platform_note = ""

            if open_window:
                notepad_exe = _locate_allowed_notepad()
                if notepad_exe is not None:
                    proc = subprocess.Popen(
                        [notepad_exe, str(resolved)],
                        shell=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                    )
                    launched = True
                    pid = proc.pid
                else:
                    platform_note = (
                        f"فایل متنی ذخیره شد، اما اجرای برنامه گرافیکی Notepad.exe فقط روی ویندوز امکان‌پذیر است (سیستم فعلی: {sys.platform})."
                    )

            return {
                "ok": True,
                "path": resolved.name,
                "backup_created": backup.name if backup else None,
                "notepad_launched": launched,
                "pid": pid,
                "platform": sys.platform,
                "note": platform_note,
                "summary": (
                    f"متن در «{resolved.name}» ذخیره شد و در Notepad ویندوز باز شد (PID={pid})."
                    if launched
                    else (platform_note or f"متن در فایل «{resolved.name}» با موفقیت ذخیره شد.")
                ),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
