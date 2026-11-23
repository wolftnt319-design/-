"""Path Sandbox & Atomic Backup Manager.

Enforces:
- Canonical path resolution
-Strict confinement to workspace / approved roots
- Symlink escape & path traversal (`..`, UNC, drive switching, null byte) blocking
- File size limits
- Automatic timestamped `.bak` backup BEFORE modifying or overwriting any existing file
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from config import BACKUP_DIR, BASE_DIR, UPLOAD_DIR, WORKSPACE_DIR, config_manager


class SandboxViolationError(PermissionError):
    """Raised when a path escapes allowed sandbox directories or violates file safety rules."""


class SandboxManager:
    """Canonical path jail and backup-before-write manager."""

    def __init__(
        self,
        allowed_roots: Optional[Iterable[str | Path]] = None,
        backup_dir: Path = BACKUP_DIR,
        max_file_size_bytes: Optional[int] = None,
    ) -> None:
        if allowed_roots is not None:
            raw_roots = list(allowed_roots)
            resolved_roots: List[Path] = []
            for r in raw_roots:
                p = Path(r).expanduser()
                if not p.is_absolute():
                    p = BASE_DIR / p
                resolved_roots.append(p.resolve())
        else:
            cfg_roots = config_manager.settings.security.allowed_roots or []
            resolved_roots = [WORKSPACE_DIR.resolve(), UPLOAD_DIR.resolve()]
            for r in cfg_roots:
                r_str = str(r).strip()
                # Ignore stale Linux absolute paths if project folder was copied to Windows
                if os.name == "nt" and r_str.startswith("/"):
                    continue
                p = Path(r_str).expanduser()
                if not p.is_absolute():
                    p = BASE_DIR / p
                rp = p.resolve()
                if rp not in resolved_roots:
                    resolved_roots.append(rp)

        self.allowed_roots = resolved_roots or [WORKSPACE_DIR.resolve()]
        for root in self.allowed_roots:
            root.mkdir(parents=True, exist_ok=True)
        self.default_root: Path = self.allowed_roots[0]
        self.backup_dir: Path = backup_dir.resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes: int = (
            max_file_size_bytes
            if max_file_size_bytes is not None
            else config_manager.settings.security.max_file_size_bytes
        )

    def resolve_safe_path(
        self,
        user_path: str | Path,
        must_exist: bool = False,
        allowed_extensions: Optional[Iterable[str]] = None,
    ) -> Path:
        """Resolve `user_path` safely inside allowed roots; block traversal and symlink escapes."""
        raw = str(user_path).strip()
        if not raw:
            raise SandboxViolationError("مسیر فایل خالی است (Empty path).")
        if "\x00" in raw:
            raise SandboxViolationError("کاراکتر نامعتبر Null در مسیر فایل وجود دارد.")

        # Block Windows UNC paths and explicit parent traversal tricks on raw input if suspicious
        if raw.startswith("\\\\") or raw.startswith("//"):
            raise SandboxViolationError("مسیر شبکه UNC مجاز نیست.")

        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.default_root / candidate

        # Check symlink escape on every existing ancestor/target before resolving
        probe = candidate
        while True:
            if probe.exists() and probe.is_symlink():
                real_target = probe.resolve()
                if not self._is_within_allowed(real_target):
                    raise SandboxViolationError(
                        f"لینک نمادین (Symlink) به خارج از محیط مجاز اشاره می‌کند: {user_path}"
                    )
            if probe.parent == probe:
                break
            probe = probe.parent

        resolved = candidate.resolve()
        if not self._is_within_allowed(resolved):
            raise SandboxViolationError(
                f"دسترسی به خارج از پوشه مجاز مسدود شد (Path Traversal Blocked): {user_path}"
            )

        if allowed_extensions is not None:
            exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed_extensions}
            if resolved.suffix.lower() not in exts:
                raise SandboxViolationError(
                    f"پسوند فایل «{resolved.suffix}» مجاز نیست. پسوندهای مجاز: {', '.join(sorted(exts))}"
                )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"فایل مورد نظر یافت نشد: {resolved.name}")

        if resolved.exists() and resolved.is_file():
            size = resolved.stat().st_size
            if size > self.max_file_size_bytes:
                raise SandboxViolationError(
                    f"حجم فایل ({size} بایت) بیشتر از سقف مجاز ({self.max_file_size_bytes} بایت) است."
                )

        return resolved

    def _is_within_allowed(self, resolved_path: Path) -> bool:
        for root in self.allowed_roots:
            try:
                resolved_path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def create_backup_if_exists(self, target_path: Path) -> Optional[Path]:
        """Create a timestamped backup copy in `data/backups/` before modifying an existing file."""
        resolved = self.resolve_safe_path(target_path, must_exist=False)
        if not resolved.exists() or not resolved.is_file():
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_stem = resolved.name.replace(os.sep, "_")
        backup_path = self.backup_dir / f"{safe_stem}.{ts}.bak"
        shutil.copy2(resolved, backup_path)
        return backup_path

    def check_payload_size(self, content: str | bytes) -> None:
        raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
        if len(raw_bytes) > self.max_file_size_bytes:
            raise SandboxViolationError(
                f"اندازه محتوا ({len(raw_bytes)} بایت) از سقف مجاز ({self.max_file_size_bytes} بایت) فراتر است."
            )
