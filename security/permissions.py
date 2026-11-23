"""Permissions, Risk Policy, Interactive Confirmation Manager, and Redacted Audit Logger."""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import AUDIT_LOG_PATH, config_manager
from security.validator import SecretRedactor


class RiskLevel(IntEnum):
    L0_READ_PURE = 0       # Pure math, memory search, workspace read/list
    L1_WORKSPACE_WRITE = 1 # Creating/editing files inside workspace (auto-backed up)
    L2_EXTERNAL_READ = 2   # Controlled web search / browser read
    L3_SENSITIVE_MUTATE = 3 # File deletion, memory important overwrite, Notepad process launch, form submit
    L4_CRITICAL = 4        # Model LoRA training, bulk memory wipe


@dataclass
class PendingConfirmation:
    token: str
    action_type: str  # "tool" | "memory" | "training"
    tool_name: str
    arguments: Dict[str, Any]
    summary_fa: str
    summary_en: str
    risk_level: int
    session_id: str
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 600

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "arguments": SecretRedactor.redact_dict(self.arguments),
            "summary_fa": self.summary_fa,
            "summary_en": self.summary_en,
            "risk_level": self.risk_level,
            "session_id": self.session_id,
            "expires_in_seconds": max(0, int(self.ttl_seconds - (time.time() - self.created_at))),
        }


class AuditLogger:
    """Writes structured, secret-redacted JSON-Lines audit events to `logs/luna.log`."""

    def __init__(self, log_path: Path = AUDIT_LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        actor: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "actor": actor,
            "status": status,
            "details": SecretRedactor.redact_dict(details or {}),
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return entry

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: List[Dict[str, Any]] = []
        for raw in reversed(lines[-limit:]):
            try:
                out.append(json.loads(raw))
            except Exception:
                continue
        return out


class ConfirmationManager:
    """Manages human-in-the-loop approval tokens for sensitive actions."""

    def __init__(self, audit_logger: Optional[AuditLogger] = None) -> None:
        self._pending: Dict[str, PendingConfirmation] = {}
        self.audit = audit_logger or AuditLogger()

    def _cleanup(self) -> None:
        expired = [k for k, v in self._pending.items() if v.is_expired]
        for k in expired:
            del self._pending[k]

    def request_confirmation(
        self,
        action_type: str,
        tool_name: str,
        arguments: Dict[str, Any],
        summary_fa: str,
        summary_en: str,
        risk_level: int = int(RiskLevel.L3_SENSITIVE_MUTATE),
        session_id: str = "default",
    ) -> PendingConfirmation:
        self._cleanup()
        token = f"confirm_{secrets.token_hex(8)}"
        ttl = config_manager.settings.security.confirmation_ttl_seconds
        item = PendingConfirmation(
            token=token,
            action_type=action_type,
            tool_name=tool_name,
            arguments=arguments,
            summary_fa=summary_fa,
            summary_en=summary_en,
            risk_level=risk_level,
            session_id=session_id,
            ttl_seconds=ttl,
        )
        self._pending[token] = item
        self.audit.log(
            event_type="confirmation_requested",
            actor="agent",
            status="pending",
            details={"token": token, "tool_name": tool_name, "action_type": action_type, "arguments": arguments},
            session_id=session_id,
        )
        return item

    def pop_decision(self, token: str, approved: bool) -> Optional[PendingConfirmation]:
        self._cleanup()
        item = self._pending.pop(token, None)
        if item is None or item.is_expired:
            self.audit.log(
                event_type="confirmation_resolved",
                actor="user",
                status="expired_or_missing",
                details={"token": token, "approved": approved},
            )
            return None
        self.audit.log(
            event_type="confirmation_resolved",
            actor="user",
            status="approved" if approved else "rejected",
            details={"token": token, "tool_name": item.tool_name, "action_type": item.action_type},
            session_id=item.session_id,
        )
        return item

    def list_pending(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        self._cleanup()
        items = [
            v.to_public_dict()
            for v in self._pending.values()
            if session_id is None or v.session_id == session_id
        ]
        return items


audit_logger = AuditLogger()
confirmation_manager = ConfirmationManager(audit_logger=audit_logger)
