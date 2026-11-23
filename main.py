"""LUNA AI — Main Entrypoint (FastAPI + WebSocket Backend & CLI).

Usage:
  python main.py                     # Starts LUNA Web Studio on 127.0.0.1:8000
  python main.py --host 0.0.0.0      # Starts server bound to all interfaces (for container preview)
  python main.py --check-env         # Prints real Stage 1 system & Ollama diagnostic JSON
  python main.py --cli               # Runs interactive terminal chat loop
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import psutil
    import uvicorn
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ModuleNotFoundError as _missing_err:
    if "--auto-install" in sys.argv:
        req_file = Path(__file__).resolve().parent / "requirements.txt"
        print(f"[LUNA Setup] Installing missing Python libraries ({_missing_err.name})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        rc = subprocess.call([sys.executable, *[a for a in sys.argv if a != "--auto-install"]])
        sys.exit(rc)
    print("========================================================")
    print(f"[خطا / Error] کتابخانه پایتون «{_missing_err.name}» هنوز نصب نشده است!")
    print("برای نصب خودکار همه پیش‌نیازها در ویندوز، یکی از این دو دستور را بزنید:")
    print("  1) دابل‌کلیک روی فایل جدید run.bat (نصب خودکار کتابخانه‌ها + اجرا)")
    print("  2) یا در CMD / PowerShell داخل همین پوشه اجرا کنید:")
    print(f'     "{sys.executable}" -m pip install -r requirements.txt')
    print("========================================================")
    sys.exit(1)

from config import FRONTEND_DIR, WORKSPACE_DIR, config_manager
from core.agent import luna_agent
from core.env_checker import check_network, inspect_environment
from core.hardware_optimizer import build_hardware_options
from core.model_manager import model_manager
from core.online_teacher import online_teacher
from core.permissions import audit_logger, confirmation_manager
from core.voice_interface import voice_runtime
from memory.long_memory import long_memory
from memory.retrieval import retriever
from memory.short_memory import short_memory
from security.sandbox import SandboxManager, SandboxViolationError

app = FastAPI(
    title="LUNA AI — Local Intelligent Assistant",
    version="1.0.0",
    description="Lightweight, secure local AI assistant powered by Ollama + qwen3:0.6b",
)

# Strict CORS policy: localhost + authorized sandbox preview hosts only
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://[a-zA-Z0-9\-]+\.e2b\.app$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

sandbox_mgr = SandboxManager()
ALLOWED_UPLOAD_EXTS = {".docx", ".xlsx", ".txt", ".md", ".csv", ".json"}


# --- Request Schemas ---
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    session_id: str = Field(default="default", min_length=1, max_length=80)


class CancelRequest(BaseModel):
    session_id: str = Field(default="default")


class ConfirmRequest(BaseModel):
    token: str = Field(..., min_length=6, max_length=80)
    approved: bool
    language: str = Field(default="fa")


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=4, max_length=1000)
    title: Optional[str] = Field(default=None, max_length=80)
    category: str = Field(default="fact")
    importance: int = Field(default=3, ge=1, le=5)


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(default=None, max_length=1000)
    title: Optional[str] = Field(default=None, max_length=80)
    category: Optional[str] = None
    importance: Optional[int] = Field(default=None, ge=1, le=5)
    is_active: Optional[bool] = None


class DistillExemplarRequest(BaseModel):
    user_prompt: str = Field(..., min_length=4, max_length=1500)
    ideal_response: str = Field(..., min_length=4, max_length=3000)
    language: str = Field(default="fa")
    source: str = Field(default="user_or_lmarena")


class FileInspectRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=200)


# --- Routes ---
@app.get("/")
async def index_page() -> Any:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "LUNA API is running; frontend/index.html not found."}, status_code=200)
    return FileResponse(str(index_file))


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    ollama_info = await model_manager.health_check()
    net_info = await asyncio.to_thread(check_network)
    vm = psutil.virtual_memory()
    settings = config_manager.settings
    return {
        "ok": True,
        "internet_online": net_info["internet_available"],
        "network_endpoints": net_info["endpoints"],
        "ollama": ollama_info,
        "operating_mode": settings.operating_mode,
        "preferred_language": settings.preferred_language,
        "teacher_enabled": settings.teacher.enabled,
        "teacher_provider": settings.teacher.provider,
        "ram": {
            "used_gb": round(vm.used / (1024 ** 3), 2),
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "percent": vm.percent,
        },
        "hardware": {
            "accelerator_mode": settings.hardware.accelerator_mode,
            "effective_options": build_hardware_options(settings.hardware),
            "processor": ollama_info.get("processor", {}),
            "ui_lite_glass": settings.hardware.ui_lite_glass,
        },
        "counts": {
            "memories": len(long_memory.list_memories(include_inactive=False, limit=500)),
            "exemplars": len(long_memory.list_teacher_exemplars(limit=500)),
            "pending_confirmations": len(confirmation_manager.list_pending()),
        },
        "voice": voice_runtime.status(),
    }


@app.get("/api/env")
async def api_environment() -> Dict[str, Any]:
    return await asyncio.to_thread(inspect_environment)


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> Dict[str, Any]:
    return await luna_agent.process_message(req.message, session_id=req.session_id)


@app.post("/api/chat/cancel")
async def api_cancel(req: CancelRequest) -> Dict[str, Any]:
    luna_agent.cancel_session(req.session_id)
    return {"ok": True, "cancelled_session": req.session_id}


@app.post("/api/confirm")
async def api_confirm(req: ConfirmRequest) -> Dict[str, Any]:
    lang = "en" if req.language == "en" else "fa"
    return await luna_agent.resolve_confirmation(req.token, approved=req.approved, language=lang)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            action = data.get("action", "chat")
            session_id = str(data.get("session_id", "default"))[:80]

            if action == "cancel":
                luna_agent.cancel_session(session_id)
                await ws.send_json({"type": "cancelled", "session_id": session_id})
                continue

            msg = str(data.get("message", "")).strip()
            if not msg:
                await ws.send_json({"type": "error", "error": "پیام خالی است."})
                continue

            async def push_event(ev: Dict[str, Any]) -> None:
                await ws.send_json(ev)

            result = await luna_agent.process_message(
                user_message=msg,
                session_id=session_id,
                event_callback=push_event,
            )
            await ws.send_json({"type": "final", "data": result})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass


# --- Sessions & History ---
@app.get("/api/sessions")
async def api_list_sessions() -> Dict[str, Any]:
    return {"ok": True, "sessions": short_memory.list_sessions(limit=40)}


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str) -> Dict[str, Any]:
    ctx = short_memory.get_context_window(session_id)
    full_msgs = short_memory.get_full_messages(session_id, limit=120)
    return {
        "ok": True,
        "session": ctx,
        "messages": full_msgs,
        "pending_confirmations": confirmation_manager.list_pending(session_id=session_id),
    }


@app.post("/api/sessions/{session_id}/clear")
async def api_clear_session(session_id: str) -> Dict[str, Any]:
    short_memory.clear_session(session_id)
    return {"ok": True, "cleared": session_id}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str) -> Dict[str, Any]:
    short_memory.delete_session(session_id)
    return {"ok": True, "deleted": session_id}


# --- Persistent Memory CRUD ---
@app.get("/api/memories")
async def api_list_memories(include_inactive: bool = True, q: str = "") -> Dict[str, Any]:
    if q.strip():
        return {"ok": True, "memories": retriever.search_memories(q.strip(), top_k=15, mark_used=False)}
    return {"ok": True, "memories": long_memory.list_memories(include_inactive=include_inactive, limit=150)}


@app.post("/api/memories")
async def api_add_memory(req: MemoryCreateRequest) -> Dict[str, Any]:
    res = long_memory.add_or_merge_memory(
        content=req.content,
        title=req.title,
        category=req.category,
        source="user_ui",
        importance=req.importance,
    )
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "خطا در ثبت حافظه"))
    return res


@app.patch("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, req: MemoryUpdateRequest) -> Dict[str, Any]:
    res = long_memory.update_memory(
        memory_id=memory_id,
        content=req.content,
        title=req.title,
        category=req.category,
        importance=req.importance,
        is_active=req.is_active,
    )
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "خطا در ویرایش حافظه"))
    return res


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int) -> Dict[str, Any]:
    res = long_memory.forget_memory(memory_id)
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail="حافظه یافت نشد")
    return res


# --- Settings & Model Parameter Lab ---
@app.get("/api/settings")
async def api_get_settings() -> Dict[str, Any]:
    health = await model_manager.health_check()
    return {
        "ok": True,
        "settings": config_manager.settings.model_dump(),
        "installed_models": health.get("installed_models", []),
    }


@app.patch("/api/settings")
async def api_patch_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    # Verify active_model only switches to an installed local Ollama model
    if "active_model" in patch:
        health = await model_manager.health_check()
        installed = health.get("installed_models", [])
        if installed and patch["active_model"] not in installed:
            raise HTTPException(
                status_code=400,
                detail=f"مدل «{patch['active_model']}» روی Ollama محلی نصب نیست. مدل‌های موجود: {', '.join(installed)}",
            )
    updated = config_manager.update_partial(patch)
    audit_logger.log("settings_updated", "user", "success", details={"keys": list(patch.keys())})
    return {"ok": True, "settings": updated.model_dump()}


# --- Workspace Files & Word/Excel Upload/Inspector ---
@app.get("/api/files")
async def api_list_workspace_files() -> Dict[str, Any]:
    return luna_agent.registry.file_tool.list_files("")


@app.post("/api/files/upload")
async def api_upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    fname = Path(file.filename or "uploaded_file").name
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(status_code=400, detail=f"پسوند «{ext}» مجاز نیست.")
    raw = await file.read()
    try:
        sandbox_mgr.check_payload_size(raw)
        dest = sandbox_mgr.resolve_safe_path(WORKSPACE_DIR / fname, must_exist=False, allowed_extensions=ALLOWED_UPLOAD_EXTS)
        backup = sandbox_mgr.create_backup_if_exists(dest)
        dest.write_bytes(raw)
        audit_logger.log("file_uploaded", "user", "success", details={"filename": dest.name, "size": len(raw)})
        return {
            "ok": True,
            "filename": dest.name,
            "size_bytes": len(raw),
            "backup_created": backup.name if backup else None,
        }
    except SandboxViolationError as sve:
        raise HTTPException(status_code=400, detail=str(sve))


@app.post("/api/files/inspect")
async def api_inspect_file(req: FileInspectRequest) -> Dict[str, Any]:
    ext = Path(req.path).suffix.lower()
    if ext == ".docx":
        return luna_agent.registry.word_tool.read_docx(req.path)
    if ext == ".xlsx":
        return luna_agent.registry.excel_tool.read_xlsx(req.path)
    return luna_agent.registry.file_tool.read_file(req.path)


# --- Teacher & LMArena Distillation Studio ---
@app.get("/api/teacher/exemplars")
async def api_list_exemplars() -> Dict[str, Any]:
    return {"ok": True, "exemplars": long_memory.list_teacher_exemplars(limit=60)}


@app.post("/api/teacher/distill")
async def api_distill_exemplar(req: DistillExemplarRequest) -> Dict[str, Any]:
    res = long_memory.save_teacher_exemplar(
        user_prompt=req.user_prompt,
        ideal_response=req.ideal_response,
        language="en" if req.language == "en" else "fa",
        teacher_source=req.source,
    )
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=f"ذخیره نمونه آموزشی ناموفق بود: {res.get('reason')}")
    return res


@app.post("/api/teacher/probe-lmarena")
async def api_probe_lmarena() -> Dict[str, Any]:
    res = await online_teacher.probe_lmarena_bridge()
    return {
        "ok": res.ok,
        "provider": res.provider,
        "captcha_detected": res.captcha_detected,
        "content": res.content,
        "error": res.error,
        "latency_ms": res.latency_ms,
    }


# --- Audit Trail ---
@app.get("/api/audit")
async def api_audit_logs() -> Dict[str, Any]:
    return {"ok": True, "logs": audit_logger.read_recent(limit=40)}


async def _run_cli() -> None:
    print("=== LUNA AI Interactive CLI (type 'exit' to quit) ===")
    sid = "cli_session"
    while True:
        try:
            user_in = input("\nشما > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_in.lower() in ("exit", "quit", "/exit"):
            break
        if not user_in:
            continue
        res = await luna_agent.process_message(user_in, session_id=sid)
        print(f"\nLUNA [{res['provenance']}] > {res['reply']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LUNA AI Local Assistant")
    parser.add_argument("--host", default=config_manager.settings.server_host)
    parser.add_argument("--port", type=int, default=config_manager.settings.server_port)
    parser.add_argument("--check-env", action="store_true", help="Run real system & Ollama diagnostics and exit")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal chat")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically once server starts")
    parser.add_argument("--auto-install", action="store_true", help="Automatically pip install missing packages")
    args = parser.parse_args()

    if args.check_env:
        print(json.dumps(inspect_environment(), ensure_ascii=False, indent=2))
        return

    if args.cli:
        asyncio.run(_run_cli())
        return

    if args.open_browser:
        target_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        url = f"http://{target_host}:{args.port}"

        def _launch_browser() -> None:
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_launch_browser, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
