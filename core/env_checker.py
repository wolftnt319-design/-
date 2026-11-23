"""Stage 1: Real System & Environment Inspector for LUNA AI.

Performs truthful runtime verification of:
- OS, kernel/release, and CPU architecture (Windows 10 vs Linux staging)
- Python & pip versions
- Total/Available RAM and Disk free space
- Ollama binary, Ollama HTTP daemon health, installed models, and qwen3:0.6b availability
- Network connectivity (for optional controlled web search & free online teacher)
- Auxiliary tools (Git, Node.js, Playwright, python-docx, openpyxl, SQLite FTS5)
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import psutil

from config import BASE_DIR, config_manager
from core.hardware_optimizer import compute_safe_cpu_threads, detect_gpus

_NET_CACHE: Optional[Dict[str, Any]] = None
_NET_CACHE_TS: float = 0.0


def _run_version(cmd: List[str]) -> Dict[str, Any]:
    exe = shutil.which(cmd[0])
    if not exe:
        return {"installed": False, "path": None, "version": None}
    try:
        proc = subprocess.run(
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
        out = (proc.stdout or proc.stderr).strip().splitlines()
        return {
            "installed": proc.returncode == 0,
            "path": exe,
            "version": out[0] if out else "",
        }
    except Exception as exc:
        return {"installed": False, "path": exe, "version": f"error: {exc}"}


def check_sqlite_fts5() -> Dict[str, Any]:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(body)")
        con.close()
        return {"supported": True, "sqlite_version": sqlite3.sqlite_version}
    except Exception as exc:
        return {"supported": False, "sqlite_version": sqlite3.sqlite_version, "error": str(exc)}


def check_ollama(base_url: str, target_model: str) -> Dict[str, Any]:
    cli_info = _run_version(["ollama", "--version"])
    result: Dict[str, Any] = {
        "cli": cli_info,
        "base_url": base_url,
        "service_running": False,
        "service_version": None,
        "installed_models": [],
        "target_model": target_model,
        "target_model_ready": False,
        "target_model_size_mb": None,
        "error": None,
    }
    try:
        with httpx.Client(timeout=4.0) as client:
            ver_resp = client.get(f"{base_url.rstrip('/')}/api/version")
            if ver_resp.status_code == 200:
                result["service_running"] = True
                result["service_version"] = ver_resp.json().get("version")

            tags_resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            if tags_resp.status_code == 200:
                models = tags_resp.json().get("models", [])
                names = []
                for m in models:
                    name = m.get("name", "")
                    size_mb = round(m.get("size", 0) / (1024 * 1024), 1)
                    names.append({"name": name, "size_mb": size_mb})
                    if name == target_model or name.startswith(target_model.split(":")[0] + ":0.6b"):
                        if name == target_model:
                            result["target_model_ready"] = True
                            result["target_model_size_mb"] = size_mb
                result["installed_models"] = names
    except Exception as exc:
        result["error"] = str(exc)
    return result


def check_network(force: bool = False) -> Dict[str, Any]:
    global _NET_CACHE, _NET_CACHE_TS
    now = time.monotonic()
    if not force and _NET_CACHE is not None and (now - _NET_CACHE_TS) < 180.0:
        return _NET_CACHE

    endpoints = {
        "duckduckgo": "https://html.duckduckgo.com/html/",
        "pollinations_teacher": "https://text.pollinations.ai/",
    }
    status: Dict[str, bool] = {}
    for name, url in endpoints.items():
        try:
            with httpx.Client(timeout=3.0, follow_redirects=True) as client:
                r = client.head(url)
                status[name] = r.status_code < 500
        except Exception:
            status[name] = False
    res = {
        "internet_available": any(status.values()),
        "endpoints": status,
    }
    _NET_CACHE, _NET_CACHE_TS = res, now
    return res


def check_python_packages() -> Dict[str, Any]:
    required = [
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "jsonschema",
        "scikit-learn",
        "beautifulsoup4",
        "python-docx",
        "openpyxl",
        "psutil",
        "playwright",
        "pytest",
    ]
    found: Dict[str, Optional[str]] = {}
    for pkg in required:
        try:
            found[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            found[pkg] = None
    return found


def inspect_environment() -> Dict[str, Any]:
    settings = config_manager.settings
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(BASE_DIR))
    sys_name = platform.system()
    is_windows = sys_name.lower() == "windows"

    ollama_status = check_ollama(settings.ollama_base_url, settings.primary_model)
    fts5_status = check_sqlite_fts5()
    packages = check_python_packages()
    network = check_network()

    issues: List[str] = []
    remediations: List[str] = []

    if not ollama_status["service_running"]:
        issues.append("سرویس Ollama در دسترس نیست.")
        if is_windows:
            remediations.append("از سایت https://ollama.com نسخه Windows را نصب کرده و دستور `ollama serve` را اجرا کنید.")
        else:
            remediations.append("دستور `ollama serve` را اجرا کنید.")

    if ollama_status["service_running"] and not ollama_status["target_model_ready"]:
        issues.append(f"مدل محلی `{settings.primary_model}` هنوز دانلود نشده است.")
        remediations.append(f"دستور `ollama pull {settings.primary_model}` را اجرا کنید.")

    missing_pkgs = [k for k, v in packages.items() if v is None]
    if missing_pkgs:
        issues.append(f"کتابخانه‌های پایتون نصب نشده‌اند: {', '.join(missing_pkgs)}")
        remediations.append("دستور `pip install -r requirements.txt` را اجرا کنید.")

    if not fts5_status["supported"]:
        issues.append("ماژول SQLite FTS5 روی این نسخه Python فعال نیست (سیستم به بازیابی TF-IDF سوییچ می‌کند).")

    if round(vm.total / (1024 ** 3), 2) < 4.0:
        issues.append(
            f"حافظه کل سیستم ({round(vm.total / (1024 ** 3), 2)} GB) کمتر از 8GB هدف است؛ حالت کم‌مصرف و تک‌مدل (Lazy Single-Model) فعال شد."
        )

    return {
        "os": {
            "system": sys_name,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.uname().machine,
            "is_windows_10_target": is_windows,
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "pip": _run_version([sys.executable, "-m", "pip", "--version"]),
        },
        "resources": {
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "ram_percent_used": vm.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 2),
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "safe_cpu_threads": compute_safe_cpu_threads(),
            "gpus": detect_gpus(),
        },
        "ollama": ollama_status,
        "sqlite_fts5": fts5_status,
        "network": network,
        "auxiliary_tools": {
            "git": _run_version(["git", "--version"]),
            "node": _run_version(["node", "--version"]),
            "notepad_exe": {
                "available": is_windows and bool(shutil.which("notepad.exe")),
                "path": shutil.which("notepad.exe") if is_windows else None,
            },
        },
        "python_packages": packages,
        "issues": issues,
        "remediations": remediations,
        "ready_for_core": ollama_status["service_running"]
        and ollama_status["target_model_ready"]
        and not missing_pkgs,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = inspect_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2))
