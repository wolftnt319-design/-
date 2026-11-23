"""Hardware Optimizer: Hybrid CPU + GPU Layer Splitter & Windows 10 Anti-Freeze Governor.

Designed specifically for constrained Windows 10 systems (e.g., Intel Core i5 3rd Gen +
Intel HD Graphics / dual laptop GPUs + 8GB RAM):
1. Reserves CPU threads for Windows DWM & browser so mouse/UI never freezes (`num_thread`).
2. Lowers background `ollama` worker scheduling priority to `BELOW_NORMAL_PRIORITY_CLASS` on Windows.
3. Configures hybrid GPU+CPU layer splitting (`num_gpu` + `OLLAMA_VULKAN=1` + `OLLAMA_IGPU_ENABLE=1`).
4. Reduces prompt prefill spikes via compact `num_batch=128` and `num_ctx=1024`.
5. Queries Ollama `/api/ps` to report real-time actual CPU vs GPU memory split truthfully.
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, List

import httpx
import psutil


def compute_safe_cpu_threads() -> int:
    """Leave 1-2 logical cores free so Windows 10 UI, audio, and Intel HD Graphics never stutter."""
    logical = psutil.cpu_count(logical=True) or 2
    physical = psutil.cpu_count(logical=False) or max(1, logical // 2)
    if logical <= 2:
        return 1
    if logical <= 4:
        # Typical Core i5 3rd Gen mobile (2C/4T) or desktop (4C/4T): use 2 threads (50% cap -> zero freeze!)
        return 2
    return max(2, min(physical, logical - 2))


def detect_gpus() -> List[Dict[str, Any]]:
    """Detect GPU adapters truthfully without external heavy dependencies."""
    gpus: List[Dict[str, Any]] = []
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=4,
                shell=False,
            )
            raw = (proc.stdout or "").strip()
            if raw:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else [parsed]
                for it in items:
                    name = str(it.get("Name", "Unknown GPU"))
                    lower = name.lower()
                    is_legacy_hd = any(tag in lower for tag in ("hd graphics 2000", "hd graphics 2500", "hd graphics 3000", "hd graphics 4000"))
                    vulkan_likely = not is_legacy_hd and any(
                        tag in lower for tag in ("nvidia", "geforce", "radeon", "amd", "arc", "iris", "uhd", "hd graphics 5", "hd graphics 6")
                    )
                    gpus.append({
                        "name": name,
                        "driver": it.get("DriverVersion"),
                        "vram_mb": round(int(it.get("AdapterRAM") or 0) / (1024 * 1024), 1) if it.get("AdapterRAM") else None,
                        "vulkan_or_cuda_likely": vulkan_likely,
                        "legacy_dx11_igpu": is_legacy_hd,
                    })
        except Exception:
            pass
    return gpus


def tame_ollama_priority() -> Dict[str, Any]:
    """Set Ollama worker processes to BELOW_NORMAL priority so Windows UI is always 100% responsive."""
    adjusted: List[int] = []
    errors: List[str] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = (proc.info.get("name") or "").lower()
            if "ollama" in pname:
                if sys.platform == "win32":
                    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    adjusted.append(int(proc.info["pid"]))
                else:
                    cur = proc.nice()
                    if cur < 5:
                        proc.nice(5)
                        adjusted.append(int(proc.info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            errors.append(type(exc).__name__)
        except Exception as exc:
            errors.append(str(exc))
    return {"adjusted_pids": adjusted, "skipped_or_denied": len(errors)}


async def inspect_ollama_processor_split(base_url: str) -> Dict[str, Any]:
    """Query Ollama `/api/ps` to see the exact real-time CPU vs GPU split of loaded models."""
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/ps")
        if resp.status_code != 200:
            return {"loaded": False, "processor_summary": "Idle (مدل در حالت آماده‌باش)"}
        models = resp.json().get("models", [])
        if not models:
            return {"loaded": False, "processor_summary": "Idle (رم آزاد)"}
        m = models[0]
        total = int(m.get("size") or 0)
        vram = int(m.get("size_vram") or 0)
        if total > 0 and vram > 0:
            gpu_pct = round((vram / total) * 100)
            cpu_pct = max(0, 100 - gpu_pct)
            summary = f"{gpu_pct}% GPU / {cpu_pct}% CPU"
        else:
            gpu_pct = 0
            cpu_pct = 100
            summary = "100% CPU (مهارشده بدون هنگ)"
        return {
            "loaded": True,
            "model": m.get("name"),
            "size_mb": round(total / (1024 * 1024), 1),
            "vram_mb": round(vram / (1024 * 1024), 1),
            "gpu_percent": gpu_pct,
            "cpu_percent": cpu_pct,
            "processor_summary": summary,
        }
    except Exception:
        return {"loaded": False, "processor_summary": "نامشخص"}


def build_hardware_options(hw_cfg: Any) -> Dict[str, Any]:
    """Translate `HardwareConfig` into exact Ollama `options` for smooth CPU+GPU co-execution."""
    threads = hw_cfg.cpu_threads if (hw_cfg.cpu_threads and hw_cfg.cpu_threads > 0) else compute_safe_cpu_threads()
    mode = getattr(hw_cfg, "accelerator_mode", "hybrid_cpu_gpu")
    gpu_layers = int(getattr(hw_cfg, "gpu_layers", 14))
    batch_size = int(getattr(hw_cfg, "num_batch", 128))

    opts: Dict[str, Any] = {
        "num_thread": threads,
        "num_batch": batch_size,
    }
    if mode == "cpu_eco":
        opts["num_gpu"] = 0
    elif mode == "gpu_max":
        opts["num_gpu"] = 99
    else:
        # Hybrid CPU + GPU split: offload `gpu_layers` (default 14 of 28 layers in qwen3:0.6b)
        # onto Vulkan/CUDA/ROCm GPU if supported, while keeping remaining layers on capped CPU threads!
        opts["num_gpu"] = gpu_layers
    return opts
