"""Optional Isolated LoRA Adapter Trainer & Rollback Manager (`training/train_lora.py`).

Safety & Hardware Guardrails:
1. NEVER imported or executed during normal LUNA startup.
2. Refuses to run without `--confirm-user-approval`.
3. Refuses to run while Ollama has a model actively loaded in RAM (`/api/ps`), preventing RAM thrashing on 8GB PCs.
4. Saves ONLY lightweight LoRA adapters + checkpoint manifests + rollback pointers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx
import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, config_manager

ADAPTERS_DIR = DATA_DIR / "adapters"


def check_ollama_idle() -> tuple[bool, str]:
    base_url = config_manager.settings.ollama_base_url.rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{base_url}/api/ps")
        if r.status_code == 200:
            running = r.json().get("models", [])
            if running:
                names = ", ".join(m.get("name", "?") for m in running)
                return False, f"مدل فعال در Ollama شناسایی شد ({names}). اجرای هم‌زمان آموزش با Ollama ممنوع است؛ ابتدا `ollama stop {running[0].get('name', '')}` را اجرا کنید."
    except Exception:
        pass
    return True, "ollama_idle"


def rollback_adapter(checkpoint_name: str) -> Dict[str, Any]:
    target = ADAPTERS_DIR / checkpoint_name
    active_link = ADAPTERS_DIR / "active_adapter.json"
    if not target.exists():
        return {"ok": False, "error": f"چک‌پوینت «{checkpoint_name}» یافت نشد."}
    if active_link.exists():
        shutil.copy2(active_link, ADAPTERS_DIR / "active_adapter.prev.bak")
    active_link.write_text(
        json.dumps(
            {"active_checkpoint": checkpoint_name, "rolled_back_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"ok": True, "active_checkpoint": checkpoint_name}


def run_lora_training(
    dataset_dir: Path,
    base_hf_model: str,
    epochs: int,
    confirm_approval: bool,
) -> Dict[str, Any]:
    if not confirm_approval:
        return {
            "ok": False,
            "error": "آموزش بدون تأیید صریح کاربر مجاز نیست. سوئیچ `--confirm-user-approval` الزامی است.",
        }

    idle_ok, idle_msg = check_ollama_idle()
    if not idle_ok:
        return {"ok": False, "error": idle_msg}

    vm = psutil.virtual_memory()
    if vm.available < 2.5 * (1024 ** 3):
        return {
            "ok": False,
            "error": f"حافظه آزاد فعلی ({round(vm.available / (1024**3), 2)} GB) کمتر از حداقل ایمن (2.5 GB) برای آموزش LoRA است.",
        }

    try:
        import torch  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as ie:
        return {
            "ok": False,
            "error": (
                f"کتابخانه‌های اختیاری آموزش نصب نیستند ({ie}). "
                "برای حفظ سبکی محیط اصلی، این کتابخانه‌ها در startup نصب نمی‌شوند. "
                "در صورت نیاز در یک محیط مجازی مجزا نصب کنید: `pip install torch transformers peft`"
            ),
        }

    train_file = dataset_dir / "train.jsonl"
    if not train_file.exists():
        return {"ok": False, "error": f"فایل دیتاست یافت نشد: {train_file}"}

    samples = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 2:
        return {"ok": False, "error": "تعداد نمونه‌های آموزشی کمتر از حداقل مورد نیاز (۲ نمونه) است."}

    t0 = time.perf_counter()
    mem_before_mb = round(psutil.Process().memory_info().rss / (1024 ** 2), 1)
    ckpt_id = datetime.now(timezone.utc).strftime("lora_%Y%m%d_%H%M%S")
    out_dir = ADAPTERS_DIR / ckpt_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_hf_model, torch_dtype=torch.float32)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lora_cfg)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=2e-4)
    peft_model.train()
    losses = []
    for _ in range(max(1, epochs)):
        for item in samples:
            rendered = "\n".join(f"{m['role']}: {m['content']}" for m in item["messages"])
            enc = tokenizer(rendered, truncation=True, max_length=384, return_tensors="pt")
            out = peft_model(**enc, labels=enc["input_ids"])
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu().item()))

    peft_model.save_pretrained(str(out_dir))
    elapsed_s = round(time.perf_counter() - t0, 2)
    mem_peak_mb = round(psutil.Process().memory_info().rss / (1024 ** 2), 1)

    report = {
        "ok": True,
        "checkpoint_id": ckpt_id,
        "base_model": base_hf_model,
        "samples_trained": len(samples),
        "epochs": epochs,
        "final_loss": round(losses[-1], 4) if losses else None,
        "mean_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "elapsed_seconds": elapsed_s,
        "rss_before_mb": mem_before_mb,
        "rss_after_mb": mem_peak_mb,
        "adapter_saved_to": str(out_dir),
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rollback_adapter(ckpt_id)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional Isolated LoRA Trainer for LUNA AI.")
    parser.add_argument("--dataset-dir", default=str(DATA_DIR / "datasets"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--confirm-user-approval", action="store_true")
    parser.add_argument("--rollback", default="", help="Checkpoint folder name to roll back to.")
    args = parser.parse_args()

    if args.rollback:
        print(json.dumps(rollback_adapter(args.rollback), ensure_ascii=False, indent=2))
    else:
        res = run_lora_training(
            dataset_dir=Path(args.dataset_dir),
            base_hf_model=args.base_model,
            epochs=args.epochs,
            confirm_approval=args.confirm_user_approval,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
