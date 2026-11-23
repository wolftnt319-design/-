"""Chronological Dataset Curator for LUNA AI (`training/prepare_dataset.py`).

Rules enforced:
- Never runs automatically at startup.
- Scrubs and rejects any record containing secrets, passwords, bank cards, or tokens.
- Uses strict chronological ordering (time-based split, NEVER random `train_test_split` across conversation turns)
  to prevent temporal data leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, SYSTEM_PROMPT_EN, SYSTEM_PROMPT_FA
from memory.database import db_manager
from security.validator import SecretRedactor


def collect_chronological_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with db_manager.connect() as conn:
        # 1. Verified Teacher Exemplars ordered by creation timestamp
        ex_rows = conn.execute(
            "SELECT language, user_prompt, ideal_response, quality_score, created_at FROM teacher_exemplars WHERE quality_score >= 0.75 ORDER BY created_at ASC, id ASC"
        ).fetchall()
        for r in ex_rows:
            u = (r["user_prompt"] or "").strip()
            a = (r["ideal_response"] or "").strip()
            if not u or not a:
                continue
            has_s1, _ = SecretRedactor.contains_secret(u)
            has_s2, _ = SecretRedactor.contains_secret(a)
            if has_s1 or has_s2:
                continue
            lang = r["language"] if r["language"] in ("fa", "en") else "fa"
            sys_p = SYSTEM_PROMPT_FA if lang == "fa" else SYSTEM_PROMPT_EN
            records.append({
                "timestamp": r["created_at"],
                "language": lang,
                "source": "teacher_exemplar",
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ],
            })

        # 2. Session-grouped verified turns (chronologically ordered by session updated_at)
        sessions = conn.execute(
            "SELECT session_id, created_at FROM conversations ORDER BY created_at ASC"
        ).fetchall()
        for s in sessions:
            sid = s["session_id"]
            msgs = conn.execute(
                "SELECT role, content, provenance, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (sid,),
            ).fetchall()
            for i in range(len(msgs) - 1):
                if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                    if msgs[i + 1]["provenance"] in ("security", "confirmation"):
                        continue
                    u = msgs[i]["content"].strip()
                    a = msgs[i + 1]["content"].strip()
                    has_s1, _ = SecretRedactor.contains_secret(u)
                    has_s2, _ = SecretRedactor.contains_secret(a)
                    if has_s1 or has_s2 or len(u) < 3 or len(a) < 5:
                        continue
                    records.append({
                        "timestamp": msgs[i + 1]["created_at"],
                        "session_id": sid,
                        "source": f"conversation:{msgs[i + 1]['provenance']}",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_FA},
                            {"role": "user", "content": u},
                            {"role": "assistant", "content": a},
                        ],
                    })

    records.sort(key=lambda x: x.get("timestamp", ""))
    return records


def build_splits(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_chronological_records()
    n = len(records)

    if n < 3:
        train, val, test = records, [], []
    else:
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        train = records[:n_train]
        val = records[n_train : n_train + n_val]
        test = records[n_train + n_val :]

    for split_name, items in (("train", train), ("validation", val), ("test", test)):
        fp = output_dir / f"{split_name}.jsonl"
        with fp.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "split_method": "strict_chronological_no_random_leakage",
        "secrets_filtered": True,
        "total_records": n,
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "output_dir": str(output_dir),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare chronological JSONL dataset for LUNA AI.")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "datasets"))
    args = parser.parse_args()
    summary = build_splits(Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
