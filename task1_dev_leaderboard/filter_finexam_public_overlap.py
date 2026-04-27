#!/usr/bin/env python3
"""Filter normalized FinExam data against the public Task 1 training collection."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
TRAIN_JSONL = APP_ROOT.parent / "task1_training" / "artifacts" / "public_task1_train.jsonl"
IN_DIR = APP_ROOT / "data" / "finexam" / "normalized"
OUT_DIR = APP_ROOT / "data" / "finexam" / "filtered"

PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*Options:", re.DOTALL | re.IGNORECASE)
OPTION_RE = re.compile(r"(?im)^\s*([A-F])\.\s*(.*?)\s*$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def public_key(row: dict) -> tuple[str, str]:
    prompt = str(row.get("prompt", ""))
    question_match = QUESTION_RE.search(prompt)
    question = question_match.group(1).strip() if question_match else ""

    options: dict[str, str] = {}
    for match in OPTION_RE.finditer(prompt):
        letter = match.group(1).upper()
        value = match.group(2)
        value = re.split(r"\s+(?:الإجابة|Answer)\s*:", value, flags=re.IGNORECASE)[0]
        options[letter] = value.strip()
    return (normalize(question), " || ".join(normalize(options[key]) for key in sorted(options) if normalize(options[key])))


def finexam_key(row: dict) -> tuple[str, str]:
    options = []
    for letter in "abcdef":
        value = row.get(f"choice_{letter}", "")
        if pd.notna(value) and normalize(value):
            options.append(normalize(value))
    return (normalize(row.get("question", "")), " || ".join(options))


def public_keys() -> set[tuple[str, str]]:
    keys = set()
    with TRAIN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                keys.add(public_key(json.loads(line)))
    return keys


def main() -> None:
    keys = public_keys()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"public_key_count": len(keys), "languages": {}}

    for language, filename in [("english", "english_single_choice.parquet"), ("chinese", "chinese_single_choice.parquet")]:
        frame = pd.read_parquet(IN_DIR / filename)
        frame["overlap_public"] = frame.apply(lambda row: finexam_key(row) in keys, axis=1)
        clean = frame.loc[~frame["overlap_public"]].drop(columns=["overlap_public"]).reset_index(drop=True)
        clean.to_parquet(OUT_DIR / filename, index=False)
        summary["languages"][language] = {
            "input_rows": int(len(frame)),
            "public_overlap_rows": int(frame["overlap_public"].sum()),
            "clean_rows": int(len(clean)),
            "overlap_ids": frame.loc[frame["overlap_public"], "id"].astype(str).tolist(),
        }

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
