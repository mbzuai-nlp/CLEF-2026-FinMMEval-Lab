#!/usr/bin/env python3
"""Create a conservative de-duplicated Arabic Task 1 pool.

The input is the Arabic supplemental pool after public-overlap filtering.
We de-duplicate at question level rather than exact row level so a public dev
item cannot expose the stem of a hidden item with slightly different choices.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
IN_PARQUET = APP_ROOT / "data" / "accounting_CLEF" / "filtered" / "arabic_nonoverlap.parquet"
OUT_PARQUET = APP_ROOT / "data" / "accounting_CLEF" / "filtered" / "arabic_clean.parquet"
OUT_SUMMARY = APP_ROOT / "data" / "accounting_CLEF" / "filtered" / "arabic_clean_summary.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    for src, dst in {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه"}.items():
        text = text.replace(src, dst)
    text = ARABIC_DIACRITICS.sub("", text)
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def option_key(row: dict) -> tuple[str, ...]:
    options = []
    for letter in ["a", "b", "c", "d", "e"]:
        value = normalize(row.get(f"choice_{letter}", ""))
        if value:
            options.append(value)
    return tuple(options)


def main() -> None:
    df = pd.read_parquet(IN_PARQUET).reset_index(drop=True)
    rows = df.to_dict(orient="records")
    question_keys = [normalize(row.get("question", "")) for row in rows]
    full_keys = [(question_key, option_key(row)) for question_key, row in zip(question_keys, rows)]

    clean_df = df.copy()
    clean_df["_question_key"] = question_keys
    clean_df = clean_df.drop_duplicates(subset=["_question_key"], keep="first").drop(columns=["_question_key"])
    clean_df = clean_df.reset_index(drop=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(OUT_PARQUET, index=False)

    summary = {
        "input_total": int(len(df)),
        "exact_duplicate_extra_rows": int(len(full_keys) - len(set(full_keys))),
        "question_duplicate_extra_rows": int(len(question_keys) - len(set(question_keys))),
        "clean_total": int(len(clean_df)),
        "clean_source_counts": clean_df["source"].value_counts().to_dict(),
        "clean_answer_counts": clean_df["correct_answer"].astype(str).str.upper().value_counts().to_dict(),
        "dedup_policy": "drop duplicate normalized question stems, keep first occurrence",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"out_parquet={OUT_PARQUET}")
    print(f"summary={OUT_SUMMARY}")


if __name__ == "__main__":
    main()
