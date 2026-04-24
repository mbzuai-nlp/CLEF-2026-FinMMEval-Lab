#!/usr/bin/env python3
"""Filter Arabic supplemental Task 1 data against the public FinMMEval collection."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = APP_ROOT.parent / "task1_training" / "cache" / "public_sources"
RAW_ARABIC = APP_ROOT / "data" / "accounting_CLEF" / "data" / "train-00000-of-00001.parquet"
OUT_DIR = APP_ROOT / "data" / "accounting_CLEF" / "filtered"
OUT_PARQUET = OUT_DIR / "arabic_nonoverlap.parquet"
OUT_SUMMARY = OUT_DIR / "arabic_nonoverlap_summary.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
OPTION_MARKER = re.compile(r"(?i)(?<![A-Za-z0-9])([A-F])\s*[\.\):：]\s*")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    for src, dst in {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه"}.items():
        text = text.replace(src, dst)
    text = ARABIC_DIACRITICS.sub("", text)
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def pack_options(values: list[str]) -> str:
    normalized = [normalize(value) for value in values if normalize(value)]
    return " || ".join(normalized)


def parse_options(block: str) -> list[str]:
    matches = list(OPTION_MARKER.finditer(block))
    values: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        value = re.sub(r"\s+", " ", block[start:end]).strip(" \n\r\t;")
        if value:
            values.append(value)
    return values


def public_arabic_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for rel in [
        "arabic_accounting/data",
        "arabic_accounting_eval/data",
        "arabic_business/data",
        "arabic_business_eval/data",
    ]:
        for parquet_path in (PUBLIC_ROOT / rel).rglob("*.parquet"):
            frame = pd.read_parquet(parquet_path)
            for row in frame.to_dict(orient="records"):
                text = str(row.get("text", ""))
                question_part = text.split("الخيارات:")[0].replace("السؤال:", "").strip()
                options_part = text.split("الخيارات:")[1] if "الخيارات:" in text else ""
                keys.add((normalize(question_part), pack_options(parse_options(options_part))))
    return keys


def supplemental_key(row: dict) -> tuple[str, str]:
    return (
        normalize(row.get("question", "")),
        pack_options(
            [
                row.get("choice_a", ""),
                row.get("choice_b", ""),
                row.get("choice_c", ""),
                row.get("choice_d", ""),
                row.get("choice_e", ""),
            ]
        ),
    )


def main() -> None:
    public_keys = public_arabic_keys()
    raw_df = pd.read_parquet(RAW_ARABIC)
    raw_df["overlap_public"] = raw_df.apply(lambda row: supplemental_key(row) in public_keys, axis=1)
    clean_df = raw_df.loc[~raw_df["overlap_public"]].drop(columns=["overlap_public"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(OUT_PARQUET, index=False)

    summary = {
        "raw_total": int(len(raw_df)),
        "public_overlap_total": int(raw_df["overlap_public"].sum()),
        "clean_total": int(len(clean_df)),
        "clean_source_counts": clean_df["source"].value_counts().to_dict(),
        "clean_answer_counts": clean_df["correct_answer"].astype(str).str.upper().value_counts().to_dict(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"out_parquet={OUT_PARQUET}")
    print(f"summary={OUT_SUMMARY}")


if __name__ == "__main__":
    main()
