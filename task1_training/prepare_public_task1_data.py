#!/usr/bin/env python3
"""Prepare public Task 1 training data from the FinMMEval HF collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "public_sources"
DEFAULT_OUT = ROOT / "artifacts" / "public_task1_train.jsonl"
DEFAULT_STATS = ROOT / "artifacts" / "public_task1_stats.json"

DATASET_SPECS = {
    "arabic_accounting": "SahmBenchmark/arabic-accounting-mcq_train",
    "arabic_business": "SahmBenchmark/arabic-business-mcq_training_standardized",
    "bhashabench": "bharatgenai/BhashaBench-Finance",
    "cfa_cpa": "Tomas08119993/finmmeval-cfa-cpa",
    "flare_es": "TheFinAI/flare-es-multifin",
    "plutus": "TheFinAI/plutus-multifin",
}

LETTER_RE = re.compile(r"^[A-Fa-f]$")
QUESTION_PATTERNS = {
    "en": re.compile(r"Question:\s*(.*?)\s*Options:\s*", re.IGNORECASE | re.DOTALL),
    "cn": re.compile(r"(?:问题|题目)：\s*(.*?)\s*(?:选项)：\s*", re.DOTALL),
}
OPTION_BLOCK_PATTERNS = {
    "en": re.compile(r"Options:\s*(.*?)\s*Answer:\s*$", re.IGNORECASE | re.DOTALL),
    "cn": re.compile(r"选项：\s*(.*?)\s*(?:答案|答)：\s*$", re.DOTALL),
}
OPTION_MARKER = re.compile(r"(?i)(?<![A-Za-z0-9])([A-F])\s*[\.\):：]\s*")

LANGUAGE_PROMPTS = {
    "arabic": "اقرأ السؤال التالي واختر الإجابة الصحيحة فقط. أجب بحرف الخيار فقط.",
    "hindi": "निम्नलिखित प्रश्न पढ़ें और सही उत्तर चुनें। केवल विकल्प का अक्षर लिखें।",
    "english": "Read the following multiple-choice question and return only the correct option letter.",
    "chinese": "请阅读下面的选择题，并只返回正确选项对应的字母。",
    "spanish": "Lee la siguiente pregunta de opción múltiple y devuelve solo la letra de la respuesta correcta.",
    "greek": "Διάβασε την παρακάτω ερώτηση πολλαπλής επιλογής και επέστρεψε μόνο το γράμμα της σωστής απάντησης.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a unified public Task 1 training file.")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--stats-file", default=str(DEFAULT_STATS))
    parser.add_argument("--validation-mod", type=int, default=20, help="Examples with hash %% mod == 0 go to validation.")
    parser.add_argument("--max-bhasha-per-language", type=int, default=4000, help="Optional cap for BhashaBench per language.")
    return parser.parse_args()


def ensure_repo_snapshot(local_name: str, repo_id: str, cache_dir: Path) -> Path:
    target = cache_dir / local_name
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target), allow_patterns=["*.parquet", "README.md"])
    return target


def load_parquets(repo_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted(repo_dir.rglob("*.parquet"))]
    if not parts:
        raise FileNotFoundError(f"No parquet files found under {repo_dir}")
    return pd.concat(parts, ignore_index=True)


def deterministic_split(source: str, row_id: str, validation_mod: int) -> str:
    digest = hashlib.sha1(f"{source}:{row_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % validation_mod
    return "validation" if bucket == 0 else "train"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_choices(options: dict[str, str]) -> int:
    return sum(1 for value in options.values() if value)


def format_prompt(language: str, question: str, options: dict[str, str]) -> str:
    header = LANGUAGE_PROMPTS[language]
    option_lines = [f"{letter}. {text}" for letter, text in options.items() if text]
    return f"{header}\n\nQuestion:\n{question}\n\nOptions:\n" + "\n".join(option_lines) + "\n\nAnswer:"


def parse_query_question(query: str, lang_code: str, fallback: str) -> str:
    text = str(query or "")
    match = QUESTION_PATTERNS[lang_code].search(text)
    if match:
        return match.group(1).strip()
    return str(fallback or "").strip()


def parse_query_options(query: str, lang_code: str) -> dict[str, str]:
    text = str(query or "")
    block_match = OPTION_BLOCK_PATTERNS[lang_code].search(text)
    block = block_match.group(1).strip() if block_match else text
    matches = list(OPTION_MARKER.finditer(block))
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        value = re.sub(r"\s+", " ", block[start:end]).strip(" \n\r\t;")
        if value:
            options[letter] = value
    return options


def normalize_standard_mcq(
    *,
    source: str,
    language: str,
    row_id: str,
    question: str,
    options: dict[str, str],
    answer: str,
    validation_mod: int,
) -> dict | None:
    answer = str(answer).strip().upper()
    options = {k.upper(): str(v).strip() for k, v in options.items() if str(v).strip()}
    if not question or not LETTER_RE.fullmatch(answer):
        return None
    if answer not in options or count_choices(options) < 2:
        return None
    prompt = format_prompt(language, question, options)
    return {
        "id": f"{source}:{row_id}",
        "source": source,
        "language": language,
        "prompt": prompt,
        "answer": answer,
        "num_choices": len(options),
        "split": deterministic_split(source, str(row_id), validation_mod),
    }


def collect_arabic_accounting(df: pd.DataFrame, validation_mod: int) -> list[dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        options = parse_query_options(row.get("query", ""), "en")
        if not options:
            options = {chr(ord("A") + i): "" for i in range(len(list(row.get("choices", []))))}
        query_options = parse_query_options(row.get("query", ""), "en")
        options = query_options if query_options else parse_query_options(row.get("query", ""), "cn")
        if not options:
            block = str(row.get("query", ""))
            options = {}
            for letter in row.get("choices", []):
                match = re.search(rf"(?im)^{re.escape(str(letter))}\.\s*(.+)$", block)
                if match:
                    options[str(letter).upper()] = match.group(1).strip()
        item = normalize_standard_mcq(
            source="arabic_accounting_train",
            language="arabic",
            row_id=str(row["id"]),
            question=str(row.get("text", "")).strip(),
            options=options,
            answer=row.get("answer", ""),
            validation_mod=validation_mod,
        )
        if item:
            rows.append(item)
    return rows


def collect_arabic_business(df: pd.DataFrame, validation_mod: int) -> list[dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        options = {}
        query = str(row.get("query", ""))
        for letter in row.get("choices", []):
            match = re.search(rf"(?im)^{re.escape(str(letter))}\.\s*(.+)$", query)
            if match:
                options[str(letter).upper()] = match.group(1).strip()
        item = normalize_standard_mcq(
            source="arabic_business_train",
            language="arabic",
            row_id=str(row["id"]),
            question=str(row.get("text", "")).strip(),
            options=options,
            answer=row.get("answer", ""),
            validation_mod=validation_mod,
        )
        if item:
            rows.append(item)
    return rows


def collect_bhashabench(df: pd.DataFrame, validation_mod: int, max_per_language: int) -> list[dict]:
    rows = []
    kept = {"en": 0, "hi": 0}
    for row in df.to_dict(orient="records"):
        language_code = str(row.get("language", "")).lower()
        if language_code not in {"en", "hi"}:
            continue
        if kept[language_code] >= max_per_language:
            continue
        language = "english" if language_code == "en" else "hindi"
        options = {
            "A": row.get("option_a", ""),
            "B": row.get("option_b", ""),
            "C": row.get("option_c", ""),
            "D": row.get("option_d", ""),
        }
        item = normalize_standard_mcq(
            source=f"bhashabench_{language_code}",
            language=language,
            row_id=str(row["id"]),
            question=str(row.get("question", "")).strip(),
            options=options,
            answer=row.get("correct_answer", ""),
            validation_mod=validation_mod,
        )
        if item:
            rows.append(item)
            kept[language_code] += 1
    return rows


def collect_cfa_cpa(df: pd.DataFrame, validation_mod: int) -> list[dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        answer = str(row.get("answer", "")).strip().upper()
        if not LETTER_RE.fullmatch(answer):
            continue
        source_sheet = str(row.get("source_sheet", "")).strip().lower()
        if source_sheet == "english":
            lang_code = "en"
            language = "english"
        elif source_sheet == "chinese":
            lang_code = "cn"
            language = "chinese"
        else:
            continue
        options = parse_query_options(row.get("query", ""), lang_code)
        item = normalize_standard_mcq(
            source=f"cfa_cpa_{language}",
            language=language,
            row_id=str(row["id"]),
            question=parse_query_question(row.get("query", ""), lang_code, row.get("text", "")),
            options=options,
            answer=answer,
            validation_mod=validation_mod,
        )
        if item:
            rows.append(item)
    return rows


def collect_classification(df: pd.DataFrame, validation_mod: int, source: str, language: str) -> list[dict]:
    rows = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        choices = [str(choice).strip() for choice in row.get("choices", []) if str(choice).strip()]
        if len(choices) < 2:
            continue
        gold = row.get("gold")
        if gold is None:
            continue
        gold_index = int(gold)
        if gold_index < 0 or gold_index >= len(choices):
            continue
        options = {chr(ord("A") + i): choice for i, choice in enumerate(choices)}
        answer = chr(ord("A") + gold_index)
        item = normalize_standard_mcq(
            source=source,
            language=language,
            row_id=str(row.get("id", idx)),
            question=str(row.get("text", "")).strip(),
            options=options,
            answer=answer,
            validation_mod=validation_mod,
        )
        if item:
            rows.append(item)
    return rows


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    out_file = Path(args.out_file).resolve()
    stats_file = Path(args.stats_file).resolve()

    dataset_dirs = {name: ensure_repo_snapshot(name, repo, cache_dir) for name, repo in DATASET_SPECS.items()}

    normalized_rows: list[dict] = []
    normalized_rows.extend(collect_arabic_accounting(load_parquets(dataset_dirs["arabic_accounting"]), args.validation_mod))
    normalized_rows.extend(collect_arabic_business(load_parquets(dataset_dirs["arabic_business"]), args.validation_mod))
    normalized_rows.extend(collect_bhashabench(load_parquets(dataset_dirs["bhashabench"]), args.validation_mod, args.max_bhasha_per_language))
    normalized_rows.extend(collect_cfa_cpa(load_parquets(dataset_dirs["cfa_cpa"]), args.validation_mod))
    normalized_rows.extend(
        collect_classification(load_parquets(dataset_dirs["flare_es"]), args.validation_mod, "flare_es", "spanish")
    )
    normalized_rows.extend(
        collect_classification(
            load_parquets(dataset_dirs["plutus"]),
            args.validation_mod,
            "plutus",
            "greek",
        )
    )

    normalized_rows.sort(key=lambda row: (row["language"], row["source"], row["id"]))
    write_jsonl(out_file, normalized_rows)

    stats: dict[str, dict] = {}
    for row in normalized_rows:
        bucket = stats.setdefault(
            row["language"],
            {"total": 0, "train": 0, "validation": 0, "sources": {}, "choice_counts": {}},
        )
        bucket["total"] += 1
        bucket[row["split"]] += 1
        bucket["sources"][row["source"]] = bucket["sources"].get(row["source"], 0) + 1
        choice_key = str(row["num_choices"])
        bucket["choice_counts"][choice_key] = bucket["choice_counts"].get(choice_key, 0) + 1

    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"out_file={out_file}")
    print(f"stats_file={stats_file}")


if __name__ == "__main__":
    main()
