#!/usr/bin/env python3
"""Normalize the private FinExam English/Chinese data for Task 1 splits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = APP_ROOT / "data" / "finexam" / "data"
DEFAULT_OUTPUT_DIR = APP_ROOT / "data" / "finexam" / "normalized"

QUESTION_PATTERNS = {
    "en": re.compile(r"Question:\s*(.*?)\s*Options:\s*", re.IGNORECASE | re.DOTALL),
    "cn": re.compile(r"(?:问|问题|题目)：\s*(.*?)\s*选项：\s*", re.DOTALL),
}
OPTIONS_PATTERNS = {
    "en": re.compile(r"Options:\s*(.*?)\s*Answer:\s*$", re.IGNORECASE | re.DOTALL),
    "cn": re.compile(r"选项：\s*(.*?)\s*(?:答案|回答|答)：\s*$", re.DOTALL),
}
OPTION_MARKER = re.compile(r"(?i)(?<![A-Za-z0-9])([A-F])\s*[\.\):：]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize the private FinExam dataset.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory with raw parquet files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for normalized parquet files.")
    return parser.parse_args()


def load_language_frame(input_dir: Path, language_code: str) -> pd.DataFrame:
    parts = []
    for path in sorted(input_dir.glob(f"*-{language_code}.parquet")):
        frame = pd.read_parquet(path)
        frame["_source_file"] = path.name
        parts.append(frame)
    if not parts:
        raise FileNotFoundError(f"No parquet files found for language code: {language_code}")
    return pd.concat(parts, ignore_index=True)


def parse_question(query: str, language_code: str, fallback: str) -> str:
    text = str(query or "")
    match = QUESTION_PATTERNS[language_code].search(text)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return re.sub(r"\s+", " ", str(fallback or "")).strip()


def parse_options(query: str, language_code: str) -> dict[str, str]:
    text = str(query or "")
    block_match = OPTIONS_PATTERNS[language_code].search(text)
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


def normalize_language(df: pd.DataFrame, language_code: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    stats = {
        "input_rows": int(len(df)),
        "kept_rows": 0,
        "filtered_non_single_answer": 0,
        "filtered_missing_question": 0,
        "filtered_bad_options": 0,
        "filtered_answer_not_in_options": 0,
    }

    for row in df.to_dict(orient="records"):
        answer = str(row.get("answer", "")).strip().upper()
        if not re.fullmatch(r"[A-F]", answer):
            stats["filtered_non_single_answer"] += 1
            continue

        question = parse_question(row.get("query", ""), language_code, row.get("text", ""))
        if not question:
            stats["filtered_missing_question"] += 1
            continue

        options = parse_options(row.get("query", ""), language_code)
        if len(options) < 2:
            stats["filtered_bad_options"] += 1
            continue
        if answer not in options:
            stats["filtered_answer_not_in_options"] += 1
            continue

        normalized = {
            "id": row["id"],
            "question": question,
            "correct_answer": answer,
            "source_sheet": row.get("source_sheet", ""),
            "source_file": row.get("_source_file", ""),
        }
        for letter in ["A", "B", "C", "D", "E", "F"]:
            normalized[f"choice_{letter.lower()}"] = options.get(letter, "")
        rows.append(normalized)

    normalized_df = pd.DataFrame(rows)
    stats["kept_rows"] = int(len(normalized_df))
    if not normalized_df.empty:
        stats["answer_counts"] = normalized_df["correct_answer"].value_counts().to_dict()
        option_cols = [f"choice_{letter.lower()}" for letter in ["A", "B", "C", "D", "E", "F"]]
        stats["num_choice_counts"] = (
            normalized_df[option_cols]
            .apply(lambda series: int(sum(bool(str(value).strip()) for value in series)), axis=1)
            .value_counts()
            .to_dict()
        )
    return normalized_df, stats


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for language_code, output_name in [("en", "english_single_choice.parquet"), ("cn", "chinese_single_choice.parquet")]:
        frame = load_language_frame(input_dir, language_code)
        normalized_df, stats = normalize_language(frame, language_code)
        output_path = output_dir / output_name
        normalized_df.to_parquet(output_path, index=False)
        summary[language_code] = {"output_path": str(output_path), **stats}

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
