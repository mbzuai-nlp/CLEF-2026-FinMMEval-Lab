#!/usr/bin/env python3
"""Create an LLM zero-shot baseline submission for Task 1 dev sets."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai


LETTER_RE = re.compile(r"\b([A-Z])\b")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choice_letters(row: dict) -> list[str]:
    options = row.get("options") or {}
    if not isinstance(options, dict) or not options:
        raise ValueError(f"Row {row.get('id', '<unknown>')} does not define options.")
    letters = [str(key).strip().upper() for key in options.keys() if str(key).strip()]
    filtered = sorted(set(letter for letter in letters if len(letter) == 1 and letter.isalpha()))
    if not filtered:
        raise ValueError(f"Row {row.get('id', '<unknown>')} has invalid option keys.")
    return filtered


def build_prompt(row: dict) -> str:
    options = row["options"]
    option_lines = [f"{letter}. {options[letter]}" for letter in choice_letters(row)]
    return (
        "Select the single best answer to this multiple-choice question.\n"
        "Return only one option letter and nothing else.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )


def parse_prediction(text: str, letters: list[str]) -> str:
    cleaned = text.strip().upper()
    if cleaned in letters:
        return cleaned
    match = LETTER_RE.search(cleaned)
    if match and match.group(1) in letters:
        return match.group(1)
    raise ValueError(f"Could not parse a valid option from model output: {text!r}")


def infer_one(
    client: openai.OpenAI,
    model: str,
    row: dict,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    letters = choice_letters(row)
    prompt = build_prompt(row)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful multilingual finance exam solver. Reply with exactly one valid option letter.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            prediction = parse_prediction(content, letters)
            return {"id": row["id"], "prediction": prediction}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(f"Failed to infer item {row['id']}: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an LLM zero-shot baseline submission.")
    parser.add_argument("--devset", required=True, help="Public dev set JSONL file.")
    parser.add_argument("--output", required=True, help="Output submission JSONL file.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent request workers.")
    parser.add_argument("--max-retries", type=int, default=3, help="Per-item retry count.")
    parser.add_argument("--retry-delay", type=float, default=1.5, help="Base retry delay in seconds.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    devset_path = Path(args.devset).resolve()
    output_path = Path(args.output).resolve()
    rows = load_jsonl(devset_path)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    client = openai.OpenAI(api_key=api_key)
    predictions_by_id: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                infer_one,
                client,
                args.model,
                row,
                args.max_retries,
                args.retry_delay,
            ): row["id"]
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            predictions_by_id[result["id"]] = result

    predictions = [predictions_by_id[row["id"]] for row in rows]
    write_jsonl(output_path, predictions)
    print(f"Wrote {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
