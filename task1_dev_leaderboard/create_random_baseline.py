#!/usr/bin/env python3
"""Create deterministic random-answer baselines for Task 1 dev sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


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


def choice_letters(row: dict) -> list[str]:
    options = row.get("options")
    if isinstance(options, dict) and options:
        letters = [str(key).strip().upper() for key in options.keys() if str(key).strip()]
        return sorted(set(letter for letter in letters if len(letter) == 1 and letter.isalpha()))

    count = int(row.get("num_choices", 0) or 0)
    if count > 0:
        return [chr(ord("A") + index) for index in range(count)]

    raise ValueError(f"Row {row.get('id', '<unknown>')} does not define answer choices.")


def build_predictions(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    predictions: list[dict] = []
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            raise ValueError("Every row must contain a non-empty id.")
        letters = choice_letters(row)
        predictions.append({"id": item_id, "prediction": rng.choice(letters)})
    return predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic random baseline submission.")
    parser.add_argument("--devset", required=True, help="Public dev set JSONL file.")
    parser.add_argument("--output", required=True, help="Output submission JSONL file.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    devset_path = Path(args.devset).resolve()
    output_path = Path(args.output).resolve()

    rows = load_jsonl(devset_path)
    predictions = build_predictions(rows, seed=args.seed)
    write_jsonl(output_path, predictions)
    print(f"Wrote {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
