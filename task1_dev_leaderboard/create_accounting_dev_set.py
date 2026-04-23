#!/usr/bin/env python3
"""Create a 100-item public dev set from accounting_CLEF.

Sampling policy:
- deterministic
- stratified by (source, correct_answer)
- largest-remainder proportional allocation

Outputs:
- public questions JSONL
- private gold JSONL
- public submission template JSON
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "accounting_CLEF"
    / "data"
    / "train-00000-of-00001.parquet"
)
DEFAULT_PUBLIC = (
    Path(__file__).resolve().parent / "dev_sets" / "accounting_clef_100_public.jsonl"
)
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parent
    / "dev_sets"
    / "accounting_clef_100_submission_template.json"
)
DEFAULT_PRIVATE = (
    Path(__file__).resolve().parent / "private" / "accounting_clef_100_gold.jsonl"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def largest_remainder_allocation(counts: dict[tuple[str, str], int], total_target: int) -> dict:
    total = sum(counts.values())
    allocations = {}
    remainders = []
    assigned = 0
    for key, count in counts.items():
        quota = count * total_target / total
        base = int(quota)
        allocations[key] = base
        assigned += base
        remainders.append((quota - base, count, key))
    remainders.sort(reverse=True)
    for _, _, key in remainders[: total_target - assigned]:
        allocations[key] += 1
    return allocations


def sample_rows(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["correct_answer"] = df["correct_answer"].str.upper()
    counts = df.groupby(["source", "correct_answer"]).size().to_dict()
    allocations = largest_remainder_allocation(counts, target_size)

    sampled_parts = []
    for (source, answer), n in sorted(allocations.items()):
        subset = df[(df["source"] == source) & (df["correct_answer"] == answer)].copy()
        if n == 0:
            continue
        sampled_parts.append(subset.sample(n=n, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled["dev_id"] = [f"acct-dev-{i:03d}" for i in range(1, len(sampled) + 1)]
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 100-item accounting Task 1 dev set.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--public-out", default=str(DEFAULT_PUBLIC))
    parser.add_argument("--private-out", default=str(DEFAULT_PRIVATE))
    parser.add_argument("--template-out", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    public_out = Path(args.public_out).resolve()
    private_out = Path(args.private_out).resolve()
    template_out = Path(args.template_out).resolve()

    df = pd.read_parquet(input_path)
    sampled = sample_rows(df, target_size=args.size, seed=args.seed)

    public_rows = []
    private_rows = []
    template_rows = []
    for row in sampled.to_dict(orient="records"):
        options = {}
        for letter in ["A", "B", "C", "D", "E"]:
            value = row.get(f"choice_{letter.lower()}", "")
            if value:
                options[letter] = value

        public_rows.append(
            {
                "id": row["dev_id"],
                "question": row["question"],
                "options": options,
                "num_choices": len(options),
            }
        )
        private_rows.append(
            {
                "id": row["dev_id"],
                "correct_answer": row["correct_answer"],
                "source": row["source"],
                "original_id": int(row["id"]),
            }
        )
        template_rows.append({"id": row["dev_id"], "prediction": ""})

    write_jsonl(public_out, public_rows)
    write_jsonl(private_out, private_rows)
    template_out.parent.mkdir(parents=True, exist_ok=True)
    with template_out.open("w", encoding="utf-8") as f:
        json.dump(template_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "size": len(sampled),
        "seed": args.seed,
        "source_counts": sampled["source"].value_counts().to_dict(),
        "answer_counts": sampled["correct_answer"].value_counts().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"public_out={public_out}")
    print(f"private_out={private_out}")
    print(f"template_out={template_out}")


if __name__ == "__main__":
    main()
