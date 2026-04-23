#!/usr/bin/env python3
"""Create public dev and private hidden-test splits for Task 1 datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {fmt}")


def largest_remainder_allocation(counts: dict[tuple, int], total_target: int) -> dict[tuple, int]:
    total = sum(counts.values())
    allocations: dict[tuple, int] = {}
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


def sample_dev_rows(df: pd.DataFrame, stratify_fields: list[str], answer_field: str, target_size: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["_row_idx"] = df.index
    df[answer_field] = df[answer_field].astype(str).str.upper()
    group_fields = stratify_fields + [answer_field]
    counts = df.groupby(group_fields).size().to_dict()
    allocations = largest_remainder_allocation(counts, target_size)

    sampled_parts = []
    for key, n in sorted(allocations.items()):
        if n == 0:
            continue
        mask = pd.Series(True, index=df.index)
        for field, value in zip(group_fields, key):
            mask &= df[field] == value
        subset = df[mask].copy()
        sampled_parts.append(subset.sample(n=n, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def build_options(row: dict, option_fields: dict[str, str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for letter, field in option_fields.items():
        value = row.get(field, "")
        if pd.notna(value) and str(value) != "":
            options[letter] = str(value)
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Task 1 dev and hidden-test splits.")
    parser.add_argument("--config", required=True, help="Split config JSON path.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    input_path = Path(config["input_path"])
    if not input_path.is_absolute():
        input_path = (config_dir / input_path).resolve()

    input_format = config["input_format"]
    dataset_key = config["dataset_key"]
    dev_prefix = config["dev_prefix"]
    test_prefix = config["test_prefix"]
    dev_size = int(config.get("dev_size", 100))
    seed = int(config.get("seed", 2026))
    id_field = config["id_field"]
    question_field = config["question_field"]
    answer_field = config["answer_field"]
    option_fields = config["option_fields"]
    meta_fields = config.get("meta_fields", [])
    stratify_fields = config.get("stratify_fields", [])

    public_out = (config_dir.parent / config["public_out"]).resolve()
    template_out = (config_dir.parent / config["template_out"]).resolve()
    dev_gold_out = (config_dir.parent / config["dev_gold_out"]).resolve()
    hidden_test_out = (config_dir.parent / config["hidden_test_out"]).resolve()

    df = load_table(input_path, input_format)
    dev_df = sample_dev_rows(df, stratify_fields=stratify_fields, answer_field=answer_field, target_size=dev_size, seed=seed)
    hidden_row_indices = set(dev_df["_row_idx"].tolist())
    hidden_df = df.loc[~df.index.isin(hidden_row_indices)].copy().reset_index(drop=True)

    dev_df["split_id"] = [f"{dev_prefix}-{i:03d}" for i in range(1, len(dev_df) + 1)]
    hidden_df["split_id"] = [f"{test_prefix}-{i:03d}" for i in range(1, len(hidden_df) + 1)]

    public_rows = []
    template_rows = []
    dev_gold_rows = []
    hidden_rows = []

    for row in dev_df.to_dict(orient="records"):
        options = build_options(row, option_fields)
        public_rows.append(
            {
                "id": row["split_id"],
                "question": row[question_field],
                "options": options,
                "num_choices": len(options),
            }
        )
        template_rows.append({"id": row["split_id"], "prediction": ""})
        gold_row = {
            "id": row["split_id"],
            "correct_answer": str(row[answer_field]).strip().upper(),
            "original_id": row[id_field],
        }
        for field in meta_fields:
            gold_row[field] = row[field]
        dev_gold_rows.append(gold_row)

    for row in hidden_df.to_dict(orient="records"):
        options = build_options(row, option_fields)
        hidden_row = {
            "id": row["split_id"],
            "question": row[question_field],
            "options": options,
            "num_choices": len(options),
            "correct_answer": str(row[answer_field]).strip().upper(),
            "original_id": row[id_field],
        }
        for field in meta_fields:
            hidden_row[field] = row[field]
        hidden_rows.append(hidden_row)

    write_jsonl(public_out, public_rows)
    public_out.parent.mkdir(parents=True, exist_ok=True)
    template_out.parent.mkdir(parents=True, exist_ok=True)
    with template_out.open("w", encoding="utf-8") as f:
        json.dump(template_rows, f, ensure_ascii=False, indent=2)
    write_jsonl(dev_gold_out, dev_gold_rows)
    write_jsonl(hidden_test_out, hidden_rows)

    summary = {
        "dataset_key": dataset_key,
        "total": int(len(df)),
        "dev_size": int(len(dev_df)),
        "hidden_test_size": int(len(hidden_df)),
        "seed": seed,
    }
    if stratify_fields:
        for field in stratify_fields:
            summary[f"{field}_dev_counts"] = dev_df[field].value_counts().to_dict()
            summary[f"{field}_hidden_counts"] = hidden_df[field].value_counts().to_dict()
    summary["answer_dev_counts"] = dev_df[answer_field].astype(str).str.upper().value_counts().to_dict()
    summary["answer_hidden_counts"] = hidden_df[answer_field].astype(str).str.upper().value_counts().to_dict()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"public_out={public_out}")
    print(f"template_out={template_out}")
    print(f"dev_gold_out={dev_gold_out}")
    print(f"hidden_test_out={hidden_test_out}")


if __name__ == "__main__":
    main()
