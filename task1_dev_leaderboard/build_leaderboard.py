#!/usr/bin/env python3
"""Build a local dev leaderboard for FinMMEval Task 1.

The tool is intentionally lightweight:
- standard-library only
- dataset-driven via a JSON config
- supports multiple MCQ datasets and multiple prediction files

Current built-in formats:
- Dataset: `mcq_jsonl`
- Predictions: `per_item_mcq_csv`, `task1_jsonl`
"""

from __future__ import annotations

import argparse
import csv
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ALL_VALID_LETTERS = set(string.ascii_uppercase)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_parquet_records(path: Path) -> List[dict]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet datasets requires pandas. Please install pandas first."
        ) from exc
    return pd.read_parquet(path).to_dict(orient="records")


def resolve_path(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def normalize_letter(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in ALL_VALID_LETTERS:
        return text
    return ""


def load_dataset(dataset_cfg: dict, config_dir: Path) -> List[dict]:
    dataset_format = dataset_cfg.get("format", "mcq_jsonl")
    path = resolve_path(config_dir, dataset_cfg["path"])
    if dataset_format == "mcq_jsonl":
        rows = load_jsonl(path)
    elif dataset_format == "mcq_parquet":
        rows = load_parquet_records(path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    id_field = dataset_cfg.get("id_field", "id")
    question_field = dataset_cfg.get("question_field", "question")
    answer_field = dataset_cfg.get("answer_field", "answer")
    option_fields = dataset_cfg.get(
        "option_fields",
        {"A": "option_A", "B": "option_B", "C": "option_C", "D": "option_D"},
    )
    valid_letters = set(option_fields.keys())
    category_field = dataset_cfg.get("category_field", "category")
    language = dataset_cfg.get("language", "")
    source = dataset_cfg.get("source", dataset_cfg["key"])

    normalized_rows: List[dict] = []
    for row in rows:
        item_id = str(row[id_field])
        options = {letter: row.get(field, "") for letter, field in option_fields.items()}
        gold = normalize_letter(row.get(answer_field))
        if gold not in valid_letters:
            raise ValueError(
                f"Dataset {dataset_cfg['key']} item {item_id} has invalid answer: {row.get(answer_field)!r}"
            )
        normalized = {
            "dataset_key": dataset_cfg["key"],
            "dataset_name": dataset_cfg.get("name", dataset_cfg["key"]),
            "id": item_id,
            "question": row.get(question_field, ""),
            "gold_letter": gold,
            "category": row.get(category_field, ""),
            "language": language,
            "source": source,
        }
        for letter, text in options.items():
            normalized[f"option_{letter}"] = text
        normalized_rows.append(
            normalized
        )
    return normalized_rows


def extract_prediction_letter(record: dict, field_candidates: Iterable[str]) -> str:
    for field in field_candidates:
        if field in record:
            letter = normalize_letter(record[field])
            if letter:
                return letter
    return ""


def load_predictions(run_cfg: dict, config_dir: Path) -> Dict[str, dict]:
    run_format = run_cfg.get("format", "task1_jsonl")
    path = resolve_path(config_dir, run_cfg["path"])
    if run_format == "per_item_mcq_csv":
        id_field = run_cfg.get("id_field", "id")
        pred_field = run_cfg.get("prediction_field", "pred_letter")
        raw_field = run_cfg.get("raw_output_field", "raw_output")
        predictions: Dict[str, dict] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = str(row[id_field])
                predictions[item_id] = {
                    "pred_letter": normalize_letter(row.get(pred_field)),
                    "raw_output": row.get(raw_field, ""),
                }
        return predictions

    if run_format == "task1_jsonl":
        id_field = run_cfg.get("id_field", "id")
        field_candidates = run_cfg.get(
            "prediction_field_candidates",
            ["prediction", "pred_letter", "answer", "answer_label", "label"],
        )
        raw_field = run_cfg.get("raw_output_field", "raw_output")
        predictions = {}
        for row in load_jsonl(path):
            item_id = str(row[id_field])
            predictions[item_id] = {
                "pred_letter": extract_prediction_letter(row, field_candidates),
                "raw_output": row.get(raw_field, ""),
            }
        return predictions

    raise ValueError(f"Unsupported prediction format: {run_format}")


def safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarise_group(rows: List[dict]) -> dict:
    total = len(rows)
    answered = sum(1 for row in rows if row["pred_letter"])
    correct = sum(1 for row in rows if row["is_correct"])
    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "coverage": safe_div(answered, total),
        "accuracy": safe_div(correct, total),
        "answered_accuracy": safe_div(correct, answered),
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_table(rows: List[dict], columns: List[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, separator] + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Task 1 dev leaderboard.")
    parser.add_argument("--config", required=True, help="Path to leaderboard config JSON.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <config_dir>/outputs/<config_stem>/",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_json(config_path)

    datasets_cfg = config.get("datasets", [])
    runs_cfg = config.get("runs", [])
    if not datasets_cfg:
        raise ValueError("Config must define at least one dataset.")
    if not runs_cfg:
        raise ValueError("Config must define at least one run.")

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (config_dir.parent / "outputs" / config_path.stem).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, List[dict]] = {}
    dataset_row_maps: Dict[str, Dict[str, dict]] = {}
    dataset_stats_rows: List[dict] = []
    dataset_keys: List[str] = []

    for dataset_cfg in datasets_cfg:
        dataset_key = dataset_cfg["key"]
        dataset_keys.append(dataset_key)
        rows = load_dataset(dataset_cfg, config_dir)
        datasets[dataset_key] = rows
        dataset_row_maps[dataset_key] = {row["id"]: row for row in rows}
        categories = sorted({row["category"] for row in rows if row["category"]})
        dataset_stats_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_cfg.get("name", dataset_key),
                "language": dataset_cfg.get("language", ""),
                "items": len(rows),
                "categories": len(categories),
                "source": dataset_cfg.get("source", dataset_key),
            }
        )

    option_columns = sorted(
        {
            key
            for dataset_rows in datasets.values()
            for row in dataset_rows
            for key in row.keys()
            if key.startswith("option_")
        }
    )

    per_item_rows: List[dict] = []
    leaderboard_rows: List[dict] = []
    by_dataset_rows: List[dict] = []
    by_category_rows: List[dict] = []

    for run_cfg in runs_cfg:
        model_name = run_cfg["model_name"]
        prediction_map = load_predictions(run_cfg, config_dir)
        requested_datasets = run_cfg.get("datasets", dataset_keys)
        model_eval_rows: List[dict] = []

        for dataset_key in requested_datasets:
            if dataset_key not in datasets:
                raise ValueError(f"Run {model_name} references unknown dataset: {dataset_key}")

            for row in datasets[dataset_key]:
                pred_info = prediction_map.get(row["id"], {})
                pred_letter = normalize_letter(pred_info.get("pred_letter"))
                is_correct = int(pred_letter != "" and pred_letter == row["gold_letter"])
                eval_row = {
                    "model_name": model_name,
                    **row,
                    "pred_letter": pred_letter,
                    "is_correct": is_correct,
                    "answered": int(bool(pred_letter)),
                    "raw_output": pred_info.get("raw_output", ""),
                }
                per_item_rows.append(eval_row)
                model_eval_rows.append(eval_row)

        overall = summarise_group(model_eval_rows)
        leaderboard_row = {
            "rank": 0,
            "model_name": model_name,
            "total": overall["total"],
            "answered": overall["answered"],
            "correct": overall["correct"],
            "coverage": round(overall["coverage"], 6),
            "accuracy": round(overall["accuracy"], 6),
            "answered_accuracy": round(overall["answered_accuracy"], 6),
        }

        grouped_by_dataset: Dict[str, List[dict]] = defaultdict(list)
        grouped_by_category: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for row in model_eval_rows:
            grouped_by_dataset[row["dataset_key"]].append(row)
            if row["category"]:
                grouped_by_category[(row["dataset_key"], row["category"])].append(row)

        for dataset_key in dataset_keys:
            summary = summarise_group(grouped_by_dataset[dataset_key])
            leaderboard_row[f"{dataset_key}__accuracy"] = round(summary["accuracy"], 6)
            leaderboard_row[f"{dataset_key}__coverage"] = round(summary["coverage"], 6)
            by_dataset_rows.append(
                {
                    "model_name": model_name,
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_row_maps[dataset_key][next(iter(dataset_row_maps[dataset_key]))]["dataset_name"],
                    "language": datasets[dataset_key][0]["language"] if datasets[dataset_key] else "",
                    "total": summary["total"],
                    "answered": summary["answered"],
                    "correct": summary["correct"],
                    "coverage": round(summary["coverage"], 6),
                    "accuracy": round(summary["accuracy"], 6),
                    "answered_accuracy": round(summary["answered_accuracy"], 6),
                }
            )

        for (dataset_key, category), rows in sorted(grouped_by_category.items()):
            summary = summarise_group(rows)
            by_category_rows.append(
                {
                    "model_name": model_name,
                    "dataset_key": dataset_key,
                    "category": category,
                    "total": summary["total"],
                    "answered": summary["answered"],
                    "correct": summary["correct"],
                    "coverage": round(summary["coverage"], 6),
                    "accuracy": round(summary["accuracy"], 6),
                    "answered_accuracy": round(summary["answered_accuracy"], 6),
                }
            )

        leaderboard_rows.append(leaderboard_row)

    leaderboard_rows.sort(
        key=lambda row: (-row["accuracy"], -row["correct"], row["model_name"].lower())
    )
    for rank, row in enumerate(leaderboard_rows, start=1):
        row["rank"] = rank

    per_item_fieldnames = [
        "model_name",
        "dataset_key",
        "dataset_name",
        "source",
        "id",
        "language",
        "category",
        "gold_letter",
        "pred_letter",
        "is_correct",
        "answered",
        "question",
    ] + option_columns + [
        "raw_output",
    ]
    leaderboard_fieldnames = [
        "rank",
        "model_name",
        "accuracy",
        "correct",
        "total",
        "coverage",
        "answered_accuracy",
        "answered",
    ] + [
        f"{dataset_key}__accuracy" for dataset_key in dataset_keys
    ] + [
        f"{dataset_key}__coverage" for dataset_key in dataset_keys
    ]

    write_csv(out_dir / "dataset_stats.csv", dataset_stats_rows, [
        "dataset_key", "dataset_name", "language", "items", "categories", "source"
    ])
    write_csv(out_dir / "leaderboard_overall.csv", leaderboard_rows, leaderboard_fieldnames)
    write_csv(out_dir / "leaderboard_by_dataset.csv", by_dataset_rows, [
        "model_name",
        "dataset_key",
        "dataset_name",
        "language",
        "accuracy",
        "correct",
        "total",
        "coverage",
        "answered_accuracy",
        "answered",
    ])
    write_csv(out_dir / "leaderboard_by_category.csv", by_category_rows, [
        "model_name",
        "dataset_key",
        "category",
        "accuracy",
        "correct",
        "total",
        "coverage",
        "answered_accuracy",
        "answered",
    ])
    write_csv(out_dir / "per_item_results.csv", per_item_rows, per_item_fieldnames)

    markdown_rows = []
    for row in leaderboard_rows:
        rendered = {
            "rank": row["rank"],
            "model_name": row["model_name"],
            "accuracy": format_pct(row["accuracy"]),
            "correct": f"{row['correct']}/{row['total']}",
            "coverage": format_pct(row["coverage"]),
        }
        for dataset_key in dataset_keys:
            rendered[f"{dataset_key}__accuracy"] = format_pct(row[f"{dataset_key}__accuracy"])
        markdown_rows.append(rendered)

    markdown_columns = [
        ("rank", "Rank"),
        ("model_name", "Model"),
        ("accuracy", "Overall Acc"),
        ("correct", "Correct/Total"),
        ("coverage", "Coverage"),
    ] + [
        (f"{dataset_key}__accuracy", f"{dataset_key} Acc") for dataset_key in dataset_keys
    ]

    readme_text = [
        "# Task 1 Dev Leaderboard",
        "",
        "## Overall",
        "",
        build_markdown_table(markdown_rows, markdown_columns),
        "",
        "## Files",
        "",
        "- `dataset_stats.csv`: dataset inventory used for this run",
        "- `leaderboard_overall.csv`: overall leaderboard with one row per model",
        "- `leaderboard_by_dataset.csv`: per-model, per-dataset breakdown",
        "- `leaderboard_by_category.csv`: per-model, per-category breakdown",
        "- `per_item_results.csv`: merged per-item evaluation records",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_text), encoding="utf-8")

    print(f"Wrote Task 1 dev leaderboard outputs to: {out_dir}")


if __name__ == "__main__":
    main()
