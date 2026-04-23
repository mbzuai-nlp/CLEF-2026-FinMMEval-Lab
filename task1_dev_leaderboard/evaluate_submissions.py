#!/usr/bin/env python3
"""Evaluate participant JSON submissions for the Task 1 dev leaderboard."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


VALID_LETTERS = {"A", "B", "C", "D", "E"}


def is_submission_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"} and not path.name.startswith("_")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
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


def normalize_letter(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return text if text in VALID_LETTERS else ""


def load_gold(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    rows = load_jsonl(path)
    gold_map = {}
    meta = {}
    for row in rows:
        item_id = str(row["id"])
        gold_map[item_id] = normalize_letter(row["correct_answer"])
        meta[item_id] = row
    return gold_map, meta


def load_submission(path: Path) -> tuple[dict[str, str], dict]:
    suffix = path.suffix.lower()
    duplicate_ids = 0
    raw_predictions = {}

    if suffix == ".jsonl":
        rows = load_jsonl(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and "predictions" in payload and isinstance(payload["predictions"], list):
            rows = payload["predictions"]
        elif isinstance(payload, dict):
            rows = [{"id": key, "prediction": value} for key, value in payload.items()]
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported submission format: {path.name}")

    for row in rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        pred = ""
        for key in ("prediction", "pred_letter", "answer", "label"):
            if key in row:
                pred = normalize_letter(row[key])
                if pred:
                    break
        if item_id in raw_predictions:
            duplicate_ids += 1
        raw_predictions[item_id] = pred

    stats = {"duplicate_ids": duplicate_ids, "rows": len(rows)}
    return raw_predictions, stats


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 1 dev submissions.")
    parser.add_argument("--gold-file", required=True, help="Private gold JSONL path.")
    parser.add_argument("--submissions-dir", required=True, help="Directory of participant JSON/JSONL files.")
    parser.add_argument("--out-dir", required=True, help="Output leaderboard directory.")
    args = parser.parse_args()

    gold_file = Path(args.gold_file).resolve()
    submissions_dir = Path(args.submissions_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_map, gold_meta = load_gold(gold_file)
    expected_ids = set(gold_map.keys())

    summary_rows = []
    per_item_rows = []
    by_source_rows = []

    submission_files = sorted(
        [
            path
            for path in submissions_dir.iterdir()
            if is_submission_file(path)
        ]
    )

    if not submission_files:
        raise ValueError(f"No .json or .jsonl submissions found in {submissions_dir}")

    for path in submission_files:
        model_name = path.stem
        predictions, load_stats = load_submission(path)
        predicted_ids = set(predictions.keys())
        unknown_ids = sorted(predicted_ids - expected_ids)
        missing_ids = sorted(expected_ids - predicted_ids)

        rows = []
        source_groups = defaultdict(list)
        for item_id in sorted(expected_ids):
            pred = predictions.get(item_id, "")
            gold = gold_map[item_id]
            source = gold_meta[item_id].get("source", "")
            is_correct = int(pred != "" and pred == gold)
            row = {
                "model_name": model_name,
                "id": item_id,
                "source": source,
                "gold_answer": gold,
                "pred_answer": pred,
                "answered": int(bool(pred)),
                "is_correct": is_correct,
            }
            rows.append(row)
            per_item_rows.append(row)
            source_groups[source].append(row)

        total = len(rows)
        answered = sum(r["answered"] for r in rows)
        correct = sum(r["is_correct"] for r in rows)
        coverage = safe_div(answered, total)
        accuracy = safe_div(correct, total)
        answered_accuracy = safe_div(correct, answered)

        summary_rows.append(
            {
                "rank": 0,
                "model_name": model_name,
                "submission_file": path.name,
                "accuracy": round(accuracy, 6),
                "correct": correct,
                "total": total,
                "coverage": round(coverage, 6),
                "answered_accuracy": round(answered_accuracy, 6),
                "answered": answered,
                "missing_ids": len(missing_ids),
                "unknown_ids": len(unknown_ids),
                "duplicate_ids": load_stats["duplicate_ids"],
                "valid_submission": int(len(unknown_ids) == 0),
            }
        )

        for source, source_rows in sorted(source_groups.items()):
            source_total = len(source_rows)
            source_answered = sum(r["answered"] for r in source_rows)
            source_correct = sum(r["is_correct"] for r in source_rows)
            by_source_rows.append(
                {
                    "model_name": model_name,
                    "source": source,
                    "accuracy": round(safe_div(source_correct, source_total), 6),
                    "correct": source_correct,
                    "total": source_total,
                    "coverage": round(safe_div(source_answered, source_total), 6),
                    "answered_accuracy": round(safe_div(source_correct, source_answered), 6),
                    "answered": source_answered,
                }
            )

        report = {
            "model_name": model_name,
            "submission_file": path.name,
            "missing_ids": missing_ids,
            "unknown_ids": unknown_ids,
            "duplicate_ids": load_stats["duplicate_ids"],
        }
        with (out_dir / f"{model_name}__validation.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    summary_rows.sort(
        key=lambda row: (-row["accuracy"], -row["correct"], -row["coverage"], row["model_name"].lower())
    )
    for idx, row in enumerate(summary_rows, start=1):
        row["rank"] = idx

    write_csv(
        out_dir / "leaderboard_overall.csv",
        summary_rows,
        [
            "rank",
            "model_name",
            "submission_file",
            "accuracy",
            "correct",
            "total",
            "coverage",
            "answered_accuracy",
            "answered",
            "missing_ids",
            "unknown_ids",
            "duplicate_ids",
            "valid_submission",
        ],
    )
    write_csv(
        out_dir / "leaderboard_by_source.csv",
        by_source_rows,
        ["model_name", "source", "accuracy", "correct", "total", "coverage", "answered_accuracy", "answered"],
    )
    write_csv(
        out_dir / "per_item_results.csv",
        per_item_rows,
        ["model_name", "id", "source", "gold_answer", "pred_answer", "answered", "is_correct"],
    )

    markdown_rows = []
    for row in summary_rows:
        markdown_rows.append(
            {
                "rank": row["rank"],
                "model_name": row["model_name"],
                "accuracy": format_pct(row["accuracy"]),
                "correct": f"{row['correct']}/{row['total']}",
                "coverage": format_pct(row["coverage"]),
                "valid": "yes" if row["valid_submission"] else "no",
            }
        )

    readme = [
        "# Task 1 Dev Leaderboard",
        "",
        build_markdown_table(
            markdown_rows,
            [
                ("rank", "Rank"),
                ("model_name", "Model"),
                ("accuracy", "Accuracy"),
                ("correct", "Correct/Total"),
                ("coverage", "Coverage"),
                ("valid", "Valid Submission"),
            ],
        ),
        "",
        "## Files",
        "",
        "- `leaderboard_overall.csv`: overall leaderboard",
        "- `leaderboard_by_source.csv`: breakdown by source split",
        "- `per_item_results.csv`: organizer-side per-item scoring results",
        "- `*__validation.json`: validation diagnostics per submission",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
