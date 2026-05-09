#!/usr/bin/env python3
"""Export organizer-only final rankings from private Task 1/2 storage.

The script intentionally exports ranking-level fields only. It does not copy
Task 1 per-item outputs because those files contain gold answers.

Example:
  python scripts/export_private_results.py \
    --config organizer_private_exports.local.json \
    --out-dir private_exports/2026-05-15

Config shape:
{
  "task1": [
    {
      "label": "English",
      "repo_id": "ORG/PRIVATE_TASK1_EN_STORAGE",
      "output_remote_dir": "outputs/english_task1_final_test",
      "registry_remote_path": "submissions/_registry.json"
    }
  ],
  "task2": [
    {
      "label": "Task2",
      "repo_id": "ORG/PRIVATE_TASK2_STORAGE",
      "private_results_dir": "private_results"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


FORBIDDEN_PUBLIC_HEADERS = {
    "answer",
    "correct_answer",
    "gold",
    "gold_answer",
    "pred_answer",
    "reference",
    "references",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str, fallback: str = "export") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or fallback


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    unsafe = FORBIDDEN_PUBLIC_HEADERS & set(fieldnames)
    if unsafe:
        raise RuntimeError(f"Refusing to export answer-bearing fields: {sorted(unsafe)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def bool_from_row(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rank_completed_rows(rows: list[dict[str, Any]], score_key: str) -> None:
    eligible = [row for row in rows if row.get("completed")]
    eligible.sort(key=lambda row: (-as_float(row.get(score_key)), str(row.get("team_name", "")).lower()))
    for rank, row in enumerate(eligible, start=1):
        row["release_rank"] = rank
    for row in rows:
        row.setdefault("release_rank", "")


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    return (
        args.hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def snapshot_hf_repo(
    repo_id: str,
    allow_patterns: list[str],
    snapshots_dir: Path,
    token: str | None,
    force_download: bool,
) -> Path:
    if not token:
        raise RuntimeError("HF token is required for Hugging Face private dataset export.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to export from Hugging Face private datasets.") from exc

    local_dir = snapshots_dir / slugify(repo_id.replace("/", "-"), "hf-repo")
    if force_download and local_dir.exists():
        shutil.rmtree(local_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
        force_download=force_download,
    )
    return local_dir


def resolve_entry_root(
    entry: dict[str, Any],
    allow_patterns: list[str],
    snapshots_dir: Path,
    token: str | None,
    force_download: bool,
) -> Path:
    local_root = entry.get("local_root")
    if local_root:
        return Path(str(local_root)).expanduser().resolve()
    repo_id = str(entry.get("repo_id", "")).strip()
    if not repo_id:
        raise RuntimeError("Each export entry must define either repo_id or local_root.")
    return snapshot_hf_repo(repo_id, allow_patterns, snapshots_dir, token, force_download)


@dataclass
class ExportResult:
    label: str
    output_files: list[str]
    row_count: int
    completed_count: int
    warnings: list[str]


def registry_team_name(registry: dict[str, Any], slug: str) -> str:
    item = registry.get(slug, {})
    if not isinstance(item, dict):
        return slug
    return str(item.get("display_name") or item.get("team_name") or slug)


def registry_uploaded_at(registry: dict[str, Any], slug: str) -> str:
    item = registry.get(slug, {})
    if not isinstance(item, dict):
        return ""
    return str(item.get("uploaded_at") or item.get("submitted_at") or item.get("last_updated") or "")


def export_task1(
    entry: dict[str, Any],
    out_dir: Path,
    snapshots_dir: Path,
    token: str | None,
    force_download: bool,
) -> ExportResult:
    label = str(entry.get("label") or "Task1").strip()
    output_remote_dir = str(entry.get("output_remote_dir") or entry.get("outputs_remote_dir") or "").strip()
    if not output_remote_dir:
        raise RuntimeError(f"Task 1 entry {label!r} must define output_remote_dir.")
    registry_remote_path = str(entry.get("registry_remote_path") or "submissions/_registry.json").strip()
    # Do not snapshot the whole outputs folder: Task 1 per_item_results.csv
    # contains gold answers. The final ranking only needs the overall CSV.
    allow_patterns = [
        f"{output_remote_dir.rstrip('/')}/leaderboard_overall.csv",
        registry_remote_path,
    ]
    root = resolve_entry_root(entry, allow_patterns, snapshots_dir, token, force_download)
    leaderboard_path = root / output_remote_dir / "leaderboard_overall.csv"
    registry_path = root / registry_remote_path
    if not leaderboard_path.exists():
        raise RuntimeError(f"Missing Task 1 leaderboard file for {label}: {leaderboard_path}")

    registry = read_json(registry_path, {}) or {}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw in read_csv(leaderboard_path):
        slug = raw.get("model_name", "")
        missing = as_int(raw.get("missing_ids"))
        unknown = as_int(raw.get("unknown_ids"))
        duplicate = as_int(raw.get("duplicate_ids"))
        valid = bool_from_row(raw.get("valid_submission"))
        completed = valid and missing == 0 and unknown == 0 and duplicate == 0 and as_float(raw.get("coverage")) >= 1.0
        row = {
            "dataset": label,
            "raw_rank": raw.get("rank", ""),
            "release_rank": "",
            "team_name": registry_team_name(registry, slug),
            "team_slug": slug,
            "accuracy": raw.get("accuracy", ""),
            "correct": raw.get("correct", ""),
            "total": raw.get("total", ""),
            "coverage": raw.get("coverage", ""),
            "answered_accuracy": raw.get("answered_accuracy", ""),
            "answered": raw.get("answered", ""),
            "completed": int(completed),
            "valid_submission": raw.get("valid_submission", ""),
            "missing_ids": raw.get("missing_ids", ""),
            "unknown_ids": raw.get("unknown_ids", ""),
            "duplicate_ids": raw.get("duplicate_ids", ""),
            "uploaded_at": registry_uploaded_at(registry, slug),
        }
        rows.append(row)
        if not completed:
            warnings.append(f"{label}: {row['team_name']} is not completed and has no release_rank.")
    rank_completed_rows(rows, "accuracy")

    file_slug = slugify(label, "task1")
    output_path = out_dir / f"task1_{file_slug}_final_ranking.csv"
    fieldnames = [
        "dataset",
        "release_rank",
        "raw_rank",
        "team_name",
        "team_slug",
        "accuracy",
        "correct",
        "total",
        "coverage",
        "answered_accuracy",
        "answered",
        "completed",
        "valid_submission",
        "missing_ids",
        "unknown_ids",
        "duplicate_ids",
        "uploaded_at",
    ]
    write_csv(output_path, rows, fieldnames)
    return ExportResult(
        label=label,
        output_files=[str(output_path)],
        row_count=len(rows),
        completed_count=sum(1 for row in rows if row.get("completed")),
        warnings=warnings,
    )


def load_task2_private_records(root: Path, private_results_dir: str) -> list[dict[str, Any]]:
    records = []
    result_root = root / private_results_dir
    for path in sorted(result_root.glob("*.json")):
        try:
            payload = read_json(path, {})
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in Task 2 private result: {path}") from exc
        if isinstance(payload, dict):
            records.append(payload)
    return records


def export_task2(
    entry: dict[str, Any],
    out_dir: Path,
    snapshots_dir: Path,
    token: str | None,
    force_download: bool,
) -> ExportResult:
    label = str(entry.get("label") or "Task2").strip()
    private_results_dir = str(entry.get("private_results_dir") or "private_results").strip()
    allow_patterns = [f"{private_results_dir.rstrip('/')}/*.json"]
    root = resolve_entry_root(entry, allow_patterns, snapshots_dir, token, force_download)
    records = load_task2_private_records(root, private_results_dir)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for record in records:
        result = record.get("result", {})
        overall = result.get("overall", {}) if isinstance(result, dict) else {}
        completion_by_tier = result.get("completion_by_tier", {}) if isinstance(result, dict) else {}
        easy = completion_by_tier.get("easy", {}) if isinstance(completion_by_tier, dict) else {}
        expert = completion_by_tier.get("expert", {}) if isinstance(completion_by_tier, dict) else {}
        completed = bool(result.get("completed")) if isinstance(result, dict) else False
        row = {
            "dataset": label,
            "release_rank": "",
            "raw_rank": "",
            "team_name": str(record.get("team_name", "")),
            "rouge1_f1": overall.get("rouge1_f1", ""),
            "rouge1_precision": overall.get("rouge1_precision", ""),
            "rouge1_recall": overall.get("rouge1_recall", ""),
            "coverage": result.get("answered_coverage", "") if isinstance(result, dict) else "",
            "easy_coverage": easy.get("answered_coverage", ""),
            "expert_coverage": expert.get("answered_coverage", ""),
            "completed": int(completed),
            "missing_count": result.get("missing_count", "") if isinstance(result, dict) else "",
            "extra_count": result.get("extra_count", "") if isinstance(result, dict) else "",
            "duplicate_count": result.get("duplicate_count", "") if isinstance(result, dict) else "",
            "blank_answer_count": result.get("blank_answer_count", "") if isinstance(result, dict) else "",
            "over_length_answer_count": result.get("over_length_answer_count", "") if isinstance(result, dict) else "",
            "updated_utc": str(record.get("updated_utc", "")),
        }
        rows.append(row)
        if not completed:
            warnings.append(f"{label}: {row['team_name']} is not completed and has no release_rank.")

    rows.sort(key=lambda row: (-as_float(row.get("rouge1_f1")), str(row.get("team_name", "")).lower()))
    for raw_rank, row in enumerate(rows, start=1):
        row["raw_rank"] = raw_rank
    rank_completed_rows(rows, "rouge1_f1")

    file_slug = slugify(label, "task2")
    output_path = out_dir / f"task2_{file_slug}_final_ranking.csv"
    fieldnames = [
        "dataset",
        "release_rank",
        "raw_rank",
        "team_name",
        "rouge1_f1",
        "rouge1_precision",
        "rouge1_recall",
        "coverage",
        "easy_coverage",
        "expert_coverage",
        "completed",
        "missing_count",
        "extra_count",
        "duplicate_count",
        "blank_answer_count",
        "over_length_answer_count",
        "updated_utc",
    ]
    write_csv(output_path, rows, fieldnames)
    return ExportResult(
        label=label,
        output_files=[str(output_path)],
        row_count=len(rows),
        completed_count=sum(1 for row in rows if row.get("completed")),
        warnings=warnings,
    )


def export_all(config: dict[str, Any], out_dir: Path, token: str | None, force_download: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = out_dir / "_hf_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generated_at": utc_now(),
        "outputs": [],
        "warnings": [],
    }

    task1_results: list[ExportResult] = []
    for entry in config.get("task1", []) or []:
        result = export_task1(entry, out_dir, snapshots_dir, token, force_download)
        task1_results.append(result)
        manifest["outputs"].append(result.__dict__)
        manifest["warnings"].extend(result.warnings)

    if task1_results:
        combined_rows: list[dict[str, Any]] = []
        for output in task1_results:
            for path in output.output_files:
                combined_rows.extend(read_csv(Path(path)))
        if combined_rows:
            combined_path = out_dir / "task1_all_final_rankings.csv"
            write_csv(combined_path, combined_rows, list(combined_rows[0].keys()))
            manifest["outputs"].append(
                {
                    "label": "Task1 combined",
                    "output_files": [str(combined_path)],
                    "row_count": len(combined_rows),
                    "completed_count": sum(1 for row in combined_rows if bool_from_row(row.get("completed"))),
                    "warnings": [],
                }
            )

    task2_results: list[ExportResult] = []
    for entry in config.get("task2", []) or []:
        result = export_task2(entry, out_dir, snapshots_dir, token, force_download)
        task2_results.append(result)
        manifest["outputs"].append(result.__dict__)
        manifest["warnings"].extend(result.warnings)

    write_json(out_dir / "export_manifest.json", manifest)
    return manifest


def build_self_test_config(tmp_root: Path) -> dict[str, Any]:
    task1_root = tmp_root / "task1_snapshot"
    task1_output = task1_root / "outputs" / "english"
    task1_output.mkdir(parents=True)
    (task1_root / "submissions").mkdir(parents=True)
    write_csv(
        task1_output / "leaderboard_overall.csv",
        [
            {
                "rank": 1,
                "model_name": "alpha",
                "submission_file": "alpha.json",
                "accuracy": 0.8,
                "correct": 160,
                "total": 200,
                "coverage": 1.0,
                "answered_accuracy": 0.8,
                "answered": 200,
                "missing_ids": 0,
                "unknown_ids": 0,
                "duplicate_ids": 0,
                "valid_submission": 1,
            },
            {
                "rank": 2,
                "model_name": "beta",
                "submission_file": "beta.json",
                "accuracy": 0.4,
                "correct": 80,
                "total": 200,
                "coverage": 0.9,
                "answered_accuracy": 0.444444,
                "answered": 180,
                "missing_ids": 20,
                "unknown_ids": 0,
                "duplicate_ids": 0,
                "valid_submission": 1,
            },
        ],
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
    write_json(
        task1_root / "submissions" / "_registry.json",
        {
            "alpha": {"display_name": "Alpha Team", "uploaded_at": "2026-05-15T00:00:00Z"},
            "beta": {"display_name": "Beta Team", "uploaded_at": "2026-05-15T00:01:00Z"},
        },
    )

    task2_root = tmp_root / "task2_snapshot"
    task2_private = task2_root / "private_results"
    task2_private.mkdir(parents=True)
    write_json(
        task2_private / "alpha.json",
        {
            "team_name": "Alpha Team",
            "email": "alpha@example.com",
            "updated_utc": "2026-05-15T00:00:00Z",
            "result": {
                "completed": True,
                "answered_coverage": 1.0,
                "missing_count": 0,
                "extra_count": 0,
                "duplicate_count": 0,
                "blank_answer_count": 0,
                "over_length_answer_count": 0,
                "completion_by_tier": {
                    "easy": {"answered_coverage": 1.0},
                    "expert": {"answered_coverage": 1.0},
                },
                "overall": {
                    "rouge1_f1": 0.62,
                    "rouge1_precision": 0.61,
                    "rouge1_recall": 0.63,
                },
            },
        },
    )
    write_json(
        task2_private / "beta.json",
        {
            "team_name": "Beta Team",
            "email": "beta@example.com",
            "updated_utc": "2026-05-15T00:02:00Z",
            "result": {
                "completed": False,
                "answered_coverage": 0.5,
                "missing_count": 128,
                "extra_count": 0,
                "duplicate_count": 0,
                "blank_answer_count": 0,
                "over_length_answer_count": 0,
                "completion_by_tier": {},
                "overall": {
                    "rouge1_f1": 0.2,
                    "rouge1_precision": 0.2,
                    "rouge1_recall": 0.2,
                },
            },
        },
    )

    return {
        "task1": [
            {
                "label": "English",
                "local_root": str(task1_root),
                "output_remote_dir": "outputs/english",
            }
        ],
        "task2": [
            {
                "label": "Task2",
                "local_root": str(task2_root),
                "private_results_dir": "private_results",
            }
        ],
    }


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="finmmeval-export-test-") as tmp:
        tmp_root = Path(tmp)
        config = build_self_test_config(tmp_root)
        out_dir = tmp_root / "exports"
        manifest = export_all(config, out_dir, token=None, force_download=False)
        expected = [
            out_dir / "task1_english_final_ranking.csv",
            out_dir / "task1_all_final_rankings.csv",
            out_dir / "task2_task2_final_ranking.csv",
            out_dir / "export_manifest.json",
        ]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(f"Self-test missing expected outputs: {missing}")
        task2_rows = read_csv(out_dir / "task2_task2_final_ranking.csv")
        if task2_rows[0]["team_name"] != "Alpha Team" or task2_rows[0]["release_rank"] != "1":
            raise RuntimeError("Self-test Task 2 ranking order is incorrect.")
        if any("email" in row for row in task2_rows):
            raise RuntimeError("Self-test detected email in public export rows.")
        if manifest["warnings"] and not any("Beta Team" in warning for warning in manifest["warnings"]):
            raise RuntimeError("Self-test warnings were not propagated.")
        print(f"Self-test passed. Generated {len(manifest['outputs'])} export groups.")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config describing Task 1/2 private storage repos.")
    parser.add_argument("--out-dir", default="private_exports/latest", help="Directory for generated ranking CSVs.")
    parser.add_argument("--hf-token", help="HF token. Defaults to HF_TOKEN/HUGGINGFACEHUB_API_TOKEN/HUGGINGFACE_TOKEN.")
    parser.add_argument("--force-download", action="store_true", help="Refresh the HF snapshot cache.")
    parser.add_argument("--self-test", action="store_true", help="Run a local synthetic export test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_test:
        return run_self_test()
    if not args.config:
        print("ERROR: --config is required unless --self-test is used.", file=sys.stderr)
        return 2
    config_path = Path(args.config).expanduser().resolve()
    config = read_json(config_path)
    if not isinstance(config, dict):
        print(f"ERROR: config must be a JSON object: {config_path}", file=sys.stderr)
        return 2
    manifest = export_all(
        config=config,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        token=resolve_hf_token(args),
        force_download=args.force_download,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
