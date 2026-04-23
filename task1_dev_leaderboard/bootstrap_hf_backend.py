#!/usr/bin/env python3
"""Bootstrap a private Hugging Face dataset repo for portal state."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from task1_dev_leaderboard.storage_backend import GOLD_REMOTE_PATH, OUTPUTS_REMOTE_DIR, REGISTRY_REMOTE_PATH


APP_ROOT = Path(__file__).resolve().parent
LOCAL_GOLD_FILE = APP_ROOT / "private" / "accounting_clef_100_gold.jsonl"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize HF dataset storage for Task 1 portal.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. org/task1-dev-storage")
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, HF_TOKEN / HUGGINGFACEHUB_API_TOKEN will be used.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the dataset repo as public. The recommended default is private.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="task1_hf_bootstrap_") as tmpdir:
        root = Path(tmpdir)
        private_dir = root / "private"
        submissions_dir = root / "submissions"
        outputs_dir = root / "outputs" / "accounting_clef_dev"
        private_dir.mkdir(parents=True, exist_ok=True)
        submissions_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        (private_dir / LOCAL_GOLD_FILE.name).write_bytes(LOCAL_GOLD_FILE.read_bytes())
        (submissions_dir / "_registry.json").write_text("{}", encoding="utf-8")

        leaderboard_fields = [
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
        ]
        (outputs_dir / "leaderboard_overall.csv").write_text(",".join(leaderboard_fields) + "\n", encoding="utf-8")
        (outputs_dir / "leaderboard_by_source.csv").write_text(
            "model_name,source,accuracy,correct,total,coverage,answered_accuracy,answered\n",
            encoding="utf-8",
        )
        (outputs_dir / "per_item_results.csv").write_text(
            "model_name,id,source,gold_answer,pred_answer,answered,is_correct\n",
            encoding="utf-8",
        )
        (outputs_dir / "README.md").write_text("# Task 1 Dev Leaderboard\n\nNo submissions yet.\n", encoding="utf-8")
        (outputs_dir / "watch_status.json").write_text(
            json.dumps(
                {
                    "backend": "hf_dataset",
                    "last_run_at": None,
                    "last_run_ok": True,
                    "returncode": 0,
                    "stdout": "Initialized empty leaderboard state.",
                    "stderr": "",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo=GOLD_REMOTE_PATH,
            path_or_fileobj=str(private_dir / LOCAL_GOLD_FILE.name),
        )
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo=REGISTRY_REMOTE_PATH,
            path_or_fileobj=str(submissions_dir / "_registry.json"),
        )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(outputs_dir),
            path_in_repo=OUTPUTS_REMOTE_DIR,
        )

    print(f"Bootstrapped HF dataset storage repo: {args.repo_id}")


if __name__ == "__main__":
    main()
