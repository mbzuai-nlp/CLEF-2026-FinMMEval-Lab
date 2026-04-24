#!/usr/bin/env python3
"""Create and upload a Task 1 Hugging Face portal and its backing dataset repo."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a Task 1 HF dataset backend and Space.")
    parser.add_argument("--variant", required=True, help="Language label, e.g. English")
    parser.add_argument("--dataset-label", required=True, help="Dataset label shown in the portal")
    parser.add_argument("--devset-filename", required=True, help="Public dev set filename in dev_sets/")
    parser.add_argument("--template-filename", required=True, help="Template filename in dev_sets/")
    parser.add_argument("--gold-filename", required=True, help="Private gold filename in private/")
    parser.add_argument("--output-subdir", required=True, help="Output subdir for leaderboard files")
    parser.add_argument("--storage-repo-id", required=True, help="HF dataset repo id for storage")
    parser.add_argument("--space-repo-id", required=True, help="HF Space repo id for the portal")
    parser.add_argument("--space-title", required=True, help="HF Space title")
    parser.add_argument("--official-site-url", default="https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/")
    parser.add_argument("--token", default=None, help="HF write token")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("A Hugging Face write token is required.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.storage_repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(repo_id=args.space_repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

    env = os.environ.copy()
    env["HF_TOKEN"] = token
    env["TASK1_GOLD_FILENAME"] = args.gold_filename
    env["TASK1_OUTPUT_SUBDIR"] = args.output_subdir
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}:{existing_pythonpath}"

    run(
        [
            "python",
            str(APP_ROOT / "bootstrap_hf_backend.py"),
            "--repo-id",
            args.storage_repo_id,
            "--gold-file",
            str(APP_ROOT / "private" / args.gold_filename),
            "--token",
            token,
        ],
        env=env,
    )

    safe_variant = args.variant.strip().lower().replace(" ", "_")
    out_dir = PROJECT_ROOT / f"task1_dev_leaderboard_hf_space_{safe_variant}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    run(
        [
            "python",
            str(APP_ROOT / "prepare_hf_space.py"),
            "--out-dir",
            str(out_dir),
            "--space-title",
            args.space_title,
        ],
        env=env,
    )

    api.upload_folder(repo_id=args.space_repo_id, repo_type="space", folder_path=str(out_dir))
    api.add_space_secret(repo_id=args.space_repo_id, key="HF_TOKEN", value=token)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_STORAGE_BACKEND", value="hf_dataset")
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_HF_REPO_ID", value=args.storage_repo_id)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_OFFICIAL_SITE_URL", value=args.official_site_url)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_DEVSET_FILENAME", value=args.devset_filename)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_TEMPLATE_FILENAME", value=args.template_filename)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_GOLD_FILENAME", value=args.gold_filename)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_OUTPUT_SUBDIR", value=args.output_subdir)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_PORTAL_VARIANT", value=args.variant)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_PORTAL_TITLE", value=args.space_title)
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_PORTAL_SUBMISSION_TITLE", value=f"{args.variant} Dev Submission Portal")
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_PORTAL_LEADERBOARD_TITLE", value=f"Task 1 {args.variant} Dev Leaderboard")
    api.add_space_variable(repo_id=args.space_repo_id, key="TASK1_PORTAL_DATASET_LABEL", value=args.dataset_label)

    print(f"Storage repo ready: {args.storage_repo_id}")
    print(f"Space deployed: https://{args.space_repo_id.replace('/', '-')}.hf.space/task1/dev")


if __name__ == "__main__":
    main()
