#!/usr/bin/env python3
"""Export a standalone Hugging Face Space folder for the Task 1 dev portal."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

SPACE_README = """---
title: FinMMEval Task 1 Dev Portal
emoji: 🧮
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# FinMMEval Task 1 Dev Portal

This Space hosts the official Task 1 dev submission portal and live leaderboard.

Required Space secrets:

- `HF_TOKEN`: write token with access to the private dataset repo used for storage

Required Space variables:

- `TASK1_STORAGE_BACKEND=hf_dataset`
- `TASK1_HF_REPO_ID=<org-or-user>/<private-dataset-repo>`
- `TASK1_OFFICIAL_SITE_URL=https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/`
"""

DOCKERFILE = """FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TASK1_RUNTIME_ROOT=/tmp/task1_dev_leaderboard_runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY task1_dev_leaderboard /app/task1_dev_leaderboard

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "task1_dev_leaderboard.web_app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

REQUIREMENTS = """fastapi
uvicorn
python-multipart
huggingface_hub
"""

PACKAGE_FILES = [
    "__init__.py",
    "bootstrap_hf_backend.py",
    "create_accounting_dev_set.py",
    "evaluate_submissions.py",
    "prepare_hf_space.py",
    "storage_backend.py",
    "web_app.py",
    "README.md",
]

PACKAGE_DIRS = [
    "dev_sets",
    "web",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a standalone HF Space folder.")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "task1_dev_leaderboard_hf_space"),
        help="Output directory for the generated Space repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_package_dir = out_dir / "task1_dev_leaderboard"
    out_package_dir.mkdir(parents=True, exist_ok=True)

    for filename in PACKAGE_FILES:
        shutil.copy2(APP_ROOT / filename, out_package_dir / filename)

    for dirname in PACKAGE_DIRS:
        shutil.copytree(APP_ROOT / dirname, out_package_dir / dirname)

    (out_dir / "README.md").write_text(SPACE_README, encoding="utf-8")
    (out_dir / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    print(f"Prepared HF Space repo at: {out_dir}")


if __name__ == "__main__":
    main()
