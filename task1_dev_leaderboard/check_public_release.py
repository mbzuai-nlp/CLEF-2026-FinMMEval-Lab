#!/usr/bin/env python3
"""Fail if organizer-private Task 1 files would be included in a public release."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


SENSITIVE_PATH_PATTERNS = [
    re.compile(r"(^|/)private(/|$)"),
    re.compile(r"(^|/)hidden(/|$)"),
    re.compile(r"(^|/)gold(/|$)"),
    re.compile(r"(^|/)[^/]*_gold\.(json|jsonl|csv)$"),
    re.compile(r"(^|/)[^/]*hidden_test[^/]*$"),
]

REQUIRED_GITIGNORE_ENTRIES = {
    "task1_dev_leaderboard/private/",
    "task1_training/artifacts/",
}


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def list_release_candidates() -> list[str]:
    output = run_git(["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    return sorted(path for path in output.split("\0") if path)


def is_sensitive_path(path: str) -> bool:
    return any(pattern.search(path) for pattern in SENSITIVE_PATH_PATTERNS)


def check_gitignore(repo_root: Path) -> list[str]:
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return sorted(REQUIRED_GITIGNORE_ENTRIES)
    entries = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return sorted(REQUIRED_GITIGNORE_ENTRIES - entries)


def main() -> int:
    repo_root = Path(run_git(["rev-parse", "--show-toplevel"]).strip())
    candidates = list_release_candidates()
    sensitive = [path for path in candidates if is_sensitive_path(path)]
    missing_gitignore_entries = check_gitignore(repo_root)

    if sensitive:
        print("Refusing public release: sensitive paths are tracked or unignored:", file=sys.stderr)
        for path in sensitive:
            print(f"  - {path}", file=sys.stderr)
    if missing_gitignore_entries:
        print("Refusing public release: required .gitignore entries are missing:", file=sys.stderr)
        for entry in missing_gitignore_entries:
            print(f"  - {entry}", file=sys.stderr)

    if sensitive or missing_gitignore_entries:
        return 1

    print("Public release check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
