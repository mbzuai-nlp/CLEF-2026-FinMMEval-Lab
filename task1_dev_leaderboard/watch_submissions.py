#!/usr/bin/env python3
"""Watch a submissions directory and auto-refresh the dev leaderboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_submission_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"} and not path.name.startswith("_")


def snapshot_submissions(submissions_dir: Path) -> dict[str, dict]:
    snapshot = {}
    for path in sorted(submissions_dir.iterdir()):
        if not is_submission_file(path):
            continue
        stat = path.stat()
        snapshot[path.name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return snapshot


def write_status(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "watch_status.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_evaluation(gold_file: Path, submissions_dir: Path, out_dir: Path) -> tuple[bool, str]:
    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "evaluate_submissions.py"),
        "--gold-file",
        str(gold_file),
        "--submissions-dir",
        str(submissions_dir),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, output.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch participant submissions and auto-refresh the Task 1 dev leaderboard."
    )
    parser.add_argument("--gold-file", required=True, help="Private gold JSONL path.")
    parser.add_argument("--submissions-dir", required=True, help="Directory containing participant JSON/JSONL files.")
    parser.add_argument("--out-dir", required=True, help="Directory where leaderboard outputs are written.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval in seconds. Default: 5",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Extra wait time before evaluation after a detected change. Default: 2",
    )
    parser.add_argument(
        "--run-on-start",
        action="store_true",
        help="Run one evaluation immediately on startup.",
    )
    args = parser.parse_args()

    gold_file = Path(args.gold_file).resolve()
    submissions_dir = Path(args.submissions_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    submissions_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_snapshot = snapshot_submissions(submissions_dir)
    last_success_at = None
    run_count = 0

    if args.run_on_start:
        ok, output = run_evaluation(gold_file, submissions_dir, out_dir)
        run_count += 1
        status = {
            "watch_started_at": utc_now_iso(),
            "last_checked_at": utc_now_iso(),
            "last_change_detected_at": None,
            "last_run_at": utc_now_iso(),
            "last_success_at": utc_now_iso() if ok else None,
            "last_run_ok": ok,
            "run_count": run_count,
            "tracked_files": last_snapshot,
            "last_output": output,
        }
        if ok:
            last_success_at = status["last_success_at"]
        write_status(out_dir, status)
    else:
        status = {
            "watch_started_at": utc_now_iso(),
            "last_checked_at": utc_now_iso(),
            "last_change_detected_at": None,
            "last_run_at": None,
            "last_success_at": None,
            "last_run_ok": None,
            "run_count": 0,
            "tracked_files": last_snapshot,
            "last_output": "",
        }
        write_status(out_dir, status)

    print(f"Watching submissions in: {submissions_dir}")
    print(f"Leaderboard output dir: {out_dir}")

    while True:
        time.sleep(args.poll_seconds)
        current_snapshot = snapshot_submissions(submissions_dir)
        if current_snapshot == last_snapshot:
            status["last_checked_at"] = utc_now_iso()
            status["tracked_files"] = current_snapshot
            write_status(out_dir, status)
            continue

        change_time = utc_now_iso()
        print(f"[{change_time}] Detected submission change. Waiting {args.settle_seconds}s to settle...")
        time.sleep(args.settle_seconds)
        current_snapshot = snapshot_submissions(submissions_dir)

        ok, output = run_evaluation(gold_file, submissions_dir, out_dir)
        run_count += 1
        last_snapshot = current_snapshot
        status = {
            "watch_started_at": status["watch_started_at"],
            "last_checked_at": utc_now_iso(),
            "last_change_detected_at": change_time,
            "last_run_at": utc_now_iso(),
            "last_success_at": utc_now_iso() if ok else last_success_at,
            "last_run_ok": ok,
            "run_count": run_count,
            "tracked_files": current_snapshot,
            "last_output": output,
        }
        if ok:
            last_success_at = status["last_success_at"]
            print(f"[{status['last_run_at']}] Leaderboard refreshed successfully.")
        else:
            print(f"[{status['last_run_at']}] Evaluation failed; watcher will continue.")
        write_status(out_dir, status)


if __name__ == "__main__":
    main()
