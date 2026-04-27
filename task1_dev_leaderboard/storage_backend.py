#!/usr/bin/env python3
"""Storage backends for the Task 1 dev portal."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


HF_REPO_TYPE = "dataset"


class PortalStorage:
    """Local filesystem storage with optional Hugging Face dataset mirroring."""

    def __init__(self, app_root: Path) -> None:
        self.app_root = app_root.resolve()
        self.mode = os.getenv("TASK1_STORAGE_BACKEND", "local").strip().lower() or "local"
        default_runtime_root = self.app_root if self.mode == "local" else Path("/tmp/task1_dev_leaderboard_runtime")
        self.runtime_root = Path(os.getenv("TASK1_RUNTIME_ROOT", str(default_runtime_root))).resolve()
        self.gold_filename = os.getenv("TASK1_GOLD_FILENAME", "hindi_mcq_100_gold.jsonl").strip()
        self.output_subdir = os.getenv("TASK1_OUTPUT_SUBDIR", "hindi_mcq_dev").strip() or "hindi_mcq_dev"
        self.hf_gold_remote_path = os.getenv(
            "TASK1_HF_GOLD_REMOTE_PATH",
            f"private/{self.gold_filename}",
        ).strip()
        self.hf_registry_remote_path = os.getenv(
            "TASK1_HF_REGISTRY_REMOTE_PATH",
            "submissions/_registry.json",
        ).strip()
        self.hf_outputs_remote_dir = os.getenv(
            "TASK1_HF_OUTPUT_REMOTE_DIR",
            f"outputs/{self.output_subdir}",
        ).strip()
        self.private_dir = self.runtime_root / "private"
        self.submissions_dir = self.runtime_root / "submissions"
        self.outputs_dir = self.runtime_root / "outputs" / self.output_subdir
        self.gold_file = self.private_dir / self.gold_filename
        self.registry_file = self.submissions_dir / "_registry.json"
        self.watch_status_file = self.outputs_dir / "watch_status.json"
        self.hf_repo_id = os.getenv("TASK1_HF_REPO_ID", "").strip()
        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        self._api = HfApi(token=self.hf_token) if self.mode == "hf_dataset" else None

    @property
    def backend_name(self) -> str:
        return self.mode

    def ensure_local_dirs(self) -> None:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def startup_sync(self) -> None:
        self.ensure_local_dirs()
        if self.mode == "hf_dataset":
            self._validate_hf_config()
            self.sync_remote_state()

    def sync_remote_state(self) -> None:
        if self.mode != "hf_dataset":
            self.ensure_local_dirs()
            return

        self._validate_hf_config()
        self._reset_local_mirror()
        snapshot_download(
            repo_id=self.hf_repo_id,
            repo_type=HF_REPO_TYPE,
            local_dir=str(self.runtime_root),
            allow_patterns=[
                self.hf_gold_remote_path,
                "submissions/*",
                f"{self.hf_outputs_remote_dir}/*",
            ],
            token=self.hf_token,
        )
        self.ensure_local_dirs()

    def delete_remote_submission(self, filename: str | None) -> None:
        if self.mode != "hf_dataset" or not filename:
            return
        try:
            self._api.delete_file(
                path_in_repo=f"submissions/{filename}",
                repo_id=self.hf_repo_id,
                repo_type=HF_REPO_TYPE,
            )
        except HfHubHTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise

    def upload_submission(self, local_path: Path) -> None:
        if self.mode != "hf_dataset":
            return
        self._upload_file(local_path, f"submissions/{local_path.name}")

    def upload_registry(self) -> None:
        if self.mode != "hf_dataset":
            return
        self._upload_file(self.registry_file, self.hf_registry_remote_path)

    def upload_outputs(self) -> None:
        if self.mode != "hf_dataset":
            return
        self._api.upload_folder(
            repo_id=self.hf_repo_id,
            repo_type=HF_REPO_TYPE,
            folder_path=str(self.outputs_dir),
            path_in_repo=self.hf_outputs_remote_dir,
        )

    def _upload_file(self, local_path: Path, path_in_repo: str) -> None:
        self._validate_hf_config()
        self._api.upload_file(
            repo_id=self.hf_repo_id,
            repo_type=HF_REPO_TYPE,
            path_in_repo=path_in_repo,
            path_or_fileobj=str(local_path),
        )

    def _reset_local_mirror(self) -> None:
        for path in (self.private_dir, self.submissions_dir, self.runtime_root / "outputs"):
            if path.exists():
                shutil.rmtree(path)
        cache_dir = self.runtime_root / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        self.ensure_local_dirs()

    def _validate_hf_config(self) -> None:
        if not self.hf_repo_id:
            raise RuntimeError("TASK1_HF_REPO_ID is required when TASK1_STORAGE_BACKEND=hf_dataset.")
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN is required when TASK1_STORAGE_BACKEND=hf_dataset.")
