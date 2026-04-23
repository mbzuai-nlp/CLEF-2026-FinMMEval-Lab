#!/usr/bin/env python3
"""Web app for Task 1 dev submissions and leaderboard."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from task1_dev_leaderboard.storage_backend import PortalStorage


APP_ROOT = Path(__file__).resolve().parent
WEB_ROOT = APP_ROOT / "web"
DEVSET_FILE = APP_ROOT / "dev_sets" / os.getenv("TASK1_DEVSET_FILENAME", "accounting_clef_100_public.jsonl")
TEMPLATE_FILE = APP_ROOT / "dev_sets" / os.getenv(
    "TASK1_TEMPLATE_FILENAME",
    "accounting_clef_100_submission_template.json",
)
EVALUATOR = APP_ROOT / "evaluate_submissions.py"
STORAGE = PortalStorage(APP_ROOT)
SUBMISSIONS_DIR = STORAGE.submissions_dir
OUTPUT_DIR = STORAGE.outputs_dir
GOLD_FILE = STORAGE.gold_file
REGISTRY_FILE = STORAGE.registry_file
WATCH_STATUS_FILE = STORAGE.watch_status_file
OFFICIAL_SITE_URL = os.getenv(
    "TASK1_OFFICIAL_SITE_URL",
    "https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/",
)
PORTAL_VARIANT = os.getenv("TASK1_PORTAL_VARIANT", "Arabic").strip() or "Arabic"
PORTAL_TITLE = os.getenv("TASK1_PORTAL_TITLE", f"Task 1 {PORTAL_VARIANT} Dev Portal").strip()
PORTAL_SUBMISSION_TITLE = os.getenv(
    "TASK1_PORTAL_SUBMISSION_TITLE",
    f"{PORTAL_VARIANT} Dev Submission Portal",
).strip()
PORTAL_LEADERBOARD_TITLE = os.getenv(
    "TASK1_PORTAL_LEADERBOARD_TITLE",
    f"Task 1 {PORTAL_VARIANT} Dev Leaderboard",
).strip()
PORTAL_DATASET_LABEL = os.getenv(
    "TASK1_PORTAL_DATASET_LABEL",
    f"{PORTAL_VARIANT} Task 1 Dev Set",
).strip()

SUBMISSION_EXTENSIONS = {".json", ".jsonl"}
SUBMISSION_LOCK = threading.Lock()

app = FastAPI(title="FinMMEval Task 1 Dev Portal", version="1.0.0")
app.mount("/task1/dev/static", StaticFiles(directory=str(WEB_ROOT / "assets")), name="task1-dev-static")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    STORAGE.ensure_local_dirs()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_page(path: Path) -> str:
    content = read_text_file(path)
    replacements = {
        "__OFFICIAL_SITE_URL__": OFFICIAL_SITE_URL,
        "__PORTAL_TITLE__": PORTAL_TITLE,
        "__PORTAL_SUBMISSION_TITLE__": PORTAL_SUBMISSION_TITLE,
        "__PORTAL_LEADERBOARD_TITLE__": PORTAL_LEADERBOARD_TITLE,
        "__PORTAL_VARIANT__": PORTAL_VARIANT,
        "__PORTAL_DATASET_LABEL__": PORTAL_DATASET_LABEL,
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text


def team_slug(value: str) -> str:
    slug = slugify(value)
    if slug:
        return slug
    digest = hashlib.sha1(value.strip().encode("utf-8")).hexdigest()[:10]
    return f"team-{digest}"


def load_registry() -> dict[str, dict]:
    return load_json(REGISTRY_FILE, {})


def save_registry(registry: dict[str, dict]) -> None:
    save_json(REGISTRY_FILE, registry)


def save_watch_status(payload: dict[str, Any]) -> None:
    save_json(WATCH_STATUS_FILE, payload)


def is_submission_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUBMISSION_EXTENSIONS and not path.name.startswith("_")


def validate_submission_bytes(content: bytes, suffix: str) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Submission file must be valid UTF-8.") from exc

    if suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON submission: {exc}") from exc
        return

    if suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSONL submission at line {line_no}: {exc}",
                ) from exc
        return

    raise HTTPException(status_code=400, detail="Unsupported submission file type.")


def run_evaluation() -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--gold-file",
        str(GOLD_FILE),
        "--submissions-dir",
        str(SUBMISSIONS_DIR),
        "--out-dir",
        str(OUTPUT_DIR),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    return {
        "ok": proc.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def maybe_number(value: str) -> Any:
    if value is None:
        return value
    if value == "":
        return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{key: maybe_number(value) for key, value in row.items()} for row in rows]


def current_leaderboard_payload() -> dict[str, Any]:
    with SUBMISSION_LOCK:
        STORAGE.sync_remote_state()
        registry = load_registry()
        overall = normalize_rows(load_csv_rows(OUTPUT_DIR / "leaderboard_overall.csv"))
        by_source = normalize_rows(load_csv_rows(OUTPUT_DIR / "leaderboard_by_source.csv"))
        status = load_json(WATCH_STATUS_FILE, {})

        for row in overall:
            slug = row.get("model_name", "")
            row["display_name"] = registry.get(slug, {}).get("display_name", slug)
            row["uploaded_at"] = registry.get(slug, {}).get("uploaded_at")
            row["original_filename"] = registry.get(slug, {}).get("original_filename")

        for row in by_source:
            slug = row.get("model_name", "")
            row["display_name"] = registry.get(slug, {}).get("display_name", slug)

        dataset_meta = {
            "devset_file": str(DEVSET_FILE),
            "template_file": str(TEMPLATE_FILE),
            "submission_count": len(
                [
                    path
                    for path in SUBMISSIONS_DIR.iterdir()
                    if is_submission_file(path)
                ]
            ),
            "last_updated": status.get("last_run_at"),
        }
        return {"overall": overall, "by_source": by_source, "status": status, "dataset": dataset_meta}


@app.on_event("startup")
def startup_event() -> None:
    STORAGE.startup_sync()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/task1/dev")


@app.get("/task1/dev", response_class=HTMLResponse, include_in_schema=False)
def task1_dev_home() -> HTMLResponse:
    return HTMLResponse(render_page(WEB_ROOT / "index.html"))


@app.get("/task1/dev/submit", response_class=HTMLResponse, include_in_schema=False)
def task1_dev_submit_page() -> HTMLResponse:
    return HTMLResponse(render_page(WEB_ROOT / "submit.html"))


@app.get("/task1/dev/leaderboard", response_class=HTMLResponse, include_in_schema=False)
def task1_dev_leaderboard_page() -> HTMLResponse:
    return HTMLResponse(render_page(WEB_ROOT / "leaderboard.html"))


@app.get("/api/task1/dev/meta")
def api_task1_dev_meta() -> JSONResponse:
    payload = current_leaderboard_payload()
    return JSONResponse(payload["dataset"])


@app.get("/api/task1/leaderboard")
def api_task1_leaderboard() -> JSONResponse:
    return JSONResponse(current_leaderboard_payload())


@app.get("/api/task1/devset/download")
def api_task1_devset_download() -> FileResponse:
    if not DEVSET_FILE.exists():
        raise HTTPException(status_code=404, detail="Dev set file not found.")
    return FileResponse(DEVSET_FILE, filename=DEVSET_FILE.name, media_type="application/json")


@app.get("/api/task1/template/download")
def api_task1_template_download() -> FileResponse:
    if not TEMPLATE_FILE.exists():
        raise HTTPException(status_code=404, detail="Submission template file not found.")
    return FileResponse(TEMPLATE_FILE, filename=TEMPLATE_FILE.name, media_type="application/json")


@app.post("/api/task1/submissions")
async def api_task1_submit(
    team_name: str = Form(...),
    prediction_file: UploadFile = File(...),
) -> JSONResponse:
    ensure_directories()

    cleaned_team_name = team_name.strip()
    if not cleaned_team_name:
        raise HTTPException(status_code=400, detail="Team name is required.")

    suffix = Path(prediction_file.filename or "").suffix.lower()
    if suffix not in SUBMISSION_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Submission must be a .json or .jsonl file.")

    slug = team_slug(cleaned_team_name)

    content = await prediction_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    validate_submission_bytes(content, suffix)

    with SUBMISSION_LOCK:
        STORAGE.sync_remote_state()
        registry = load_registry()
        previous_filename = registry.get(slug, {}).get("stored_filename")

        for existing in SUBMISSIONS_DIR.glob(f"{slug}.*"):
            if existing.is_file():
                existing.unlink()

        save_path = SUBMISSIONS_DIR / f"{slug}{suffix}"
        save_path.write_bytes(content)

        registry[slug] = {
            "display_name": cleaned_team_name,
            "uploaded_at": utc_now_iso(),
            "original_filename": prediction_file.filename,
            "stored_filename": save_path.name,
        }
        save_registry(registry)
        if previous_filename and previous_filename != save_path.name:
            STORAGE.delete_remote_submission(previous_filename)
        STORAGE.upload_submission(save_path)
        STORAGE.upload_registry()

        result = run_evaluation()
        status_payload = {
            "backend": STORAGE.backend_name,
            "last_run_at": utc_now_iso(),
            "last_run_ok": result["ok"],
            "returncode": result["returncode"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
        }
        save_watch_status(status_payload)
        STORAGE.upload_outputs()
        if not result["ok"]:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Submission saved but evaluation failed.",
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                },
            )

    payload = current_leaderboard_payload()
    row = next((item for item in payload["overall"] if item.get("model_name") == slug), None)
    validation_path = OUTPUT_DIR / f"{slug}__validation.json"
    validation = load_json(validation_path, {})

    return JSONResponse(
        {
            "message": "Submission received and leaderboard refreshed.",
            "team_slug": slug,
            "team_name": cleaned_team_name,
            "leaderboard_row": row,
            "validation": validation,
            "last_updated": payload["dataset"].get("last_updated"),
        }
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "time": utc_now_iso(),
            "backend": STORAGE.backend_name,
            "hf_repo_id": STORAGE.hf_repo_id or None,
            "devset_exists": DEVSET_FILE.exists(),
            "gold_exists": GOLD_FILE.exists(),
            "submission_dir": str(SUBMISSIONS_DIR),
            "output_dir": str(OUTPUT_DIR),
            "official_site_url": OFFICIAL_SITE_URL,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("task1_dev_leaderboard.web_app:app", host="0.0.0.0", port=8091, reload=True)
