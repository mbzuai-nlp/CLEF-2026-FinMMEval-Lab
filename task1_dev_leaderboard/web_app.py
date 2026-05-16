#!/usr/bin/env python3
"""Web app for Task 1 dev submissions and leaderboard."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import smtplib
import subprocess
import sys
import threading
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from task1_dev_leaderboard.storage_backend import PortalStorage


APP_ROOT = Path(__file__).resolve().parent
WEB_ROOT = APP_ROOT / "web"
DEVSET_FILE = APP_ROOT / "dev_sets" / os.getenv("TASK1_DEVSET_FILENAME", "hindi_mcq_100_public.jsonl")
TEMPLATE_FILE = APP_ROOT / "dev_sets" / os.getenv(
    "TASK1_TEMPLATE_FILENAME",
    "hindi_mcq_100_submission_template.json",
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
PORTAL_MODE = os.getenv("TASK1_PORTAL_MODE", "dev").strip().lower() or "dev"
if PORTAL_MODE not in {"dev", "test"}:
    raise RuntimeError("TASK1_PORTAL_MODE must be either dev or test.")
PORTAL_VARIANT = os.getenv("TASK1_PORTAL_VARIANT", "Hindi").strip() or "Hindi"
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
TEAM_CODES_FILE = Path(
    os.getenv("TASK1_TEAM_CODES_FILE", str(STORAGE.private_dir / STORAGE.team_codes_filename))
).resolve()
REQUIRE_TEAM_CODE = os.getenv("TASK1_REQUIRE_TEAM_CODE", "").strip().lower() in {"1", "true", "yes", "on"}
REQUIRE_EMAIL = PORTAL_MODE == "test" or os.getenv("TASK1_REQUIRE_EMAIL", "").strip().lower() in {"1", "true", "yes", "on"}

SUBMISSION_EXTENSIONS = {".json", ".jsonl"}
VALID_LETTERS = {"A", "B", "C", "D", "E", "F"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBMISSION_LOCK = threading.Lock()

NOTIFY_SMTP_HOST = os.getenv("FINMMEVAL_NOTIFY_SMTP_HOST", "").strip()
NOTIFY_SMTP_PORT = int(os.getenv("FINMMEVAL_NOTIFY_SMTP_PORT", "587"))
NOTIFY_SMTP_USERNAME = os.getenv("FINMMEVAL_NOTIFY_SMTP_USERNAME", "").strip()
NOTIFY_SMTP_PASSWORD = os.getenv("FINMMEVAL_NOTIFY_SMTP_PASSWORD", "").strip()
NOTIFY_FROM = os.getenv("FINMMEVAL_NOTIFY_FROM", NOTIFY_SMTP_USERNAME).strip()
NOTIFY_REPLY_TO = os.getenv("FINMMEVAL_NOTIFY_REPLY_TO", "").strip()
NOTIFY_STARTTLS = os.getenv("FINMMEVAL_NOTIFY_STARTTLS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SUBMISSION_DEADLINE_TEXT = os.getenv("FINMMEVAL_SUBMISSION_DEADLINE_TEXT", "20 May 2026 AoE").strip()

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
    leaderboard_is_test = PORTAL_MODE == "test"
    replacements = {
        "__OFFICIAL_SITE_URL__": OFFICIAL_SITE_URL,
        "__PORTAL_TITLE__": PORTAL_TITLE,
        "__PORTAL_SUBMISSION_TITLE__": PORTAL_SUBMISSION_TITLE,
        "__PORTAL_LEADERBOARD_TITLE__": PORTAL_LEADERBOARD_TITLE,
        "__PORTAL_VARIANT__": PORTAL_VARIANT,
        "__PORTAL_DATASET_LABEL__": PORTAL_DATASET_LABEL,
        "__LEADERBOARD_KICKER__": "Submission Status" if leaderboard_is_test else "Live Ranking",
        "__LEADERBOARD_INTRO__": (
            "Final-test submissions are accepted and evaluated privately. Scores, ranks, and correct counts remain hidden until the official release."
            if leaderboard_is_test
            else f"Official organizer-side ranking page for the released {PORTAL_VARIANT} Task 1 dev set. Results refresh automatically after each successful submission."
        ),
        "__LEADERBOARD_NOTE__": (
            "Each submission must include an email address. The email is used as the unique submission key, so a later upload from the same email replaces the earlier active submission."
            if leaderboard_is_test
            else "Organizer note: each registered team should use one consistent official team name. Duplicate aliases, test entries, or obvious non-team submissions may be merged, renamed, or removed from this dev leaderboard."
        ),
        "__LEADERBOARD_TABLE_TITLE__": "Submission Status" if leaderboard_is_test else "Overall Ranking",
        "__LEADERBOARD_TABLE_HEAD__": (
            "<tr><th>Team</th><th>Updated</th><th>Coverage</th><th>Completed</th></tr>"
            if leaderboard_is_test
            else "<tr><th>Rank</th><th>Team</th><th>Accuracy</th><th>Correct / Total</th><th>Coverage</th><th>Valid</th><th>Uploaded At</th></tr>"
        ),
        "__LEADERBOARD_EMPTY_COLSPAN__": "4" if leaderboard_is_test else "7",
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


def normalize_email(value: str) -> str:
    return value.strip().lower()


def validate_email(value: str) -> str:
    email = normalize_email(value)
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    if not EMAIL_RE.fullmatch(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    return email


def email_submission_slug(email: str) -> str:
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]
    return f"email-{digest}"


def notification_configured() -> bool:
    return bool(NOTIFY_SMTP_HOST and NOTIFY_FROM)


def send_email_notification(to_email: str, subject: str, body: str) -> dict[str, Any]:
    if not to_email:
        return {"sent": False, "reason": "missing_recipient"}
    if not notification_configured():
        return {"sent": False, "reason": "smtp_not_configured"}

    msg = EmailMessage()
    msg["From"] = NOTIFY_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    if NOTIFY_REPLY_TO:
        msg["Reply-To"] = NOTIFY_REPLY_TO
    msg.set_content(body)

    try:
        if NOTIFY_SMTP_PORT == 465:
            with smtplib.SMTP_SSL(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=20) as server:
                if NOTIFY_SMTP_USERNAME and NOTIFY_SMTP_PASSWORD:
                    server.login(NOTIFY_SMTP_USERNAME, NOTIFY_SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=20) as server:
                if NOTIFY_STARTTLS:
                    server.starttls()
                if NOTIFY_SMTP_USERNAME and NOTIFY_SMTP_PASSWORD:
                    server.login(NOTIFY_SMTP_USERNAME, NOTIFY_SMTP_PASSWORD)
                server.send_message(msg)
    except Exception as exc:  # pragma: no cover - depends on deployment SMTP
        print(f"format notification failed for {to_email}: {exc}", file=sys.stderr)
        return {"sent": False, "reason": str(exc)}
    return {"sent": True}


def first_items(values: list[str], limit: int = 5) -> str:
    if not values:
        return ""
    shown = ", ".join(values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown


def task1_format_issue_lines(validation: dict[str, Any]) -> list[str]:
    lines = []
    missing_ids = validation.get("missing_ids") or []
    unknown_ids = validation.get("unknown_ids") or []
    duplicate_id_values = validation.get("duplicate_id_values") or []
    invalid_prediction_ids = validation.get("invalid_prediction_ids") or []

    if missing_ids:
        lines.append(f"- Missing expected question IDs: {len(missing_ids)}")
        lines.append(f"  Examples: {first_items(missing_ids)}")
    if unknown_ids:
        lines.append(f"- Unknown question IDs not in the official test set: {len(unknown_ids)}")
        lines.append(f"  Examples: {first_items(unknown_ids)}")
    if validation.get("duplicate_ids", 0):
        lines.append(f"- Duplicate question ID entries: {validation['duplicate_ids']}")
        if duplicate_id_values:
            lines.append(f"  Examples: {first_items(duplicate_id_values)}")
    if invalid_prediction_ids:
        lines.append(f"- Invalid predictions: {len(invalid_prediction_ids)}")
        lines.append(f"  Examples: {first_items(invalid_prediction_ids)}")

    return lines or ["- The submission did not pass the final-test format check."]


def notify_task1_format_issue(to_email: str, team_name: str, validation: dict[str, Any]) -> dict[str, Any]:
    coverage = validation.get("coverage")
    coverage_text = f"{float(coverage) * 100:.2f}%" if coverage is not None else "not available"
    subject = f"FinMMEval Task 1 {PORTAL_VARIANT} Submission Format Check"
    issue_text = "\n".join(task1_format_issue_lines(validation))
    body = f"""Dear {team_name} team,

We received your FinMMEval Task 1 {PORTAL_VARIANT} final-test submission, but it did not pass the organizer-side format check.

Detected issues:
{issue_text}

Current answered coverage: {coverage_text}
Rows received: {validation.get("rows", "not available")}
Expected test items: {validation.get("total", "not available")}

Please update the file so that each official test question ID appears exactly once and each prediction is one of A/B/C/D/E/F. You can resubmit before {SUBMISSION_DEADLINE_TEXT}; only the latest submission from the same email will be used.

This email is only a format-check notification. Scores and ranks remain hidden until the official release.

Best regards,
FinMMEval Organizers
"""
    return send_email_notification(to_email, subject, body)


def load_registry() -> dict[str, dict]:
    return load_json(REGISTRY_FILE, {})


def save_registry(registry: dict[str, dict]) -> None:
    save_json(REGISTRY_FILE, registry)


def save_watch_status(payload: dict[str, Any]) -> None:
    save_json(WATCH_STATUS_FILE, payload)


def is_submission_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUBMISSION_EXTENSIONS and not path.name.startswith("_")


def normalize_team_code(value: str) -> str:
    return value.strip()


def load_team_codes() -> dict[str, dict[str, str]]:
    if not TEAM_CODES_FILE.exists():
        return {}
    payload = load_json(TEAM_CODES_FILE, {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Team code file must be a JSON object.")

    codes: dict[str, dict[str, str]] = {}
    for raw_code, raw_team in payload.items():
        code = normalize_team_code(str(raw_code))
        if not code:
            continue
        if isinstance(raw_team, str):
            display_name = raw_team.strip()
            slug = team_slug(display_name)
        elif isinstance(raw_team, dict):
            display_name = str(raw_team.get("display_name") or raw_team.get("team_name") or "").strip()
            slug = str(raw_team.get("team_slug") or raw_team.get("slug") or "").strip()
            if not slug and display_name:
                slug = team_slug(display_name)
        else:
            continue
        if not display_name or not slug:
            raise HTTPException(status_code=500, detail=f"Invalid team code entry for code: {code}")
        codes[code] = {"display_name": display_name, "team_slug": slugify(slug) or team_slug(display_name)}
    return codes


def team_code_required() -> bool:
    return REQUIRE_TEAM_CODE or TEAM_CODES_FILE.exists()


def resolve_team_identity(team_name: str, submission_code: str | None) -> tuple[str, str]:
    cleaned_team_name = team_name.strip()
    codes = load_team_codes()
    if team_code_required():
        code = normalize_team_code(submission_code or "")
        if not code:
            raise HTTPException(status_code=403, detail="A team submission code is required.")
        team = codes.get(code)
        if team is None:
            raise HTTPException(status_code=403, detail="Invalid team submission code.")
        return team["team_slug"], team["display_name"]

    if not cleaned_team_name:
        raise HTTPException(status_code=400, detail="Team name is required.")
    return team_slug(cleaned_team_name), cleaned_team_name


def resolve_submission_identity(
    team_name: str,
    submission_code: str | None,
    contact_email: str,
) -> tuple[str, str, str | None]:
    if REQUIRE_EMAIL:
        email = validate_email(contact_email)
        display_name = team_name.strip()
        if not display_name:
            raise HTTPException(status_code=400, detail="Team name is required.")
        if len(display_name) > 160:
            raise HTTPException(status_code=400, detail="Team name must be 160 characters or fewer.")
        return email_submission_slug(email), display_name, email

    slug, display_name = resolve_team_identity(team_name, submission_code)
    if len(display_name) > 160:
        raise HTTPException(status_code=400, detail="Team name must be 160 characters or fewer.")
    email = normalize_email(contact_email) if contact_email.strip() else None
    if email and not EMAIL_RE.fullmatch(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    return slug, display_name, email


def parse_submission_rows(content: bytes, suffix: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Submission file must be valid UTF-8.") from exc

    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON submission: {exc}") from exc
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            return [row for row in payload["predictions"] if isinstance(row, dict)]
        if isinstance(payload, dict):
            return [{"id": key, "prediction": value} for key, value in payload.items()]
        raise HTTPException(status_code=400, detail="Unsupported JSON submission structure.")

    if suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSONL submission at line {line_no}: {exc}",
                ) from exc
            if not isinstance(row, dict):
                raise HTTPException(status_code=400, detail=f"JSONL line {line_no} must be an object.")
            rows.append(row)
        return rows

    raise HTTPException(status_code=400, detail="Unsupported submission file type.")


def validate_submission_bytes(content: bytes, suffix: str) -> None:
    parse_submission_rows(content, suffix)


def normalize_prediction(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return text if text in VALID_LETTERS else ""


def extract_prediction(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("prediction", "pred_letter", "answer", "label"):
        if key in row:
            return str(row[key]).strip().upper(), normalize_prediction(row[key])
    return "", ""


def load_expected_ids(path: Path) -> set[str]:
    if not path.exists():
        raise HTTPException(status_code=500, detail="Gold/answer key file is not configured for validation.")
    expected = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=500, detail=f"Invalid organizer gold file at line {line_no}: {exc}") from exc
            expected.add(str(row["id"]))
    return expected


def validate_submission_for_test_mode(content: bytes, suffix: str) -> dict[str, Any]:
    rows = parse_submission_rows(content, suffix)
    expected_ids = load_expected_ids(GOLD_FILE)
    predictions: dict[str, str] = {}
    duplicate_ids = 0
    duplicate_id_values = []
    invalid_prediction_ids = []

    for row in rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        raw_pred, pred = extract_prediction(row)
        if item_id in predictions:
            duplicate_ids += 1
            duplicate_id_values.append(item_id)
        predictions[item_id] = pred
        if raw_pred and not pred:
            invalid_prediction_ids.append(item_id)

    submitted_ids = set(predictions)
    unknown_ids = sorted(submitted_ids - expected_ids)
    missing_ids = sorted(expected_ids - submitted_ids)
    answered_ids = sorted(item_id for item_id in expected_ids if predictions.get(item_id))
    total = len(expected_ids)
    answered = len(answered_ids)
    return {
        "mode": "test",
        "rows": len(rows),
        "total": total,
        "answered": answered,
        "coverage": round(answered / total, 6) if total else 0.0,
        "missing_ids": missing_ids,
        "unknown_ids": unknown_ids,
        "duplicate_ids": duplicate_ids,
        "duplicate_id_values": sorted(set(duplicate_id_values)),
        "invalid_prediction_ids": sorted(set(invalid_prediction_ids)),
        "invalid_prediction_count": len(set(invalid_prediction_ids)),
        "valid_submission": int(
            not missing_ids and not unknown_ids and not duplicate_ids and not invalid_prediction_ids
        ),
    }


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
        status = load_json(WATCH_STATUS_FILE, {})
        public_status = {
            "backend": status.get("backend", STORAGE.backend_name),
            "last_run_at": status.get("last_run_at"),
            "last_run_ok": status.get("last_run_ok", True),
            "returncode": status.get("returncode"),
            "message": status.get("message"),
        }

        for row in overall:
            slug = row.get("model_name", "")
            row["display_name"] = registry.get(slug, {}).get("display_name", slug)
            row["uploaded_at"] = registry.get(slug, {}).get("uploaded_at")
            row["original_filename"] = registry.get(slug, {}).get("original_filename")

        dataset_meta = {
            "devset_filename": DEVSET_FILE.name,
            "template_filename": TEMPLATE_FILE.name,
            "portal_mode": PORTAL_MODE,
            "requires_team_code": False if REQUIRE_EMAIL else team_code_required(),
            "requires_email": REQUIRE_EMAIL,
            "submission_count": len(
                [
                    path
                    for path in SUBMISSIONS_DIR.iterdir()
                    if is_submission_file(path)
                ]
            ),
            "last_updated": status.get("last_run_at"),
        }
        return {"overall": overall, "status": public_status, "dataset": dataset_meta}


def current_test_status_payload() -> dict[str, Any]:
    with SUBMISSION_LOCK:
        STORAGE.sync_remote_state()
        registry = load_registry()
        status = load_json(WATCH_STATUS_FILE, {})
        submissions = []
        for slug, meta in sorted(registry.items(), key=lambda item: (item[1].get("uploaded_at") or "", item[0]), reverse=True):
            format_validation = load_json(OUTPUT_DIR / f"{slug}__format_validation.json", {})
            evaluation_validation = load_json(OUTPUT_DIR / f"{slug}__validation.json", {})
            format_valid = bool(format_validation and format_validation.get("valid_submission"))
            evaluation_completed = bool(evaluation_validation)
            submissions.append(
                {
                    "team_name": meta.get("display_name", slug),
                    "uploaded_at": meta.get("uploaded_at"),
                    "coverage": format_validation.get("coverage") if format_validation else None,
                    "completed": bool(format_valid and evaluation_completed),
                    "format_valid": format_valid,
                    "evaluation_completed": evaluation_completed,
                    "missing_ids": len(format_validation.get("missing_ids") or []),
                    "unknown_ids": len(format_validation.get("unknown_ids") or []),
                    "duplicate_ids": int(format_validation.get("duplicate_ids") or 0),
                    "invalid_prediction_count": int(format_validation.get("invalid_prediction_count") or 0),
                }
            )

        public_status = {
            "backend": status.get("backend", STORAGE.backend_name),
            "last_run_at": status.get("last_run_at"),
            "last_run_ok": status.get("last_run_ok", True),
            "returncode": status.get("returncode"),
            "message": status.get("message"),
        }
        dataset_meta = {
            "devset_filename": DEVSET_FILE.name,
            "template_filename": TEMPLATE_FILE.name,
            "portal_mode": PORTAL_MODE,
            "requires_team_code": False if REQUIRE_EMAIL else team_code_required(),
            "requires_email": REQUIRE_EMAIL,
            "submission_count": len(
                [
                    path
                    for path in SUBMISSIONS_DIR.iterdir()
                    if is_submission_file(path)
                ]
            ),
            "last_updated": status.get("last_run_at"),
        }
        return {"submissions": submissions, "status": public_status, "dataset": dataset_meta}


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
    if PORTAL_MODE == "test":
        return JSONResponse(current_test_status_payload())
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
    team_name: str = Form(""),
    contact_email: str = Form(""),
    submission_code: str | None = Form(None),
    prediction_file: UploadFile = File(...),
) -> JSONResponse:
    ensure_directories()

    slug, display_name, email = resolve_submission_identity(team_name, submission_code, contact_email)

    suffix = Path(prediction_file.filename or "").suffix.lower()
    if suffix not in SUBMISSION_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Submission must be a .json or .jsonl file.")

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
            "display_name": display_name,
            "uploaded_at": utc_now_iso(),
            "original_filename": prediction_file.filename,
            "stored_filename": save_path.name,
        }
        if email:
            registry[slug]["contact_email"] = email
            registry[slug]["email_hash"] = hashlib.sha256(email.encode("utf-8")).hexdigest()
        save_registry(registry)
        if previous_filename and previous_filename != save_path.name:
            STORAGE.delete_remote_submission(previous_filename)
        STORAGE.upload_submission(save_path)
        STORAGE.upload_registry()

        if PORTAL_MODE == "test":
            validation = validate_submission_for_test_mode(content, suffix)
            save_json(OUTPUT_DIR / f"{slug}__format_validation.json", validation)
            result = run_evaluation()
            notification = None
            if email and not validation.get("valid_submission"):
                notification = notify_task1_format_issue(email, display_name, validation)
                save_json(OUTPUT_DIR / f"{slug}__format_notification.json", notification)
            status_payload = {
                "backend": STORAGE.backend_name,
                "mode": PORTAL_MODE,
                "last_run_at": utc_now_iso(),
                "last_run_ok": result["ok"],
                "returncode": result["returncode"],
                "message": "Submission saved, format-validated, and evaluated. Scores are hidden until the deadline."
                if result["ok"]
                else "Submission saved and format-validated, but organizer-side evaluation did not complete.",
            }
            save_watch_status(status_payload)
            STORAGE.upload_outputs()
            if not result["ok"]:
                return JSONResponse(
                    {
                        "message": "Submission received and format-validated, but organizer-side evaluation did not complete.",
                        "team_slug": slug,
                        "team_name": display_name,
                        "validation": validation,
                        "evaluation": {
                            "completed": False,
                            "last_updated": status_payload["last_run_at"],
                        },
                        "notification_sent": bool(notification and notification.get("sent")),
                    },
                    status_code=202,
                )
            return JSONResponse(
                {
                    "message": "Submission received. Format was validated and organizer-side evaluation completed; scores are hidden until the deadline.",
                    "team_slug": slug,
                    "team_name": display_name,
                    "validation": validation,
                    "evaluation": {
                        "completed": True,
                        "last_updated": status_payload["last_run_at"],
                    },
                    "notification_sent": bool(notification and notification.get("sent")),
                    "last_updated": status_payload["last_run_at"],
                }
            )

        result = run_evaluation()
        status_payload = {
            "backend": STORAGE.backend_name,
            "mode": PORTAL_MODE,
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
            "team_name": display_name,
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
            "portal_mode": PORTAL_MODE,
            "requires_team_code": False if REQUIRE_EMAIL else team_code_required(),
            "requires_email": REQUIRE_EMAIL,
            "devset_ready": DEVSET_FILE.exists(),
            "leaderboard_ready": (OUTPUT_DIR / "leaderboard_overall.csv").exists(),
            "official_site_url": OFFICIAL_SITE_URL,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("task1_dev_leaderboard.web_app:app", host="0.0.0.0", port=8091, reload=True)
