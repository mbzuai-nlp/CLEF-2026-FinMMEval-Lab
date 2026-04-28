# Task 1 Dev Leaderboard

Local tooling for a Task 1 style development leaderboard.

Important policy note:
- The FinMMEval Hugging Face collection is the official public data release for Task 1. Participants may use those released datasets for training and may reorganize or re-split them as needed.
- The dev sets served by this leaderboard are separate organizer-held evaluation data. They are intended for validation only and should not be added back into training.
- The remaining hidden test sets are reserved for final evaluation.

The goal is to make it easy to:
- register one or more MCQ datasets
- register one or more prediction files from local model runs
- compute overall and subgroup accuracy
- export a lightweight local leaderboard
- deploy the Task 1 portal either locally or on Hugging Face Spaces

## Current Scope

This scaffold is deliberately conservative:
- it assumes Task 1 style multiple-choice data
- it does not require confidence scores
- it uses only the Python standard library

The included starter config uses a local Arabic MCQ set from `SAHM_private`:
- dataset: `SAHM_private/data/islamic_finance/fatwa/fatwa_mcq.jsonl`
- runs: `SAHM_private/runs/*_mcq/per_item_mcq.csv`

The loader also supports local parquet MCQ datasets, which is useful for private Hugging Face data drops such as `Raniahossam33/accounting_CLEF`.
There is also a ready-to-edit template at `task1_dev_leaderboard/configs/accounting_clef_template.json`.

## Quick Start

From the repo root:

```bash
python task1_dev_leaderboard/build_leaderboard.py \
  --config task1_dev_leaderboard/configs/sahm_local.json
```

Outputs go to:

```bash
task1_dev_leaderboard/outputs/sahm_local/
```

Main files:
- `leaderboard_overall.csv`
- `leaderboard_by_dataset.csv`
- `leaderboard_by_category.csv`
- `per_item_results.csv`
- `README.md`

## Released Dev Sets

At present, only the Hindi Task 1 dev leaderboard is public. English, Chinese, and Arabic dev leaderboards are temporarily hidden while the organizer side completes final data-pool review and split validation.

Public files:
- `task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl`
- `task1_dev_leaderboard/dev_sets/hindi_mcq_100_submission_template.json`

Organizer-private file:
- `task1_dev_leaderboard/private/hindi_mcq_100_gold.jsonl`

Sampling policy:
- deterministic
- seed = `2026`
- target size = `100`

### Participant Submission Format

Participants can submit either `.json` or `.jsonl`.

For public deployments, organizers may require a private team submission code. When enabled, the code determines the canonical team slug and display name used on the leaderboard.

Minimal per-item schema:

```json
{
  "id": "acct-dev-001",
  "prediction": "B"
}
```

Accepted prediction field names:
- `prediction`
- `pred_letter`
- `answer`
- `label`

For `.json`, the evaluator accepts:
- a list of objects
- an object with a `predictions` list
- a flat mapping from `id -> prediction`

### Organizer Evaluation Workflow

Put participant files under:

```bash
task1_dev_leaderboard/submissions/
```

Run:

```bash
python task1_dev_leaderboard/evaluate_submissions.py \
  --gold-file task1_dev_leaderboard/private/hindi_mcq_100_gold.jsonl \
  --submissions-dir task1_dev_leaderboard/submissions \
  --out-dir task1_dev_leaderboard/outputs/hindi_mcq_dev
```

This writes:
- `leaderboard_overall.csv`
- `leaderboard_by_source.csv`
- `per_item_results.csv`
- one `*__validation.json` per submission
- a rendered `README.md`

### Auto-Refresh Mode

If you want organizer-side near-real-time leaderboard refresh, run the watcher:

```bash
python task1_dev_leaderboard/watch_submissions.py \
  --gold-file task1_dev_leaderboard/private/hindi_mcq_100_gold.jsonl \
  --submissions-dir task1_dev_leaderboard/submissions \
  --out-dir task1_dev_leaderboard/outputs/hindi_mcq_dev \
  --run-on-start
```

Behavior:
- watches `submissions/` for new or updated `.json` / `.jsonl` files
- waits briefly for files to finish writing
- reruns evaluation automatically
- refreshes leaderboard outputs in place
- writes `watch_status.json` with watcher state and last run result

This gives you organizer-side automatic ranking updates. If you later want a public live page, you can point a static site, Hugging Face Space, or dashboard script at the files in `task1_dev_leaderboard/outputs/accounting_clef_dev/`.

## Web Portal

There is also a lightweight web app with:
- participant upload page
- leaderboard page
- upload API that immediately reruns evaluation

The portal now supports two storage modes:

- `local`: keep submissions, gold, and leaderboard files on the local filesystem
- `hf_dataset`: keep organizer-private state in a private Hugging Face dataset repo while serving the public web app from a Hugging Face Space

Start it from the repo root:

```bash
uvicorn task1_dev_leaderboard.web_app:app --host 0.0.0.0 --port 8091
```

Pages:
- `/task1/dev`
- `/task1/dev/submit`
- `/task1/dev/leaderboard`

Key API routes:
- `POST /api/task1/submissions`
- `GET /api/task1/leaderboard`
- `GET /api/task1/devset/download`
- `GET /api/task1/template/download`

Current behavior:
- each team uploads one `.json` or `.jsonl` file
- re-submitting from the same team replaces the previous file
- the leaderboard is refreshed immediately after a successful upload
- the leaderboard page polls the latest results automatically

### Hugging Face Storage Mode

Set these environment variables before starting the app:

```bash
export TASK1_STORAGE_BACKEND=hf_dataset
export TASK1_HF_REPO_ID=<org-or-user>/<private-dataset-repo>
export HF_TOKEN=<write-token>
```

Recommended layout:

- public Hugging Face Space for the web app
- private Hugging Face dataset repo for:
  - organizer gold file
  - participant submissions
  - leaderboard outputs

The public dev set and submission template remain in the app repo. The private gold file does not.

### Bootstrap The Private HF Dataset Repo

Create and initialize the organizer-private dataset repo:

```bash
python task1_dev_leaderboard/bootstrap_hf_backend.py \
  --repo-id <org-or-user>/<private-dataset-repo>
```

By default the dataset repo is created as private. This is the recommended setting.

### Prepare A Standalone HF Space Repo

Generate a clean Space folder that contains only the files needed by the portal:

```bash
python task1_dev_leaderboard/prepare_hf_space.py
```

This creates:

```bash
task1_dev_leaderboard_hf_space/
```

That folder contains:

- a Docker Space `README.md`
- a `Dockerfile`
- a minimal `requirements.txt`
- the public `task1_dev_leaderboard` app package

Push that generated folder to a new Hugging Face Space repo.

### Hugging Face Space Runtime Notes

- use a `Docker Space`
- set the Space secret `HF_TOKEN`
- set the Space variables:
  - `TASK1_STORAGE_BACKEND=hf_dataset`
  - `TASK1_HF_REPO_ID=<org-or-user>/<private-dataset-repo>`
- the app runs fine on free CPU hardware
- the private dataset repo should remain private because it stores gold labels and submissions

## Supported Input Formats

### Dataset format: `mcq_jsonl`

Expected JSONL fields by default:

```json
{
  "id": 0,
  "question": "...",
  "option_A": "...",
  "option_B": "...",
  "option_C": "...",
  "option_D": "...",
  "answer": "A",
  "category": "..."
}
```

You can override field names in the config.

### Dataset format: `mcq_parquet`

Useful when the dataset comes as a parquet file. Example schema:

```json
{
  "id": 1,
  "question": "...",
  "choice_a": "...",
  "choice_b": "...",
  "choice_c": "...",
  "choice_d": "...",
  "choice_e": "...",
  "correct_answer": "c"
}
```

Map the fields in config, for example:

```json
{
  "key": "accounting_clef_ar",
  "format": "mcq_parquet",
  "path": "../data/accounting_CLEF/data/train-00000-of-00001.parquet",
  "answer_field": "correct_answer",
  "option_fields": {
    "A": "choice_a",
    "B": "choice_b",
    "C": "choice_c",
    "D": "choice_d",
    "E": "choice_e"
  }
}
```

### Prediction format: `per_item_mcq_csv`

Expected CSV fields by default:

```text
id,category,gold_letter,pred_letter,is_correct,raw_output
```

Only `id` and `pred_letter` are required for scoring.

### Prediction format: `task1_jsonl`

Useful for future Task 1 runs. Each line should contain:

```json
{
  "id": "...",
  "prediction": "B"
}
```

Accepted prediction field candidates by default:
- `prediction`
- `pred_letter`
- `answer`
- `answer_label`
- `label`

## Config Shape

```json
{
  "datasets": [
    {
      "key": "dataset_key",
      "name": "Human-readable name",
      "format": "mcq_jsonl",
      "path": "relative/or/absolute/path.jsonl",
      "language": "en"
    }
  ],
  "runs": [
    {
      "model_name": "model_x",
      "format": "per_item_mcq_csv",
      "path": "relative/or/absolute/path.csv",
      "datasets": ["dataset_key"]
    }
  ]
}
```

Paths are resolved relative to the config file.

## Recommended Next Step

As more internal Task 1-style datasets become available, add them as new dataset entries instead of hardcoding logic into the script. That keeps the leaderboard reproducible and makes it easier to separate:
- official-like dev sets
- auxiliary private sets
- language-specific stress tests
