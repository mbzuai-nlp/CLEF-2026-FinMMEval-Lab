# Task 1 Random Baselines

Deterministic random-answer baselines for the current public Task 1 dev sets.

Files:
- `random_arabic_submission.jsonl`
- `random_hindi_submission.jsonl`

Regenerate them from the repo root:

```bash
python task1_dev_leaderboard/create_random_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/random_arabic_submission.jsonl \
  --seed 2026

python task1_dev_leaderboard/create_random_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/random_hindi_submission.jsonl \
  --seed 2026
```
