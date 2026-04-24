# Task 1 Baselines

Deterministic baselines for the current public Task 1 dev sets.

Files:
- `random_arabic_submission.jsonl`
- `random_hindi_submission.jsonl`
- `random_english_submission.jsonl`
- `random_chinese_submission.jsonl`
- `always_a_<language>_submission.jsonl`
- `round_robin_<language>_submission.jsonl`

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

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/always_a_arabic_submission.jsonl \
  --strategy always_a

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/round_robin_arabic_submission.jsonl \
  --strategy round_robin
```
