# Task 1 Baselines

Lightweight baselines for the currently public Task 1 dev leaderboards.

Current public languages:
- `Arabic`
- `Hindi`

Files:
- `random_arabic_submission.jsonl`
- `random_hindi_submission.jsonl`
- `always_a_arabic_submission.jsonl`
- `always_a_hindi_submission.jsonl`
- `round_robin_arabic_submission.jsonl`
- `round_robin_hindi_submission.jsonl`
- `small_qwen_zero_shot_arabic_submission.jsonl`
- `small_qwen_zero_shot_hindi_submission.jsonl`

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

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/always_a_hindi_submission.jsonl \
  --strategy always_a

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/round_robin_hindi_submission.jsonl \
  --strategy round_robin

python task1_dev_leaderboard/create_local_zero_shot_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/small_qwen_zero_shot_arabic_submission.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct

python task1_dev_leaderboard/create_local_zero_shot_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/small_qwen_zero_shot_hindi_submission.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct
```
