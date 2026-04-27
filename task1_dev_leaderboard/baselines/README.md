# Task 1 Baselines

Lightweight baselines for the currently public Task 1 dev leaderboards.

Current public languages:
- `English`
- `Chinese`
- `Arabic`
- `Hindi`

Files:
- `random_english_submission.jsonl`
- `random_chinese_submission.jsonl`
- `random_arabic_submission.jsonl`
- `random_hindi_submission.jsonl`
- `always_a_english_submission.jsonl`
- `always_a_chinese_submission.jsonl`
- `always_a_arabic_submission.jsonl`
- `always_a_hindi_submission.jsonl`
- `round_robin_english_submission.jsonl`
- `round_robin_chinese_submission.jsonl`
- `round_robin_arabic_submission.jsonl`
- `round_robin_hindi_submission.jsonl`
- `small_qwen_zero_shot_english_submission.jsonl`
- `small_qwen_zero_shot_chinese_submission.jsonl`
- `small_qwen_zero_shot_arabic_submission.jsonl`
- `small_qwen_zero_shot_hindi_submission.jsonl`

Baseline definitions:
- `Random Baseline`: chooses uniformly from the valid options shown for each item with a fixed seed.
- `Always A`: always predicts option `A` when it is available.
- `Round Robin`: cycles through the valid option letters in item order.
- `Qwen2.5-0.5B-Instruct Zero shot`: runs the small Qwen2.5 instruction model without task-specific fine-tuning.

Regenerate them from the repo root:

```bash
python task1_dev_leaderboard/create_random_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/english_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/random_english_submission.jsonl \
  --seed 2026

python task1_dev_leaderboard/create_random_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/chinese_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/random_chinese_submission.jsonl \
  --seed 2026

python task1_dev_leaderboard/create_random_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/random_arabic_submission.jsonl \
  --seed 2026

python task1_dev_leaderboard/create_random_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/random_hindi_submission.jsonl \
  --seed 2026

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/english_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/always_a_english_submission.jsonl \
  --strategy always_a

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/chinese_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/always_a_chinese_submission.jsonl \
  --strategy always_a

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/always_a_arabic_submission.jsonl \
  --strategy always_a

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/english_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/round_robin_english_submission.jsonl \
  --strategy round_robin

python task1_dev_leaderboard/create_rule_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/chinese_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/round_robin_chinese_submission.jsonl \
  --strategy round_robin

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
  --devset task1_dev_leaderboard/dev_sets/english_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/small_qwen_zero_shot_english_submission.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct

python task1_dev_leaderboard/create_local_zero_shot_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/chinese_finexam_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/small_qwen_zero_shot_chinese_submission.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct

python task1_dev_leaderboard/create_local_zero_shot_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/accounting_clef_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/small_qwen_zero_shot_arabic_submission.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct

python task1_dev_leaderboard/create_local_zero_shot_baseline.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --output task1_dev_leaderboard/baselines/small_qwen_zero_shot_hindi_submission.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct
```
