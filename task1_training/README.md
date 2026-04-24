# Task 1 Qwen Baseline

This directory contains a lightweight training pipeline for a public-data-only Task 1 baseline using:

- `Qwen/Qwen2.5-3B-Instruct`
- QLoRA / LoRA
- only public datasets from the FinMMEval collection

Recommended environment on this machine:

- Python: `/home/zhuohanx/.conda/envs/defame/bin/python`

## 1. Prepare public training data

```bash
/home/zhuohanx/.conda/envs/defame/bin/python task1_training/prepare_public_task1_data.py
```

This writes:

- `task1_training/artifacts/public_task1_train.jsonl`
- `task1_training/artifacts/public_task1_stats.json`

## 2. Train a Qwen2.5-3B LoRA baseline

```bash
/home/zhuohanx/.conda/envs/defame/bin/python task1_training/train_qwen_lora.py \
  --train-file task1_training/artifacts/public_task1_train.jsonl \
  --output-dir task1_training/artifacts/qwen25_3b_task1_lora \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --bf16 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2
```

For a short smoke test:

```bash
/home/zhuohanx/.conda/envs/defame/bin/python task1_training/train_qwen_lora.py \
  --train-file task1_training/artifacts/public_task1_train.jsonl \
  --output-dir task1_training/artifacts/qwen25_3b_task1_lora_smoke \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --bf16 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-train-examples 64 \
  --max-eval-examples 16 \
  --max-steps 2
```

## 3. Generate leaderboard submissions

```bash
/home/zhuohanx/.conda/envs/defame/bin/python task1_training/predict_qwen_lora.py \
  --devset task1_dev_leaderboard/dev_sets/english_cfa_cpa_100_public.jsonl \
  --adapter-dir task1_training/artifacts/qwen25_3b_task1_lora/final_adapter \
  --output task1_training/artifacts/english_qwen25_3b_predictions.jsonl
```

Then submit the generated JSONL to the Task 1 portal or evaluate locally with:

```bash
python task1_dev_leaderboard/evaluate_submissions.py \
  --gold-file task1_dev_leaderboard/private/english_cfa_cpa_100_gold.jsonl \
  --submissions-dir <dir-containing-jsonl> \
  --out-dir <eval-output-dir>
```
