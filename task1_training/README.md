# Task 1 LoRA Baselines

This directory contains a lightweight training pipeline for public-data-only Task 1 baselines using:

- `Qwen/Qwen2.5-3B-Instruct`
- `mistralai/Ministral-3-3B-Instruct-2512`
- QLoRA / LoRA
- only public datasets from the FinMMEval collection

Policy note:

- The FinMMEval Hugging Face collection is the official public data release for Task 1.
- The original split names on dataset cards do not restrict participant usage; the released collection may be reorganized or re-split for training.
- The separate Task 1 leaderboard dev sets are organizer-held evaluation data and should not be mixed back into training.

Recommended environment on this machine:

- Python: `/home/zhuohanx/.conda/envs/finance/bin/python`

## 1. Prepare public training data

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/prepare_public_task1_data.py
```

This writes:

- `task1_training/artifacts/public_task1_train.jsonl`
- `task1_training/artifacts/public_task1_stats.json`

## 2. Train a Qwen2.5-3B LoRA baseline

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/train_qwen_lora.py \
  --train-file task1_training/artifacts/public_task1_train.jsonl \
  --output-dir task1_training/artifacts/qwen25_3b_task1_lora \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --bf16 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2
```

Official organizer baseline configuration used for the current public Task 1 leaderboards:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning: LoRA
- Precision: `bf16`
- Epochs: `2`
- Training source: only the official public FinMMEval Hugging Face collection
- Excluded data: all organizer-held Task 1 leaderboard dev sets and all hidden test sets

## 3. Train a Ministral 3 3B LoRA baseline

`mistralai/Ministral-3-3B-Instruct-2512` requires a recent Transformers version with Mistral3 support. On this machine, the `finance` environment has been upgraded for this run; keep it separate from any vLLM-serving environment if dependency constraints matter.

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/train_qwen_lora.py \
  --train-file task1_training/artifacts/public_task1_train.jsonl \
  --output-dir task1_training/artifacts/ministral3_3b_2512_task1_lora \
  --model-name mistralai/Ministral-3-3B-Instruct-2512 \
  --bf16 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100
```

Current Hindi dev result from the public-data-only LoRA run:

- Model: `mistralai/Ministral-3-3B-Instruct-2512`
- Adapter: `task1_training/artifacts/ministral3_3b_2512_task1_lora/final_adapter`
- Training loss: `0.5216`
- Final validation loss: `0.6166`
- Hindi dev accuracy: `61/100 = 61%`

For comparison, the existing Qwen2.5-3B public baseline file scored `45/100 = 45%` on the same Hindi dev set.

For a short smoke test:

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/train_qwen_lora.py \
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

## 4. Generate leaderboard submissions

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/predict_qwen_lora.py \
  --devset task1_dev_leaderboard/dev_sets/hindi_mcq_100_public.jsonl \
  --adapter-dir task1_training/artifacts/ministral3_3b_2512_task1_lora/final_adapter \
  --base-model mistralai/Ministral-3-3B-Instruct-2512 \
  --output task1_training/artifacts/ministral3_3b_2512_hindi_predictions.jsonl
```

Then submit the generated JSONL to the Task 1 portal or evaluate locally with:

```bash
python task1_dev_leaderboard/evaluate_submissions.py \
  --gold-file task1_dev_leaderboard/private/english_cfa_cpa_100_gold.jsonl \
  --submissions-dir <dir-containing-jsonl> \
  --out-dir <eval-output-dir>
```

## 5. Release safety check

Before preparing a public release or Hugging Face Space folder, run:

```bash
python task1_dev_leaderboard/check_public_release.py
```

The check fails if organizer-private paths such as `private/`, `*_gold.jsonl`, or `*hidden_test*` are tracked or otherwise unignored.
