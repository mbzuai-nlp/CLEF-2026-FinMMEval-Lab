#!/usr/bin/env python3
"""Train a Task 1 LoRA baseline with Transformers Trainer."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from transformers import FineGrainedFP8Config
except ImportError:  # Older Transformers versions do not expose FP8 dequantization.
    FineGrainedFP8Config = None

try:
    from transformers import Mistral3ForConditionalGeneration
except ImportError:  # Older Transformers versions do not support Ministral 3.
    Mistral3ForConditionalGeneration = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA baseline for Task 1.")
    parser.add_argument("--train-file", required=True, help="Prepared JSONL training file.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and adapter.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    return parser.parse_args()


def load_prepared_dataset(path: Path) -> tuple[Dataset, Dataset]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "validation"]
    return Dataset.from_pandas(pd.DataFrame(train_rows)), Dataset.from_pandas(pd.DataFrame(eval_rows))


def build_messages(example: dict) -> tuple[list[dict], list[dict]]:
    prompt_messages = [{"role": "user", "content": example["prompt"]}]
    full_messages = prompt_messages + [{"role": "assistant", "content": example["answer"]}]
    return prompt_messages, full_messages


def preprocess_dataset(tokenizer: AutoTokenizer, dataset: Dataset, max_seq_length: int) -> Dataset:
    def _process(example: dict) -> dict:
        prompt_messages, full_messages = build_messages(example)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_length)
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(_process, remove_columns=dataset.column_names)


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            pad_len = max_len - len(f["labels"])
            labels.append(f["labels"] + [-100] * pad_len)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_file = Path(args.train_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, eval_dataset = load_prepared_dataset(train_file)
    if args.max_train_examples:
        train_dataset = train_dataset.select(range(min(args.max_train_examples, len(train_dataset))))
    if args.max_eval_examples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_examples, len(eval_dataset))))

    tokenizer_kwargs = {}
    if "ministral-3" in args.model_name.lower():
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = preprocess_dataset(tokenizer, train_dataset, args.max_seq_length)
    eval_dataset = preprocess_dataset(tokenizer, eval_dataset, args.max_seq_length)

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    config = AutoConfig.from_pretrained(args.model_name)
    model_cls = AutoModelForCausalLM
    if getattr(config, "model_type", None) == "mistral3":
        if Mistral3ForConditionalGeneration is None:
            raise RuntimeError(
                "This model requires a Transformers version with Mistral3ForConditionalGeneration."
            )
        model_cls = Mistral3ForConditionalGeneration
        if quantization_config is None:
            if FineGrainedFP8Config is None:
                raise RuntimeError(
                    "This FP8 Ministral 3 checkpoint requires FineGrainedFP8Config(dequantize=True)."
                )
            quantization_config = FineGrainedFP8Config(dequantize=True)

    model = model_cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
    )
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "steps" if len(eval_dataset) else "no"
    elif "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "steps" if len(eval_dataset) else "no"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
