#!/usr/bin/env python3
"""Generate Task 1 predictions from a trained Qwen LoRA adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LETTER_RE = re.compile(r"\b([A-Z])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task 1 predictions with a Qwen LoRA adapter.")
    parser.add_argument("--devset", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row: dict) -> str:
    options = row["options"]
    option_lines = [f"{letter}. {options[letter]}" for letter in sorted(options.keys())]
    return (
        "Read the following multiple-choice question and return only the correct option letter.\n\n"
        f"Question:\n{row['question']}\n\nOptions:\n" + "\n".join(option_lines) + "\n\nAnswer:"
    )


def parse_prediction(text: str, valid_letters: set[str]) -> str:
    cleaned = text.strip().upper()
    if cleaned in valid_letters:
        return cleaned
    match = LETTER_RE.search(cleaned)
    if match and match.group(1) in valid_letters:
        return match.group(1)
    raise ValueError(f"Could not parse a valid option letter from: {text!r}")


def main() -> None:
    args = parse_args()
    devset_path = Path(args.devset).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    output_path = Path(args.output).resolve()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    predictions = []
    for row in load_jsonl(devset_path):
        prompt = build_prompt(row)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = parse_prediction(generated, {key.upper() for key in row["options"].keys()})
        predictions.append({"id": row["id"], "prediction": prediction})

    write_jsonl(output_path, predictions)
    print(f"Wrote {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
