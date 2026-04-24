#!/usr/bin/env python3
"""Create a local small-LLM zero-shot baseline submission for Task 1 dev sets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTER_RE = re.compile(r"\b([A-Z])\b")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choice_letters(row: dict) -> list[str]:
    options = row.get("options")
    if isinstance(options, dict) and options:
        letters = [str(key).strip().upper() for key in options.keys() if str(key).strip()]
        filtered = sorted(set(letter for letter in letters if len(letter) == 1 and letter.isalpha()))
        if filtered:
            return filtered

    count = int(row.get("num_choices", 0) or 0)
    if count > 0:
        return [chr(ord("A") + index) for index in range(count)]

    raise ValueError(f"Row {row.get('id', '<unknown>')} does not define answer choices.")


def build_prompt(row: dict) -> str:
    letters = choice_letters(row)
    options = row["options"]
    option_lines = [f"{letter}. {options[letter]}" for letter in letters]
    return (
        "Answer this multiple-choice finance question.\n"
        "Return only one option letter.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )


def parse_prediction(text: str, letters: list[str]) -> str:
    cleaned = text.strip().upper()
    if cleaned in letters:
        return cleaned
    match = LETTER_RE.search(cleaned)
    if match and match.group(1) in letters:
        return match.group(1)
    return letters[0]


def model_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local small-Qwen zero-shot baseline submission.")
    parser.add_argument("--devset", required=True, help="Public dev set JSONL file.")
    parser.add_argument("--output", required=True, help="Output submission JSONL file.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Transformers model id for local zero-shot inference.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Maximum generation length.")
    args = parser.parse_args()

    devset_path = Path(args.devset).resolve()
    output_path = Path(args.output).resolve()
    rows = load_jsonl(devset_path)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto" if model_device() != "cpu" else None,
    )
    if model_device() == "cpu":
        model.to("cpu")
    model.eval()

    predictions: list[dict] = []
    for row in rows:
        letters = choice_letters(row)
        messages = [
            {
                "role": "system",
                "content": "You solve multilingual finance multiple-choice questions. Reply with exactly one valid option letter.",
            },
            {"role": "user", "content": build_prompt(row)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append({"id": row["id"], "prediction": parse_prediction(text, letters)})

    write_jsonl(output_path, predictions)
    print(f"Wrote {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
