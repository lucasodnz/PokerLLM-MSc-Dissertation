#!/usr/bin/env python3
"""
hybrid_evaluator.py

Hybrid action-and-sizing evaluator for LoRA-adapted or few-shot-prompted
causal language models on textual poker decision states:

  1. The action (fold / call / check / bet / raise) is selected by comparing
     the next-token log-probability of each available action -- no text is
     generated for this step.
  2. Only when the selected or reference action is aggressive (bet/raise) is
     a short deterministic generation used to produce the numeric size.
  3. Predictions are scored with the dataset-agnostic functions in
     metrics.py. The headline, reported metrics are the continuous
     action-and-sizing score (Ac-s) and the conditional sizing score;
     exact-match rate and BB-tolerance bands are additional, non-headline
     diagnostics retained for internal analysis only.

This file owns parsing, prompting, model loading, and I/O. metrics.py owns
only the scoring math and has no dependency on this file, PokerBench, or
Transformers/torch.

Usage:
  python evaluation/hybrid_evaluator.py \\
      --mode lora --adapter /path/to/adapter \\
      --dataset examples/preflop.json --predictions-csv out.csv --metrics-json out.json

  python evaluation/hybrid_evaluator.py \\
      --mode fewshot --train-dataset /path/to/train.json \\
      --dataset examples/postflop.json --predictions-csv out.csv --metrics-json out.json
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics

BASE_ACTIONS_PREFLOP = ["fold", "call", "check", "raise"]
BASE_ACTIONS_POSTFLOP = ["fold", "call", "check", "bet", "raise"]
AGGRESSIVE_ACTIONS = ("bet", "raise")
ANSWER_CUE = "Your optimal action is:"


# ============================================================================
# Parsing (raw instruction / gold label / generated text -> structured values)
# ============================================================================

def extract_base_action(text: str) -> str:
    text = text.strip().lower()
    words = text.split()
    first = words[0] if words else ""
    if first in BASE_ACTIONS_POSTFLOP:
        return first
    for action in BASE_ACTIONS_POSTFLOP:
        if text.startswith(action):
            return action
    return first or "unknown"


def parse_label(label: str) -> Tuple[str, Optional[float]]:
    """Parses a gold label such as 'raise 12.5' into (action, size)."""
    label = label.strip().lower()
    action = extract_base_action(label)
    size = None
    if action in AGGRESSIVE_ACTIONS:
        parts = label.split()
        if len(parts) >= 2:
            try:
                size = float(parts[1])
            except ValueError:
                size = None
    return action, size


def parse_sizing_value(generated_text: str) -> Optional[float]:
    """Extracts the first numeric value from a short sizing-generation output."""
    match = re.search(r"(\d+(?:\.\d+)?)", generated_text.strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_position_from_instruction(instruction: str) -> str:
    match = re.search(r"your position is\s+(UTG|HJ|CO|BTN|SB|BB)", instruction, re.IGNORECASE)
    return match.group(1).upper() if match else "unknown"


def extract_street_from_instruction(instruction: str) -> str:
    text = instruction.split("Now it is your turn")[0] if "Now it is your turn" in instruction else instruction
    patterns = {"river": r"[Tt]he river comes", "turn": r"[Tt]he turn comes", "flop": r"[Tt]he flop comes"}
    last_positions = {}
    for street, pat in patterns.items():
        found = list(re.finditer(pat, text))
        if found:
            last_positions[street] = found[-1].start()
    if last_positions:
        return max(last_positions, key=last_positions.get)
    if "before the flop" in text.lower() or "preflop" in text.lower():
        return "preflop"
    return "unknown"


def infer_available_actions(instruction: str, base_actions: List[str]) -> List[str]:
    """Restricts the action set if the instruction text explicitly lists one
    (e.g. 'Available actions: fold, call, raise' or the equivalent 'Legal
    actions:' wording). Falls back to the full base action set otherwise --
    this is the common case for PokerBench-style prompts, which do not
    include an explicit action list."""
    text = instruction.lower()
    for marker in ("available actions:", "legal actions:"):
        if marker in text:
            tail = text.split(marker)[-1].split("\n")[0]
            available = [a for a in base_actions if a in tail]
            if available:
                return available
    return base_actions


# ============================================================================
# Prompting
# ============================================================================

def build_prompt_for_logprob(instruction: str) -> str:
    """Appends the answer cue unconditionally. PokerBench-style instructions
    already end with this cue, so the resulting prompt intentionally contains
    it twice -- this matches the exact construction the reference adapters
    were trained and evaluated against; changing it would not reproduce the
    published numbers."""
    return f"{instruction.strip()}\n{ANSWER_CUE}"


def build_prompt_for_sizing(instruction: str, action: str) -> str:
    return (
        f"{instruction.strip()}\n"
        f"{ANSWER_CUE} {action}\n"
        "Return only the size in chips as a number (e.g., 10 or 12.5). No words.\n"
        "Size:"
    )


def sample_fewshot_examples(train_data: List[dict], seed: int, base_actions: List[str]) -> List[dict]:
    """Samples one example per action class for a few-shot prompt."""
    rng = random.Random(seed)
    examples = []
    for action in base_actions:
        candidates = [ex for ex in train_data if ex["output"].strip().lower().startswith(action)]
        if candidates:
            examples.append(rng.choice(candidates))
    return examples


def build_prompt_fewshot_logprob(instruction: str, fewshot_examples: List[dict]) -> str:
    prompt = ""
    for ex in fewshot_examples:
        action = extract_base_action(ex["output"])
        prompt += f"{ex['instruction'].strip()}\n{ANSWER_CUE} {action}\n\n"
    return prompt + f"{instruction.strip()}\n{ANSWER_CUE}"


def build_prompt_fewshot_sizing(instruction: str, action: str, fewshot_examples: List[dict]) -> str:
    prompt = ""
    for ex in fewshot_examples:
        if not ex["output"].strip().lower().startswith(action):
            continue
        parts = ex["output"].strip().split()
        size = parts[1] if len(parts) >= 2 else "10"
        prompt += f"{ex['instruction'].strip()}\n{ANSWER_CUE} {action}\nSize: {size}\n\n"
    prompt += (
        f"{instruction.strip()}\n{ANSWER_CUE} {action}\n"
        "Return only the size in chips as a number (e.g., 10 or 12.5). No words.\nSize:"
    )
    return prompt


# ============================================================================
# Model inference (torch/transformers imported lazily -- not required to
# import or test the rest of this module)
# ============================================================================

def get_best_action_logprob(prompt: str, model, tokenizer, available_actions: List[str]) -> str:
    """Classifies the action via next-token log-probability over the
    available action set. No text is generated."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    scores = {}
    for action in available_actions:
        token_ids = tokenizer.encode(f" {action}", add_special_tokens=False) or \
                    tokenizer.encode(action, add_special_tokens=False)
        if token_ids:
            scores[action] = log_probs[token_ids[0]].item()
    return max(scores, key=scores.get)


def generate_size(prompt: str, model, tokenizer, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def load_model(model_name: str, mode: str, adapter_path: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    if mode == "lora":
        from peft import PeftModel
        if not adapter_path:
            raise ValueError("--adapter is required when --mode lora")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()
    return model, tokenizer


# ============================================================================
# Resumable, deduplicated CSV output
# ============================================================================

def make_sample_id(instruction: str, gold_label: str) -> str:
    """Stable identifier for one (instruction, gold label) pair: SHA-256 of the
    stripped instruction and the normalized gold label. Used to detect rows a
    previous run already completed, so a rerun can resume without
    duplicating them."""
    payload = (instruction.strip() + "\x1f" + gold_label.strip().lower()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_existing_sample_ids(predictions_csv_path: str) -> set:
    """Reads sample_id values already present in an existing predictions CSV.
    Returns an empty set if the file does not exist yet."""
    ids = set()
    if not os.path.isfile(predictions_csv_path):
        return ids
    with open(predictions_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("sample_id")
            if sid:
                ids.add(sid)
    return ids


class IncrementalCsvWriter:
    """Writes predictions.csv one row at a time, flushing and fsyncing after
    each row. If the file already has content, new rows are appended instead
    of overwriting it -- combined with make_sample_id / load_existing_sample_ids,
    this is what makes an interrupted run resumable without duplicate rows.
    The header is written only when the file is created, never on append."""

    def __init__(self, path: str, fieldnames: List[str]):
        file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
        self._fh = open(path, "a" if file_exists else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()
            self._fh.flush()

    def write_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_predictions_csv(path: str) -> List["metrics.SampleResult"]:
    """Reads this evaluator's own predictions CSV back into SampleResult
    objects, so aggregate metrics are computed over the complete file --
    including rows written by an earlier, interrupted run -- rather than
    only the rows produced in the current pass."""
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold_action = row["gold_action"]
            samples.append(metrics.SampleResult(
                action_correct=row["action_correct"] == "1",
                gold_is_aggressive=gold_action in AGGRESSIVE_ACTIONS,
                gold_size=_parse_optional_float(row.get("gold_size")),
                pred_size=_parse_optional_float(row.get("pred_size")),
            ))
    return samples


# ============================================================================
# Main
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["lora", "fewshot"], default="lora")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--adapter", default=None, help="Path to a LoRA adapter (required for --mode lora)")
    p.add_argument("--dataset", required=True, help="Path to a JSON list of {instruction, output}")
    p.add_argument("--train-dataset", default=None, help="Training set to sample few-shot examples from (--mode fewshot)")
    p.add_argument("--stage", choices=["preflop", "postflop"], default=None,
                   help="Overrides stage inference from the dataset filename.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens-sizing", type=int, default=10)
    p.add_argument("--predictions-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default="metrics.json")
    return p


def main():
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples:
        data = data[: args.max_samples]

    if args.stage:
        stage = args.stage
    else:
        stage = "postflop" if "postflop" in args.dataset.lower() else "preflop"
        print(f"[warn] --stage not given -- inferring stage='{stage}' from the dataset "
              f"filename ({args.dataset}). Pass --stage explicitly to avoid relying on this.",
              file=sys.stderr)
    base_actions = BASE_ACTIONS_POSTFLOP if stage == "postflop" else BASE_ACTIONS_PREFLOP

    fewshot_examples = []
    if args.mode == "fewshot":
        if not args.train_dataset:
            raise ValueError("--train-dataset is required when --mode fewshot")
        with open(args.train_dataset, encoding="utf-8") as f:
            train_data = json.load(f)
        fewshot_examples = sample_fewshot_examples(train_data, args.seed, base_actions)

    model, tokenizer = load_model(args.model, args.mode, args.adapter)

    fieldnames = ["sample_idx", "sample_id", "position", "street", "gold_action", "gold_size",
                  "pred_action", "pred_size", "action_correct", "ac_s_i"]

    existing_ids = load_existing_sample_ids(args.predictions_csv)
    if existing_ids:
        print(f"[resume] {len(existing_ids)} samples already present in {args.predictions_csv} "
              "-- skipping those.")
    writer = IncrementalCsvWriter(args.predictions_csv, fieldnames)

    n_processed_this_run = 0
    for i, item in enumerate(data):
        instruction = item["instruction"]
        sample_id = make_sample_id(instruction, item["output"])
        if sample_id in existing_ids:
            continue

        gold_action, gold_size = parse_label(item["output"])
        position = extract_position_from_instruction(instruction)
        street = extract_street_from_instruction(instruction)

        available_actions = (infer_available_actions(instruction, base_actions)
                              if stage == "postflop" else base_actions)

        if args.mode == "lora":
            prompt = build_prompt_for_logprob(instruction)
        else:
            prompt = build_prompt_fewshot_logprob(instruction, fewshot_examples)
        pred_action = get_best_action_logprob(prompt, model, tokenizer, available_actions)

        pred_size = None
        if gold_action in AGGRESSIVE_ACTIONS or pred_action in AGGRESSIVE_ACTIONS:
            sizing_prompt = (build_prompt_for_sizing(instruction, pred_action) if args.mode == "lora"
                              else build_prompt_fewshot_sizing(instruction, pred_action, fewshot_examples))
            generated = generate_size(sizing_prompt, model, tokenizer, args.max_new_tokens_sizing)
            pred_size = parse_sizing_value(generated)

        action_correct = pred_action == gold_action
        gold_is_aggressive = gold_action in AGGRESSIVE_ACTIONS
        ac_s_i = metrics.compute_continuous_acs(action_correct, gold_is_aggressive, gold_size, pred_size)

        writer.write_row({
            "sample_idx": i, "sample_id": sample_id, "position": position, "street": street,
            "gold_action": gold_action, "gold_size": gold_size,
            "pred_action": pred_action, "pred_size": pred_size,
            "action_correct": int(action_correct), "ac_s_i": round(ac_s_i, 6),
        })
        existing_ids.add(sample_id)
        n_processed_this_run += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(data)} processed", flush=True)

    writer.close()
    print(f"{n_processed_this_run} new predictions written this run.")

    summary = metrics.summarize(read_predictions_csv(args.predictions_csv))
    summary["stage"] = stage
    summary["mode"] = args.mode
    summary["model"] = args.model
    summary["adapter"] = args.adapter

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Ac={summary['accuracy_action']:.4f} "
          f"Ac-s(continuous)={summary['acs_continuous_mean']:.4f} "
          f"conditional_sizing_score={summary['conditional_sizing_score']}")
    print(f"Predictions: {args.predictions_csv}")
    print(f"Metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
