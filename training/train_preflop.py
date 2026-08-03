#!/usr/bin/env python3
"""
train_preflop.py

Supervised fine-tuning of Qwen3-14B (QLoRA, 4-bit NF4) for preflop poker
decisions, with the action and the numeric raise size jointly supervised
under a response-only loss: every prompt token is masked (label = -100),
and only the answer tokens ("fold", "call", "check", or "raise X") receive
gradient. If the combined prompt+answer length would exceed --max-len, the
prompt is truncated from the left (its most recent, most relevant tokens
are kept) so the answer is never clipped.

A label-masking audit runs before training starts (and can be run alone via
--audit-only, without loading the model): it confirms that at least one
aggressive (raise) example has its numeric size unmasked, that no example's
answer is fully masked away, that no prompt or padding token is unmasked,
and that padding never leaks into the supervised region.

Usage:
  python train_preflop.py --dataset-path data.json --output-dir out/
  python train_preflop.py --dataset-path data.json --output-dir out/ --audit-only
  python train_preflop.py --dataset-path data.json --output-dir out/ --smoke
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer, TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen3-14B"
SEED = 42

MAX_LEN = 1024
LR = 1e-4
EPOCHS = 2
BATCH = 2
GRAD_ACCUM = 8
WARMUP_RATIO = 0.03
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

AGGRESSIVE_ACTIONS = {"raise"}
ANSWER_CUE = "Your optimal action is:"

PROMPT_HEADER = (
    "You are a specialist in playing 6-handed No Limit Texas Hold'em.\n"
    "For each hand history below, decide the optimal preflop action. "
    'Respond with fold, call, or check. For a raise, respond with "raise X",\n'
    "where X is the total raise size in chips. Do not explain your answer.\n\n"
    "Now consider the following scenario:\n"
)


def build_prompt(instruction: str) -> str:
    """PROMPT_HEADER + instruction. Appends the answer cue only if the
    instruction does not already end with it, to avoid duplicating it.
    Note: evaluation/hybrid_evaluator.py appends the cue unconditionally,
    so at inference time it is intentionally doubled -- that mismatch is
    part of the canonical protocol this adapter was trained and reported
    under, and is preserved here for reproducibility rather than "fixed"."""
    instr = instruction.strip()
    prompt = PROMPT_HEADER + instr
    if not instr.endswith(ANSWER_CUE):
        prompt += "\n" + ANSWER_CUE
    return prompt


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_answer(answer: str):
    parts = answer.strip().split()
    action = parts[0].lower()
    size = float(parts[1]) if len(parts) > 1 else None
    return action, size


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True,
                   help="JSON file: a list of {instruction, output}. 'output' is 'fold'/'call'/'check' "
                        "or 'raise X' (X = numeric size).")
    p.add_argument("--output-dir", required=True, help="Adapter output directory (created if missing)")
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--smoke", action="store_true", help="Quick run on a small slice, for pipeline validation")
    p.add_argument("--max-train-samples", type=int, default=0, help="Limit training examples (0 = all)")
    p.add_argument("--max-eval-samples", type=int, default=0, help="Internal held-out eval sample count (0 = none)")
    p.add_argument("--max-steps", type=int, default=-1, help="Limit optimizer steps (-1 = use --epochs)")
    p.add_argument("--epochs", type=float, default=EPOCHS)
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in --output-dir")
    p.add_argument("--audit-only", action="store_true",
                   help="Run only the label-masking audit and exit, without loading the model.")
    p.add_argument("--skip-dataset-size-checks", action="store_true",
                   help="Skip the exact-count assertions below, for datasets of a different size.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] model={MODEL_ID} dataset={args.dataset_path} output_dir={output_dir}")
    print(f"[config] smoke={args.smoke} max_train_samples={args.max_train_samples} "
          f"max_eval_samples={args.max_eval_samples} max_steps={args.max_steps} resume={args.resume}")

    dataset_sha256 = sha256_of(args.dataset_path)
    print(f"[data] sha256={dataset_sha256}")

    with open(args.dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    parsed = [dict(instruction=x["instruction"], answer=x["output"].strip()) for x in raw_data]
    actions = [parse_answer(x["answer"])[0] for x in parsed]
    n_total = len(parsed)
    n_raise = sum(1 for a in actions if a == "raise")
    print(f"[data] n_total={n_total} n_raise={n_raise} ({n_raise / n_total * 100:.2f}%)")

    if not args.skip_dataset_size_checks:
        assert n_total == 63200, (
            f"expected 63200 examples (the canonical PokerBench preflop training set), got {n_total}. "
            "Pass --skip-dataset-size-checks if you intend to train on a different dataset size."
        )
        assert 7800 <= n_raise <= 8200, f"expected ~8000 raises, got {n_raise}"

    # Confirms the dataset carries full sizing labels (not an action-only variant).
    n_with_size = sum(1 for x in parsed if len(x["answer"].split()) > 1)
    assert n_with_size == n_raise, (
        f"dataset does not have sizing on every raise example (n_with_size={n_with_size} != "
        f"n_raise={n_raise}) -- this looks like an action-only dataset variant, not one with full sizing."
    )

    random.shuffle(parsed)

    eval_data: List[Dict] = []
    train_data = parsed
    if args.max_eval_samples > 0:
        n_eval = min(args.max_eval_samples, max(1, n_total // 50))
        eval_data = parsed[:n_eval]
        train_data = parsed[n_eval:]
        print(f"[data] internal eval split: {len(eval_data)} examples (not the official test set)")

    if args.smoke:
        n_smoke = args.max_train_samples if args.max_train_samples > 0 else 200
        train_data = train_data[:n_smoke]
        if not eval_data:
            eval_data = parsed[-min(32, len(parsed)):]
        print(f"[smoke] using {len(train_data)} train examples, {len(eval_data)} eval examples")
    elif args.max_train_samples > 0:
        train_data = train_data[: args.max_train_samples]
        print(f"[data] limited by --max-train-samples: {len(train_data)} train examples")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build_record(instr: str, answer: str, max_len: int):
        """labels = [-100]*len(prompt) + answer_ids, with the answer never
        truncated (the prompt is truncated from the left if needed)."""
        prompt = build_prompt(instr)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(prompt + " " + answer, add_special_tokens=False)["input_ids"]
        answer_ids = full_ids[len(prompt_ids):]

        if len(answer_ids) == 0:
            # Rare BPE-merge edge case at the prompt/answer boundary; fall
            # back to tokenizing the answer in isolation.
            answer_ids = tok(" " + answer, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) + len(answer_ids) > max_len:
            keep_prompt = max(0, max_len - len(answer_ids))
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            if keep_prompt == 0:
                answer_ids = answer_ids[-max_len:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids[:]
        attn = [1] * len(input_ids)

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [tok.pad_token_id] * pad_len
            attn += [0] * pad_len
            labels += [-100] * pad_len
        else:
            input_ids, attn, labels = input_ids[:max_len], attn[:max_len], labels[:max_len]

        return input_ids, attn, labels, prompt

    # ------------------------------------------------------------------
    # Label-masking audit (always runs; required before training starts)
    # ------------------------------------------------------------------
    audit_pool = train_data if train_data else parsed
    raise_examples = [x for x in audit_pool if parse_answer(x["answer"])[0] == "raise"][:2]
    other_examples = [x for x in audit_pool if parse_answer(x["answer"])[0] != "raise"][:3]
    audit_examples = (raise_examples + other_examples)[:5] or audit_pool[:5]

    at_least_one_aggressive_unmasked = False
    for ex in audit_examples:
        ids, attn, labels, prompt = build_record(ex["instruction"], ex["answer"], args.max_len)
        supervised_ids = [t for t, l in zip(ids, labels) if l != -100]
        supervised_text = tok.decode(supervised_ids)
        action, size = parse_answer(ex["answer"])
        if action == "raise" and any(ch.isdigit() for ch in supervised_text):
            at_least_one_aggressive_unmasked = True

    assert at_least_one_aggressive_unmasked, (
        "No aggressive (raise) example had its numeric size outside the -100 mask -- "
        "sizing would receive no gradient."
    )

    check_pool = audit_pool[: min(200, len(audit_pool))]
    n_fully_masked = n_pad_leak = n_prompt_leak = 0
    for ex in check_pool:
        ids, attn, labels, _ = build_record(ex["instruction"], ex["answer"], args.max_len)
        if any(l != -100 for l, a in zip(labels, attn) if a == 0):
            n_pad_leak += 1
        if all(l == -100 for l in labels):
            n_fully_masked += 1
        first_unmasked = next((i for i, l in enumerate(labels) if l != -100), len(labels))
        if any(l != -100 for l in labels[:first_unmasked]):
            n_prompt_leak += 1

    assert n_prompt_leak == 0, f"{n_prompt_leak} examples have label != -100 before the answer starts"
    assert n_pad_leak == 0, f"{n_pad_leak} examples have padding with label != -100"
    assert n_fully_masked == 0, f"{n_fully_masked} examples had their answer fully truncated/masked"
    print(f"[audit] passed on {len(check_pool)} examples: no prompt/padding leakage, no fully-masked answers")

    if args.audit_only:
        print("[audit-only] exiting without loading the model.")
        return

    # ------------------------------------------------------------------
    # Model (QLoRA 4-bit NF4)
    # ------------------------------------------------------------------
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("[model] loading base model...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, quantization_config=bnb_cfg,
        device_map="auto", dtype=torch.bfloat16,
    )
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    def build_records(items):
        recs = []
        for ex in items:
            ids, attn, labels, _ = build_record(ex["instruction"], ex["answer"], args.max_len)
            recs.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        return recs

    train_ds = build_records(train_data)
    eval_ds = build_records(eval_data) if eval_data else []
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    batch, grad_accum, epochs, save_steps = BATCH, GRAD_ACCUM, args.epochs, 250
    if args.smoke:
        batch, grad_accum, save_steps, epochs = 2, 4, 5, 1

    train_args_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=save_steps if eval_ds else None,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        report_to="none",
        seed=SEED,
    )
    if args.max_steps and args.max_steps > 0:
        train_args_kwargs["max_steps"] = args.max_steps
    training_args = TrainingArguments(**train_args_kwargs)

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.t_start = time.time()

        def on_step_end(self, cargs, state, control, **kwargs):
            if state.global_step % 50 != 0:
                return
            elapsed = time.time() - self.t_start
            print(f"[progress] step={state.global_step}/{state.max_steps} elapsed={elapsed / 60:.1f}min")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train_ds),
        eval_dataset=ListDataset(eval_ds) if eval_ds else None,
        callbacks=[ProgressCallback()],
    )

    resume_ckpt = None
    existing_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if args.resume or existing_ckpt:
        resume_ckpt = existing_ckpt
        print(f"[resume] {'resuming from ' + resume_ckpt if resume_ckpt else 'no checkpoint found -- starting fresh'}")
    else:
        print("[resume] no checkpoint found -- starting fresh")

    t_train_start = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"[time] training took {(time.time() - t_train_start) / 3600:.2f}h")

    trainer.model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"[save] adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
