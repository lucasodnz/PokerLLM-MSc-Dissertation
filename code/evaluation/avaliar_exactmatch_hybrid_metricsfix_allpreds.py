#!/usr/bin/env python3
"""
Portable evaluator wrapper for the hybrid exact-match evaluation workflow.
This release copy preserves the entry-point shape while avoiding hard-coded
local paths from the original dissertation workspace.
"""

from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "samples" / "preflop_example.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "eval.json"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
parser.add_argument("--output_file", type=str, default=str(DEFAULT_OUTPUT))
args = parser.parse_args()

out_path = Path(args.output_file)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text('{"status": "placeholder", "dataset": args.dataset}\n', encoding="utf-8")
print(f"Wrote placeholder evaluation output to {out_path}")
