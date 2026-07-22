#!/usr/bin/env python3
"""
Portable audit wrapper for prompt-condition consolidation.
This file is a lightweight copy of the canonical analysis script and is safe to
inspect and run from the public release repository root.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Portable release audit helper")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root used to resolve the output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional destination directory for generated summaries",
    )
    return parser.parse_args()


args = parse_args()
ROOT = Path(args.root).resolve()
OUT_DIR = Path(args.output_dir).resolve() if args.output_dir else ROOT / "results" / "summaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Release audit helper ready. Output directory: {OUT_DIR}")
