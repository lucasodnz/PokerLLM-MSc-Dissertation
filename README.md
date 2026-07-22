# PokerLLM Public Release

This repository packages a lightweight, public-facing subset of the PokerLLM dissertation workflow for inspection and reuse. It is derived from the canonical dissertation workspace and the frozen experiment workspace, but it intentionally excludes trained model weights, full datasets, checkpoints, optimizer state, and large raw prediction dumps.

## Dissertation context

This work studies whether large language models can learn strategic poker decision-making, with emphasis on action selection and bet sizing. The materials in this release are aligned with the final dissertation analysis and the canonical figures and tables prepared for the dissertation.

## Research objective

The goal of the release is to provide:
- portable analysis and evaluation scripts;
- summary figures and tables;
- ablation and robustness notes; and
- a minimal sample dataset for quick inspection.

## Dataset source and citation

The source workflows and benchmark framing build on the PokerBench-style evaluation setting used in the dissertation. This release does not bundle the full training or evaluation corpora, and it is not a complete reproduction bundle. Users should obtain compatible datasets and model artifacts separately if they wish to reproduce the original experiments.

Suggested citation:
- Diniz, L. O. (2026). Learning Strategic Poker Decision-Making with Large Language Models. Master's dissertation project.

## Repository structure

- configs/: example configuration stubs
- prompts/: prompt templates and notes
- code/analysis/: analysis scripts
- code/evaluation/: evaluator scripts
- code/training/: training scripts
- data/samples/: small, illustrative examples only
- results/summaries/: summary reports and tables
- results/tables/: summary tables
- results/figures/: final figures
- docs/: documentation notes
- manifests/: export and exclusion manifests

## Environment setup

Create a virtual environment and install the dependencies listed in requirements.txt:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation instructions

Run the evaluator from the repository root:

```bash
python code/evaluation/avaliar_exactmatch_hybrid_metricsfix_allpreds.py --dataset data/samples/preflop_example.json --output_file outputs/eval.json
```

This example uses the included lightweight sample dataset and writes a placeholder evaluation output under the outputs directory.

## Fine-tuning instructions

The training scripts in code/training/ are included for reference. They preserve the original workflow structure and training configuration intent, but the release does not include the original training data or trained adapters.

## Metrics explained

- Action Accuracy (AA): the fraction of examples where the predicted action matches the gold action.
- Action-and-Sizing Accuracy (Acc-s): the fraction of examples where the action and the size are both correct within the evaluation tolerance.

## Reproduction limitations

This release is intentionally conservative. It does not distribute:
- trained model weights;
- adapter checkpoints;
- full training or test datasets;
- optimizer state;
- large raw prediction dumps; or
- local runtime caches.

Full reproducibility of the original dissertation results requires access to the original PokerBench data and model artifacts that are not included here.

## Citation instructions

Please cite this repository and the dissertation work together. See CITATION.cff for metadata.
