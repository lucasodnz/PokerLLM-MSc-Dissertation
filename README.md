# PokerLLM — MSc Dissertation Research

Code accompanying the MSc dissertation:

**Learning Strategic Poker Decision-Making with Large Language Models**

This project investigates open large language models as predictors of strategic decisions in 6-max No-Limit Texas Hold'em. It compares few-shot prompting with supervised fine-tuning and evaluates categorical action prediction separately from numerical bet and raise sizing.

This repository presents one focused MSc research project within a broader research agenda on poker, imperfect-information games, and large language models.

## Main Contributions

- A hybrid evaluation pipeline that selects actions using log-probabilities over the legal action set and invokes deterministic text generation only when numerical sizing is required.
- A controlled comparison of six open LLMs under few-shot prompting.
- Stage-specific supervised fine-tuning of Qwen3-14B for preflop and postflop decisions.
- Separate evaluation of categorical action prediction and numerical sizing.
- Diagnostic analyses across action classes, positions, postflop streets, prompt formulations, decoding settings, and training-data fractions.

## Key Results

Supervised fine-tuning increased Action Accuracy from:

- **76.0% to 95.7% preflop**
- **51.5% to 91.8% postflop**

The results show that open LLMs can achieve high agreement with benchmark reference actions after supervised adaptation, while precise numerical sizing remains more difficult and sensitive to prompt formulation.

## Relationship to PokerBench

This work builds on the textual poker-state formulation and datasets introduced by **PokerBench**:

- GitHub: https://github.com/pokerllm/pokerbench
- Paper: https://doi.org/10.1609/aaai.v39i24.34814

PokerBench provides textualized preflop and postflop decision states with reference outputs. The dissertation extends this experimental setting through a controlled few-shot comparison, stage-specific supervised adaptation, a hybrid action-and-sizing evaluation pipeline, and more granular error, contextual, and robustness analyses.

## Research Outputs

This research developed through the following outputs:

- **ENIAC 2025** — *Evaluation of LLMs for Effective Recommendation of Poker Strategies* — https://doi.org/10.5753/eniac.2025.12461
- **BRACIS 2026** — *Supervised Fine-Tuning of Large Language Models for Strategic Decision-Making in Poker* — accepted for presentation and publication.
- **IEEE Transactions on Games** — *PokerLLM: Learning Strategic Poker Play with Large Language Models* — manuscript based on the complete experimental framework.

## Academic Context

This repository accompanies the MSc dissertation developed at the Federal University of Uberlândia under the supervision of Prof. Dr. Murillo Guimarães Carneiro.

The project focuses on isolated decision prediction rather than complete poker play, online re-solving, exploitability evaluation, or opponent-adaptive strategy.

## Repository Contents

- `code/evaluation/hybrid_evaluator.py`  
  Hybrid action and sizing evaluation pipeline.

- `code/training/train_preflop.py`  
  Preflop supervised fine-tuning script.

- `code/training/train_postflop.py`  
  Postflop supervised fine-tuning script.

Users should obtain the compatible PokerBench data and model checkpoints separately.

The original PokerBench datasets, trained adapters, model weights, and large prediction files are not redistributed in this repository.

## Usage

Inspect the available arguments with:

```bash
python code/evaluation/hybrid_evaluator.py --help
python code/training/train_preflop.py --help
python code/training/train_postflop.py --help
```
