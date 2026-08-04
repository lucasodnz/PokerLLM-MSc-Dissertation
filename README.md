# PokerLLM: Learning Strategic Poker Decision-Making with Large Language Models

Supervised fine-tuning and hybrid evaluation of open large language models as
predictors of PokerBench-derived reference decisions for isolated 6-max
No-Limit Texas Hold'em states, covering both preflop and postflop decisions.
Categorical action prediction and numerical bet/raise sizing are reported
both separately and through a continuous joint score.

See `CITATION.cff` for citation information.

## Main contributions

- Systematic evaluation of few-shot prompting and supervised adaptation in
  preflop and postflop.
- Hybrid inference and evaluation pipeline separating categorical action
  scoring from numerical sizing generation.
- Continuous Ac-s metric and conditional proportional sizing diagnostic.

## Method overview

Two independent QLoRA (4-bit NF4) adapters are trained on top of Qwen3-14B:
one for preflop decisions and one for postflop decisions, using
PokerBench-derived reference decisions expressed as `action` or
`action size` (e.g., `raise 7.0`). The preflop trainer
(`training/train_preflop.py`) supervises the complete response, including
both the action and the numeric sizing. Both training scripts reproduce the
configurations used to produce the results below.

At evaluation time, `evaluation/hybrid_evaluator.py` scores each decision in
two stages: the action is selected by comparing the next-token
log-probability of each available action, with no text generated; a short,
deterministic generation is then used to produce the numeric size only when
the selected or reference action is aggressive. `evaluation/metrics.py`
scores the resulting action/size pairs and has no dependency on this
pipeline, PokerBench, or Transformers/torch.

## Key results

| Stage    | Setting                | Ac         | Ac-s       |
|----------|-------------------------|-----------:|-----------:|
| Preflop  | Few-shot Qwen3-14B      | 76.1 ± 2.6 | 71.4 ± 2.6 |
| Preflop  | Fine-tuned Qwen3-14B    | 93.3       | 93.2       |
| Postflop | Few-shot Qwen3-14B      | 51.5 ± 2.4 | 50.3 ± 2.5 |
| Postflop | Fine-tuned Qwen3-14B    | 91.8       | 91.6       |

Conditional sizing scores reach 0.994 preflop over 221 eligible decisions
and 0.992 postflop over 2,168 decisions. Once the aggressive action is
correctly identified, both stage-specific adapters recover its numerical
sizing with high proportional agreement.

## Continuous Action–Sizing accuracy (Ac-s)

**Sizing similarity.** For a reference size $s > 0$ and a predicted size $\hat{s} > 0$:

$$\mathrm{sim}(s, \hat{s}) = \exp\left(-\left|\log(\hat{s} / s)\right|\right)$$

This equals 1 for an exact match and decays smoothly and symmetrically as the
predicted size diverges from the reference size in either direction (in log space).

**Per-sample Ac-s.** For sample $i$ with reference action $a_i$ and predicted action $\hat{a}_i$:

$$
\mathrm{Ac\text{-}s}_i =
\begin{cases}
0 & \hat{a}_i \neq a_i \\
1 & \hat{a}_i = a_i \text{ and } a_i \notin \{\mathrm{bet}, \mathrm{raise}\} \\
\mathrm{sim}(s_i, \hat{s}_i) & \hat{a}_i = a_i,\ a_i \in \{\mathrm{bet}, \mathrm{raise}\},\ \text{and } s_i, \hat{s}_i \text{ both valid} \\
0 & \hat{a}_i = a_i,\ a_i \in \{\mathrm{bet}, \mathrm{raise}\},\ \text{but } \hat{s}_i \text{ invalid}
\end{cases}
$$

**Global Ac-s.** The mean over all $N$ samples in the evaluation set:

$$\mathrm{Ac\text{-}s} = \frac{1}{N}\sum_{i=1}^{N} \mathrm{Ac\text{-}s}_i$$

**Valid sizing.** A size is valid when it is a finite number, greater than 0,
and at most 100 (this dataset expresses sizes as a chip count normalized to
the big blind). An invalid or missing predicted size is never silently
treated as an exact match or as zero error -- it always scores 0 in Ac-s.

**Conditional sizing score.** The mean sizing similarity restricted to the
eligible subset $E = \{i : \hat{a}_i = a_i,\ a_i \text{ aggressive},\ s_i \text{ and } \hat{s}_i \text{ both valid}\}$:

$$\mathrm{Conditional\ Sizing\ Score} = \frac{1}{|E|}\sum_{i \in E} \mathrm{sim}(s_i, \hat{s}_i)$$

It is undefined (not zero) when $E$ is empty, and isolates numerical sizing
quality from action-classification errors.

## Repository structure

```
README.md
requirements.txt
CITATION.cff
training/
evaluation/
examples/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Fine-tuned adapter, hybrid evaluation
python evaluation/hybrid_evaluator.py \
    --mode lora --adapter /path/to/adapter \
    --dataset examples/preflop.json \
    --predictions-csv preds.csv --metrics-json metrics.json

# Few-shot baseline (base model, no adapter)
python evaluation/hybrid_evaluator.py \
    --mode fewshot --train-dataset /path/to/train.json \
    --dataset examples/postflop.json \
    --predictions-csv preds.csv --metrics-json metrics.json

# Preflop training
python training/train_preflop.py --dataset-path /path/to/preflop_train.json --output-dir out/preflop

# Postflop training
python training/train_postflop.py --train-path /path/to/postflop_train.json --test-path /path/to/postflop_test.json --output-dir out/postflop
```

## Data and model availability

The experiments use PokerBench-derived decision data and Qwen3-14B base
checkpoints, which are not redistributed in this repository. The small files
under `examples/` only illustrate the expected input format and support
lightweight execution tests.

## Reproducibility and scope

The repository provides the training scripts, hybrid evaluator, metric
implementation, and small format examples. Reproducing the reported results
requires compatible PokerBench-formatted data, the corresponding base-model
checkpoints, and GPU compute.

Reported scores measure agreement with PokerBench-derived reference decisions
for isolated states. They do not establish poker solving, equilibrium
computation, exploitability guarantees, opponent adaptation, or
complete-match performance.

## Research outputs

Related research outputs include:

- **ENIAC 2025** -- *Evaluation of LLMs for Effective Recommendation of Poker
  Strategies* -- [DOI](https://doi.org/10.5753/eniac.2025.12461) -- published.
- **BRACIS 2026** -- *Supervised Fine-Tuning of Large Language Models for
  Strategic Decision-Making in Poker* -- accepted for presentation and
  publication.
- **MSc dissertation** -- *Learning Strategic Poker Decision-Making with
  Large Language Models* -- published in the UFU repository, with DOI.
- **IEEE Transactions on Games manuscript** -- under review.

## Citation

```
Title: Learning Strategic Poker Decision-Making with Large Language Models
Author: Lucas de Oliveira Diniz
Institution: Federal University of Uberlândia
Year: 2026
Dissertation DOI: 10.14393/ufu.di.2026.510
ORCID: 0000-0001-7233-8906
Advisor: Murillo Guimarães Carneiro
```

See `CITATION.cff` for the machine-readable version of this citation metadata.
