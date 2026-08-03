# PokerLLM

Supervised fine-tuning and hybrid evaluation of open large language models on
textual poker decision states (6-max No-Limit Texas Hold'em), covering both
preflop and postflop decisions. Categorical action prediction and numerical
bet/raise sizing are reported both separately and through a continuous joint
score.

See `CITATION.cff` for citation information.

## Hybrid evaluation pipeline

Each decision is scored in two stages:

1. **Action.** The action (fold / call / check / bet / raise) is selected by
   comparing the next-token log-probability of each available action -- no
   text is generated for this step.
2. **Sizing.** A short deterministic generation is invoked whenever the
   selected or reference action is aggressive, allowing sizing information
   to be retained for evaluation and diagnostics.

`evaluation/hybrid_evaluator.py` owns parsing, prompting, model loading, and
I/O for this pipeline. `evaluation/metrics.py` owns only the scoring math
below and has no dependency on this pipeline, PokerBench, or
Transformers/torch -- it operates on already-parsed action/size pairs.

## Continuous Action–Sizing accuracy (Ac-s)

**Sizing similarity.** For a gold size $s > 0$ and a predicted size $\hat{s} > 0$:

$$\mathrm{sim}(s, \hat{s}) = \exp\left(-\left|\log(\hat{s} / s)\right|\right)$$

This equals 1 for an exact match and decays smoothly and symmetrically as the
predicted size diverges from the gold size in either direction (in log space).

**Per-sample Ac-s.** For sample $i$ with gold action $a_i$ and predicted action $\hat{a}_i$:

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

## Results

Supervised fine-tuning brings substantial gains over few-shot prompting in
both preflop and postflop decision-making, with the corrected preflop trainer
(`training/train_preflop.py`) jointly supervising the action and the numeric
sizing under a response-only loss. The hybrid evaluator
(`evaluation/hybrid_evaluator.py`) treats action selection and sizing
generation as separate steps and then combines them through the continuous
Action–Sizing accuracy (Ac-s) defined above.

Detailed results, statistical significance tests, robustness checks, and
ablation studies are reported in the dissertation and the associated papers
(see Research outputs below).

## Research outputs

Related research outputs include:

- **ENIAC 2025** -- *Evaluation of LLMs for Effective Recommendation of Poker
  Strategies* -- [DOI](https://doi.org/10.5753/eniac.2025.12461) -- published.
- **BRACIS 2026** -- *Supervised Fine-Tuning of Large Language Models for
  Strategic Decision-Making in Poker* -- accepted for presentation and
  publication.
- An associated MSc dissertation, *Learning Strategic Poker Decision-Making
  with Large Language Models*.

## Data

PokerBench ([repository](https://github.com/pokerllm/pokerbench),
[DOI](https://doi.org/10.1609/aaai.v39i24.34814)) supplies the textual
poker-state formulation and reference decisions used to produce the results
above, as the source of the experimental data rather than a code dependency.
Model adapters, PokerBench datasets, and prediction files are not
redistributed in this repository.

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
