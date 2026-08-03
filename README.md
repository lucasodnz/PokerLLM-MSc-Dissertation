# PokerLLM

Supervised fine-tuning and hybrid evaluation of open large language models on
textual poker decision states (6-max No-Limit Texas Hold'em), covering both
preflop and postflop decisions. Categorical action prediction and numerical
bet/raise sizing are reported both separately and through a continuous joint
score.

## Repository contents

```
training/
  train_preflop.py      Preflop supervised fine-tuning (Qwen3-14B, QLoRA, response-only loss;
                         action and sizing jointly supervised)
  train_postflop.py      Postflop supervised fine-tuning (Qwen3-14B, QLoRA)
evaluation/
  hybrid_evaluator.py    Action via log-probability + sizing via deterministic generation
  metrics.py             Dataset-agnostic scoring functions (no PokerBench/Transformers dependency)
examples/
  preflop.json           Two illustrative (synthetic) preflop examples
  postflop.json          Two illustrative (synthetic) postflop examples
```

Model adapters, PokerBench datasets, and prediction files are not redistributed here.

## Continuous action-and-sizing score (Ac-s)

For each sample, the continuous Ac-s is:

- `0` if the predicted action is wrong
- `1` if the action is correct and the reference action is not aggressive (not bet/raise)
- `exp(-|log(pred_size / gold_size)|)` if the action is correct, aggressive, and both sizes are valid
- `0` if the action is correct and aggressive but the predicted size is invalid

A size is valid when it is a finite number, greater than 0, and at most 100. The **conditional
sizing score** is the mean of the third case above, computed only over samples where the action is
correct, aggressive, and both sizes are valid (it is not defined, rather than zero, when there are no
such samples). This scoring logic (`evaluation/metrics.py`) is dataset-agnostic: it operates on
already-parsed action/size pairs and has no dependency on PokerBench or on Transformers/torch.

## Relationship to PokerBench

The textual poker-state formulation and reference decisions used to produce the results below come from
**PokerBench** (https://github.com/pokerllm/pokerbench, https://doi.org/10.1609/aaai.v39i24.34814).
PokerBench is the source of the experimental data, not a code dependency: `metrics.py` does not import or
require any PokerBench code.

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

## Results

| Stage | Setting | Ac | Continuous Ac-s | Conditional sizing score |
|---|---|---|---|---|
| Preflop | Few-shot | 76.1% | 71.4% | — |
| Preflop | Fine-tuned | 93.3% | 93.2% | 99.4% |
| Postflop | Few-shot | 51.5% | 50.3% | — |
| Postflop | Fine-tuned | 91.8% | 85.3% | 70.0% |

## Citation

See `CITATION.cff`.
