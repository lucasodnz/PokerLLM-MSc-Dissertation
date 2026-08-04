"""
metrics.py

Dataset-agnostic scoring functions for action-and-sizing prediction tasks
(e.g. poker decisions expressed as an action plus, for aggressive actions,
a numeric size). No dependency on PokerBench, torch, or transformers --
only the Python standard library. Parsing of raw model output, prompt
construction, and model loading live in hybrid_evaluator.py; this module
only scores already-parsed (action, size) pairs.

Continuous action-and-sizing score (Ac-s), per sample:
  0                                      if the predicted action is wrong
  1                                      if the action is correct and the reference action is not aggressive
  exp(-|log(pred_size / gold_size)|)     if the action is correct, aggressive, and both sizes are valid
  0                                      if the action is correct and aggressive but the predicted size is invalid

A size is valid when it is a finite number, greater than 0, and at most
SIZE_MAX_VALID. Invalid or missing sizes are never silently treated as
zero-error matches -- they contribute 0 to Ac-s and are excluded from the
conditional sizing score.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple

SIZE_MIN_VALID = 0.0    # exclusive -- must be > 0
SIZE_MAX_VALID = 100.0  # inclusive -- 100 is valid, values above are not


def is_valid_size(value: Optional[float]) -> Tuple[bool, str]:
    """Returns (valid, reason). Valid only if: not None, finite, > 0, <= SIZE_MAX_VALID."""
    if value is None or value == "":
        return False, "missing"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, "not_numeric"
    if math.isnan(v):
        return False, "nan"
    if math.isinf(v):
        return False, "infinite"
    if v <= SIZE_MIN_VALID:
        return False, "non_positive"
    if v > SIZE_MAX_VALID:
        return False, "exceeds_max"
    return True, ""


def compute_relative_sizing_score(gold_size: Optional[float], pred_size: Optional[float]) -> Optional[float]:
    """exp(-|log(pred/gold)|) when both sizes are valid, regardless of whether
    the action was predicted correctly. None when either size is invalid --
    never silently scored as 0."""
    gold_ok, _ = is_valid_size(gold_size)
    pred_ok, _ = is_valid_size(pred_size)
    if not (gold_ok and pred_ok):
        return None
    return math.exp(-abs(math.log(float(pred_size) / float(gold_size))))


def compute_continuous_acs(action_correct: bool, gold_is_aggressive: bool,
                            gold_size: Optional[float], pred_size: Optional[float]) -> float:
    """Per-sample continuous Ac-s. See module docstring for the four cases."""
    if not action_correct:
        return 0.0
    if not gold_is_aggressive:
        return 1.0
    score = compute_relative_sizing_score(gold_size, pred_size)
    return score if score is not None else 0.0


def sizing_within_tolerance(gold_size: float, pred_size: float, tolerance: float) -> bool:
    """True if |pred - gold| <= tolerance. Caller must ensure both sizes are valid."""
    return abs(float(pred_size) - float(gold_size)) <= tolerance


class SampleResult:
    """One scored sample. Callers (hybrid_evaluator.py) build these from
    already-parsed predictions; this class and summarize() below never parse
    raw text or touch a model."""

    __slots__ = ("action_correct", "gold_is_aggressive", "gold_size", "pred_size")

    def __init__(self, action_correct: bool, gold_is_aggressive: bool,
                 gold_size: Optional[float] = None, pred_size: Optional[float] = None):
        self.action_correct = action_correct
        self.gold_is_aggressive = gold_is_aggressive
        self.gold_size = gold_size
        self.pred_size = pred_size


def summarize(samples: Sequence[SampleResult]) -> dict:
    """Aggregates a sequence of SampleResult into the summary metrics used
    throughout this project:
      Headline metrics (used in reported results):
      - accuracy_action: fraction of samples with the correct action
      - acs_continuous_mean: mean per-sample continuous Ac-s (denominator = all samples)
      - conditional_sizing_score: mean relative sizing score over ELIGIBLE
        samples only (action correct, aggressive, both sizes valid) -- and its
        count
      Supplementary, non-headline diagnostics (not part of the reported results):
      - exact_match_rate: fraction of samples where the action is correct AND,
        for aggressive samples, the predicted size exactly equals the gold size
      - within_0_5bb / within_1bb / within_2bb: fraction of eligible samples
        whose absolute sizing error is within each tolerance
    """
    n = len(samples)
    if n == 0:
        return {
            "n_samples": 0, "accuracy_action": None, "acs_continuous_mean": None,
            "exact_match_rate": None, "conditional_sizing_score": None,
            "conditional_sizing_n": 0, "within_0_5bb": None, "within_1bb": None, "within_2bb": None,
        }

    n_correct = 0
    acs_values = []
    n_exact = 0
    eligible_scores = []
    eligible_errors = []

    for s in samples:
        if s.action_correct:
            n_correct += 1

        acs_values.append(compute_continuous_acs(
            s.action_correct, s.gold_is_aggressive, s.gold_size, s.pred_size))

        if not s.gold_is_aggressive:
            if s.action_correct:
                n_exact += 1
            continue

        gold_ok, _ = is_valid_size(s.gold_size)
        pred_ok, _ = is_valid_size(s.pred_size)
        if s.action_correct and gold_ok and pred_ok:
            if float(s.gold_size) == float(s.pred_size):
                n_exact += 1
            score = compute_relative_sizing_score(s.gold_size, s.pred_size)
            eligible_scores.append(score)
            eligible_errors.append(abs(float(s.pred_size) - float(s.gold_size)))

    n_eligible = len(eligible_scores)

    def _frac_within(tol: float) -> Optional[float]:
        if n_eligible == 0:
            return None
        return sum(1 for e in eligible_errors if e <= tol) / n_eligible

    return {
        "n_samples": n,
        "accuracy_action": n_correct / n,
        "acs_continuous_mean": sum(acs_values) / n,
        "exact_match_rate": n_exact / n,
        "conditional_sizing_score": (sum(eligible_scores) / n_eligible) if n_eligible else None,
        "conditional_sizing_n": n_eligible,
        "within_0_5bb": _frac_within(0.5),
        "within_1bb": _frac_within(1.0),
        "within_2bb": _frac_within(2.0),
    }
