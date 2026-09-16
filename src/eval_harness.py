"""
Evaluation harness.

Three things live here, matching the assignment's deliverable #3 exactly:
  1. Automated metrics vs the golden set (intent accuracy/F1, escalation
     precision/recall/F1) -- cheap, deterministic, run on every commit.
  2. LLM-as-judge rubric for reply quality (groundedness, correctness,
     tone, completeness -> overall 1-5).
  3. judge_vs_human_agreement() -- correlates the judge's scores against
     data/human_judge_calibration.csv so the judge's numbers are
     falsifiable, not just asserted. See decision_log.md #11 for why we
     report this as a correlation on a small set rather than claiming
     the judge is "validated."
"""
import csv
import json
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from src.llm_client import complete
from src.io_utils import read_csv_rows

JUDGE_SYSTEM = """You are a strict quality judge for AppleSupport agent
replies. You will get: the customer message, a gold summary of what a
correct reply must contain, and a candidate reply. Score the candidate
1-5 on each axis:
- groundedness: does it reflect an actual, brand-consistent resolution
  path (not a vague platitude)?
- correctness: does it match the gold_reply_summary's required content?
- tone: appropriate warmth/concision for the situation (more gravity for
  angry/urgent messages)?
- completeness: does it give the customer something actionable?
Return ONLY JSON: {"groundedness": int, "correctness": int, "tone": int,
"completeness": int, "overall": int, "rationale": "one sentence"}"""


def judge_reply(customer_text: str, gold_summary: str, candidate_reply: str) -> dict:
    """Score a candidate reply, returning a zero score for malformed judge JSON."""
    user = (f"Customer message: {customer_text}\n"
            f"Gold summary (what a correct reply needs): {gold_summary}\n"
            f"Candidate reply: {candidate_reply}")
    raw = complete(JUDGE_SYSTEM, user, max_tokens=200)
    try:
        result = json.loads(raw)
        required = ("groundedness", "correctness", "tone", "completeness", "overall")
        if any(not isinstance(result[key], int) or not 1 <= result[key] <= 5
               for key in required):
            raise ValueError("judge scores must be integers from 1 to 5")
        return result
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"groundedness": 0, "correctness": 0, "tone": 0,
                "completeness": 0, "overall": 0, "rationale": "unparseable judge output"}


def intent_metrics(golden_rows: list[dict], predictions_by_id: dict) -> dict:
    correct = 0
    per_intent = Counter()
    per_intent_correct = Counter()
    for row in golden_rows:
        gold = row["intent"]
        pred = predictions_by_id[row["tweet_id"]]["intent"]
        per_intent[gold] += 1
        if pred == gold:
            correct += 1
            per_intent_correct[gold] += 1
    n = len(golden_rows)
    accuracy = correct / n if n else 0.0
    per_intent_recall = {
        k: per_intent_correct[k] / v for k, v in per_intent.items()
    }
    return {"accuracy": accuracy, "n": n, "per_intent_recall": per_intent_recall}


def escalation_metrics(golden_rows: list[dict], predictions_by_id: dict) -> dict:
    tp = fp = fn = tn = 0
    for row in golden_rows:
        gold = row["should_escalate"] == "True"
        pred = predictions_by_id[row["tweet_id"]]["escalate"]
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1  # WORST case: should've gone to a human, didn't
        elif not gold and pred:
            fp += 1  # costs human time but is safe
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "note": "fn (missed escalations) is the metric to watch most closely"}


def judge_vs_human_agreement(
    calibration_csv: str | Path | None = None,
    golden_csv: str | Path | None = None,
) -> dict:
    """Compare deterministic mock/live judge scores with human calibration scores."""
    repo_root = Path(__file__).resolve().parents[1]
    calibration_csv = calibration_csv or repo_root / "data" / "human_judge_calibration.csv"
    golden_csv = golden_csv or repo_root / "data" / "golden_set.csv"
    golden_rows = read_csv_rows(
        golden_csv, ("tweet_id", "text", "gold_reply_summary")
    )
    rows = read_csv_rows(
        calibration_csv, ("tweet_id", "candidate_reply", "human_overall_1to5")
    )
    golden_by_id = {r["tweet_id"]: r for r in golden_rows}
    diffs = []
    pairs = []
    for r in rows:
        if r["tweet_id"] not in golden_by_id:
            raise ValueError(
                f"Calibration row references unknown tweet_id {r['tweet_id']}"
            )
        gold_summary = golden_by_id[r["tweet_id"]]["gold_reply_summary"]
        customer_text = golden_by_id[r["tweet_id"]]["text"]
        judge = judge_reply(customer_text, gold_summary, r["candidate_reply"])
        human = int(r["human_overall_1to5"])
        diffs.append(abs(judge["overall"] - human))
        pairs.append((judge["overall"], human))
    mae = sum(diffs) / len(diffs) if diffs else float("nan")
    exact_or_off_by_one = sum(1 for d in diffs if d <= 1) / len(diffs) if diffs else float("nan")
    return {
        "n": len(rows), "pairs_judge_human": pairs,
        "mean_abs_error": mae,
        "pct_within_1_point": exact_or_off_by_one,
    }


if __name__ == "__main__":
    print("Judge/human agreement on calibration set:")
    print(json.dumps(judge_vs_human_agreement(), indent=2))
