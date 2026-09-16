"""
python3 scripts/run_eval.py
Produces metrics.json: agent vs trivial_baseline vs simple_baseline on
the golden set, for intent accuracy and escalation precision/recall.
"""
import csv
import json
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import run_one
from src.retrieve import Playbook
from src.baselines import trivial_baseline, simple_baseline
from src.eval_harness import intent_metrics, escalation_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "data" / "golden_set.csv"


def run_all(golden_rows, playbook):
    agent_preds, trivial_preds, simple_preds = {}, {}, {}
    for row in golden_rows:
        tid, text = row["tweet_id"], row["text"]
        agent_preds[tid] = run_one(text, playbook)
        trivial_preds[tid] = trivial_baseline(text)
        simple_preds[tid] = simple_baseline(text)

    results = {}
    for name, preds in [("agent", agent_preds),
                         ("trivial_baseline", trivial_preds),
                         ("simple_baseline", simple_preds)]:
        results[name] = {
            "intent": intent_metrics(golden_rows, preds),
            "escalation": escalation_metrics(golden_rows, preds),
        }
    return results


def print_results(label, results):
    print(f"\n=== {label} ===")
    for name in results:
        im = results[name]["intent"]
        em = results[name]["escalation"]
        print(f"{name}:")
        print(f"  intent accuracy: {im['accuracy']:.2f} (n={im['n']})")
        print(f"  escalation precision/recall/f1: "
              f"{em['precision']:.2f}/{em['recall']:.2f}/{em['f1']:.2f} "
              f"(missed escalations fn={em['fn']})")


def main():
    if not GOLDEN.is_file():
        raise FileNotFoundError(f"Golden set not found: {GOLDEN}")
    with GOLDEN.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        raise ValueError(f"Golden set is empty: {GOLDEN}")
    verified_rows = [r for r in all_rows if r.get("verified") == "True"]
    playbook = Playbook(REPO_ROOT / "data" / "playbook.csv")

    output = {}
    if verified_rows:
        output["human_verified_headline"] = run_all(verified_rows, playbook)
        print_results(f"HUMAN-VERIFIED subset (n={len(verified_rows)}) "
                       f"-- treat this as the headline number", output["human_verified_headline"])

    output["full_first_pass_unverified"] = run_all(all_rows, playbook)
    print_results(f"FULL set incl. unverified first-pass labels (n={len(all_rows)}) "
                   f"-- noisier, do not treat as ground truth", output["full_first_pass_unverified"])

    with (REPO_ROOT / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull breakdown written to metrics.json. "
          f"{len(verified_rows)}/{len(all_rows)} golden rows are human-verified; "
          f"the rest carry only the programmatic first-pass label (~64% raw "
          f"agreement measured across two manual review batches, 75 rows total "
          f"-- see report).")


if __name__ == "__main__":
    main()
