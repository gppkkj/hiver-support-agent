"""
Builds data/playbook.csv: (customer_text, brand_reply, intent) triples,
the grounding store used by src/retrieve.py.

Two modes:
- Real data present (data/resolved_pairs.csv, from scripts/filter_brand.py
  against the actual Kaggle export): subsamples N pairs and labels them
  with the same rule-based first-pass labeler used for the golden set
  (scripts/label_golden_set.py's classify()), for consistency between how
  the golden set and the playbook are labeled. This is a FIRST-PASS label
  on unverified playbook entries -- acceptable here because playbook noise
  degrades retrieval quality gracefully (worst case: a mediocre exemplar
  is retrieved), unlike golden-set noise which corrupts the metrics
  directly. See decision_log.md #4 and #9.
- If the filtered real-data output is missing, the script stops with an
  actionable error rather than silently building an incomplete playbook.
"""
import csv
import sys
import os
import random
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.label_golden_set import classify

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVED_REAL = REPO_ROOT / "data" / "resolved_pairs.csv"
OUT = REPO_ROOT / "data" / "playbook.csv"
SUBSAMPLE_N = 3000
random.seed(42)


def main():
    if os.path.exists(RESOLVED_REAL):
        with RESOLVED_REAL.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{RESOLVED_REAL} is empty; cannot build a playbook")
        print(f"Found real resolved_pairs.csv: {len(rows)} pairs")
        if len(rows) > SUBSAMPLE_N:
            rows = random.sample(rows, SUBSAMPLE_N)
        pairs = [(r["customer_text"], r["brand_reply"]) for r in rows]
    else:
        raise FileNotFoundError(
            f"{RESOLVED_REAL} not found; run scripts/filter_brand.py first"
        )

    if not pairs:
        raise ValueError("No usable customer/reply pairs were found")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_text", "brand_reply", "intent"])
        for customer_text, brand_reply in pairs:
            label = classify(customer_text)
            w.writerow([customer_text, brand_reply, label])

    print(f"[1/1] Wrote {len(pairs)} resolved pairs to {OUT} "
          f"(subsampled to {SUBSAMPLE_N} max, first-pass labeled)")


if __name__ == "__main__":
    main()
