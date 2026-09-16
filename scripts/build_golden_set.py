"""
Golden evaluation set construction.

SAMPLING METHOD (write-up also goes in report/REPORT.md):
1. Filter the raw dataset to inbound tweets @-mentioning the chosen brand
   handle that are the FIRST message in their thread (not follow-ups),
   so each golden example is a realistic "new incoming ticket."
2. Stratified sample across:
   a. Calendar time (avoid over-sampling one bot outage / event day)
   b. Message length tercile (short one-liners behave differently from
      long multi-issue rants -- both need to be represented)
   c. A cheap keyword-based pre-tag (angry-word count, question-mark
      count) so the golden set isn't accidentally 90% "easy" cases.
3. Target size: 200 examples (mid-point of the 150-250 range) --
   150 randomly stratified + 50 deliberately oversampled from the
   pre-tag's "angry" and "ambiguous" buckets, because escalation
   precision/recall is the metric we most need to trust, and rare-but-
   critical cases are exactly what random sampling under-represents.
4. LABELLING (by hand, one pass by me + a second pass re-labelling a
   random 20% for self-agreement, since there's no second labeller
   available for this take-home -- see report's "what's misleading"
   section for why that's a limitation, not a substitute for real IRR):
   - intent: one of the 7 taxonomy labels
   - gold_reply_summary: 1-2 sentence description of what a GOOD reply
     must contain (not a full reply -- a rubric a judge/human can score
     a candidate reply against)
   - should_escalate: bool
   - escalation_reason: free text, required if should_escalate=True

This script only implements the *sampling* (step 1-3); the labelling
(step 4) is a manual spreadsheet pass and is intentionally not
automated -- see decision_log.md #9.
"""
import csv
import random
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "first_contact.csv"
OUT = REPO_ROOT / "data" / "golden_set_sample_for_labelling.csv"
TARGET_RANDOM = 150
TARGET_OVERSAMPLE = 50   # from angry/short/ambiguous buckets specifically

random.seed(13)


def pretag(text: str) -> str:
    t = text.lower()
    angry_words = ["unacceptable", "ridiculous", "fraud", "worst", "never",
                   "!!", "scam", "lawsuit", "lawyer", "chargeback", "sue"]
    if any(w in t for w in angry_words):
        return "angry"
    if len(text.split()) < 6:
        return "short"
    if "?" in text:
        return "question"
    return "other"


def main():
    if not RAW.is_file():
        raise FileNotFoundError(
            f"{RAW} not found; run scripts/filter_brand.py first"
        )
    with RAW.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "tweet_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{RAW} must contain tweet_id and text columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{RAW} is empty; cannot build a golden set")
    print(f"First-contact population: {len(rows)}")

    buckets = defaultdict(list)
    for r in rows:
        buckets[pretag(r["text"])].append(r)
    print({k: len(v) for k, v in buckets.items()})

    random.shuffle(rows)
    random_sample = rows[:TARGET_RANDOM]
    random_ids = {r["tweet_id"] for r in random_sample}

    oversample_pool = [r for r in buckets["angry"] + buckets["short"]
                        if r["tweet_id"] not in random_ids]
    random.shuffle(oversample_pool)
    oversample = oversample_pool[:TARGET_OVERSAMPLE]

    final = random_sample + oversample
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "text", "pretag", "intent", "gold_reply_summary",
                     "should_escalate", "escalation_reason"])
        for r in final:
            w.writerow([r["tweet_id"], r["text"], pretag(r["text"]), "", "", "", ""])

    print(f"[1/1] Wrote {len(final)} sampled examples "
          f"({len(random_sample)} random + {len(oversample)} oversampled "
          f"angry/short) to {OUT} for manual labelling.")


if __name__ == "__main__":
    main()
