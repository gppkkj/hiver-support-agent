import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classify import classify
from src.retrieve import Playbook
from src.draft import draft_reply
from src.escalate import decide
from src.io_utils import read_csv_rows, require_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK_PATH = REPO_ROOT / "data" / "playbook.csv"


def run_one(text: str, playbook: Playbook) -> dict:
    classification = classify(text)
    exemplars = playbook.top_k(classification["intent"], text, k=3)
    draft = draft_reply(text, exemplars)
    decision = decide(text, classification, draft)
    return {
        "input": text,
        "intent": classification["intent"],
        "intent_confidence": classification["confidence"],
        "reply": draft["reply"],
        "grounded": draft["grounded"],
        "n_exemplars": draft["n_exemplars"],
        "escalate": decision["escalate"],
        "escalation_reasons": decision["reasons"],
    }


def run_batch(input_csv: str, output_jsonl: str):
    playbook = Playbook(PLAYBOOK_PATH)
    rows = read_csv_rows(input_csv, ())
    require_rows(rows, f"Input CSV {input_csv}")
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for r in rows:
            text = r.get("text") or r.get("customer_text")
            if not text:
                raise ValueError("Input CSV contains a row with no text/customer_text")
            result = run_one(text, playbook)
            out.write(json.dumps(result) + "\n")
    print(f"[1/1] Wrote {len(rows)} results to {output_jsonl}")


if __name__ == "__main__":
    # Quick manual demo against a couple of held-out-style examples.
    playbook = Playbook(PLAYBOOK_PATH)
    demo_messages = [
        "@AppleSupport my phone screen is completely black and won't respond to anything",
        "@AppleSupport this is the third time I've emailed about my refund, I'm done, filing a chargeback today",
        "@AppleSupport does the new Watch work with Android phones?",
    ]
    for msg in demo_messages:
        print(json.dumps(run_one(msg, playbook), indent=2))
        print("-" * 60)
