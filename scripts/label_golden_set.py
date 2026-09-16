"""
First-pass programmatic labeller for data/golden_set_sample_for_labelling.csv,
built from patterns actually read in the 200-row real sample (this data is
dominated by the iOS 11 rollout: battery drain, crashes/freezes, the
autocorrect-"I" bug, WiFi/Bluetooth drops, restart loops -- very different
from what a generic keyword list would guess).

This is a FIRST PASS, not the final label. Output goes to
data/golden_set_firstpass.csv and is then manually spot-checked (see
data/golden_set.csv + report/REPORT.md sampling note) -- single-labeller
process, documented as a limitation in decision_log.md #9, not presented
as validated inter-rater-reliable ground truth.
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "data" / "golden_set_sample_for_labelling.csv"
OUT_PATH = REPO_ROOT / "data" / "golden_set_firstpass.csv"

DEVICE_TROUBLE = [
    "crash", "freez", "restart", "reboot", "won't turn on", "wont turn on",
    "black screen", "battery", "drain", "slow", "laggy", "lag", "hang",
    "wifi", "wi-fi", "bluetooth", "won't connect", "wont connect",
    "autocorrect", "i️", "\u2139", "keyboard", "glitch", "bug", "buggy",
    "stuck", "dies fast", "won't charge", "wont charge", "not charging",
    "camera", "microphone", "can't hear", "cant hear", "no sound",
    "won't load", "wont load", "screen", "shutting off", "shut off",
    "vibrat", "tone", "call fail", "won't play", "wont play", "froze",
    "doesn't work", "won't work", "wont work", "not working", "unresponsive",
    "disappeared", "stopped", "won't load", "wont load", "keeps telling me",
    "won't install", "wont install", "activation", "face id", "don't work",
    "malfunctio", "no longer works", "isn't working", "isnt working",
    "not respond", "doesn't respond", "won't respond", "wrong artwork",
    "not receiving", "won't receive", "not receiving my calls",
]
ACCOUNT_ACCESS = [
    "apple id", "icloud", "2 step", "two factor", "2fa", "two-factor",
    "sign in", "signing in", "log in", "login", "can't login", "verification",
    "password", "locked out", "authentic",
]
BILLING = [
    "charge", "charged", "purchase i didn't mean", "refund", "subscription",
    "billed", "payment", "$", "money back", "wasn't authorized",
]
FEATURE_HOWTO = [
    "how do i", "how can i", "is it possible", "does it support", "can i",
    "where's", "where is", "any tips", "any way to", "what is mackeeper",
    "is there any chance", "how to", "can the iphone", "possible on",
    "is the warranty", "warranty still valid", "supposed to do", "will they still get",
]
REPAIR_ORDER = [
    "sent my", "sent it in", "repair", "genius bar", "order", "shipped",
    "delivery", "fedex", "store pickup", "pick up", "waiting for my dm",
    "still waiting", "days later", "no response",
]
ESCALATE_SIGNALS = [
    "switch to android", "switching brands", "android shopping",
    "lawsuit", "lawyer", "sue", "never buying", "worst", "hung up",
    "days later", "still waiting", "week ago", "reservation", "hang up",
    "fedex lost", "hate this company", "cancel my", "asap", "urgent",
    "desperate", "unacceptable",
]


def score(text, keywords):
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def classify(text):
    scores = {
        "device_troubleshooting": score(text, DEVICE_TROUBLE),
        "account_access": score(text, ACCOUNT_ACCESS),
        "billing_subscription": score(text, BILLING),
        "feature_howto": score(text, FEATURE_HOWTO),
        "repair_order_status": score(text, REPAIR_ORDER),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other_general"
    return best


def should_escalate(text, intent):
    t = text.lower()
    hits = [kw for kw in ESCALATE_SIGNALS if kw in t]
    # very short "help!!!" style cries with no diagnosable content ->
    # can't be auto-handled meaningfully, needs a human to ask what's wrong
    if len(text.split()) <= 3 and ("help" in t or "sos" in t):
        hits.append("too-short-to-diagnose")
    if intent == "billing_subscription" and any(w in t for w in ["refund", "charged", "wasn't authorized"]):
        hits.append("billing dispute")
    return hits


def gold_summary(intent, text):
    templates = {
        "device_troubleshooting": "Must acknowledge the specific symptom and give a "
            "concrete first troubleshooting step (not a generic 'sorry to hear that')",
        "account_access": "Must direct to the correct Apple ID/iCloud recovery or "
            "2FA path, not a generic password reset",
        "billing_subscription": "Must direct to reportaproblem.apple.com or request "
            "account details to investigate the charge, without promising a refund outright",
        "feature_howto": "Must correctly answer the specific product question asked",
        "repair_order_status": "Must ask for the order/repair confirmation number and "
            "acknowledge any stated delay, not offer unrelated troubleshooting",
        "other_general": "Must acknowledge the tweet's actual content, not a fully generic reply",
    }
    return templates[intent]


def main():
    if not IN_PATH.is_file():
        raise FileNotFoundError(
            f"{IN_PATH} not found; run scripts/build_golden_set.py first"
        )
    with IN_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{IN_PATH} is empty; nothing to label")
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "text", "pretag", "intent", "gold_reply_summary",
                     "should_escalate", "escalation_reason"])
        for r in rows:
            intent = classify(r["text"])
            reasons = should_escalate(r["text"], intent)
            w.writerow([
                r["tweet_id"], r["text"], r["pretag"], intent,
                gold_summary(intent, r["text"]),
                bool(reasons), "; ".join(reasons),
            ])
    print(f"[1/1] Wrote first-pass labels for {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
