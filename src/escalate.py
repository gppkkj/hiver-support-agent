"""
Escalation decision.

Design choice (see decision_log.md #6): this is a deterministic rule
engine over the classifier + retrieval + draft outputs, NOT another LLM
call. An escalation gate is a safety-critical control point; it should be
auditable and testable with plain unit tests, and its false-negative rate
(auto-handling something that should've gone to a human) is the single
number this whole project is most accountable for. An LLM-judged gate
would be harder to regression-test and harder to defend live.
"""

CONFIDENCE_FLOOR = 0.6
HIGH_RISK_INTENTS = {"complaint_escalation"}
SENSITIVE_KEYWORDS = [
    "lawyer", "legal action", "sue", "fraud", "chargeback",
    "self harm", "suicide", "unsafe", "fire", "smoke", "injur",
]


def decide(customer_text: str, classification: dict, draft: dict) -> dict:
    """Apply additive safety rules and return an auditable decision."""
    text = customer_text.lower()
    reasons = []

    if classification.get("intent") in HIGH_RISK_INTENTS:
        reasons.append("intent classified as complaint_escalation")

    confidence = classification.get("confidence", 0.0)
    if confidence < CONFIDENCE_FLOOR:
        reasons.append(
            f"classifier confidence {confidence:.2f} "
            f"below floor {CONFIDENCE_FLOOR}"
        )

    if not draft.get("grounded", False):
        reasons.append("no historical exemplar found -- reply is not grounded")

    hit = [kw for kw in SENSITIVE_KEYWORDS if kw in text]
    if hit:
        reasons.append(f"sensitive keyword(s) detected: {hit}")

    escalate = len(reasons) > 0
    return {
        "escalate": escalate,
        "reasons": reasons if escalate else ["all checks passed"],
    }
