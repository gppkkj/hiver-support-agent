"""
Two baselines, required by the report's 'results vs baselines' section.

trivial_baseline: the floor. Anyone could ship this in 5 minutes.
simple_baseline:  a competent non-LLM system -- keyword rules for both
                   intent and escalation, canned template replies per
                   intent. This is the bar a from-scratch LLM agent
                   actually needs to clear to justify its cost/latency.
"""

CANNED_REPLIES = {
    "device_troubleshooting": "Sorry to hear that! Please try a restart, and "
                               "DM us your device model if it persists.",
    "account_access": "Sorry for the trouble logging in -- please try "
                       "resetting your password from a browser, not the app.",
    "billing_subscription": "We understand the concern -- please DM us the "
                             "account email so we can look into the charge.",
    "feature_howto": "Happy to help -- could you tell us a bit more about "
                      "what you're trying to do?",
    "repair_order_status": "Sorry for the wait -- please DM us your order "
                            "or repair confirmation number.",
    "complaint_escalation": "We're sorry to hear this. A member of our team "
                             "will follow up with you directly.",
    "other_general": "Thanks for reaching out! Let us know how we can help.",
}

ESCALATION_KEYWORDS = [
    "lawyer", "legal", "sue", "fraud", "chargeback", "unacceptable",
    "ridiculous", "worst", "never buying", "switching to android",
]


def trivial_baseline(text: str) -> dict[str, object]:
    return {
        "intent": "other_general",
        "reply": "Thanks for reaching out! Our team will get back to you shortly.",
        "escalate": False,
    }


def simple_baseline(text: str) -> dict[str, object]:
    t = text.lower()
    if any(w in t for w in ["charged", "refund", "subscription", "cancel", "billed"]):
        intent = "billing_subscription"
    elif any(w in t for w in ["password", "login", "locked", "log into"]):
        intent = "account_access"
    elif any(w in t for w in ["repair", "order", "no updates", "sent my"]):
        intent = "repair_order_status"
    elif any(w in t for w in ["how do i", "how much", "does it support", "where"]):
        intent = "feature_howto"
    elif any(w in t for w in ["won't turn on", "crash", "not charging", "stopped working", "black screen"]):
        intent = "device_troubleshooting"
    else:
        intent = "other_general"

    escalate = any(kw in t for kw in ESCALATION_KEYWORDS)
    return {
        "intent": intent,
        "reply": CANNED_REPLIES[intent],
        "escalate": escalate,
    }
