"""
Intent taxonomy for the support agent.

Built bottom-up by open-coding ~150 sampled AppleSupport threads (see
scripts/build_golden_set.py for the sampling procedure), then merging
near-duplicate categories until label counts stabilized. Seven intents
covers >90% of sampled traffic without needing Banking77-style granularity
-- this brand's Twitter traffic is operationally coarse (device broke,
can't log in, billing dispute, how-do-i, angry escalation, everything else).
"""

INTENTS = {
    "device_troubleshooting": {
        "description": "Hardware/software malfunction: won't turn on, crashes, "
                        "battery, connectivity, broken feature after an update.",
        "default_risk": "low",
    },
    "account_access": {
        "description": "Login, password reset, 2FA, locked/hacked account.",
        "default_risk": "medium",  # touches security -> be conservative
    },
    "billing_subscription": {
        "description": "Charges, refunds, subscription cancellation, disputed "
                        "payments.",
        "default_risk": "medium",
    },
    "feature_howto": {
        "description": "How do I do X / does the product support X -- no problem, "
                        "just a question.",
        "default_risk": "low",
    },
    "repair_order_status": {
        "description": "Status of a repair, replacement, or store order already "
                        "in progress.",
        "default_risk": "medium",
    },
    "complaint_escalation": {
        "description": "Explicit anger, threats (chargeback, legal, 'switching "
                        "brands'), repeated unresolved contact, fraud claims.",
        "default_risk": "high",
    },
    "other_general": {
        "description": "Doesn't fit cleanly above: store locations, praise, "
                        "off-topic mentions, ambiguous one-liners.",
        "default_risk": "low",
    },
}

INTENT_LABELS = list(INTENTS.keys())
