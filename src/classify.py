import json
from typing import Any

from src.intents import INTENTS, INTENT_LABELS
from src.llm_client import complete

SYSTEM = f"""You are an intent classifier for AppleSupport customer tweets.
Classify the message into exactly one of these intents:
{json.dumps(INTENTS, indent=2)}

Return ONLY a JSON object with keys "intent" (one of {INTENT_LABELS}) and
"confidence" (float 0-1, your honest estimate of how unambiguous this
classification is -- not just always 0.9). No other text."""


def classify(text: str) -> dict[str, Any]:
    """Classify one customer message, failing closed on invalid model output."""
    if not isinstance(text, str) or not text.strip():
        return {"intent": "other_general", "confidence": 0.0}

    raw = complete(SYSTEM, text, max_tokens=100)
    try:
        out = json.loads(raw)
        intent = out["intent"]
        confidence = float(out["confidence"])
        if intent not in INTENT_LABELS or not 0 <= confidence <= 1:
            raise ValueError("intent or confidence is outside the allowed range")
        return {"intent": intent, "confidence": confidence}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Fail closed: unparseable model output should not silently become
        # a confident wrong label -- treat as lowest-confidence "other".
        return {"intent": "other_general", "confidence": 0.0}
