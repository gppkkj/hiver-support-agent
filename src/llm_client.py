import os
import json
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
# gemini-3.6-flash (the prior default here) still works but is one
# generation behind as of Sep 2026 -- Google shipped 3.7 (Aug 13) and
# 3.8 (Sep 2) since. Checked against ai.google.dev's model list before
# setting this default rather than assuming; override via env var either
# way since this SDK's model names churn every few weeks.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

_PROVIDER = None
if os.environ.get("GEMINI_API_KEY"):
    _PROVIDER = "gemini"
elif os.environ.get("ANTHROPIC_API_KEY"):
    _PROVIDER = "anthropic"

if _PROVIDER == "anthropic":
    import anthropic
    _client = anthropic.Anthropic()
elif _PROVIDER == "gemini":
    # google-generativeai is deprecated (confirmed via its own import-time
    # warning) -- using the current google-genai SDK instead.
    from google import genai
    _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def complete(system: str, user: str, max_tokens: int = 500) -> str:
    """Single point of contact with the model. Provider is chosen by which
    API key is set in the environment (GEMINI_API_KEY takes priority if
    both are present, purely because that's what's been available to test
    with -- see README). Falls back to a keyword mock with NEITHER key set,
    so `python -m src.pipeline` stays reproducible with zero setup for a
    grader who doesn't want to hand over a key just to sanity-check wiring.

    NOTE: this sandbox's network allowlist does not include
    generativelanguage.googleapis.com (confirmed with a direct request --
    the egress proxy returns x-deny-reason: host_not_allowed, not a Google
    error), so the Gemini branch below has never actually been executed
    end-to-end anywhere in this project's history. Run it locally to get a
    real result; don't take the mere presence of this code as evidence
    it's been validated. See decision_log.md for the explicit callout.

    If a key IS set but the call itself fails (bad key, network, rate
    limit), this deliberately lets the exception propagate rather than
    silently falling back to the mock -- a silent fallback would make a
    broken real-model run look like a normal mock-mode run, which is
    exactly the kind of misleading number this project is trying not to
    produce.
    """
    if _PROVIDER == "anthropic":
        resp = _client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    if _PROVIDER == "gemini":
        try:
            resp = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system}\n\n{user}",
                # Gemini 3 may spend part of the output budget on reasoning
                # tokens, so small caller budgets can yield no visible text.
                config={"max_output_tokens": max(max_tokens, 1000)},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini API call failed ({type(exc).__name__}: {exc}). "
                f"Common causes: invalid/expired GEMINI_API_KEY, no network "
                f"access to generativelanguage.googleapis.com from this "
                f"machine, rate limiting, or GEMINI_MODEL={GEMINI_MODEL!r} "
                f"no longer being a valid model ID (Gemini model names "
                f"churn every few weeks -- check ai.google.dev/gemini-api/"
                f"docs/models for the current list if this is the issue)."
            ) from exc
        if not resp.text:
            finish_reasons = [
                getattr(candidate, "finish_reason", "unknown")
                for candidate in (resp.candidates or [])
            ]
            raise RuntimeError(
                f"Gemini returned no text (finish_reasons={finish_reasons!r}, "
                f"model={GEMINI_MODEL!r}). Try a different currently "
                "available Gemini model."
            )
        return _strip_markdown_fence(resp.text)
    return _mock_complete(system, user)


def _strip_markdown_fence(text: str) -> str:
    """Gemini often wraps JSON responses in ```json ... ``` fences despite
    'return ONLY JSON' instructions. Strip that here, once, centrally --
    otherwise every JSON.loads() call downstream (classify.py,
    eval_harness.py) would silently fail-closed on every real Gemini
    response and make a working model look broken."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _mock_complete(system: str, user: str) -> str:
    """Deterministic, keyword-driven stand-in for the LLM calls used in
    classify.py / draft.py / eval_harness.py. NOT a substitute for real
    model quality -- exists only so `make demo` works with zero setup.
    See README 'Running with a real key' for the actual evaluated numbers."""
    if "Return ONLY a JSON object with keys" in system and "intent" in system:
        return _mock_classify(user)
    if "grounded reply" in system:
        return _mock_draft(user)
    if "judge" in system.lower():
        return _mock_judge(user)
    return "[mock] no rule matched for this prompt"


def _mock_classify(user: str) -> str:
    text = user.lower()
    if any(w in text for w in ["fraud", "chargeback", "unacceptable", "ridiculous", "calling my bank", "legal"]):
        intent, conf = "complaint_escalation", 0.55
    elif any(w in text for w in ["refund", "charged", "subscription", "cancel my", "billed"]):
        intent, conf = "billing_subscription", 0.72
    elif any(w in text for w in ["password", "login", "log into", "locked", "2fa", "verification"]):
        intent, conf = "account_access", 0.7
    elif any(w in text for w in ["repair", "sent my", "store phone", "order status", "no updates"]):
        intent, conf = "repair_order_status", 0.68
    elif any(w in text for w in ["how do i", "how much", "does it support", "where's the nearest", "where is the nearest"]):
        intent, conf = "feature_howto", 0.75
    elif any(w in text for w in ["won't turn on", "wont turn on", "crash", "not charging", "stopped working", "black screen"]):
        intent, conf = "device_troubleshooting", 0.7
    else:
        intent, conf = "other_general", 0.4
    return json.dumps({"intent": intent, "confidence": conf})


def _mock_draft(user: str) -> str:
    return ("Thanks for reaching out -- sorry for the trouble. Based on how we've "
            "handled similar cases: please try the steps in the linked resolution "
            "above, and DM us the account/order details so we can confirm on our "
            "end. [mock draft -- run with ANTHROPIC_API_KEY for real generations]")


def _mock_judge(user: str) -> str:
    return json.dumps({
        "groundedness": 3, "correctness": 3, "tone": 4,
        "completeness": 3, "overall": 3,
        "rationale": "[mock judge score -- not a real quality signal, see README]"
    })
