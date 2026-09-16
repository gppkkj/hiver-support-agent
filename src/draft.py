from typing import Any

from src.llm_client import complete

SYSTEM = """You write a grounded reply as the AppleSupport Twitter support
agent. You will be given the customer's message and 0-3 historical
exemplars of how this brand actually resolved similar issues.

Rules:
- Base your reply on the exemplars' actual resolution steps where they apply.
- If NO exemplars are given, do not invent a resolution -- write a short
  reply that acknowledges the issue and asks for the specific detail needed
  to help (order #, account email, device model), and say so plainly.
- Keep it under 280 characters where possible, brand voice: warm, concise,
  no over-promising, no fake ticket numbers.
- Do not claim to have looked anything up on their account."""


def draft_reply(customer_text: str, exemplars: list[dict[str, Any]]) -> dict[str, Any]:
    """Draft a reply using only the retrieved exemplars as grounding context."""
    if exemplars:
        ex_block = "\n".join(
            f"- Customer: {e['customer_text']}\n  Brand resolved with: {e['brand_reply']}"
            for e in exemplars
        )
        user = f"Customer message:\n{customer_text}\n\nHistorical exemplars:\n{ex_block}"
        grounded = True
    else:
        user = f"Customer message:\n{customer_text}\n\nHistorical exemplars: none found."
        grounded = False

    reply = complete(SYSTEM, user, max_tokens=250)
    return {"reply": reply, "grounded": grounded, "n_exemplars": len(exemplars)}
