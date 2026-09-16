"""
Grounding retrieval.

Design choice (see decision_log.md #4): we do NOT embed the whole 3M-tweet
corpus. We build a small "resolution playbook" -- one exemplar
(customer msg -> brand's actual reply) per intent, mined from real
inbound/response pairs in the dataset, refreshed periodically offline.
At inference time we do lightweight lexical (TF-IDF cosine) retrieval
of the top-k most similar historical resolutions *within the predicted
intent bucket* and pass those verbatim as grounding context. This keeps
latency and cost low and keeps the grounding auditable (a human can read
the exact 3 exemplars that produced a given draft).
"""
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.io_utils import read_csv_rows, require_rows


class Playbook:
    """TF-IDF index over the intent-bucketed resolution playbook."""

    def __init__(self, resolved_pairs_csv: str | Path):
        self.by_intent = defaultdict(list)
        rows = read_csv_rows(
            resolved_pairs_csv, ("customer_text", "brand_reply", "intent")
        )
        require_rows(rows, f"Playbook {resolved_pairs_csv}")
        for row in rows:
            if row["customer_text"].strip() and row["intent"].strip():
                self.by_intent[row["intent"]].append(row)
        if not self.by_intent:
            raise ValueError(f"Playbook {resolved_pairs_csv} has no usable rows")

        self._vectorizers = {}
        self._matrices = {}
        for intent, rows in self.by_intent.items():
            texts = [r["customer_text"] for r in rows]
            vec = TfidfVectorizer(stop_words="english")
            try:
                mat = vec.fit_transform(texts)
            except ValueError as exc:
                raise ValueError(
                    f"Playbook intent '{intent}' has no usable vocabulary"
                ) from exc
            self._vectorizers[intent] = vec
            self._matrices[intent] = mat

    def top_k(self, intent: str, query: str, k: int = 3) -> list[dict[str, str]]:
        """Return up to ``k`` positive-similarity exemplars for an intent."""
        if k <= 0 or not isinstance(query, str) or not query.strip():
            return []
        rows = self.by_intent.get(intent, [])
        if not rows or intent not in self._vectorizers:
            return []
        vec = self._vectorizers[intent]
        mat = self._matrices[intent]
        q = vec.transform([query])
        sims = cosine_similarity(q, mat).flatten()
        ranked = sorted(zip(sims, rows), key=lambda x: -x[0])
        return [r for s, r in ranked[:k] if s > 0]  # sim==0 -> no real evidence
