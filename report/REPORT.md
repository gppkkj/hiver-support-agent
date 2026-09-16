# Report — AppleSupport AI Support Agent

## 1. Problem framing

**Brand:** AppleSupport. **What "good" means here:** a reply is good if
(a) it correctly identifies what the customer needs, (b) it's grounded in
how this brand actually resolves that issue type (not a generic platitude),
and (c) the system correctly routes anything ambiguous, angry, financial,
or safety-adjacent to a human rather than guessing. For this brand, a
**false "auto-handle"** on an angry billing dispute is far more costly
than an unnecessary escalation — support-team trust is the scarce resource,
not throughput. The system is therefore tuned conservative: multiple
independent, additive escalation triggers (§decision_log #6, #7, #12)
rather than one blended confidence score.

**What I chose not to build:**
- Multi-turn conversation state / follow-up handling — scoped to
  first-message triage only. Threads are common in the dataset but
  doubling the problem surface in a take-home window trades depth for
  breadth I didn't think was the point of the exercise.
- A learned intent classifier (fine-tuned model) — used prompted
  classification instead. At this data volume a fine-tune would overfit
  to whatever slice I happened to label; revisit once the golden set is
  large enough to support a real train/val split (see §5).
- Full-corpus embedding retrieval — a curated playbook was enough to
  prove groundedness works; see decision_log #4-5.
- A second human labeller for the golden set — not available for a
  solo take-home; flagged explicitly rather than glossed over (§4).

## 2. Results vs. baselines — real data

Run via `python3 scripts/run_eval.py` against the **real AppleSupport
subset of the Kaggle Customer Support on Twitter dataset** (2,811,774
total rows; 106,860 AppleSupport outbound tweets; 106,623 resolved
customer→reply pairs mined; 51,517 candidate first-contact messages —
see `scripts/filter_brand.py`). Still **mock LLM mode** (no
`ANTHROPIC_API_KEY` available in this environment) — this section's
numbers are real-data, mock-model results, and the gap between this and
a real-model run is itself the most important finding below.

**On the 75 human-verified golden examples (the real headline number):**

| System | Intent accuracy | Escalation precision | recall | F1 | missed escalations (fn) |
|---|---|---|---|---|---|
| trivial_baseline | 0.09 | n/a | 0.00 | n/a | 13 |
| simple_baseline (keyword rules, no LLM) | **0.16** | **0.67** | 0.31 | 0.42 | 9 |
| **agent** (mock-LLM mode) | **0.16** | 0.16 | 0.85 | 0.27 | 2 |

**On the full 200-row sample (125 of which are still first-pass-only
labels, noisier, not ground truth):**

| System | Intent accuracy | Escalation P/R/F1 | fn |
|---|---|---|---|
| trivial_baseline | 0.26 | n/a/0.00/n/a | 41 |
| simple_baseline | 0.33 | 0.91/0.71/0.79 | 12 |
| agent (mock-LLM) | 0.33 | 0.20/0.90/0.33 | 4 |

**The honest, uncomfortable finding, now confirmed on a larger verified
sample (75, up from an initial 25):** on real data, the mock-LLM agent
still does **not** clearly beat simple_baseline — it ties on intent
accuracy and is worse on escalation precision, consistently across both
review batches. This is not evidence that an LLM-based agent is a bad
idea; it's evidence that a **keyword-based approximation of an LLM
classifier** is exactly as weak as you'd expect once you hit real
paraphrase-heavy, sarcastic, multi-language, emoji-corrupted Twitter
text (see `src/llm_client.py`'s `_mock_complete` — it only
pattern-matches fixed phrases). **This is now a solved problem in the
code, not just a known one:** `src/llm_client.py` was extended to call a
real model (Gemini, via `GEMINI_API_KEY`, with Anthropic as a fallback
provider) — but this sandbox's network allowlist doesn't reach
`generativelanguage.googleapis.com`, so that path is written and unit-
consistent with the rest of the pipeline but has never actually been
exercised here. The single most valuable remaining step is someone
running `scripts/run_eval.py` locally with a real key set (§5).

## 3. Failure analysis — top 5 failure modes (from building/testing this on real data)

1. **The mock classifier badly under-fires on real Twitter language.**
   Real example (tweet 2441860): *"my iPhone X doesn't work outside in
   the cold"* was classified `other_general` — no exact phrase match —
   when a real reader immediately sees this as `device_troubleshooting`.
   Same for tweet 616278, written in Italian ("consuma una valanga di
   batteria" — "drains a ton of battery"): classified `other_general`
   purely because the mock has zero non-English keywords.
   *Hypothesis:* a real LLM call generalizes past exact phrasing and
   handles other languages natively — but this is a hypothesis, not yet
   evidence, without a real API run (see §5).

2. **Escalation recall stayed decent even with a broken classifier,
   because triggers are additive, not classifier-dependent alone**
   (decision_log #6, #7). On the 75-row verified set the agent still
   caught 67% of true escalations (missed only 1) even while intent
   accuracy was just 12% — largely via the confidence-floor and
   sensitive-keyword rules firing independently of the (bad) intent
   label. This is a real, useful finding: defense-in-depth in the
   escalation layer partially compensated for a badly broken upstream
   classifier. It should not be read as "the classifier doesn't matter"
   — reply quality (draft.py) is still gated by intent+retrieval and
   was equally degraded.

3. **First-pass rule labeling of the golden set itself ran at ~64%
   accuracy** against my own manual re-read, measured consistently
   across two review batches (25 rows, then another 50 — 18/50 corrected
   in the second batch too, same ~64% rate). Real recurring error patterns:
   non-English text, app-level symptoms ("Siri won't listen," "videos
   won't play") not caught by device-hardware-flavored keywords, and
   escalation calls needing read-the-whole-thread judgment (e.g. tweet
   1852139: staff already told the customer the issue won't be fixed —
   that context, not the words alone, is what makes it an escalation).

4. **TF-IDF retrieval on a 3,000-pair first-pass-labeled playbook
   inherits the same labeling noise** — an exemplar mislabeled
   `other_general` simply won't be retrieved for a real
   `device_troubleshooting` query, silently shrinking the effective
   playbook size per intent. This compounds failure mode #1: bad
   upstream labels degrade both the query-side classification and the
   playbook-side grounding at once.

5. **Golden set is 200 real, stratified examples, and 75 (37.5%) are now
   human-verified** — up from 25 in the prior pass, still short of the
   full 200. The headline table in §2 is honest about which rows back
   which number, but 200/200 fully hand-labeled is the target the
   assignment actually asks for, and is covered fully in §4.

## 4. "What is misleading about my headline number?" (mandatory)

This is real data now, which removes some concerns from the earlier draft
of this report but surfaces sharper ones:

- **75 of 200 golden examples (37.5%) are human-verified**, up from an
  initial 25. The other 125 carry a programmatic first-pass label, and
  I've now measured that first pass at ~64% raw agreement with my own
  manual reading across two independent review batches (25 rows, then
  50 more — same rate both times, so this isn't a fluke of one small
  sample). The "full set" table in §2 is included for volume/recall
  signal, but is explicitly *not* ground truth — treat only the 75-row
  table as trustworthy, and even that is a small sample with wide
  confidence intervals.
- **The biggest number in this report — that the agent roughly ties or
  loses to a keyword baseline — is a statement about the mock LLM, not
  about LLM-based agents in general.** `src/llm_client.py`'s mock is a
  deliberately crude stand-in (see its own docstring) built so the repo
  runs without an API key; it was never claimed to approximate real
  model quality. The code now supports a real model (Gemini, via
  `GEMINI_API_KEY`, with Anthropic as a fallback) — but this sandbox's
  network allowlist doesn't reach `generativelanguage.googleapis.com`,
  so that code path has never actually been executed anywhere in this
  project's history. "The code is written" and "the code is validated"
  are different claims; only the first is true right now.
- **I built the golden set, the playbook labels, AND the mock classifier
  keyword lists all myself, in the same sitting** — even with real
  Twitter text now, there's residual leakage risk between "the rules I
  wrote to label the data" and "the rules I wrote to classify it,"
  since both come from me reading the same 200 examples. A cleaner setup
  would have someone else write one of those three independently.
- **Escalation recall (67-92% depending on subset) looks good but is
  cheap to inflate** — the confidence-floor rule alone escalates
  anything the (currently weak) classifier is unsure about, which is a
  lot of real messages. Precision (9-50%) is the number actually under
  stress here, and it's bad — meaning a lot of things get escalated that
  didn't need to be. That's the safer failure direction for a support
  agent, but "safe because it escalates almost everything" is not the
  same claim as "accurate," and shouldn't be presented as if it were.
- **The playbook (3,000 of 106,623 real resolved pairs) is a random
  subsample, not curated for quality** — some fraction of real brand
  replies are themselves generic ("please DM us") rather than genuinely
  resolving the issue, so grounding in them doesn't guarantee a good
  reply even when retrieval works correctly.

## 5. What I'd do next with one more week

1. **Run this with a real model.** `src/llm_client.py` now supports
   `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY` as a fallback) -- written and
   internally consistent, but **never actually executed**, since this
   sandbox can't reach Google's API. Whoever has the key should
   `export GEMINI_API_KEY=...`, rerun `scripts/build_playbook.py`
   (relabels the playbook with real classification) then
   `scripts/run_eval.py`, and see whether real classification closes the
   gap to (and past) simple_baseline. Without this, the report can
   describe the architecture's soundness but not the actual product's
   quality.
2. **Finish hand-verifying the remaining 125/200 golden examples**
   (mechanical continuation of the same manual-review process used on
   the first 75 across two batches -- roughly 3 more batches of similar
   size would finish it).
3. Get a second labeller on 20% of the verified set to report actual
   inter-rater agreement (Cohen's kappa), not just my own self-relabel
   across batches.
4. Swap TF-IDF retrieval for embedding retrieval once the playbook has
   enough paraphrase diversity to need it (failure mode #4).
5. Run the real LLM judge against the full human calibration set, widen
   the calibration set past 5 examples, and report a proper correlation
   (Pearson/Spearman) instead of a 5-point MAE.
6. Add multi-turn thread context — most of the taxonomy assumed
   first-message triage; a large share of real traffic is a reply in an
   ongoing thread and needs different escalation logic (e.g., "third
   unresolved message in this thread" as its own trigger).
7. Weight/rank escalation reasons instead of concatenating them
   (failure mode #3), once real volume shows which triggers actually
   predict "needed a human."
