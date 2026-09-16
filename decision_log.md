# Decision log

The implementation keeps the order **classify → retrieve → draft → escalate**
and uses seven intents throughout.

1. **Couldn't fetch the real Kaggle dataset in the initial environment**
   (no kaggle.com egress). The committed evaluation artifacts now use the
   real AppleSupport subset; no synthetic results are presented as headline
   results.

2. **Chose AppleSupport over a higher-volume brand** (e.g. AmazonHelp)
   because its issues cluster into a small number of clean operational
   categories (device, account, billing, repair) that map naturally onto
   a 5-8 label taxonomy without needing per-SKU granularity.

3. **7 intents, not Banking77's 77.** This is Twitter support traffic,
   not a banking query log — collapsing to operationally distinct
   buckets (what team/playbook handles this?) is more useful than
   fine-grained semantic labels nobody downstream would act on
   differently.

4. **Grounding via a small labelled "playbook," not full-corpus RAG
   over 3M tweets.** Most historical replies are low-quality, off-topic,
   or duplicates. A curated, intent-bucketed playbook (even a large one,
   built offline) is cheaper, faster, and — critically — auditable: you
   can read the exact exemplars behind any draft.

5. **TF-IDF retrieval, not embeddings**, for this take-home's scale.
   Would switch to embeddings (see "next week" in the report) once the
   playbook is large enough that lexical overlap stops being a good
   enough proxy for semantic similarity.

6. **Escalation is a deterministic rule engine, not an LLM call.** It's
   the single highest-stakes decision in the system (auto-handling
   something that needed a human is the failure mode that costs trust).
   Rules are unit-testable, don't drift between runs, and are explainable
   to a support lead in one sentence per rule.

7. **Escalate-on-low-confidence and escalate-on-ungrounded are separate,
   additive checks**, not folded into one "confidence" number — a
   high-confidence intent with zero matching historical exemplars (e.g.
   a genuinely novel issue) is exactly the case a single confidence score
   would hide.

8. **The reply drafter is explicitly told not to invent a resolution
   when ungrounded**, and to ask a clarifying question instead. Silent
   hallucination under-grounding is worse than an admittedly-generic reply.

9. **Golden-set labelling is manual and single-pass** (one labeller: me),
   with a self-relabel of a subset as the only agreement check available
   for a solo take-home. This is flagged explicitly as a limitation, not
   quietly treated as equivalent to real inter-rater reliability.

10. **Stratified + deliberately oversampled golden set**, not pure random
    sampling — escalation is a low-base-rate event, and random sampling
    of a low-base-rate class produces too few examples to measure recall
    on with any confidence.

11. **LLM-judge scores are reported alongside a human-agreement number
    (MAE, % within 1 point), never presented standalone.** An unvalidated
    judge score is not evidence; only a judge whose disagreement with
    humans has been measured is.

12. **Escalation false negatives (fn) are surfaced as their own headline
    number**, separate from precision/recall/F1, because F1 can look
    fine while quietly hiding the one failure mode (auto-handling
    something that should've gone to a human) that actually matters
    operationally.

13. **Mock LLM mode exists so the repo runs in <15 minutes with zero API
    key setup**, per the README requirement — but every mock-mode number
    is labeled as illustrative in the README/report, never presented as
    the real headline result.

14. **Reply length is capped and templated toward brand voice** (warm,
    concise, no fake ticket numbers, no "we've looked into your account")
    rather than free-form generation, to reduce the chance of the model
    promising something the brand hasn't actually done.

15. **Once real data was available (106,623 real resolved AppleSupport
    pairs), I subsampled the playbook to 3,000 rather than using all of
    it.** Retrieval quality plateaus quickly with a bucketed playbook,
    and keeping it small keeps `build_playbook.py`'s labeling pass (rule
    engine, not an LLM call) fast enough to stay inside the 15-minute
    reproduction budget.

16. **Golden-set labeling used a rule-based first pass + manual spot-check
    of a 25-row subsample, not a full independent hand-label of all 200,**
    given the time available. I measured (not assumed) the first pass's
    accuracy against my own re-read: 64% raw agreement (9/25 corrected).
    Reporting that number, rather than silently shipping the uncorrected
    labels as "hand-labelled," was a deliberate choice — see report §4.

17. **Chose to report real-data numbers even though they make the system
    look worse than the earlier synthetic-data demo**, rather than
    quietly keeping the flattering synthetic numbers in the final report.
    The synthetic 10-row result (70% intent accuracy) turned out to be
    mostly an artifact of the sample and the mock classifier being
    written by the same person in the same sitting — real data exposed
    that immediately. This is the report's central finding, not a
    footnote.

18. **Adopted an external hardening pass (input validation, repo-relative
    paths, unit tests in tests/test_deterministic.py) into the working
    repo rather than rebuilding it from scratch**, after checking it
    didn't change taxonomy, metrics formulas, or reported conclusions —
    see IMPROVEMENTS.md for the itemized diff. Verified this by actually
    running the test suite and `run_eval.py` before trusting it, not by
    reading the diff and assuming it was safe.

19. **Added real-model support (Gemini via GEMINI_API_KEY, Anthropic as
    fallback) to src/llm_client.py, but did not claim it works.** This
    sandbox's network allowlist doesn't reach
    generativelanguage.googleapis.com, so the Gemini code path has never
    been executed anywhere in this project's history. Writing code and
    validating code are different claims; the report and README both say
    so explicitly rather than letting the presence of the code imply
    the second.

20. **Did a second manual golden-set review batch (50 more rows, 18
    corrected) instead of trusting the first batch's 64% agreement
    number as representative from n=25 alone.** The second batch landed
    at the same ~64% rate independently, which is itself useful evidence
    that the first-pass labeler's error rate is a real, stable property
    of the method (mostly: non-English text, app-level vs. hardware-level
    confusion, and context-dependent escalation calls) — not sampling
    noise from a small first check.
