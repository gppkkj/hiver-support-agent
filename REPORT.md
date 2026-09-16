1. Problem Framing
What “Good” Means for AppleSupport

The goal is to build a support agent that can:

Classify incoming customer messages into a small set of AppleSupport-specific intents.
Draft replies grounded in historically resolved AppleSupport conversations.
Decide whether to auto-handle the request or escalate it to a human, with a clear reason.

For AppleSupport, a good system should provide relevant, grounded, concise, and safe responses while avoiding unsupported claims. When confidence is low or a request is sensitive or ambiguous, the system should prefer human escalation over an unreliable automated response.

What We Chose Not to Build

To keep the system focused and reproducible, we did not build:

Full multi-turn conversation memory.
A fine-tuned classification model.
A large-scale vector database.
LLM-based escalation decisions.
A production deployment or real-time Twitter integration.

The implementation instead focuses on a lightweight pipeline using LLM classification and generation, TF-IDF historical retrieval, and deterministic escalation rules.

2. Results vs. Baselines

The system is evaluated against two simpler approaches:

Trivial Baseline

A default/majority prediction strategy that does not use the full support pipeline.

Simple Baseline

A lightweight keyword/rule-based approach without the complete LLM + retrieval pipeline.

AI Support Agent

The proposed system combines:

LLM-based intent classification
TF-IDF retrieval from historical AppleSupport resolutions
LLM-based grounded reply generation
Deterministic escalation rules
Evaluation Metrics

The evaluation measures:

Intent accuracy
Per-intent performance
Escalation precision
Escalation recall
Escalation F1
Missed escalations
Reply quality using an LLM-as-a-judge

The current mock-model evaluation produced:

Metric	AI Agent
Intent Accuracy	16.0%
Precision	15.9%
Recall	84.6%
F1	26.8%
Missed Escalations	2

The real Gemini API connection was successfully validated, but the complete real-model evaluation was not completed because the classification response reached the configured output-token limit. Therefore, no unsupported real-Gemini performance numbers are reported.

3. Failure Analysis
Failure Mode 1 — Keyword-Based Classification Misses Paraphrases

Some customer messages describe an issue without using the exact keywords expected by the classifier.

Example: A customer may describe an iPhone battery problem without explicitly using terms such as “battery” or “charging.”

Hypothesis: Semantic classification should improve robustness to paraphrasing compared with fixed keyword matching.

Failure Mode 2 — Multilingual and Informal Messages

Twitter support messages frequently contain short, informal, abbreviated, or multilingual text.

Example: A customer may describe severe battery drain using a mixture of regional language and English.

Hypothesis: A language model can better interpret variations in wording than a fixed keyword system.

Failure Mode 3 — High Escalation Recall Can Reduce Precision

The escalation rules are intentionally conservative.

The current evaluation shows:

Escalation recall: 84.6%
Escalation precision: 15.9%

This means the system catches many escalation cases but also escalates many messages that may not require human intervention.

Hypothesis: More human-labelled escalation examples could be used to calibrate the escalation rules and reduce unnecessary escalations.

Failure Mode 4 — Ambiguous Messages

Some customer messages are too short or ambiguous to determine the correct intent without additional conversation context.

Example: A message such as “It still doesn't work” may require the previous conversation to determine whether the issue concerns billing, login, repair, or a device problem.

Hypothesis: Adding multi-turn conversation context would improve classification and escalation decisions.

Failure Mode 5 — Retrieval Quality Depends on Historical Data

The response generator depends on retrieved historical resolutions.

If the retrieved example is generic, incomplete, or not sufficiently similar, the generated response may also be less useful.

Hypothesis: Better-quality historical labelling and semantic embedding retrieval could improve retrieval quality while preserving the same overall architecture.

4. What Is Misleading About My Headline Number?

A single accuracy number does not fully describe the quality of a support agent.

First, the golden set contains 200 examples, but human verification is still being expanded. First-pass labels should not be treated as equivalent to independently verified ground truth.

Second, the current measured agent results are from the deterministic mock model. They should not be interpreted as measured performance of Gemini. The Gemini API itself has been successfully connected and tested, but the full evaluation has not yet produced valid real-model metrics.

Third, escalation recall alone can be misleading. A system could achieve very high recall simply by escalating most messages. Precision, recall, F1, and missed escalations therefore need to be considered together.

Finally, historical grounding does not guarantee that a retrieved response is correct for every new customer situation. Historical responses provide evidence, not certainty.

5. What I'd Do With One More Week
1. Complete Real-Model Evaluation

Increase the classification output budget and add robust handling for incomplete LLM responses, then run the complete evaluation using Gemini.

2. Complete Human Verification

Finish manually verifying the golden set so that at least 150 examples are genuinely hand-labelled.

3. Add Independent Human Evaluation

Have a second human independently label 15–20 examples and calculate Cohen’s kappa to measure inter-rater agreement.

4. Improve Retrieval

Compare TF-IDF retrieval against embedding-based semantic retrieval while keeping the existing retrieval interface unchanged.

5. Add Multi-Turn Context

Use previous messages from the conversation when the first customer message is insufficient to determine the intent or escalation decision.

6. Improve Reply Evaluation

Expand human calibration for the LLM-as-a-judge and measure agreement across a larger set of generated responses.