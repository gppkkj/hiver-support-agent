# Improvements

## Modified files

- `src/io_utils.py` — added shared CSV existence, header, parse, and empty-input validation.
- `src/classify.py` — added input validation, explicit model-output validation, and type hints.
- `src/retrieve.py` — added playbook validation, empty-query handling, documentation, and safer TF-IDF errors.
- `src/escalate.py` — added a docstring and defensive access to classifier/draft fields.
- `src/draft.py` — added a precise return type and docstring.
- `src/baselines.py` — added return type hints.
- `src/pipeline.py` — made paths repository-relative and validated batch input rows.
- `src/eval_harness.py` — made paths repository-relative, validated judge scores, and improved CSV errors.
- `scripts/filter_brand.py` — added CLI help, raw-data validation, output-directory creation, and clearer progress output.
- `scripts/build_golden_set.py` — made paths repository-relative and added missing/empty input checks.
- `scripts/label_golden_set.py` — made paths repository-relative and added missing/empty input checks.
- `scripts/build_playbook.py` — made paths repository-relative and replaced the broken missing-demo fallback with an actionable error.
- `scripts/run_eval.py` — made paths repository-relative and added golden-set/playbook validation.
- `requirements.txt` — declared the existing `pandas` dependency used by `filter_brand.py`.
- `README.md` — corrected stale dataset/layout wording and added setup assumptions, troubleshooting, and common errors.
- `decision_log.md` — clarified the current real-data state and documented the fixed pipeline order.
- `report/REPORT.md` — corrected duplicated next-step numbering without changing conclusions or metrics.
- `metrics.json` — regenerated through the existing evaluation command; metric values remain unchanged.
- `tests/test_deterministic.py` — added lightweight tests for escalation, retrieval, and metric calculations.

## Why

These changes improve fresh-clone reproducibility, make malformed or missing
inputs fail with useful messages, clarify the mock-mode boundary, and cover
deterministic behavior without requiring an API key. Existing metric formulas,
dataset artifacts, taxonomy, retrieval strategy, escalation rules, and report
claims were left intact.

## Remaining limitations

- The headline evaluation still has only 25 human-verified examples.
- Mock mode is intentionally keyword-driven and is not evidence of real LLM quality.
- The report's documented limitations remain: no multi-turn context, no second
  labeller, TF-IDF retrieval, and no live API evaluation in this environment.
- `filter_brand.py` still requires the large Kaggle export and pandas.

## Intentionally untouched

No major architecture, pipeline stage, taxonomy, retrieval method, escalation
strategy, evaluation methodology, existing metrics, or conclusions were changed.
