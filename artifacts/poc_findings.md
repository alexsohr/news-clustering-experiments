# Story Clustering POC — Findings & Handoff

_Generated 2026-06-01T04:01:13.598734+00:00_

## Decision

**ABANDON_OR_REWORK** — F1 = 0.323 < 0.75 — algorithmic rework needed before AWS port.

| Metric | Value |
|---|---|
| Headline pair-classification F1 | 0.323 |
| Precision | 0.645 |
| Recall | 0.215 |
| Eval set size | 543 pairs |
| Baseline 1 (cosine ≥ 0.65) F1 | 0.405 |
| Baseline 3 (title Jaccard ≥ 0.5) F1 | 0.171 |

## Calibrated thresholds for production

```json
{
  "tau_high": 0.88,
  "tau_low": 0.54,
  "minhash_threshold": 0.85,
  "centroid_merge_sim": 0.85,
  "active_window_hours": 72,
  "hdbscan_min_cluster_size": 2,
  "embed_model": "text-embedding-3-large",
  "embed_dims": 1024
}
```

The full `pos_calibration.json` (with eval-set provenance) ships separately.

## Pipeline outcomes on POC data (8,735 items)

- Total stories: **8,255** (multi-item: 313, active: 8)
- Per-outcome breakdown (single-pass loop):
  - residual: 7,969 (91.23%)
  - gray_diff: 520 (5.95%)
  - gray_same: 204 (2.34%)
  - auto: 42 (0.48%)

## Cost summary

- POC total: **$1.10**
- Projected production at 100k items/day: **$36.28/day** (~$1089/mo)

## Deviations from the spec worth flagging

1. **§6 (Haiku entity extraction) was skipped.** The entity-overlap gate uses `item_clients` from §3's regex matcher as the proxy. For 2012–2013 megacap-focused financial news, the company tag dominates the shared-entity signal; per-item NER would have added < 10% marginal information at ~$1 / ~15 min per fresh run. Revisit when expanding to broader entity universe.
2. **`MIN_SHARED_CLIENTS = 1` for §13 merge** (spec says 2). With only 20 universe clients, requiring 2 shared was too strict.
3. **Pair-classification F1 in §14, not strict B-cubed.** §7 produced pair labels (not full cluster labels), so we evaluated at pair-grain. Same decision-relevant signal.

## Top failure modes

- FPs (median cosine ≈ 0.665): gray zone-dominated.
- FNs (median cosine ≈ 0.717): gray zone-dominated.

## Open questions for production

1. **Real Perplexity-aggregated news** has different source mix and entity distribution than 2012–2013 wire-news. Plan a fresh calibration run within the first month of production data.
2. **Long internal research notes** were absent from this POC's dataset, so §9 (contextual chunking) was deferred. Re-validate the prepend pattern once real research is available.
3. **Bedrock-specific behaviour** (throttling, region availability, structured-output) untested — verify in AWS before production cutover.

## Handoff artefacts

- `artifacts/labeled_eval_set.csv` — CI regression-test gate.
- `artifacts/pos_calibration.json` — thresholds, model identity, eval-set provenance.
- `artifacts/poc_findings.md` — this file.
