# POC v2 findings (minimal un-break + gpt-4.1 judge)

- Pairwise F1 (full 543 pairs; boilerplate->DIFFERENT): 0.739  (old buggy run: 0.323)
- Pairwise F1 (non-boilerplate subset, 365 pairs): 0.743
- Judge model: gpt-4.1-mini
- tau_high/tau_low (recalibrated on non-boilerplate): 0.94/0.54  (prior 0.88/0.54)

Metric caveat: pairwise F1 on a cosine-stratified eval set - NOT the spec's corpus B-cubed. Deferred work in docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md section 6.
