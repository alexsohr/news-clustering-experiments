# v4 Bloomberg-only Chunk Experiment — Findings

**Date:** 2026-06-21
**Notebook:** `story_clustering_poc_v4.ipynb` §19 (+ reused §18 chunk machinery)
**Motivation:** Production data is body-rich (internal research articles), so Bloomberg (body-rich,
unique editorial articles) is a far better proxy than the Reuters-dominated mix (58% title-only).
Reuters dropped entirely.

## Setup
- **3,000 Bloomberg items** (filtered from 27,963 BB items in the client universe), judge = gpt-4.1-mini.
- BB is genuinely body-rich: **13,710 chunks (4.57/item; 2,693 multi-chunk items)** vs 2.13/item in the
  mixed corpus. Near-zero near-dups (2) and boilerplate (1) — clean unique articles.
- **Fresh Bloomberg eval** (`artifacts/v4/bloomberg_eval.csv`): the mixed-corpus eval had only 56
  BB-BB pairs (10 SAME). Built a new one — kNN same-client candidate pairs (where same-event pairs
  live) + chunk-advantage pairs, labeled by **gpt-4.1** (stronger than & independent of the
  gpt-4.1-mini judge). Result: **463 pairs, 29 SAME** — a *hard* eval (SAME mean cosine 0.817,
  DIFFERENT 0.733; all candidates are similar).

## Result 1 — Vector separability (§18.2, on the BB eval)
| | chunk max-pool AUC | single-vec AUC | SAME−DIFFERENT cosine gap |
|---|---|---|---|
| **Bloomberg (body-rich)** | **0.907** | 0.905 | 0.102 (chunk) vs 0.084 (single) |
| News (body-poor) | 0.889 | 0.912 | — |

On body-rich data chunk vectors are **as good or marginally better** at separating SAME/DIFFERENT —
the opposite of the news corpus, where chunk vectors were *worse*.

## Result 2 — End-to-end §10 (single-vector baseline vs chunk_pair)
| Run | F1 | P | R | TP | FP | FN |
|---|---|---|---|---|---|---|
| single-vector baseline | 0.667 | 0.842 | 0.552 | 16 | 3 | 13 |
| chunk + chunk_pair | **0.680** | 0.810 | 0.586 | 17 | 4 | 12 |

chunk_pair edges the baseline by **+0.013 F1** (catches one more SAME pair: recall 0.552→0.586).

## Headline
**On body-rich Bloomberg the gap flips/closes:** chunking goes from clearly *losing* on news
(0.834 → 0.790, −0.044) to *marginally winning* on Bloomberg (0.667 → 0.680, +0.013). This supports
the hypothesis that on the production corpus (body-rich research artifacts), body-paragraph chunking
is at least competitive and likely beneficial — the news null result was a proxy artifact (title-only
Reuters + title-excluded chunks).

## Honest caveat (important)
The BB eval has only **29 SAME pairs**, so the +0.013 (one TP) is **within noise** — this is a
*directional* result, not a statistically confident win. To confirm: build a larger BB labeled eval
(target ≥80–100 SAME pairs, e.g. label more kNN candidates and/or a larger BB sample), ideally with a
multi-model ensemble labeler instead of single gpt-4.1.

## Follow-ups (ran 2026-06-22, via subagents driving the live kernel)

**End-to-end §10 on the 463-pair eval (29 SAME — directional):**
| arm | F1 | P | R | TP | FP | FN |
|---|---|---|---|---|---|---|
| single-vector baseline | 0.667 | 0.842 | 0.552 | 16 | 3 | 13 |
| chunk + chunk_pair | 0.680 | 0.810 | 0.586 | 17 | 4 | 12 |
| combo (single-vec candidates + chunk_pair judge) | 0.680 | 0.810 | 0.586 | 17 | 4 | 12 |
| chunk + full_body | 0.694 | 0.850 | 0.586 | 17 | 3 | 12 |

All chunk arms beat the baseline (within noise on 29 SAME); full_body marginally best (recovers 1 FP
via full context). The combo ("better vectors + better judge text") tied chunk_pair — no extra lift.

**Confident signals — bigger eval (1,100 pairs, 68 SAME, kNN top-cosine):**
- **Vector separability (raw cosine AUC): chunk max-pool 0.666 vs single-vector 0.624 (+0.042).**
  On a statistically sound eval, chunk vectors separate SAME/DIFFERENT *better* on body-rich data —
  the opposite of news (chunk 0.889 < single 0.912). This is the headline confirmation.
- **Judge-isolated A/B (n=1100): chunk_pair acc 0.878 / F1 0.472 ≈ full_body acc 0.876 / F1 0.473**,
  but full_body = 1953 tok/call ($2.15) vs chunk_pair 662 tok/call ($0.73) — **chunk_pair = same
  quality at ⅓ the cost.** (Both low precision on this FP-prone high-cosine pool; the fusion gate
  supplies precision in the full pipeline.)

**Confident END-TO-END (1,100 pairs / 68 SAME) — added 2026-06-22 (`v4_confident_endtoend.json`):**
| arm | F1 | P | R | TP | FP | FN |
|---|---|---|---|---|---|---|
| single-vector baseline | 0.605 | 0.706 | 0.529 | 36 | 15 | 32 |
| chunk + chunk_pair | 0.600 | 0.692 | 0.529 | 36 | 16 | 32 |
| chunk + full_body | 0.627 | 0.740 | 0.544 | 37 | 13 | 31 |

On the confident eval the three are **statistically tied** (spread 0.027 over 68 SAME; TP differ by ≤1).
⚠️ **Correction:** the directional 463-eval read (chunk_pair 0.680 > baseline 0.667) was **noise** — at
68 SAME the end-to-end arms are indistinguishable.

## Bottom line (honest, confident)
On body-rich Bloomberg, **chunk vectors are a better SEPARATOR** of same-event pairs (cosine AUC
0.666 vs 0.624, +0.042 — confident), the opposite of news. But that advantage **does not convert into
an end-to-end F1 win**: with the fusion gate + LLM judge in place, baseline ≈ chunk_pair ≈ full_body
(~0.60–0.63, tied). So chunking is **neutral-to-positive end-to-end and strictly better at the vector
level** — a safe/good choice for body-rich production (and likely a clearer win on a corpus with more
same-event density or where candidate-retrieval quality matters more than here). **`chunk_pair` is the
judge text to use** (ties full_body at ⅓ the cost). News (body-poor) remains the case where chunking
actively hurts.

## Artifacts
`bloomberg_eval.csv` (463), `bloomberg_eval_large.csv` (1100), `bb_followup_sa{1,2,3}.json`.

## Cost ≈ $7 total BB (initial ~$1.7 + follow-ups: SA1 ~$1.5, SA2 ~$0.1, SA3 ~$3.9 incl. gpt-4.1 labeling + large judge A/B).

---

# UPDATE 2026-06-23 — Representative eval (the apples-to-apples number)

**Why:** every prior BB end-to-end F1 (0.40–0.63) was measured on a *hard-negative-mined* eval
(`bloomberg_eval_large.csv`: 6.2% SAME, SAME−DIFF cosine gap **0.021** — near-inseparable by
construction). That number is a floor, not a production estimate, and is NOT comparable to the news
F1=0.834/0.870 (which used a representative, cosine-stratified eval). So we built the missing eval.

**How (`artifacts/v4/eval_repr/`):** 521 candidate pairs drawn from the *real* candidate rule
(shared-client + 72h window) over the 3,000 BB items, then **stratified across single-vec cosine
buckets** (mirrors the v2 news eval that produced 0.870) — not top-cosine mined. Labeled SAME/DIFFERENT
by **Claude Sonnet subagents on the Claude Code subscription** (11 parallel agents; not the OpenAI key).
Result: **520 scored pairs, 125 SAME (24.0%)**, SAME cosine 0.772 vs DIFF 0.584 → **gap 0.188** (wider
than news 0.148), SAME-rate monotone by bucket (0%→5%→15%→42%→73%). Chart:
`artifacts/v4/eval_repr/repr_vs_hard.png`.

**Scorer:** identical pairwise gate + **gpt-4.1-mini** judge (no gpt-5.4-mini escalation) applied to
BOTH evals, so the only variable is eval composition. (`scripts/score_repr_eval.py`,
`artifacts/v4/repr_vs_hard_scored.json`.)

| arm | **representative** F1 / P / R | hard F1 / P / R |
|---|---|---|
| single_vec (single-vec cosine, full-body judge) | **0.834** / 0.864 / 0.806 | 0.391 / 0.941 / 0.247 |
| chunk_pair (max-pool cosine, matched-chunk judge) | 0.773 / 0.872 / 0.694 | 0.418 / 0.912 / 0.271 |
| full_body  (max-pool cosine, full-body judge) | 0.828 / 0.848 / 0.809 | 0.410 / 0.956 / 0.261 |

## Headline
**On a representative eval the pipeline scores F1 ≈ 0.83 — the same league as the news baseline 0.834.**
The "low" 0.4–0.6 numbers were entirely a hard-eval artifact; nothing regressed. Same pipeline, same
scorer, swap a hard-negative-mined eval for a representative one → F1 roughly doubles.

## Refinement that CHANGES the recommendation
On representative-difficulty pairs the **chunk_pair judge text HURTS** (F1 0.773 vs full_body 0.828 /
single_vec 0.834) — and it's all recall (**48 missed SAME vs 25**). The single best-matching chunk pair
is too thin to confirm SAME once pairs are no longer near-duplicates; the full body is needed. At the
*vector/gate* level, chunk max-pool ≈ single-vec (full_body 0.828 ≈ single_vec 0.834 — tied). So:

- ❌ The earlier hard-eval-based call ("adopt body chunking + **chunk_pair** judge — same quality at ⅓
  cost") does **not** hold on representative data. chunk_pair's ⅓-cost win was an artifact of an eval
  made of near-duplicates, where the matched chunk already contains the whole story.
- ✅ **Best representative config: single-vector candidates + full-body judge (F1 0.834).** Chunking the
  vectors is neutral (tied); chunk_pair judge text costs ~6 F1 pts of recall.

**Caveat:** still a Bloomberg proxy. Chunk vectors' raw-cosine separability edge on body-rich data
(AUC 0.666 vs 0.624) remains real but does NOT convert to F1 here. For the user's own research-artifact
corpus, recalibrate on a representative eval of THAT data before committing to chunking. Cost of this
run: ~$0.40 (gpt-4.1-mini judge) + Sonnet labeling on subscription.
