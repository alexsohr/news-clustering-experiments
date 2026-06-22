# v4 Chunk Experiment — Findings

**Date:** 2026-06-21
**Notebook:** `story_clustering_poc_v4.ipynb` §18
**Spec/plan:** `docs/superpowers/specs/2026-06-17-...`, `docs/superpowers/plans/2026-06-17-...`

## Setup
- Fresh v4 notebook (clean port of the v3 ship pipeline; dead experiments dropped).
- **3,000 items**, eval-seeded so all 543 labeled pairs stay measurable (933/933 eval items present).
- **Judge held at gpt-4.1-mini** (gpt-5.2 escalation disabled) to keep the experiment cheap and to
  isolate the chunk-text variable. Absolute F1 is therefore below the v3 0.870 ship line by design.
- **Port-correctness gate:** v3-exact 10k single-vector run reproduced **F1=0.870** exactly — the port
  is faithful.
- Total spend ≈ **$1.5** (gpt-4.1-mini judge + ~$0.12 chunk embeddings).

## Result 1 — End-to-end (does chunking beat single-vector?)

| Run (3k, gpt-4.1-mini) | F1 | P | R | FP | FN |
|---|---|---|---|---|---|
| **single-vector baseline** | **0.834** | 0.783 | 0.892 | 23 | 10 |
| chunk + chunk_pair judge | 0.790 | 0.723 | 0.871 | 31 | 12 |
| chunk + full_body judge | 0.794 | 0.716 | 0.892 | 33 | 10 |

**DECISION: KEEP single-vector.** Both chunk arms lose ~0.04 F1, entirely via **precision** (false
merges 23 → 31/33); recall is flat. Body-only chunk vectors **over-merge** at candidate selection.

Diagnostic: deep-body matches *did* fire (26% of links came from `chunk_idx > 0`), but they produced
false positives rather than recall gains.

## Result 2 — Judge-isolated A/B (chunk_pair vs full_body, 543 pairs)

| Judge text | Accuracy | tok/call | $ (543 pairs) |
|---|---|---|---|
| title + lede (single-vector judge) | 0.904 | — | — |
| **chunk_pair** (title+date+matched chunk) | **0.928** | 317 | $0.17 |
| full_body (title+date+full body) | 0.910 | 605 | $0.33 |

**JUDGE-TEXT VERDICT: chunk_pair wins** — higher accuracy than both full_body and title+lede, at
**half** the tokens of full_body. Full bodies dilute the signal; the matched chunk pair is the
sharpest evidence.

## Key insight (the two findings point in opposite directions)
- The **chunk_pair JUDGE TEXT is the best** of the three (0.928 > title+lede 0.904 > full_body 0.910).
- But the **body-only chunk VECTOR representation hurts** clustering (max-pool over-merges).
- So the loss is in the *candidate-selection* vectors, **not** the judge. They are separable.

## Why the news corpus is a weak proxy
- 58% of items are **Reuters title-only** → chunking falls back to title-as-one-chunk (≈ single-vector).
- For Bloomberg (body-bearing), chunks are **body-only by design** — excluding the title removes the
  strongest same-event signal (titles share entities/event terms), so body paragraphs match on generic
  financial language → over-merge.
- The production target (internal research articles, body-rich, lede irrelevant) is the opposite regime;
  this news result should **not** be generalized to it.

## Recommended follow-ups
1. **Single-vector candidates + chunk_pair judge** (untested combination). Vectors: keep v3 single-vector
   (better candidate precision); judge: switch to chunk_pair (0.928 > 0.904). Plausibly beats 0.834 —
   the cheapest promising next experiment.
2. **Title-prepended chunks** instead of body-only — re-introduce the title into chunk vectors to fix the
   over-merge; A/B against body-only.
3. **Re-validate on a body-rich research-artifact corpus** — the regime chunking was designed for.
4. Optional **gpt-5.2 confirmation** of the baseline-vs-chunk gap (unlikely to flip a 0.04 gap).
