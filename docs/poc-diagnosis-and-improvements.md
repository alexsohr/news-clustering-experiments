# Story-Clustering POC — Diagnosis of the F1 = 0.32 Result & Ways to Improve

**Status:** investigation complete · **Notebook unchanged** (analysis only) · date: 2026-05-31

This document explains *why* `story_clustering_poc.ipynb` produced F1 ≈ 0.32 (vs the ≥ 0.85
go/no-go bar) and lays out a prioritized, evidence-backed plan to fix it. It folds in a
10-agent investigation (5 adversarial verifiers run against the real artifacts + 5 web-research
agents on SOTA + a completeness critic) and a set of direct measurements on the cached data.

> **Headline:** the reported 0.32 is **mostly a software bug, not a verdict on the algorithm.**
> The notebook's own `ABANDON_OR_REWORK` decision was reached on a **broken run** and should not
> be trusted yet. Fix the bug, fix the data/eval setup, add a couple of cheap signals, and the
> architecture plausibly reaches the 0.80s on this very dataset (a learned classifier already
> hits **0.70 F1 / 0.92 AUC** on the existing labels with no new model).

---

## 0. The numbers everyone should know

| Measurement | Value | Source |
|---|---|---|
| Reported pipeline F1 (pairwise, on 543 labeled pairs) | **0.323** (P 0.645 / R 0.215) | §14.1 |
| Trivial baseline `cosine ≥ 0.65` | **0.405** | §14.2 |
| Best *possible* single cosine threshold | **0.43** @ τ=0.62 | measured |
| Cosine ROC-AUC (same-event separability) | **0.748** | §8 / reproduced |
| `τ_high` "calibrated" | **0.88** — a degenerate `argmax(precision)` fallback (no τ ever hit P≥0.95) | §8.4 |
| Single-pass loop outcome | **91% residual**, 8,300 / 8,489 stories are singletons | §10.5 |
| **Multi-channel learned classifier (cheap features, 5-fold CV)** | **0.70 F1 / 0.92 AUC** | measured (this report) |

The pipeline **scoring below a one-line cosine baseline** is the tell: a sophisticated system
should never lose to `cosine ≥ 0.65`. That only happens if the pipeline is broken — which it is.

---

## 1. PRIMARY ROOT CAUSE — a vector-indexing bug in the assignment loop (§10)

**This is the most important finding and it was invisible to the research panel** (they reasoned
about a correctly-running pipeline). It is a genuine bug, reproduced from first principles.

### What the code does (cells 167–168)

```python
sorted_items = canonical_items.sort_values("published_at").copy()   # DEFAULT kind="quicksort" → UNSTABLE
sorted_items["pos"] = np.arange(len(sorted_items))                  # comment claims "original index" — it is NOT
...
item_pos = int(row["pos"])
item_vec = assignment_vecs[item_pos]                               # indexes the vector array by re-sorted RANK
```

`assignment_vecs` is built in `canonical_items` order (cell 87), and `canonical_items` is already
**stably** time-sorted (cell 55 `sort_values(..., kind="stable")` → cell 81 boolean filter +
`reset_index`). The loop then **re-sorts with the default *unstable* quicksort** and indexes the
vector array by a freshly-minted `arange` rank.

### Why it is catastrophic on this data

For any items with **tied timestamps**, an unstable sort reorders them within the tie-group, so
`pos` no longer lines up with the vector array — **each tied item is clustered using a *different
same-day item's* embedding.**

Confirmed with hard evidence:

- **78.9 %** of working-item timestamps are **exactly midnight** (date-precision) → enormous tie-groups.
- Synthetic reproduction with that tie structure: **~40–70 % of items get the wrong vector**;
  switching to `kind="stable"`/`"mergesort"` → **0 wrong**.
- **12 of the 19** trivial near-duplicate SAME pairs (e.g. *"UPDATE 2-Boeing 787's dimmable windows…"*
  vs *"UPDATE 1-Boeing 787's…"*, cosine **0.984**; *"U.S. sues Wells Fargo"* vs
  *"U.S. files civil mortgage fraud lawsuit against Wells Fargo"*, **0.918**) share an **identical
  timestamp** — exactly the population the bug scrambles. That is why 15 of ~16 high-cosine SAME
  pairs are false negatives.
- The bug **propagates into HDBSCAN** (cell 180 `residual_vecs = assignment_vecs[singleton_member_idxs]`
  reuses the buggy indices), so the 60 % noise / singleton-collapse is also largely an artifact.

Meanwhile §7's eval cosines and §8's thresholds are computed independently and are **correct** —
which is exactly why the correct-cosine baseline (0.40) beats the scrambled-vector pipeline (0.32).

### Impact (corrected-loop estimate on the 543 eval pairs)

| Configuration | F1 |
|---|---|
| Reported (buggy) pipeline | 0.32 |
| **Bug fixed**, current Haiku judge | **~0.47** (recall ~doubles) |
| Bug fixed, competent gray-zone judge (oracle) | ~0.90¹ |

¹ The 0.90 is an upper bound and is partly *circular* — see §3.2.

### The fix (one line)

```python
sorted_items = canonical_items.sort_values("published_at", kind="stable").copy()
# or, more defensively, index vectors by the true position:
#   item_pos = row.name           # canonical_items has a RangeIndex == assignment_vecs position
#   item_vec = assignment_vecs[item_pos]
```

**Action P0: apply this, re-run §10–§15, and re-read every downstream number before drawing any
conclusion.** Everything below is what still needs attention *after* the run is no longer broken.

---

## 2. Reconciling the research panel's "the architecture is net-negative"

Three of the verifier agents concluded the LLM-judge + HDBSCAN stages are *actively subtractive*
(because the pipeline scores below plain cosine). **That conclusion is an artifact of the §10 bug.**
Once vectors are indexed correctly, the bug-fixed pipeline (~0.47) **beats** the best cosine
threshold (0.43), and the architecture has headroom via the gray-zone judge. Do **not** abandon the
single-pass + judge + residual design on the strength of the current number — it was never given a
fair run.

That said, the panel's *secondary* findings (below) are real, were reproduced against the
artifacts, and limit the achievable ceiling even after the bug is fixed.

---

## 3. SECONDARY ISSUES — real ceilings that remain after the bug fix

### 3.1 The embedding represents *topic*, not *event* (the deepest issue)

- ROC-AUC **0.748**, best single-threshold F1 **0.43**, precision ceiling ~**0.55**.
- `P(SAME | cosine bin)` *plateaus* at ~0.25 across 0.60–0.85 and only reaches **0.40** in the top
  `[0.85,1.0)` bin; even cosine ≥ 0.95 is only **44 %** SAME. This is **not** a sampling artifact
  (the conditional is sample-size-invariant) — `text-embedding-3-large` on title+lede genuinely
  cannot separate *same event* from *same topic/sector*.
- This matches Steck & Ekanadham, *"Is cosine-similarity of embeddings really about similarity?"*
  (arXiv 2403.05440) — cosine on a single dense vector conflates many notions of similarity.

**Fix:** stop relying on a single cosine cut. Fuse the dense vector with **event-discriminative
channels** (see §4) — this is the highest-leverage, lowest-cost change and the most consistent
SOTA finding (Miranda 2018 reports TF-IDF + entity + time ≈ 0.92 B-cubed vs dense-alone ≈ 0.69).

### 3.2 The gray-zone judge (Claude Haiku, sole arbiter) is a recall sink

- With `τ_high = 0.88`, the gray zone **[0.54, 0.88)** swallows **71 of 93** SAME pairs — their
  fate rests entirely on one Haiku binary verdict.
- The judge cache skews **520 DIFFERENT / 204 SAME**, and on the genuinely hard gray-zone pairs its
  effective recall is ~0.03 — it under-merges exactly where it matters.
- A *competent* gray-zone judge would lift F1 toward 0.90, but **that estimate is circular**: the
  `final_label` is itself the Sonnet/GPT-5.2/Gemini ensemble majority (confirmed:
  `final_label == ensemble_majority` for 100 % of rows), so "a Sonnet-class judge nearly solves it"
  is true partly by construction. Treat it as *suggestive*, not proof.

**Fix:** (a) lower `τ_high` off the degenerate 0.88 so the dense 0.86–0.99 SAME mass auto-merges;
(b) replace the sole-Haiku verdict with a **calibrated escalation cascade** — cheap judge (or a
cross-encoder) decides confident pairs, escalate only the truly ambiguous middle to a stronger
model/ensemble (cf. *Trust-or-Escalate*, arXiv 2407.18370); (c) calibrate the score to `P(SAME)`
(isotonic / Platt) so the decision threshold respects the ~17 % base rate.

### 3.3 Boilerplate pollution drags the threshold to the ceiling

- **67 % of pairs at cosine ≥ 0.80 are DIFFERENT** — templated Reuters wire: *"REG – … Form 8.5
  (EPT/RI)"*, *"… Net Asset Value(s)"*, *"Verizon … $525 Million"* vs *"… $105 Million"*,
  *"4G LTE Expands in <city>"*. All 15 false positives at τ_high are templated. This is what forced
  `τ_high → 0.88` (no real P≥0.95 operating point exists on the polluted distribution).
- Removing templated pairs raises ROC-AUC **0.748 → ~0.88–0.90** and restores a genuine `τ_high`
  around 0.78–0.85.
- **Caveat (from the panel, important):** do **NOT** treat wire tags `UPDATE-N / WRAPUP / FACTBOX /
  CORRECTED` as boilerplate-to-drop — **32 % of true SAME pairs carry these tags** on legitimate
  same-story follow-ups. Target *structural templates* (regulatory forms, NAV feeds, PR templates),
  not editorial wire tags.

**Fix:** a **curation gate before embedding** that routes templated/non-editorial items
(Form 8.x / Rule 8 dealing disclosures, NAV feeds, conference-notice & network-expansion PRs,
Zacks blog) to a separate pool. Note `trafilatura`/`justext` strip HTML chrome, **not** document
templates (Froebe et al., arXiv 2111.10864), so use pattern/template detection, not just extraction.

### 3.4 The corpus is temporally too sparse for streaming assignment

- The notebook **random-samples ~10k items across a 2-year span** (2012–2013) — ~208× sparser than
  the source feed. Busiest client ≈ 17 items/day in the full pool; after down-sampling, **well under
  1 item/day/client** → most items have *no* same-client neighbor in their 72 h window → the
  measured 91 % residual.
- Also: the feed is **96 % single-source Reuters** (not the multi-source mix the spec assumed), and
  **~95 % of items mention no client at all** — the "client" tag is thin and noisy.
- SOTA stream clustering assumes dense event bursts (8–18 articles/story/day); this corpus violates
  that assumption.

**Fix:** evaluate on a **contiguous, dense time-slice** (e.g. all client-items from a few busy weeks
of the full feed), not a 2-year random sample. The ablation shows ~50k contiguous items restores
72 h neighbor availability to ~0.69, ~200k to ~0.96.

### 3.5 The evaluation measures the wrong thing on the wrong set

| Defect | Why it matters |
|---|---|
| Reports **pairwise F1**, not the spec's corpus **B-cubed F1** | Different metric; not comparable to the 0.85 bar. |
| Computed on a **cosine-stratified, hard-pair-enriched** 543-pair set | Not representative; a representative sample would score *differently* (pairwise *lower*). |
| **Same pairs calibrate τ *and* report F1** | Calibration/eval leakage → optimistic. |
| `final_label` is byte-identical to the **LLM ensemble** (humans never overrode) | Label circularity — can't fairly score an LLM judge against LLM-made labels. |
| §7.1 within-72 h pair enumeration uses a **non-resetting two-pointer** | Silently **skips many valid close pairs**, biasing the eval pool. Fix: reset `j = i+1` per `i`. |

**Fix:** build a small **representative, contiguously-sampled, independently-human-labeled**
clustered gold set; compute **B-cubed P/R/F1** (scoring non-singleton clusters separately from
singleton detection — Cattan 2021, arXiv 2106.04192 — so the 98 % singleton rate doesn't inflate
the score); split **calibration / eval** by *story and time*, not by pair.

---

## 4. The decisive experiment — feature fusion already clears the cosine ceiling

The completeness critic's #1 point: the go/no-go hinges on a number nobody computed — the CV
performance of a **multi-channel pair classifier**. We computed it (5-fold CV, 543 labeled pairs,
cheap features only: cosine, time-delta, title/lede token-Jaccard, numeric-mismatch flag,
boilerplate flag):

| Model | AUC | best-F1 | P / R |
|---|---|---|---|
| **GBM (fusion)** | **0.919** | **0.704** | 0.73 / 0.68 |
| LogReg (fusion) | 0.903 | 0.688 | 0.68 / 0.70 |
| cosine-only (same CV) | 0.746 | 0.419 | — |

**+0.28 F1 over cosine-alone, with no new embedding model.** The learned weights are telling:
`title_jac` and `boilerplate` carry **negative** weight toward SAME — conditioned on cosine, near-
identical wording signals a *templated clone* (DIFFERENT), while moderate lexical overlap signals a
genuine *paraphrase* (SAME). This is precisely the SOTA "learned binary merge classifier" pattern
(Miranda 2018 SVM-merge; Saravanakumar 2021 entity-aware).

**Caveats (be honest):** the best-F1 threshold is tuned on the OOF predictions (mildly optimistic);
pairs sharing an item can leak across folds; and it's still the stratified set. AUC (0.90,
threshold-independent) is the robust signal. Adding a real **entity-overlap** channel (per-article
NER, not the coarse client tag — which is non-discriminative here because pairs share a client by
construction) and a domain/finer embedding should push this further.

---

## 5. Prioritized action plan

| # | Action | Effort | Expected impact |
|---|---|---|---|
| **P0** | **Fix the §10 vector-indexing bug** (`kind="stable"` or index by `row.name`); re-run §10–§15. | 1 line | Pipeline goes from *broken* to ~0.47; un-blocks every downstream conclusion. |
| **P1** | **Replace the single cosine gate with a learned multi-channel pair classifier** (cosine + time-decay + entity-overlap + numeric + boilerplate, calibrated to P(SAME)). | Small | 0.42 → **0.70+** demonstrated; the highest-leverage change. |
| **P2** | **Add a boilerplate curation gate** before embedding (regulatory forms / NAV / PR templates — *not* UPDATE/WRAPUP tags). | Small | Restores a non-degenerate τ_high; fixes the precision half. |
| **P3** | **Replace sole-Haiku gray-zone judge** with a calibrated escalation cascade / stronger judge / cross-encoder; lower τ_high. | Medium | Largest recall recovery; gray zone holds 71/93 SAME pairs. |
| **P4** | **Re-do the corpus**: contiguous dense time-slice (busy weeks), not a 2-year random sample. | Medium | Un-starves streaming assignment; drops the 91 % residual. |
| **P5** | **Re-do the evaluation**: representative independently-labeled gold set, corpus **B-cubed**, story/time-split calibration vs eval; fix the §7.1 enumeration bug. | Medium | Makes the go/no-go number trustworthy (and defensible either way). |

**Two cheap experiments to run first** (de-risk before big workstreams):
1. Embed **full body** (or title+body) from `canonical.parquet` and re-measure ROC-AUC — tests "title+lede is too coarse."
2. Re-run the bug-fixed loop on a **dense slice** and re-measure residual % — confirms density was the binding constraint.

---

## 6. What we still cannot claim

- **That 0.85 is reachable.** The 0.70/0.92 classifier and the oracle-judge 0.90 are *encouraging
  but caveated* (threshold tuning, label circularity, stratified set, dense-corpus SOTA comparisons).
  The honest position: **fix P0–P2, re-measure with a proper B-cubed metric on a representative set,
  then decide.** Don't promise 0.85; don't abandon at 0.32.
- **That the architecture is net-negative.** Disproven as stated — that finding was bug-induced.
- **The exact corpus boilerplate rate / B-cubed.** Estimated ~26 % of titles templated; the stratified
  eval set's 60 % over-states it. Needs measurement on the representative corpus.

---

## 7. References (from the research pass)

- Miranda et al. 2018, *Multilingual Clustering of Streaming News* — https://aclanthology.org/D18-1483/
- Saravanakumar et al. 2021, *Event-Driven News Stream Clustering using Entity-Aware Contextual Embeddings* — https://aclanthology.org/2021.eacl-main.198/
- Yoon et al. 2023, *USTORY: Unsupervised Story Discovery* — https://arxiv.org/abs/2304.04099
- Yoon et al. 2023, *SCStory* — https://arxiv.org/html/2312.03725
- Steck & Ekanadham 2024, *Is Cosine-Similarity of Embeddings Really About Similarity?* — https://arxiv.org/html/2403.05440v1
- Tang & Yang 2025, *FinMTEB: Finance Massive Text Embedding Benchmark* — https://arxiv.org/abs/2502.10990
- *Trust or Escalate: LLM Judges with Provable Guarantees* — https://arxiv.org/abs/2407.18370
- *Overconfidence in LLM-as-a-Judge* — https://arxiv.org/html/2508.06225v2
- Amigó et al. 2009, *A comparison of extrinsic clustering evaluation metrics (B-cubed)* — https://link.springer.com/article/10.1007/s10791-008-9066-8
- Cattan et al. 2021, *Realistic Evaluation Principles for Cross-document Coreference* — https://arxiv.org/abs/2106.04192
- Froebe et al. 2021, *Impact of Main Content Extraction on Near-Duplicate Detection* — https://arxiv.org/abs/2111.10864
- Abbas et al. 2023, *SemDeDup* — https://arxiv.org/abs/2303.09540 · SemHash — https://github.com/MinishLab/semhash

*Investigation artifacts: workflow script `scripts/poc_diagnosis_workflow.js`; full agent output retained in the task transcript.*
