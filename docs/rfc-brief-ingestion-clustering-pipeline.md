# RFC Evidence Brief — News Ingestion & Story-Clustering Pipeline

_Produced 2026-06-12 from the validated POC (`story_clustering_poc_v3.ipynb`; final clean-kernel ledger record `P6_final-ship`, 2026-06-11). Self-contained: every value, prompt, and parameter needed to improve the RFC is embedded here. Repo paths appear only as provenance._

---

## 0. How to use this document

This brief carries two layers for two jobs:

- **Prescriptive layer** — the validated design, stage by stage, with exact parameters, prompts, and schemas. Use it to draft missing RFC sections.
- **Evidence layer** — measured results, rejected alternatives, and pitfalls. Use it to strengthen existing RFC text and answer reviewer challenges.

**Epistemic markers** used throughout:

| Marker | Meaning |
|---|---|
| `[MEASURED]` | A benchmark fact from the POC evaluation. Citable as-is. |
| `[VALIDATED DESIGN]` | A design choice that survived measured iteration. Transplant with confidence. |
| `[PRODUCTION DELTA]` | A judgment call where production differs from the POC; adapt, don't copy. |
| `[OPEN]` | Unresolved; the RFC should address it explicitly. |

**RFC mapping table** (matched to the target RFC's draft sections):

| Target RFC section | Read first | Supporting |
|---|---|---|
| 1. Deployment topology | **Ch 16** | Ch 8 (judge ops), Ch 14.5–14.6 |
| 2. Event-driven trigger flow | **Ch 17** | Ch 14.5 |
| 3. DataFlow — persistence boundaries | **Ch 18** | Ch 4 (dup-map), Ch 8 (verdict ledger) |
| 4. Data model — ERD | **Ch 19** | Ch 3, Ch 7 (schemas) |
| 5. Ingestion — per-client & generic branches | **Ch 20.1** | Ch 3–5 |
| 6. Synthesis — content → context enrichment | **Ch 20.2** | Ch 6, Ch 10 |
| 7. Aggregation — clustering + scoring | **Ch 7–9** | Ch 11–13 (validation, journey, alternatives) |
| (cross-cutting) Risks / Open questions | Ch 14, 15 | |

Part I (Ch 1–15) is the validated algorithm + evidence. Part II (Ch 16–20) is systems guidance derived from it for the RFC's infrastructure sections — mostly `[PRODUCTION DELTA]` inference anchored on `[MEASURED]` POC properties; it should be cited as *grounded design guidance*, not as benchmark fact.

---

## 1. System context & production mapping

**Production flow:** the firm maintains a **client portfolio** (tracked companies). A per-client **news search** runs against the news providers; results arrive **already tagged** with the client they concern. The pipeline then deduplicates, curates, embeds, clusters into stories, and enriches — producing a per-client story feed.

**POC proxy `[MEASURED]`:** two years (2012-01 → 2013-12) of Bloomberg + Reuters wire coverage — 2,075,631 raw items — filtered to a 20-client universe (JPMorgan, Apple, Goldman Sachs, Morgan Stanley, Bank of America, Citigroup, Google, Facebook, Microsoft, Boeing, Ford, AT&T, Verizon, Wells Fargo, Amazon, Toyota, Twitter, Walmart, IBM, ExxonMobil). 84,764 items mention ≥1 client (4.1%); sampled to **10,000 items** for end-to-end processing. Client tagging in the POC was alias-regex mention scanning — a stand-in for production's search-side tagging.

**What transfers vs what doesn't:**

| Transfers `[VALIDATED DESIGN]` | Does NOT transfer `[PRODUCTION DELTA]` |
|---|---|
| Stage architecture and ordering | Calibrated thresholds (τ, fusion gates) — corpus-specific; **re-calibrate on real production data within the first month** |
| Dedup mechanics (URL canon + MinHash) | The 72h window semantics (POC corpus was temporally sparse and day-quantized) |
| Curation guard concept and patterns | Cost/volume figures (POC mix ≠ production mix) |
| Fusion-scorer feature set + training recipe | The exact label set (rebuild eval pairs from production data) |
| Judge rubric, escalation, caching discipline | Batch-mode merge/expiry cadence (streaming needs its own; see Ch 14) |
| Evaluation methodology (Ch 11) | |

**Known input-data caveat `[MEASURED]`:** 88.2% of POC timestamps are date-quantized (midnight; `time_precision: day` for 1.83M of 2.08M rows). All time windows therefore operated at calendar-day resolution. Production should ingest and preserve true timestamps and record a precision field per item.

---

## 2. Architecture at a glance

```
client portfolio ──► news search (items arrive client-tagged)
        │
        ▼
 [3] INGEST    item schema: id, title, body, source, published_at(+precision), client_tags
        ▼
 [4] DEDUP     URL canonicalization (exact) + MinHash/LSH (near-dup wire copies)
        ▼
 [5] CURATE    template/filing detector → non-editorial items set aside (searchable, not clustered)
        ▼
 [6] EMBED     title + 600-char lede → text-embedding-3-large, 1024-dim unit vector (cached)
        ▼
 [7] ASSIGN    single-pass, chronological. Candidates: shared client + active window.
        │      fusion score p → p≥p_high: auto-join │ p≥p_low: LLM judge │ else: new story
        ▼
 [9] RESIDUAL  HDBSCAN proposes groups among leftover singletons → judge-verified
        ▼
[10] MERGE+ENRICH  judge-gated story merges; schema-enforced LLM metadata
        ▼
 per-client story feed
```

**Headline result `[MEASURED]`** (frozen 543-pair benchmark, final configuration): **Precision 0.917 · Recall 0.828 · F1 0.870** against a pre-agreed ship bar of F1 ≥ 0.85. Baseline at iteration start: P 0.682 / R 0.806 / F1 0.739. Long-range guard slice (Ch 11): **0 false merges** (was 46 at baseline). Final feed shape: 7,276 stories over 7,871 items (439 multi-item; largest story 13 items).

---

## 3. Stage: Ingestion & client tagging

**Input contract the pipeline assumes `[VALIDATED DESIGN]`:**

```
item = {
  item_id          stable unique id (POC: uuid5 of canonical URL)
  title            headline
  body             full text (may be empty for title-only wires)
  source           provider id
  published_at     timestamp + time_precision ∈ {minute, day, ...}
  item_clients     set of client tags (from search-side tagging)
  url              original URL
}
```

- `[PRODUCTION DELTA]` In production, `item_clients` comes from the search layer. The POC's downstream stages use it in three load-bearing places: the assignment candidate gate (Ch 7), the residual-cluster coherence check (Ch 9), and merge candidacy (Ch 10). **Tag quality directly caps recall** — an item missing its client tag can never reach the right story. The RFC should state the tagging precision/recall expectation and a monitoring plan.
- `[MEASURED]` Multi-client items exist (957/10,000 in the POC) and are handled naturally: an item can be a candidate for stories of any of its clients.
- Timestamp precision: see Ch 1 caveat. Store it; don't infer.

---

## 4. Stage: Deduplication

Two independent layers, both `[VALIDATED DESIGN]`:

**4a. Exact: URL canonicalization.** Lowercase scheme/host; strip tracking query parameters (`utm_*`, `fbclid`, …) and fragments; normalize trailing slashes; hash the result. Items sharing a hash are exact duplicates. `[MEASURED]` On the POC corpus this merged 0 rows (wire feeds had clean URLs) — keep it anyway; it is free and production web sources will need it.

**4b. Near-duplicate: MinHash + LSH.**
- Tokenize lowercased `title + body` with `[a-z0-9]+`; build **5-gram shingles**; hash into a **128-permutation MinHash**.
- LSH index at threshold 0.70 produces candidate pairs (11,705 on the POC); pairs with estimated **Jaccard ≥ 0.85** are duplicates; union-find picks one canonical item per cluster.
- `[MEASURED]` 235 duplicate clusters covering 1,500 items → **1,265 items (12.65%) removed as re-published wire copies**. Threshold stability sweep: 0.70→1,325, 0.75→1,308, 0.80→1,292, 0.85→1,265, 0.90→1,231 removed — the choice of 0.85 is not knife-edge.
- **Keep, don't delete:** duplicates are mapped (`dup_map: duplicate_id → canonical_id`), never dropped. This matters twice: feeds can still resolve any item to its story through the map, and evaluation must resolve labeled pairs through it (Ch 11).

**Pitfall `[MEASURED]`:** near-duplicate ≠ same story. Templated items (recurring filings, per-town press releases) have high shingle overlap but are *different events*. Dedup catches only ≥85% overlap; the template problem is handled by curation (Ch 5) and the fusion scorer (Ch 7), not here.

---

## 5. Stage: Curation (boilerplate set-aside)

A pattern detector routes **non-editorial wire** out of the clustering corpus before embedding `[VALIDATED DESIGN]`:

- Patterns: `REG-…` regulatory wires, `Form 8.3 / 8.5 / 8 (DD)` disclosures, `Net Asset Value(s)` / NAV notices, `Rule 38.5`, "X to Present at … Conference" PRs, analyst-alert templates (Zacks etc.).
- Guardrail: legitimate editorial markers (`UPDATE n-`, `WRAPUP`, `RPT-`) are **not** flagged.
- `[MEASURED]` 864/8,735 canonical items (9.9%) set aside; clustering corpus = 7,871.
- Set-asides remain searchable and are *not* deleted — they are excluded from story formation only.

**Why this stage earns its place:** templated items are the densest source of false merges (high mutual similarity, different events). Removing them up front killed an entire false-merge class that no downstream threshold could otherwise separate. The flip side is a structural blind spot in evaluation (Ch 11): pairs touching set-asides cannot be predicted SAME.

---

## 6. Stage: Embedding

- **Model:** OpenAI `text-embedding-3-large`, truncated to **1024 dims**, unit-normalized (so dot product = cosine). `[VALIDATED DESIGN]`
- **Input:** `title + first 600 chars of body` ("lede"). Full bodies add boilerplate and cost without adding event signal at this stage.
- **Caching:** by content hash; each unique item embedded exactly once. POC cache: 8,735 entries / 36 MB. Cost ≈ $0.00002/item at $0.00013 per 1k tokens.
- **Known limits, both `[MEASURED]`:**
  1. *Topic ≠ event.* Embeddings place same-company, same-topic items close together even when they are different events. A raw-cosine single threshold ceilings at **F1 0.640** (best τ = 0.62) on the benchmark. This is the core motivation for the fusion scorer + judge.
  2. *Distance concentration.* In 1024 dims the same-event vs different-event cosine gap is small relative to spread; density methods (Ch 9) see ~62% of residual items as "noise". Treat that number as a property of the geometry, not a quality metric.

---

## 7. Stage: Story assignment (the core loop)

**Shape `[VALIDATED DESIGN]`:** single chronological pass (stable sort by `published_at`). For each item:

```
candidates = stories that are open
             AND share ≥1 client with the item
             AND whose newest member is within the active window (POC: 72h)
if no candidates:            → new story                       (16% of items [MEASURED])
best = candidate story with max cosine(item, story centroid)
p    = fusion_score(item, best)                                # see below
if p ≥ p_high:               → auto-join best                  (≈2% of items)
elif p ≥ p_low:              → LLM judge decides               (≈32% of items)
else:                        → new story                       (≈50% of items)
on join: append member, recompute story centroid (mean of member vectors, re-normalized)
```

`[MEASURED]` final-run outcome counts (7,871 items): no-candidates 1,286 · auto 173 · judged 2,496 (of which SAME 418) · below-floor 3,916.

**The candidate gate runs before any similarity** — shared client + recency are hard prerequisites. `[MEASURED]` It is not the recall bottleneck (only 16% of items found no candidate; 66% found one but scored below the floor), but it makes client-tag quality and window choice load-bearing (Ch 3, Ch 14).

### The fusion scorer `[VALIDATED DESIGN]`

Logistic regression over 7 features computable at decision time (item vs the story's cosine-nearest member for lexical features; centroid for cosine):

| # | Feature | Definition | Coefficient (standardized) |
|---|---|---|---|
| 1 | `cosine` | embedding cosine to story centroid | **+2.652** |
| 2 | `title_jac` | token-set Jaccard of titles | **−1.674** |
| 3 | `minhash_jac` | MinHash-estimated body Jaccard | +0.378 |
| 4 | `dt_days` | |Δ published| in days, capped at 7 | −1.060 |
| 5 | `len_ratio` | min/max of title+lede lengths | −1.160 |
| 6 | `num_mismatch` | 1 − Jaccard of numeric tokens | +0.325 |
| 7 | `caps_mismatch` | 1 − Jaccard of capitalized tokens (entity proxy) | −0.444 |

Intercept −2.824. **The sign of `title_jac` is the headline insight:** *conditioned on high embedding similarity, high literal title overlap is evidence AGAINST the same event* — that is the signature of templated look-alikes. No similarity-only system can express this.

- **Training:** the 543-pair labeled benchmark (Ch 11); StandardScaler + LogisticRegression; **GroupKFold (5 folds) over connected components of the pair graph** (390 components) so pairs sharing an item never straddle train/test — the leakage that inflated earlier estimates.
- `[MEASURED]` Out-of-fold **AUC 0.912** vs **0.748** for cosine alone.
- **Gate calibration:** on OOF probabilities, `p_high` = lowest p keeping precision ≥ 0.99 → **0.807**; `p_low` = highest p keeping recall ≥ 0.95 → **0.030**.
- Legacy raw-cosine gates for reference (superseded but recorded): τ_high 0.94 (P 1.00 / R 0.054), τ_low 0.54 (P 0.384 / R 0.957).
- `[PRODUCTION DELTA]` Re-fit and re-calibrate on a production-labeled pair set; with day-quantized POC timestamps, `dt_days` should be re-derived once true timestamps exist.

---

## 8. Stage: The LLM judge

**Why a judge at all `[MEASURED]`:** with the auto gate calibrated to ≥99% precision, nearly every contested merge is the judge's call. At iteration start the judge was simultaneously the **largest false-merge source** (23/35 FPs) and the **largest missed-merge blocker** (11/18 FNs). Every accepted improvement after the structural fixes was a judge improvement. *Judge quality ≈ pipeline quality.*

### The rubric prompt (v2, shipped) `[VALIDATED DESIGN]` — verbatim

```
You are determining whether two financial news items describe the SAME news event.

ITEM A:
  title: {title_a}
  lede:  {first 600 chars of body_a}
  published_at: {pub_a}

ITEM B:
  title: {title_b}
  lede:  {first 600 chars of body_b}
  published_at: {pub_b}

Two items are the SAME story if they describe the same underlying news event
involving the same primary entities — even if framing, source, or details differ.
They are DIFFERENT if the primary event differs (e.g., "Apple Q3 earnings beat" vs
"Apple Q3 earnings preview" are different events despite same topic; follow-ups in an
ongoing saga that report distinct developments are DIFFERENT events).

Reply with a single word: SAME or DIFFERENT.
```

**Design rationale:** this prompt *mirrors the rubric the ground-truth labelers used* (Ch 11). The superseded v1 prompt was a bare "same story or different?" with 200-char bodies and no dates; its two error modes were exact mirror images — saga follow-ups judged SAME (false merges: 8 of the worst FPs were JPMorgan-"London Whale" sub-events), and cross-source restatements of one event judged DIFFERENT (missed merges, e.g. "Amazon buys Ivona" pairs at cosine 0.83). Aligning the judge's definition of "story" with the labelers' definition was worth **+0.027 F1** by itself.

### Models & escalation `[VALIDATED DESIGN]`

- Base judge: `gpt-4.1-mini`, temperature 0, single-word completion.
- **Escalation:** all judged pairs escalate to `gpt-5.2` (longer per-request timeout; max_completion_tokens 4000 to allow reasoning). Escalation was first validated on the weak cosine band [0.60, 0.75) (+0.027 F1), then extended to the full range (+0.033 F1).
- On escalated-call failure (timeout/connection after retries), **fall back to the base judge** rather than aborting.

**Judge accuracy ladder `[MEASURED]`** (vs the 543 human-validated labels): v1 prompt 0.904 → v2 rubric 0.923 → + band escalation 0.943 → + full escalation **0.965**. Final per-cosine-bin accuracy:

| cosine bin | n | accuracy |
|---|---|---|
| 0.20–0.40 | 60 | 1.000 |
| 0.40–0.50 | 60 | 0.983 |
| 0.50–0.55 | 60 | 0.967 |
| 0.55–0.60 | 55 | 0.982 |
| 0.60–0.65 | 60 | 0.917 |
| 0.65–0.70 | 52 | 0.885 |
| 0.70–0.75 | 60 | 0.950 |
| 0.75–0.80 | 46 | 0.978 |
| 0.80–0.85 | 42 | 1.000 |
| 0.85–1.00 | 48 | 1.000 |

The residual weakness is the 0.60–0.70 band — genuinely ambiguous adjacent-event pairs.

### Operational discipline `[VALIDATED DESIGN]`

- **Caching & determinism:** every verdict cached on disk keyed by `sha256(model | prompt_version | item_a_id | item_b_id)`. Same pair → same answer, forever; re-runs are free; every merge decision is auditable.
- **The cache-key trap:** the prompt text MUST be represented in the key (a version string suffices). An early version keyed by `(model, pair)` only — a prompt edit would silently reuse stale verdicts. For story-vs-story judging (Ch 10), where the prompt content depends on whether enrichment ran, key by a **hash of the rendered prompt**.
- **Retry policy:** 429 → exponential backoff (60s × 1.5ⁿ); connection/timeout errors → short backoff (5s × n); cap retries, then fall back (escalated → base) or surface.
- Cost `[MEASURED]`: base judge ≈ $0.00017/call; judge calls are the only rate-limited, only non-cacheable-in-advance path in the pipeline.
- `[OPEN]` **Circularity caveat:** `gpt-5.2` was one of the three label-ensemble vendors, so part of its 0.965 agreement is self-agreement. Production must validate the escalated judge against freshly collected human labels before trusting that figure. (The labels were 3-vendor majority + human review of the ambiguous band — not a single model's output — so the result is directionally robust.)

---

## 9. Stage: Residual clustering (HDBSCAN + judge gate)

After assignment, most items are single-item stories (the POC corpus is genuinely singleton-heavy). HDBSCAN proposes groups among them:

- Parameters: `min_cluster_size=2, min_samples=2, cluster_selection_method="eom"`, cosine distances on the 1024-dim vectors. `[VALIDATED DESIGN]`
- `[MEASURED]` 596 raw clusters; **62.6% of residuals labeled noise.** Do not target noise-% — it is dominated by distance concentration (Ch 6) and genuine singletons. The metric that matters is the missed-merge / false-merge rate on labeled pairs.
- **Client-overlap alone is an insufficient coherence check** `[MEASURED]`: requiring a shared client across members still passed template families (per-town press releases of one company). At baseline, HDBSCAN clusters contributed 12 of 35 false positives and 46 long-range false merges.
- **Judge gate `[VALIDATED DESIGN]`** — every surviving cluster is verified before becoming a story, ≤4 judge calls each:
  - n=2: judge the pair; DIFFERENT → dissolve.
  - n≥3: judge medoid (member nearest the centroid) vs farthest member; on DIFFERENT, peel the farthest and re-judge (max 2 peels, then dissolve).
  - n≥4 and passing: one extra seeded-random medoid-vs-member spot check; DIFFERENT → peel that member.
  - Peeled/dissolved members return to the singleton pool.
- `[MEASURED]` Under the final judge, the gate kept only **3 of 373** proposals — and cluster-stage false positives went **12 → 0**. The strictness is a feature: in this corpus, density-coherent same-client groups are overwhelmingly template families, not stories. `[PRODUCTION DELTA]` Expect a higher keep-rate on a richer editorial mix; the gate adapts automatically because the judge decides, not a threshold.

---

## 10. Stage: Merge pass & enrichment

**Merge pass (story ↔ story):** candidate pairs = active multi-item stories with centroid cosine ≥ 0.85 and ≥1 shared client; each candidate judged (story-level prompt over title+summary, or first member's title when enrichment hasn't run); approved pairs merged via union-find. `[MEASURED]` In the shipped batch configuration this stage is nearly inert (4 candidates) because the expiry sweep closes stories first; the attempted repair was measured and rejected (Ch 13.2). `[OPEN]` Streaming production needs its own merge cadence design (Ch 14).

**Enrichment:** `gpt-4.1` with schema-enforced output generates per-story metadata: `title` ≤ 80 chars, `summary` ≤ 400 chars (two factual sentences), one-word `topic`, `entities`. Calls cached by story membership; ≈ $0.0024/story; POC enriched the top-100 stories by size with a trivial title-fallback for the rest. `[PRODUCTION DELTA]` Production should enrich all multi-item stories and refresh metadata when membership changes materially.

---

## 11. Evaluation methodology

This is the RFC's validation backbone; the design is as important as the results.

**Frozen pair benchmark `[VALIDATED DESIGN]`:**
- Pairs enumerated from the corpus: same-client pairs within 72h, stratified across cosine-similarity bins (~60/bin) so hard cases aren't drowned by easy ones. **543 pairs: 93 SAME / 450 DIFFERENT.**
- Labels: a **3-vendor LLM ensemble** (Claude Sonnet 4.6, GPT-5.2, Gemini 3.5 Flash) with structured output; 2-of-3 majority; 2×UNCLEAR or 1-1-1 splits routed to **human review**, focused on the threshold-critical cosine band (0.55–0.75). Labeling rubric = the same event-level definition the judge uses (Ch 8) — by design.
- The set is **frozen** and serves as a CI regression gate: any model/threshold change must re-run it.

**Ensemble labeling prompt:** the judge v2 prompt (Ch 8) was *derived from* this labeling prompt. The labeling version is identical except it (a) lacks the saga-follow-up clause — that was added to the judge later, after error analysis, and is consistent with how the ensemble actually labeled saga pairs — and (b) requests JSON output `{"verdict": "SAME"|"DIFFERENT"|"UNCLEAR", "reason": "<one sentence>"}` instead of a single word.

**Metric:** pairwise classification P/R/F1 over the labeled pairs, where a pair is predicted SAME iff both items resolve to the same story id (resolution passes through the dedup map, Ch 4). Items set aside by curation can never co-cluster; such pairs predict DIFFERENT and are **counted** (178/543 pairs touch set-asides — measured, not hidden).

**Known blind spots `[MEASURED]` — state these in the RFC:**
1. *Same-client only.* Cross-client merges are unmeasurable until a cross-client slice exists.
2. *Within-72h only.* Long-range recall is invisible to the headline metric — which motivated the guard slice below.
3. *Pairwise ≠ corpus-level.* This is a stratified pair metric, not B-Cubed over the full corpus.

**Supplemental long-range guard slice `[MEASURED]`:** 199 ensemble-labeled same-client pairs with 3–30-day gaps, stratified by cosine × gap. Finding: **only 1 of 199 is genuinely SAME** — long-gap high-similarity pairs in this corpus are almost entirely recurring templates. The slice therefore guards *precision* (the pipeline's false merges on it went 46 → 0) and proves window-widening is not a recall opportunity here.

**Per-stage attribution `[VALIDATED DESIGN]`:** every false positive is attributed to the stage that created the merge (auto-gate / judge / residual cluster / merge pass) and every false negative to its blocker (judge-said-no / below-floor / set-aside / window / cluster-dissolved). This turned experiment selection from guesswork into arithmetic and is worth a paragraph in the RFC's operability section: per-decision audit data should be a first-class output.

**Accept rule used during iteration:** a change is accepted only if ΔF1 ≥ +0.02 on the frozen set, or smaller with clear structural evidence — and never if the non-boilerplate slice or the guard slice regresses.

---

## 12. Evidence journey (the iteration ledger)

Eleven measured runs, each a single flagged change, each accepted or rejected against the rule above `[MEASURED]`:

| Run | Change | F1 | Verdict | One-line takeaway |
|---|---|---|---|---|
| R0 | baseline (v2 algorithm, eval hygiene proven no-op) | 0.739 | root | FP-dominated: 35 FP / 18 FN |
| R1a | judge-gate HDBSCAN clusters | 0.768 | ✅ | cluster FPs 12→1; long-range FPs 46→13 |
| R1b | merge-pool expansion @0.80 | 0.764 | ❌ | 50 merges, 0 true pairs, +4 long-range FPs |
| R1c | second assignment sweep | aborted | ❌ | provably couldn't flip any FN (identical judge cache keys) |
| R2 | fusion gates replace raw-cosine τ | 0.784 | ✅ | recall +0.032 at flat precision; OOF AUC 0.912 |
| R3 | judge rubric v2 (labeler-aligned + saga clause) | 0.810 | ✅ | the conceptual fix; judge acc 0.904→0.923 |
| R4 | nearest-member judge representative | 0.802 | ❌ | helped false pairs as much as true ones |
| R5 | gpt-5.2 escalation, weak band only | 0.837 | ✅ | precision 0.775→0.846 |
| R6 | gpt-5.2 escalation, full range | **0.870** | ✅ SHIP | P 0.917; judge acc 0.965; guard slice 0 FP |
| P6 | final clean-kernel full run | 0.870 | confirmed | reproducible end-to-end |

Total iteration LLM spend ≈ $15. The arc to quote in the RFC: **structural guards bought precision; scorer+judge quality bought everything else.** F1 0.739 → 0.870; precision 0.682 → 0.917; long-range false merges 46 → 0.

---

## 13. Alternatives considered & rejected (with measurements)

RFC-ready "alternatives considered" material — each was *measured*, not argued away:

1. **Optimizing HDBSCAN noise-%.** Rejected as a target metric: noise can be driven down trivially by force-assignment at precision's expense; ~60% noise is partly geometry (Ch 6) and partly genuine singletons. Optimize missed-merge/false-merge rates on labeled pairs instead.
2. **Merge-pool expansion** (include closed stories; lower threshold 0.85→0.80). `[MEASURED]` 71 candidates, 50 judge-approved merges — recovering **zero** true pairs while adding +1 benchmark FP and +4 long-range FPs. The merge judge of that era approved template-family merges. Re-test queued now that the judge is rubric-aligned (Ch 15).
3. **Second assignment sweep** (re-offer singletons to stories after the pass). Rejected by analysis before completion: the dominant missed merges were judge rejections, and the sweep re-asks the *same judge the same cached question* (identical cache key) — it cannot flip them. Also predicted to add long-range FPs. Re-test only with a changed judge and a tight window.
4. **Nearest-member judge representative** (vs first member). `[MEASURED]` F1 0.810→0.802: recall unchanged, precision down — the nearest member is more similar for *false* pairs too.
5. **Window widening (72h → 7/14/30d) for recall.** Killed by the guard-slice finding (1/199 SAME): in this corpus there is essentially no long-range same-story mass to recover, only template traps.
6. **Deferred, not rejected:** UMAP/HDBSCAN geometry sweeps (residual stage now contributes ~0 FPs and few merges — low ceiling); richer embedding input (invalidates every calibration and cache; the fusion scorer attacks the same weakness for ~$0); gradient-boosted fusion (doc'd AUC 0.919 vs LogReg 0.912 — not worth the interpretability loss).

---

## 14. Risks & operational pitfalls

1. **Judge circularity** `[OPEN]` — see Ch 8. Validate the escalated judge on fresh human labels; do not quote 0.965 in production claims until then.
2. **Re-calibration triggers** `[VALIDATED DESIGN]`: embedding-model version change; domain shift (e.g., beyond financial news); frozen-set F1 regression > 0.03. Thresholds are corpus artifacts, not constants.
3. **Cache-key discipline** — model id AND prompt version (or rendered-prompt hash) must be in every verdict cache key. The POC hit this trap twice; both are fixed patterns now (Ch 8, Ch 10).
4. **Template traps recur at every stage.** Dedup misses them (<85% overlap), embeddings love them, density clustering groups them. Defense in depth: curation patterns + the negative `title_jac` fusion feature + the judge rubric + the long-range guard slice in CI.
5. **Streaming vs batch deltas** `[PRODUCTION DELTA]`: the POC ran batch. Production must re-design: (a) window/expiry semantics — POC "active window" assumed chronological replay; (b) merge-pass cadence (periodic job vs event-driven) and the pool-starvation interaction with expiry (Ch 10); (c) judge latency on the ingest path — auto-gate and below-floor decisions are instant, but ~32% of items awaited a judge call in the POC mix; the RFC needs an async-assignment or provisional-story story.
6. **Single-vendor concentration:** embeddings + judge are both OpenAI. The labeling ensemble is deliberately 3-vendor; consider the same hedge for the judge (the architecture is model-agnostic — rubric + cache + escalation transfer).
7. **Eval-set maintenance:** the frozen set is the regression gate; it ages with the corpus. Schedule periodic refreshes through the same ensemble+human machinery, versioned (never edited in place).
8. **Client-tag dependency** (Ch 3): the candidate gate makes tagging a hard recall ceiling; monitor tag precision/recall in production.

---

## 15. Open questions the RFC should answer

1. **Story granularity is a product decision, not a technical one.** The labelers (and therefore the judge) define a story as a *single news event*; saga follow-ups are separate stories (the JPMorgan "London Whale" saga = several stories: loss disclosure, FBI probe, executive departure, charges). If the product wants saga-level feeds, build it as a **second-level grouping over event stories** — do not loosen the event definition (that path measurably destroys precision).
2. **Cross-client merging:** currently unmeasurable (Ch 11). Decide whether multi-client stories matter for the product; if yes, budget a cross-client eval slice first.
3. **Judge latency/SLA in streaming** (Ch 14.5) and the fallback behavior when the judge is unavailable (provisional singleton + later reconciliation is the natural answer; specify it).
4. **Re-test list with the current judge:** merge-pool expansion (Ch 13.2) and a constrained second sweep (top-1 candidate, ≤7-day window) — both were rejected under the weaker judge; the calculus may have changed.
5. **Human-label refresh cadence** for both the eval set and judge validation (Ch 14.1, 14.7).
6. **Tag-quality monitoring** and the contract with the search layer (Ch 3).

---

# Part II — Systems guidance for the RFC's infrastructure sections

_Derived from measured POC properties; epistemic markers apply. Where Part II makes an architectural recommendation, the anchor facts are cited from Part I._

---

## 16. Computational shape & deployment topology (→ RFC §1)

Per-stage operational profile:

| Stage | State | Parallelism | External dependency | Failure behavior |
|---|---|---|---|---|
| Dedup (4) | LSH index (rebuildable from digests) | parallel per shard; LSH join global | none | recompute-safe |
| Curate (5) | none (pure function) | embarrassingly parallel | none | recompute-safe |
| Embed (6) | shared content-hash cache | embarrassingly parallel | embeddings API (high limits) | idempotent retries |
| **Assign (7)** | **story store — the only order-dependent, mutable stage** | sequential within an ordering domain | judge API (rate-limited) | see Ch 17 provisional pattern |
| Residual cluster + gate (9) | snapshot batch | single job | judge API | re-runnable on snapshot |
| Merge (10) | snapshot batch | single job | judge API | re-runnable |
| Enrich (10) | metadata cache by membership | parallel per story | LLM API | idempotent; refresh on change |

**The serialization point `[MEASURED]`:** assignment is a single chronological pass mutating the story store; it is the stage that cannot be naively parallelized. Client-partitioned sharding is *almost* possible — but **957 of 10,000 items carried multiple client tags** and would couple shards (an item joining a story in shard A must be visible to shard B's candidate gate). Topology options, in order of fidelity to the validated behavior: (a) one global assignment worker per environment (validated; 10k items replay end-to-end in ~8 min on warm caches `[MEASURED]`); (b) micro-batched assignment every N minutes, ordered by timestamp within a batch (closest scalable equivalent); (c) client-sharded workers with multi-client fan-out and idempotent join reconciliation (highest throughput, new consistency design — treat as `[OPEN]`).

**The judge is the only queue-worthy path `[MEASURED]`:** ~32% of items needed a judge call; observed throughput ~60 calls/min at a 50-RPM limiter with ~1 s/call latency (escalated calls slower). Everything else is fast, local, or cacheable. Deploy the judge as a worker pool with a rate budget, retry policy (Ch 8), and the verdict ledger in front — cached verdicts shortcut the queue entirely (a full pipeline replay made **0 paid judge calls** `[MEASURED]`).

**Shared caches are infrastructure, not optimizations:** embedding cache (36 MB / 8,735 items), MinHash digests, judge-verdict ledger. They carry determinism, auditability, and the marginal-cost profile (re-runs ≈ free). Treat them as durable services, not local files.

---

## 17. Event-driven trigger flow (→ RFC §2)

Batch→event translation of the validated pipeline:

| Trigger | Work | Idempotency key | Emits |
|---|---|---|---|
| `item.ingested` | dedup check → curate flag → embed | content hashes | `item.ready` (or `item.set_aside`, `item.duplicate_of`) |
| `item.ready` | assignment decision (Ch 7 gates) | item_id (decision logged once) | `story.created` \| `story.updated` \| `judgment.requested` |
| `judgment.completed` | reconcile provisional assignment | verdict-ledger key | `story.updated` |
| schedule: expiry sweep | close stale stories | run window | `story.closed` |
| schedule: residual scan | HDBSCAN + judge gate over singletons | snapshot id | `story.created` (gated) |
| schedule: merge scan | story-pair judging | snapshot id | `story.merged` |
| `story.updated` (membership change, debounced) | enrichment refresh | membership hash | `story.metadata_updated` |
| schedule: nightly eval | replay frozen + guard sets | eval-set sha | alert if F1 drops > 0.03 |

**The async-judge pattern `[PRODUCTION DELTA]`:** the POC awaited the judge inline. In an event flow, an item whose decision needs a verdict should be assigned a **provisional singleton story** immediately (feed-visible, marked provisional), with `judgment.requested` emitted; on `judgment.completed` = SAME, the provisional story is absorbed into the target story (`story.updated`). This makes judge latency a feed-quality lag, not an ingest blocker. The cost is transient story churn; the guard is the same one the POC proved: nothing merges without a verdict.

**Ordering requirement `[MEASURED]`:** single-pass assignment is order-sensitive (the POC measured order effects; see Ch 13.3). Event delivery must preserve approximate chronological order within an ordering domain (Ch 16) or accept jitter and rely on the periodic residual/merge scans as the reconciliation mechanism — which is exactly what they exist for.

**Replayability:** every stage is idempotent given the caches: content-hashed embeddings, append-only verdict ledger keyed by `(model, prompt_version, pair)`, decision log keyed by item. A full replay of the POC pipeline from caches reproduced identical results in minutes `[MEASURED]` — design the event flow to keep this property; it is the cheapest disaster-recovery and backfill story available.

---

## 18. Persistence inventory & boundaries (→ RFC §3)

| # | Store | Key | Mutability | Writers → Readers | Notes |
|---|---|---|---|---|---|
| 1 | Item store | item_id | immutable (+tag corrections) | ingest → all | system of record |
| 2 | **Duplicate map** | dup item_id → canonical | append-only | dedup → feed resolution, eval | **load-bearing**: any item must resolve to its story through it (Ch 4) |
| 3 | Curation flags | item_id | append-only | curate → assign (exclusion), search | set-asides stay searchable |
| 4 | Embedding cache | content hash | immutable | embed → assign, residual | rebuildable at cost |
| 5 | MinHash digests | item_id | immutable | dedup → LSH, fusion features | rebuildable |
| 6 | **Story store** | story_id | **mutable** (members, centroid, status, timestamps) | assign, gate, merge, expiry → feeds, enrich | the only truly mutable state in the system |
| 7 | **Decision log** (story membership + provenance) | (story_id, item_id) | append-only | assign, gate, merge → audit, attribution, eval | per-decision: gate taken, fusion score, verdict ref. This log is what made the POC's per-stage FP/FN attribution possible (Ch 11) — make it first-class |
| 8 | **Judge verdict ledger** | model \| prompt_version \| pair | append-only | judge → assign, gate, merge, audit | determinism + cost control (Ch 8) |
| 9 | Model artifacts | version | versioned immutable | calibration → assign gates, CI | fusion coefficients, gates, prompts, τ values |
| 10 | Eval sets + run ledger | set version / run id | frozen / append-only | eval → CI | the regression gate |

**Boundary framing for the RFC:** three planes. The **content plane** (1–5) is immutable or content-addressed — freely replicable and rebuildable. The **story plane** (6–7) holds the only mutable state, and even there the decision log is append-only; only the story store itself mutates. The **decision/model plane** (8–10) is append-only and versioned. Consequence: snapshots + the append-only ledgers give full replay/audit without event-sourcing the world.

---

## 19. Data model — ERD-ready entities (→ RFC §4)

```
CLIENT          (client_id PK, name, aliases[], active)
ITEM            (item_id PK, canonical_url, url_hash, title, body, source,
                 published_at, time_precision, is_boilerplate, embed_hash)
ITEM_CLIENT     (item_id FK, client_id FK)                  -- M:N tags from search/routing
DUPLICATE_OF    (dup_item_id PK → ITEM, canonical_item_id FK → ITEM)
STORY           (story_id PK, status ∈ {open, closed}, first_seen_at, last_seen_at,
                 centroid_ref, created_by ∈ {assign, cluster_gate, merge})
STORY_MEMBERSHIP(story_id FK, item_id FK, admitted_at,
                 admitted_by ∈ {auto, judge, cluster_gate, merge},
                 fusion_score, verdict_id FK nullable)      -- decision provenance, append-only
JUDGE_VERDICT   (verdict_id PK, item_a_id, item_b_id, model, prompt_version,
                 verdict ∈ {SAME, DIFFERENT}, decided_at)
STORY_METADATA  (story_id FK, membership_hash, title, summary, topic, entities[],
                 model, generated_at)                       -- versioned by membership_hash
EVAL_PAIR       (pair_id PK, item_a_id, item_b_id, final_label, ensemble_verdicts,
                 human_label nullable, set_version)
MODEL_ARTIFACT  (artifact_id PK, kind ∈ {fusion, calibration, judge_prompt},
                 version, payload, trained_on_sha)
```

Relationships: `ITEM ↔ CLIENT` M:N via `ITEM_CLIENT`; `ITEM` 0..1 `DUPLICATE_OF`; `STORY` 1—N `STORY_MEMBERSHIP` N—1 `ITEM`; memberships optionally reference the `JUDGE_VERDICT` that admitted them; `STORY_METADATA` versions per membership change.

**The schema lesson worth a sentence in the RFC `[VALIDATED DESIGN]`:** membership rows carry *how* the item got in (`admitted_by`, score, verdict reference). That provenance is what turns "why are these two articles together?" from an investigation into a lookup, and it is the substrate for per-stage quality attribution (Ch 11) and for CI on any future model change.

---

## 20. Ingestion branches & enrichment placement (→ RFC §5, §6)

### 20.1 Per-client vs generic branches

- **Per-client branch:** items arrive client-tagged from search. This is the branch the entire Part I validates end-to-end.
- **Generic branch (untagged market-wide news) `[OPEN]` with one validated bridge:** the clustering candidate gate's **blocking key is the client tag** (Ch 7) — untagged items cannot enter per-client assignment at all. Two-step guidance:
  1. **Mention-routing bridge** — scan generic items for portfolio-client mentions (alias matching; upgradeable to NER) and route hits into the per-client branch. This is effectively validated: it is exactly how the POC corpus was built (alias-regex over 2.07M raw wire items routed 4.1% to ≥1 client `[MEASURED]`), and the entire benchmark consists of items that entered this way.
  2. **True market-wide remainder** — needs its own clustering universe with a different blocking key (topic/sector, or windowed all-pairs at modest volume) and has **zero evaluation coverage**. Recommendation: scope it out of story formation in v1 (search-only surface), or budget a dedicated labeled slice before making quality claims. Do not silently extend the per-client quality numbers to it.
- Fallback lane: items whose per-client tagging failed should flow through the same mention-routing as a safety net before landing in the generic pool.

### 20.2 Item-level "content → context" enrichment

The RFC places enrichment *before* clustering. The POC's evidence, applied:

- **The consumption contract:** aggregation consumes, per item: title, 600-char lede, 1024-dim embedding, MinHash digest, client tags, timestamp (+precision), source, curation flag. Whatever the synthesis stage produces, it must emit/preserve exactly these — they are the clustering interface.
- **A measured warning `[MEASURED]`:** an earlier POC pass ran a controlled benchmark of prepending LLM-generated document context to text *before embedding* (50 long documents, retrieval task): hits@5 lift was **−0.03** — the enrichment-into-embedding path was tested and rejected. Do not assume context-enriched text embeds better; measure on a retrieval/pairing task first.
- **Where enrichment provably pays:**
  1. **Judge-visible context** `[MEASURED]` — giving the judge fuller ledes + publication dates (not richer embeddings) drove the 0.904 → 0.965 accuracy gain. Item enrichment that produces clean, information-dense ledes directly feeds this.
  2. **Structured features for the fusion scorer** `[OPEN]`, promising — real entity extraction would replace the weak regex entity-proxy feature (`caps_mismatch`, coefficient −0.444) with true entity-overlap, aimed exactly at the template failure mode. Untested; cheap to A/B once entities exist.
  3. **Curation signals** — template/filing detection is itself an item-level enrichment output (Ch 5).
- **Sequencing rule `[VALIDATED DESIGN]`:** anything that changes the *embedded text* invalidates every calibrated threshold, the fusion model, and the embedding cache (Ch 14.2). Finalize the enrichment design **before** running calibration; any later change to embedded content re-triggers full re-calibration. Enrichment that only *adds fields* (entities, flags, context for the judge) does not invalidate anything — prefer it.

---

## Appendix-grade provenance

All numbers trace to: the frozen benchmark `labeled_eval_set.csv` (543 pairs, sha-pinned in run records), the supplemental slice `labeled_eval_supplemental.csv` (199 pairs), eleven experiment ledger JSONs (`artifacts/v3/experiments/`, each with config snapshot, per-stage counts, metrics, attribution, accept/reject notes), `fusion_model.json` (coefficients, scaler, gates, CV scheme), `pos_calibration.json`, and the fully executed `story_clustering_poc_v3.ipynb` (§17 renders the cross-run ledger and progression chart).
