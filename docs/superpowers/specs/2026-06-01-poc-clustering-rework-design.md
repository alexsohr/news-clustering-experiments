# POC Story-Clustering — v2 Rework Design (minimal un-break + GPT-4.1 judge)

**Date:** 2026-06-01 · **Status:** approved design, ready for implementation plan
**Author:** pairing session (Claude) · **Repo:** `/Users/alex/Projcts/news-clustering` (NOT a git repo)
**Companion docs:** `docs/poc-diagnosis-and-improvements.md` (full diagnosis + evidence),
`docs/story-clustering-poc-spec.md` (original POC spec), memory `project_poc_clustering_rootcause`.

---

## 0. TL;DR

`story_clustering_poc.ipynb` reported pairwise F1 ≈ **0.32** and an `ABANDON_OR_REWORK` verdict.
The investigation showed that verdict was reached on a **broken run**. This pass makes a **minimal,
scoped set of changes** in a **copy** of the notebook to get a *trustworthy* number, plus one
requested model swap:

1. **P0** — fix the §10 vector-indexing bug (the dominant cause).
2. **P2** — curate out templated "boilerplate" wire items (Approach A: set aside).
3. **Judge swap** — replace the Anthropic **Haiku** judge with **OpenAI `gpt-4.1`** (both judge call sites).
4. **DataFrame-centric** code only in the cells the above touch.

Everything is done in **`story_clustering_poc_v2.ipynb`** writing to **`artifacts/v2/`**; the original
is preserved for old-vs-new comparison. **Deferred** items (the bigger algorithmic/eval work) are
catalogued in §6 with enough detail to resume in a fresh session.

---

## 1. Background & problem (why we're here)

The notebook clusters financial-news items into "stories" (one real-world event each) via a
single-pass nearest-centroid assignment loop + an LLM gray-zone judge + an HDBSCAN residual pass,
with cosine thresholds calibrated from a 543-pair labeled set. Target: B-cubed F1 ≥ 0.85.

Measured failure (all reproduced against the artifacts — see companion diagnosis doc):
- Reported **pairwise** F1 = 0.323 (P 0.645 / R 0.215), **below** a one-line `cosine ≥ 0.65` baseline (0.405).
- Single-pass loop → **91% residual**, **8,300 / 8,489 stories are singletons**.
- `τ_high` "calibrated" to **0.88** — a degenerate `argmax(precision)` fallback (no τ ever reached P ≥ 0.95).
- ROC-AUC of cosine as a same-event separator = **0.748** (weak but not hopeless).

**Root cause found by static analysis (the panel of research agents did NOT catch this):** a
vector-indexing bug in §10 (cells 167–168). `assignment_vecs` is built in `canonical_items` order
(already stable-time-sorted), but the loop re-sorts with the **default *unstable* quicksort** and
indexes vectors by a fresh `arange` rank. **78.9% of timestamps are exactly midnight (date-precision)**
→ huge tie-groups → ~40–70% of items clustered with the **wrong same-day embedding**; the bug
propagates into HDBSCAN (cell 180). This is why the pipeline lost to plain cosine (the baseline uses
§7's *correct* cosine; the loop used scrambled vectors).

**Corrected-loop estimate (measured on the 543 eval pairs):** bug fixed + current judge ≈ **0.47**
(recall ~doubles); a strong gray-zone judge pushes higher. A learned multi-channel classifier on the
existing labels already hits **0.70 F1 / 0.92 AUC** — so a path to 0.85 plausibly exists (see §6).

---

## 2. Decisions locked (from brainstorming)

| Decision | Choice |
|---|---|
| Ambition of this pass | **Minimal un-break** (P0 + P2 only) + the GPT-4.1 judge swap |
| DataFrame refactor scope | **Touched cells only** |
| Where to make changes | **A v2 copy** (`story_clustering_poc_v2.ipynb`, outputs → `artifacts/v2/`) |
| Boilerplate handling | **Approach A** — set aside (route out of clustering), report two numbers, $0 API |
| Judge model | **OpenAI `gpt-4.1`** (full, not mini/nano) replacing Haiku at both judge call sites |

---

## 3. In-scope design

### 3.1 Shape & layout
- Copy `story_clustering_poc.ipynb` → `story_clustering_poc_v2.ipynb`. Original untouched.
- New outputs → `artifacts/v2/`. Original `artifacts/` (and the old 0.32 numbers) preserved.
- Reuse `.cache/embeddings.pkl` (content-hash keyed) → re-embedding is **free**.
- Add a top **"v2 changelog" markdown cell** listing the 3 changes + a pointer to this spec.
- Notebook-style memories apply: markdown cell before each code cell; short, single-concern cells;
  prefer `insert_cell`/`edit_cell_source` over giant overwrites.

### 3.2 P0 — fix the §10 indexing bug (id-keyed + regression guard)
Replace the unstable-sort + `arange` position assignment in cell 167 with an `item_id`-keyed mapping:
```python
pos_of_id = pd.Series(np.arange(len(canonical_items)), index=canonical_items["item_id"])
sorted_items = canonical_items.sort_values("published_at", kind="stable").copy()
sorted_items["pos"] = sorted_items["item_id"].map(pos_of_id)   # TRUE position, not re-sorted rank
# permanent regression guard — makes the bug impossible to reintroduce silently:
assert (canonical_items["item_id"].to_numpy()[sorted_items["pos"].to_numpy()]
        == sorted_items["item_id"].to_numpy()).all(), "item↔vector mapping broken"
```
- In the §10 loop, access the vector by id (e.g. a small `vec_for(item_id)` helper backed by
  `assignment_vecs[pos_of_id[item_id]]`) rather than by re-sorted position.
- Because `member_idxs` then store **correct** positions, **§11 (HDBSCAN, cell 180/184) and §13
  (merge, cell 216) need no change** — the fix propagates downstream.
- `assignment_vecs` stays an `(N,1024)` array internally (correct, built in canonical order); the
  `item_id`→position Series is the portable bridge that satisfies the "embeddings keyed by id" intent.

### 3.3 DataFrame touches (ONLY in cells P0/P2 touch)
- **`pos_of_id`** — Series keyed by `item_id` (the id→vector bridge). New.
- **`stories_df`** — materialize the `stories` list-of-dicts to a DataFrame at the §10.5 and §11.4
  boundaries: columns `story_id, n_items, member_ids, member_idxs, item_clients, first_seen_at,
  last_seen_at, closed_at` (+ optionally a separate centroid matrix; do NOT stuff 1024-d arrays into
  a display DataFrame). The **live loop keeps fast objects**; `stories_df` is for inspection/QA/export.
- **`outcomes`** — a column on the sorted items frame instead of a parallel Python list.
- **`is_boilerplate`** — boolean column on `canonical_items` and on the eval-pair frame.
- Everything else (entity sets, MinHash structures, in-loop candidate lists) is left as-is
  (that's the deferred notebook-wide sweep, §6).

### 3.4 P2 — boilerplate curation (Approach A: set aside)
- **Detector — structural templates ONLY.** Flag titles/ledes matching: `^REG -`, `Form 8`
  (incl. `Form 8.5`, `Form 8.3`), `(EPT/`, `(DD)`, `Net Asset Value`/`\bNAV\b`, conference/PR notices
  (`to Participate`, `Invites You`, `to Present at`, `One-on-One`), network-expansion PRs
  (`4G LTE .* Expands`, `Expands .* 4G LTE`), `Total Voting Rights`, `Transaction in Own Shares`,
  `Daily Share`, Zacks analyst-blog patterns.
  **MUST NOT flag** editorial wire tags: `UPDATE-N`, `WRAPUP`, `FACTBOX`, `CORRECTED`, `TAKE A LOOK`
  — **32% of true-SAME pairs carry these** and they are legitimate same-story follow-ups.
- **Partition.** Flagged items → a `boilerplate_df`, excluded from §10/§11/§13 (mirror the existing
  no-client "noise" partition pattern in §3).
- **Hand-audit cell.** Print: count + % of corpus flagged; a random sample of flagged items; and a
  random sample of *high-cosine items that were NOT flagged* — to catch both over- and under-flagging.
- **Recalibration.** Re-run §8 calibration on the **non-boilerplate** pairs → expected to restore a
  **non-degenerate `τ_high`** (~0.78–0.85 per research, vs today's degenerate 0.88). Report old vs new
  τ. (Deterministic, no API.) The §10 loop uses the recalibrated τ.
- **Eval reporting — two numbers, same pairwise metric, explicitly labeled "pairwise on the stratified
  eval set, NOT B-cubed":**
  1. **Full 543 pairs** (vs old 0.32) — isolates the **bug-fix** effect. Boilerplate items, now
     unclustered, resolve to "not co-clustered" → predict DIFFERENT (correct TNs for the templated
     DIFFERENT pairs).
  2. **Non-boilerplate subset** (drop pairs where either item `is_boilerplate`) — the **curation** effect.
  3. Baselines (`cosine ≥ 0.65`, title-Jaccard) computed on **both** sets for honest comparison.

### 3.5 Judge swap — Haiku → OpenAI `gpt-4.1`
- Add `CONFIG["judge_model"] = "gpt-4.1"`. Keep `haiku_model` defined (still used by optional §9 unless
  also swapped — see note).
- New **OpenAI judge helper** used by BOTH call sites:
  - §10 gray-zone `haiku_judge_same` (cell 165) → `gpt_judge_same(item_row, rep_row)`.
  - §13 merge judge (cell 214) → same helper / OpenAI call.
  - Reuse the `AsyncOpenAI` client + 50-RPM limiter already present (§7/§9). Same prompts,
    `temperature=0`, parse `SAME`/`DIFFERENT` from the response.
- **Cache namespacing is automatic:** both cache keys already hash the model id
  (`_judge_key(model,…)`; `sha256("merge|…|model")`), so switching the model id yields new keys; old
  Haiku caches stay on disk, harmless. Gray-zone/merge calls re-run on `gpt-4.1` (modest OpenAI spend).
- **Cost cells (§15):** update the judge lines from Haiku pricing to `gpt-4.1` pricing
  (gray_judge, merge_judge; and doc_context only if §9 is run).
- **§9 note:** the optional long-doc retrieval experiment also calls Haiku (doc-context question
  generation, cells 154 etc.). Swap those to `gpt-4.1` too for consistency, BUT §9 does not affect the
  clustering number and only runs if §9 is executed.

### 3.6 Acceptance criteria ("done")
- `story_clustering_poc_v2.ipynb` runs top-to-bottom without errors.
- The §10 **regression `assert` passes**.
- Reported in a clearly-labeled results cell:
  - (a) corrected pipeline F1 on the **full** 543-pair set vs the old 0.32;
  - (b) corrected pipeline F1 on the **non-boilerplate** subset;
  - (c) baselines on both sets;
  - (d) new outcome mix (residual % / #singletons) vs old (91% / 8,300);
  - (e) old vs new `τ_high`/`τ_low`;
  - (f) boilerplate hand-audit (counts + samples).
- A `artifacts/v2/v2_findings.md` summarizing the above and explicitly stating the metric caveat.

### 3.7 Error handling / determinism
- Reuse existing 429-retry + sliding-window rate limiter for the OpenAI judge.
- `kind="stable"` sort + unchanged seeds (`random_state=42`) keep re-runs deterministic; judge calls
  cached at `temperature=0`.
- Over-flagged boilerplate **fails safe** (an item just becomes a singleton, never a wrong merge).
- Embedding cache reused (content-hash keyed) → no re-embedding cost.

---

## 4. Risks & mitigations
- **Boilerplate detector over-/under-flags** → hand-audit cell + fail-safe direction; iterate patterns.
- **GPT-4.1 judge cost** → only gray-zone + merge candidates hit it; cached; bounded by gray-zone size.
- **Eval set still stratified/pairwise** → explicitly labeled; not used as the final go/no-go (that's
  deferred P5). The two-number reporting separates bug-fix from curation effects.
- **No git** → the v2-copy strategy is the safety net; original notebook & artifacts preserved.

---

## 5. Explicitly OUT of scope (this pass)
Learned multi-channel classifier (P1), dense contiguous corpus (P4), corpus-level B-cubed + independent
gold set (P5), notebook-wide DataFrame sweep, embedding-model experiments. All detailed in §6.

---

## 6. DEFERRED BACKLOG — detailed, for a future session

> Resume context: read `docs/poc-diagnosis-and-improvements.md` first (full evidence). The working
> set is 8,735 items; embeddings live in `.cache/embeddings.pkl` as `{content_hash → np.float32[1024]}`
> (NOT keyed by item_id — they're hashes of `embed_input`). The 543 labeled pairs are in
> `artifacts/labeled_eval_set.csv` (450 DIFFERENT / 93 SAME). `.cache/canonical.parquet` is the FULL
> 8.2M-row raw feed (2006–2016, 96% Reuters), NOT the working set. `.cache/mentions_slice.parquet`
> has `item_id, published_at, mentioned_clients` BUT its `item_id` does NOT match the notebook's
> regenerated ids — key joins on `url` instead. Investigation workflow: `scripts/poc_diagnosis_workflow.js`.

### D1 — Learned multi-channel pair classifier (replaces the single cosine gate) — HIGHEST LEVERAGE
- **Why:** cosine alone is intrinsically weak for *same-event* (ROC-AUC 0.748, best single-threshold
  F1 0.43, P(SAME|cosine) plateaus ~0.25). A quick 5-fold CV on the existing 543 labels with cheap
  features already gave **GBM AUC 0.919 / F1 0.704** and **LogReg AUC 0.903 / F1 0.688** vs
  cosine-only 0.42 — a **+0.28 F1** lift with no new embedding model.
- **Features that worked:** `cosine`, `time_delta_hours`, `title_token_jaccard`, `lede_token_jaccard`,
  numeric-mismatch flag (both titles have numbers but disjoint → DIFFERENT), `is_boilerplate` flag.
  Notable learned signal: `title_jaccard` and `boilerplate` carry **negative** weight toward SAME —
  conditioned on cosine, near-identical wording signals a *templated clone* (DIFFERENT), moderate
  overlap signals a *paraphrase* (SAME).
- **Add next:** a real **entity-overlap** channel from per-article NER (NOT the coarse `client` tag,
  which is non-discriminative here because eval pairs share a client by construction); a soft
  time-decay similarity.
- **Architecture change:** replace `τ_high`/`τ_low`/gray-zone gating with a calibrated `P(SAME)` from
  the classifier (Miranda 2018 "SVM-merge" pattern; logistic/GBM). Calibrate with isotonic regression.
- **Eval caveats to fix when measuring:** the best-F1 threshold above is tuned on OOF (mildly
  optimistic); pairs share items across folds (split by story/time, not by pair); still the stratified
  set. Use proper grouped CV.
- **Expected impact:** the single most promising path to ≥ 0.85.

### D2 — Stronger / escalating gray-zone judge (beyond a single binary verdict)
- **Why:** with `τ_high` high, the gray zone [0.54, 0.88) holds **71 of 93** SAME pairs; their fate
  rests on one judge. (This pass swaps Haiku→GPT-4.1, a partial mitigation.) Full fix: a **calibrated
  escalation cascade** (cheap judge or cross-encoder decides confident pairs; escalate the ambiguous
  middle to a stronger model/ensemble — cf. "Trust or Escalate", arXiv 2407.18370), with isotonic
  calibration to respect the ~17% SAME base rate.
- **Note on measuring the judge:** `final_label` IS the Sonnet/GPT/Gemini ensemble majority, so
  "a frontier judge nearly solves it" is partly circular — needs an **independent** gold set (see D4).
- **Tools:** cross-encoder rerankers (bge-reranker-v2, mxbai-rerank, Cohere Rerank 3.5, Voyage rerank).

### D3 — Dense contiguous corpus (fix the 91% residual)
- **Why:** the notebook **random-samples 10k items across a 2-year span** (~208× sparser than the
  feed) → most items have no same-client neighbor in their 72h window → residual collapse. Busiest
  client (JPMorgan) ≈ 17 items/day in the full pool; median ≈ 4/day; after down-sample, < 1/day/client.
- **Fix:** select a **contiguous, dense time-slice** (e.g. all client-mentioning items from a few busy
  weeks of `.cache/mentions_slice.parquet`), not a 2-year random sample. Ablation: ~50k contiguous
  items restores 72h neighbor availability to ~0.69, ~200k to ~0.96.
- **Keep** the 72h active window (it captured 100% of true-SAME pairs; max gap was exactly 72h).
- **Caveat:** also note the feed is **96% single-source Reuters** and **~95% of items mention no
  client** — the "client" signal is thin; consider deriving entities from text instead.

### D4 — Honest evaluation (corpus B-cubed + independent gold set + no leakage)
- **Why:** current eval is **pairwise F1 on a cosine-stratified, hard-pair-enriched** set (not the
  spec's corpus B-cubed), the **same pairs calibrate τ and report F1** (leakage), and `final_label`
  is byte-identical to the LLM ensemble (label circularity). A representative sample would score
  *differently* (a reweighting sim suggested pairwise F1 *lower*, ~0.26–0.38; B-cubed can look
  deceptively high because 98% of stories are singletons → easy TNs).
- **Fix:** build a small **representative, contiguously-sampled, independently human-labeled** clustered
  gold set (annotators do NOT see the LLM verdicts); compute **B-cubed P/R/F1** scoring non-singleton
  clusters separately from singleton detection (Cattan 2021, arXiv 2106.04192); split
  **calibration / eval by story and time**, not by pair.
- **Also fix the §7.1 pair-enumeration bug:** the within-72h two-pointer does NOT reset `j` per `i`
  (it carries the high-water mark), so it **skips many valid close pairs**, biasing the eval pool.
  Correct form: for each `i`, scan `j = i+1` forward while `ts[j]-ts[i] <= window`.

### D5 — Notebook-wide DataFrame sweep (the broader ergonomics refactor)
- Convert remaining list/dict/set state to DataFrames where sensible: entity sets, MinHash structures,
  in-loop candidate lists, `embed_cache` access, etc. Keep the hot story accumulator as objects but
  expose `stories_df` everywhere downstream. Strongly consider storing embeddings as a DataFrame/Series
  keyed by `item_id` globally (this pass only does the id→position bridge in the touched cells).

### D6 — Representation experiments (cheap de-risking, do before D1's embedding work)
- (a) Embed **full body** (or title+body) from `.cache/canonical.parquet` and re-measure ROC-AUC
  (tests "title+lede is too coarse for event vs topic").
- (b) Evaluate a finance/domain or finer embedding (voyage-finance-2 / Fin-E5 / voyage-3-large /
  Cohere embed-v4) **on the 543 pairs** — FinMTEB shows general-benchmark scores don't transfer.
- (c) Structural **template-masking** before embedding (embed only the variable slots of templated text).

### D7 — Re-label a fresh, representative eval set (Approach C, if needed)
- If/when the corpus changes (D3) or boilerplate is curated globally, re-sample stratified pairs from
  the clean pool and re-run the 3-vendor ensemble (Sonnet 4.6 + GPT-5.2 + Gemini 3.5 Flash),
  `temperature=0`, ~$45 / ~50 min. Keep the ensemble≠Haiku guardrail (no circular calibration).

---

## 7. Key facts / gotchas a fresh session needs
- **Bug essence:** cell 167 `sort_values("published_at")` is unstable; 78.9% midnight timestamps →
  ties scrambled → wrong vectors. Fixed via stable sort + `item_id`→position map + an `assert`.
- **Embeddings cache** is keyed by **content hash** of `embed_input`, not item_id (see `embed_cache_key`).
- **mentions_slice `item_id` ≠ notebook ids** — join on `url`.
- **canonical_items order** = stable published_at (cell 55 → cell 81). `assignment_vecs` follows it.
- **Judge cache keys** include the model id → swapping models auto-namespaces the cache.
- **Numbers to beat:** old pipeline 0.32; cosine baseline 0.405; best cosine 0.43; corrected-loop
  estimate ~0.47; multi-channel classifier 0.70/0.92.

## 8. References & artifacts
- Diagnosis + evidence: `docs/poc-diagnosis-and-improvements.md`
- Investigation workflow: `scripts/poc_diagnosis_workflow.js`
- Original spec: `docs/story-clustering-poc-spec.md`
- Memory: `project_poc_clustering_rootcause`, `feedback_notebook_style`, `feedback_short_cells`,
  `project_anthropic_tier1_tpm`
- Literature: Miranda 2018 (aclanthology.org/D18-1483), Saravanakumar 2021 (aclanthology.org/2021.eacl-main.198),
  USTORY 2023 (arxiv 2304.04099), Steck & Ekanadham 2024 (arxiv 2403.05440), FinMTEB (arxiv 2502.10990),
  Trust-or-Escalate (arxiv 2407.18370), Cattan 2021 (arxiv 2106.04192), Amigó 2009 B-cubed
  (doi 10.1007/s10791-008-9066-8).
