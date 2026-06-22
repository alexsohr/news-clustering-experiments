# POC Clustering v4 — Paragraph Chunking + Judge-Text A/B

**Date:** 2026-06-17
**Status:** Design approved, ready for implementation plan
**Predecessor:** v3 shipped at F1 = 0.870 (P = 0.917) — see `artifacts/v3/`, `project_v3_ship_result` memory
**Notebook:** `story_clustering_poc_v4.ipynb` (authored fresh; the current file is a copy of v3 and will be replaced)

---

## 1. Motivation

v3 embeds **one vector per item** from `title + body[:lede_chars]` (a truncated lede). Two
problems motivate v4:

1. **Production data is not just news.** The real corpus includes internal research articles and
   similar long-form artifacts where the discriminating signal lives deep in the body, not in the
   title or lede. A title+lede embedding misses it by construction.
2. **Chunking lets a match anywhere in the body link two items.** If we split the body into
   paragraph chunks and embed each, two items are the same story when *any* chunk pair is close
   enough — coverage the single truncated-lede vector cannot give.

v4 reworks the vector path to **paragraph-level chunks (multiple vectors per item)** and runs a
controlled **A/B on what text the LLM judge compares** (matched chunk pair vs. full body),
deciding by measured accuracy and cost rather than opinion.

---

## 2. Goals / Non-goals

**Goals**
- Replace the single-vector-per-item representation with **body paragraph chunks** wherever a
  vector drives a clustering decision: §6 (embedding), §8 (calibration), §10 (main loop), §11
  (HDBSCAN).
- Implement **two judge-text arms** behind a config flag and measure each on accuracy + cost.
- Keep a **single-vector baseline path** (`use_chunking=False`) that reproduces ≈0.870, so the
  chunking delta is cleanly attributable.
- Reduce the corpus to **3000 items** (repeatable seed) while keeping the 543-pair labeled eval
  set fully measurable end-to-end.
- Author a **clean, focused notebook** — only the validated v3 ship pipeline + the new chunking
  experiment; dead v3 experiments excluded.

**Non-goals**
- No change to dedup (§4 URL, §5 MinHash/LSH) — those already operate on the full body.
- No re-labeling of the eval set; we reuse `labeled_eval_set.csv` (543 pairs).
- No change to §12 metadata generation, §13 merge/expiry structure, §15 cost model structure.
- Not chasing a new F1 record; the deliverable is a **measured decision** about chunking and the
  judge-text mode.

---

## 3. Scope of change (by section)

| Section | v3 today | v4 |
|---|---|---|
| §3.6 Sampling | random 10k | **3000**, seed-repeatable, **seeded with eval items** |
| §4 URL dedup | URL hash | unchanged |
| §5 MinHash/LSH | `title + body` shingles (full body) | unchanged |
| §6 Embedding | 1 vec/item from `title + body[:lede]` | **N vecs/item from body paragraphs** |
| §8 Calibration | τ + fusion on item cosine | **re-fit on chunk max-pool cosine** |
| §10 Main loop | `item_vec · story_centroid` | **max over chunk pairs (nearest-chunk)** |
| §11 HDBSCAN | cosine over 1-vec/singleton | **cosine over chunk vecs → union items** |
| §10/§11/§13 Judges | title + lede text | **`judge_text_mode` A/B: chunk_pair \| full_body** |
| §14 Eval | item-pair P/R/F1 | unchanged (representation-agnostic) |

A **`use_chunking` master switch** selects between the v3 single-vector path (baseline) and the
chunk path throughout §6/§8/§10/§11.

---

## 4. Dataset reduction & eval seeding

- `CONFIG["target_canonical_items"] = 3000`; sampling uses the existing `CONFIG["random_seed"]`
  for repeatability.
- `df.sample(n=3000)` is **not** a prefix of `n=10000`, so a naive reduction would strand the eval
  set: the 543 labeled pairs reference 10k item_ids; at 3000 only ~9% of pairs retain both
  endpoints (~50 pairs). Unusable.
- **Fix:** build the corpus as **(all unique items referenced by `labeled_eval_set.csv`) ∪
  (random fill to 3000)**, fill drawn with the seed. The eval items are canonical by construction
  (they survived v3's §3–§5 filters), so they pass through again and the full 543-pair eval stays
  measurable end-to-end.
- **Documented caveat (in-notebook):** seeding enriches the corpus with near-dup-candidate items,
  so cluster density is not a pure random draw. Acceptable for measuring F1; flagged honestly in
  the §14 reporting.

---

## 5. Notebook structure

Authored **fresh** into `story_clustering_poc_v4.ipynb`, reusing on-disk `.cache/` and
`artifacts/` (embed cache, judge cache, `labeled_eval_set.csv`, minhash cache).

**Excluded vs. v3:** §9 removed-chunking stub; the ABORTED P2c 14-day second-sweep; superseded
fusion/escalation/judge **variant** cells (final ship config only); the `REUSE_EXISTING_EVAL=False`
re-labeling branch; the v2→v3 overlay archaeology (collapsed into one clean `CONFIG`).

**Style:** follows the user's notebook conventions — markdown cell before every code cell, liberal
inline comments, short single-concern cells (~10–25 lines), and **a chart with every measurement**.

**Two distinct baselines (the v3 0.870 was measured at 10k):**
- **Port-correctness gate** — run `use_chunking=False` at the **full 10k** corpus with v3's ship
  `judge_prompt_version` (so the embed + judge caches hit → near-free) and confirm F1 reproduces
  ≈0.870. If it doesn't, the fresh port is wrong — fix before proceeding.
- **Experiment baseline** — the `use_chunking=False` run at the **3k-seeded** corpus. This is the
  fair, same-corpus reference the chunk variants must beat. It will *not* equal 0.870 (smaller,
  differently composed corpus); that's expected.

---

## 6. Paragraph chunking (§6 rework)

**Chunks are body-only.** The title is deliberately excluded from the vector representation — it
aligns with the production goal (body-borne signal) and the title is not lost from the pipeline
(the judge still receives it; the §8 fusion scorer still uses it for `title_jac`).

**Procedure per item:**
1. Split body on paragraph boundaries (`\n\s*\n`; fallback single `\n` for sources without double
   breaks). Strip, drop empties.
2. Merge tiny paragraphs (`< min_chunk_tokens`, default **25**) into the neighbor so we don't embed
   3-word fragments.
3. Sentence-split huge paragraphs (`> max_chunk_tokens`, default **400**).
4. Cap at `max_chunks_per_item` (default **12**) — embedding cost is negligible (~$0.2 total);
   the cap bounds §10/§11 compute.
5. **Title-only / empty-body carve-out:** if an item has no usable body, fall back to
   **title-as-the-single-chunk**. This is the floor that keeps title-only items (common — most
   Reuters items) clusterable; with a real body the title never enters the vectors.

**Output:** `chunks_df` (`item_id`, `chunk_idx`, `chunk_text`) + a `chunk_vecs` matrix with an
`item_id → row-range` index, embedded through the existing async cached embedder
(`text-embedding-3-large`, unit-normalized).

---

## 7. Chunk-level similarity

### 7.1 §10 main loop
- Story state replaces the single `centroid` with the **stacked chunk vectors of all members**
  (`story["chunk_vecs"]`, updated on assign).
- **item↔story similarity = max over (item-chunk, story-chunk) cosine** =
  `(item_chunks @ story_chunks.T).max()`. The faithful form of "any chunk pair matches → linked."
- The **`argmax` pair is the matched chunk pair** that feeds the judge in `chunk_pair` mode.
- Gates (`τ_high`/`τ_low`, fusion) run on this max-pool similarity. Candidate filtering
  (shared client + 72h window) unchanged.
- `use_chunking=False` reverts story state to the v3 single centroid (baseline path).

### 7.2 §11 HDBSCAN
- Input: **all residual chunk vectors** of singleton items (not one per item). Precomputed
  cosine-distance → HDBSCAN over chunks.
- **Map back via union-find over items:** items owning chunks in the same HDBSCAN cluster are
  unioned into one story.
- **Over-merge guard (measure-first):** a generic paragraph shared across unrelated items could
  chain a giant component. Three existing gates already defend: §6.1b boilerplate routing removes
  templated wire items *before* clustering; §11.3 client-overlap gate drops no-shared-client
  clusters; §11.3b judge gate peels incoherent members. **Add a diagnostic** that logs the size of
  the transitive merge components; only build a dedicated "generic-chunk" filter **if** the
  diagnostic shows blowup. Do not pre-build the filter.

### 7.3 §8 recalibration (mandatory)
- Max-pool cosine is systematically **higher** than v3 centroid cosine (best-of-many-pairs), so
  `τ_high`/`τ_low` and the fusion model **must be re-fit** on chunk max-pool similarity over the
  543 labeled pairs (embed both items' chunks, take max-pool). §8 sweep/KDE/ROC/fusion cells run
  unchanged on the redefined similarity.
- **Two calibration profiles** (chunk vs. single-vector), selected by `use_chunking`.

---

## 8. Judge-text A/B

`CONFIG["judge_text_mode"]` applies to **all** gray-zone judges (§10.2, §11.3b, §13.0, §13.2):

- **`chunk_pair`** — per side: `title + published_at + the matched chunk(s)`. The matched pair is
  §10's `argmax` chunk pair; for §11/§13 it's the nearest chunk pair between the two stories. **No
  lede** — generalizes to research artifacts.
- **`full_body`** — per side: `title + published_at + full body`. Maximum context, maximum cost.

`judge_prompt_version` is **bumped per arm** so the two never collide in the judge cache.

---

## 9. Measurement (charted)

**Layer 1 — Judge-isolated** (cheap, decisive on the judge itself; corpus-size-independent):
run *both* arms across the 543 labeled pairs → **accuracy, P/R, tokens-per-call, $** per arm,
broken out by cosine bin (extends the §10.2c harness). Chart: per-bin accuracy + cost per arm.

**Layer 2 — End-to-end** (real impact; the judge only fires in the gray zone): run the full
pipeline on the eval-seeded 3000 corpus for each arm → **F1, P, R, total $**. Plus one
**baseline** run (`use_chunking=False`).

**Ledger runs** (§14.7 / §1.4 machinery): `baseline (single-vector)`,
`chunk + chunk_pair judge`, `chunk + full_body judge`. Chart: grouped P/R/F1 with the 0.85 ship
line and the 0.870 baseline marked.

---

## 10. Success criteria & decision rule

- **Chunking wins** only if its best end-to-end F1 **beats** the **3k-seeded single-vector
  experiment baseline** (same corpus), not merely ties. (The 10k≈0.870 run is only the
  port-correctness gate, not the comparison target.)
- **Judge arm wins** on end-to-end F1; if the two arms are within noise (~±0.01), take the
  cheaper one (`chunk_pair`).
- Report all three runs + both charts against the **0.85 ship line** in §14.7 / §16.
- If chunking does **not** beat baseline, that is a valid, documented outcome (keep single-vector;
  record why) — consistent with the v3 ledger discipline.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Title-only items → zero chunks → unclusterable | Title-as-single-chunk fallback (§6) |
| HDBSCAN over-merge via generic shared paragraph | Existing 3 gates + transitive-component diagnostic; filter only if needed |
| Re-using v3 τ on max-pool cosine (too low) | Mandatory §8 recalibration; two profiles |
| Eval set stranded at 3000 | Seed corpus with eval items (§4) |
| Fresh notebook regresses the 0.870 baseline | Baseline-validation gate before chunk comparison; reuse on-disk caches/artifacts |
| `full_body` judge token blow-up / 30k-TPM tier-1 cap | Judge fires only in gray zone (bounded volume); background-warm cache, no blocking batches |

---

## 12. New / changed CONFIG keys

| Key | Default | Purpose |
|---|---|---|
| `target_canonical_items` | `3000` | corpus size (was 10000) |
| `use_chunking` | `True` | master switch; `False` = v3 single-vector baseline |
| `chunk_min_tokens` | `25` | merge tiny paragraphs |
| `chunk_max_tokens` | `400` | sentence-split huge paragraphs |
| `max_chunks_per_item` | `12` | bound §10/§11 compute |
| `judge_text_mode` | `chunk_pair` | A/B arm: `chunk_pair` \| `full_body` |
| `judge_prompt_version` | bumped per arm | keep judge cache arms separate |

---

## 13. Open follow-ups (out of scope for this build)

- Title-prepended chunks as a third embedding variant, *if* paragraph-alone underperforms.
- Dedicated generic-chunk filter for §11, *if* the over-merge diagnostic fires.
- Re-validating chunking on a true research-artifact corpus once available (the news eval set is a
  proxy).
