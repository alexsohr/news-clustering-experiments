# Cross-Company Story Clustering Pipeline — Design Spec

**Date:** 2026-06-23
**Status:** Approved (design); pending spec review → implementation plan.

**Goal:** Cluster a normalized news corpus into *stories* (groups of items about the same
real-world event) as a deterministic batch pipeline orchestrated by Prefect, with each stage an
atomic, independently-testable task, terminating in a versioned write to PostgreSQL.

**Architecture:** A single Prefect `@flow` runs ~12 `@task` nodes. Each task reads its input
artifact(s) from `runs/<run_id>/`, calls a pure function (the clustering logic lives in
`scripts/bloomberg_clustering.py`), and writes exactly one output artifact. Tasks pass artifact
*paths*, not DataFrames. The terminal task loads the batch into Postgres as a new immutable version.

**Tech stack:** Python 3.12, Prefect (orchestration), pandas/pyarrow (artifacts), numpy (vectors),
OpenAI `text-embedding-3-large` (embeddings) + `gpt-4.1-mini` / `gpt-5.4-mini` (judge, Responses API),
`datasketch` MinHash/LSH, scikit-learn LogisticRegression gate (pre-fit), `hdbscan` (optional
residual), PostgreSQL (sink, via SQLAlchemy/psycopg).

## Global Constraints

- **Run model:** batch one-shot. Every run reprocesses the entire corpus and rebuilds all stories
  from scratch. No persistent story state.
- **Story scope:** cross-company (global). `company_id` is a *feature/metadata*, NOT a candidate
  filter. The same event under different `company_id`s must be able to merge into one story.
- **Scale:** small (≤ ~20k items/batch). Candidate generation is exact brute-force cosine kNN within
  a publish-date window — no ANN index. HDBSCAN residual is feasible on the full set.
- **Gate:** pure inference. The pipeline loads a pre-fit `fusion_model.json`. Calibration is out of
  scope (separate, run-once concern).
- **Winning config baked in:** single-vector embeddings + **full-body** LLM judge + union-find
  clustering. No body chunking, no `chunk_pair` judge text. (Measured 2026-06-23: on a representative
  Bloomberg eval, single-vec+full-body F1=0.834 = news baseline; chunk_pair judge costs ~6 F1 pts of
  recall — see `artifacts/v4/bloomberg_experiment_findings.md`.)
- **Determinism:** same input + `config_hash` ⇒ identical story partitions (story_id UUIDs aside).
- **Secrets:** DB credentials + OpenAI key via env / Prefect Secret block; never committed.

## Resolved Design Decisions

| Fork | Decision |
|---|---|
| Run model | Batch one-shot (rebuild every run) |
| Story scope | Cross-company / global (embedding kNN candidate gen) |
| Scale | Small ≤20k (exact brute-force kNN) |
| Gate calibration | Load pre-fit artifact (pipeline = pure inference) |
| Orchestrator | Prefect (`@flow` / `@task`) |
| Clustering | Union-find over confirmed-SAME edges + cohesion split (replaces the streaming assignment loop) |
| Sink | PostgreSQL, **versioned batches** (new `run_id`, flip "current" on success) |
| Eval | Optional pre-load QA gate (task 11b) |

## Source Schema (normalized input)

`company_id, title, content, url, publish_date, ingestion_date` — one row per (article, company)
tag. The same article may appear under multiple `company_id`s; near-dup dedup collapses these and
aggregates the companies into a `company_ids` set on the representative.

## Atomic Tasks (DAG order)

Each task: **inputs → output artifact**, idempotent, cache-keyed on (input digest + `config_hash`),
independently unit-testable. Pure logic in `bloomberg_clustering.py`; each `@task` is a thin
load → call → persist wrapper.

| # | `@task` | Input → Output | Responsibility |
|---|---|---|---|
| 1 | `load_normalized` | source → `00_items.parquet` | Validate schema; coerce `publish_date`/`ingestion_date` to tz-aware datetimes; drop rows missing url or (title & content); assign stable `item_id = hash(canonical_url)`; quarantine malformed rows to `rejected.parquet`. |
| 2 | `exact_dedup` | 00 → `01_canonical.parquet` (+ `01_url_dupes.parquet`) | Canonicalize url, hash, drop exact-url dupes (keep earliest `publish_date`). |
| 3 | `near_dedup` | 01 → `02_items_dedup.parquet` (+ `02_dup_map.parquet`) | MinHash+LSH over **content**; collapse near-dup components to a representative; **aggregate `company_id` → `company_ids` set** across members; record member→rep map. |
| 4 | `embed_items` | 02 → `03_vectors.npy` (+ `03_item_order.parquet`) | Single-vector embed of `title + "\n" + content`; cache by content hash; L2-normalize; row-aligned to item order. |
| 5 | `generate_candidates` | 02+03 → `04_candidates.parquet` | **Cross-company** exact cosine kNN within ±`window_hours` of `publish_date`; keep pairs (a<b) with cosine ≥ `cosine_floor`. |
| 6 | `pair_features` | 02+04 → `05_features.parquet` | 7 fusion features per candidate pair (`bc.pair_features`: cosine, title_jac, minhash_jac, dt_days, len_ratio, num_mismatch, caps_mismatch). |
| 7 | `gate_decide` | 05 + `fusion_model.json` → `06_gated.parquet` | Load pre-fit gate; per pair label `auto_same` (p≥p_high) / `gray` (p_low≤p<p_high) / `auto_reject`. |
| 8 | `judge_gray` | 06+02 → `07_judged.parquet` | Full-body LLM judge (`gpt-4.1-mini` base; optional `gpt-5.4-mini` escalation band) on **gray pairs only**; disk-cached; UNCLEAR/unparseable → not-SAME; Prefect retries + concurrency limit + optional judge-budget cap. |
| 9 | `assemble_stories` | 06+07+03 → `08_assignments.parquet` | Edges = `auto_same` ∪ judge-SAME → **union-find** → `story_id`; **post-merge cohesion split** of low-cohesion components (Clustering Semantics below). |
| 10 | `residual_cluster` *(optional)* | 03+08 → `09_assignments.parquet` | HDBSCAN over still-singleton vectors → merge missed same-events via union-find. Config-toggled. |
| 11 | `persist_outputs` | 08/09+00+02_dup_map → `items_scored.parquet`, `stories.parquet`, `metrics.json` | Expand representatives back to **every original row** (re-attach near-dup + per-company members) so all rows get a `story_id`; emit stories table + run metrics. |
| 11b | `evaluate` *(optional QA gate)* | eval set + 08 → `metrics_eval.json`, `eval_chart.png` | If a labeled eval exists, score pairwise P/R/F1 + chart; **may fail the flow before the DB write**. |
| 12 | `load_to_postgres` | `items_scored`/`stories` parquet → Postgres | Transactional, versioned-batch bulk load; idempotent by `run_id`; terminal sink (DB Sink below). |

### Config (`RunConfig`, frozen, hashed into `run_id`)
`source_uri, window_hours (default 72), cosine_floor, gate_path, embed_model, judge_base_model,
judge_escalation_model|None, judge_escalation_band, judge_budget|None, split_threshold,
residual_enabled (bool), eval_path|None, eval_min_f1|None, database_url (secret), runs_dir`.

## Clustering Semantics — union-find + cohesion split

Batch + global scope ⇒ stories are **connected components of a confirmed-SAME pair graph**, not the
output of a time-ordered assignment loop. Order-independent and reproducible.

- **Edges:** only high-precision pairs — `auto_same` (gate p≥p_high) and judge-confirmed SAME.
  Gray-rejected and auto-reject pairs are never edges.
- **Transitive over-merge risk:** a single weak bridge can chain two tight sub-clusters (A–B–C with
  A–C unrelated). Mitigation (default-on, configurable): after union-find, for each component compute
  the centroid; if its weakest bridging edge / min member-to-centroid cosine < `split_threshold`,
  re-split that component at a stricter cosine floor. This is the primary merge-*precision* knob,
  tuned against the eval.

## DB Sink — versioned batches (step 12)

**Why versioned:** `story_id`s are not stable across runs (union-find components shift with the
corpus), so upsert-by-story_id is invalid. Each run writes a new immutable batch tagged `run_id`; a
single "current" pointer flips on success.

**Load mechanics (one transaction):** insert `clustering_runs` row (`status=running`) → `COPY`
`stories` + `story_items` → set `finished_at`, `status=succeeded`, flip `is_current` → commit.
Idempotent: skip if `run_id` already `succeeded`. On failure: `status=failed`, `is_current`
untouched, so consumers always read a complete batch. Connection from `database_url` secret; Prefect
retries on transient DB errors.

### ⚠️ PLACEHOLDER schema — replace with the real ERD when available

Only task 12 (and possibly column names in task 11) depend on this; upstream logic is ERD-agnostic.

```sql
clustering_runs                     -- batch provenance / "current" pointer
  run_id            text  PRIMARY KEY      -- = config+source hash
  config_hash       text
  source_snapshot   text                   -- corpus/version clustered
  started_at        timestamptz
  finished_at       timestamptz
  n_items           int
  n_stories         int
  is_current        bool                    -- exactly one true (partial unique index)
  status            text                    -- running | succeeded | failed

stories
  story_id          uuid  PRIMARY KEY       -- fresh per run, globally unique
  run_id            text  REFERENCES clustering_runs
  size              int
  company_ids       jsonb                    -- aggregated cross-company set
  first_publish     timestamptz
  last_publish      timestamptz
  representative_item_id  text
  representative_title    text

story_items                          -- one row per ORIGINAL item
  id                bigserial PRIMARY KEY
  run_id            text  REFERENCES clustering_runs
  story_id          uuid  REFERENCES stories
  item_id           text                     -- stable hash(canonical_url)
  company_id        text                     -- original (pre-aggregation) company
  url               text
  publish_date      timestamptz
  ingestion_date    timestamptz
  is_representative bool
  near_dup_of       text  NULL               -- representative item_id, if collapsed
  -- indexes: (run_id, story_id), (item_id), (company_id)
```

## Error Handling & Robustness

- **Idempotency/resume:** per-task `cache_key_fn` over (input digest + `config_hash`); failed flows
  resume from the last good artifact under `runs/<run_id>/`.
- **Judge:** disk-cached verdicts (re-runs free), retries+backoff on rate-limit/timeout, concurrency
  limit, conservative UNCLEAR→not-SAME, optional judge-budget cap (overflow gray → reject).
- **Embeddings:** batched + retried, cached by content hash → cheap partial-failure resume.
- **DB:** staging + transactional pointer flip; idempotent by `run_id`; failed load never flips
  `is_current`.
- **Validation:** task 1 fails fast on missing required columns; malformed rows quarantined
  (configurable quarantine-vs-hard-fail). Empty paths valid: zero candidates → all singletons; zero
  gray → no judge calls.
- **Observability:** `metrics.json` stage counts + Prefect run states/logging.

## Testing Strategy

Embedder and judge are **injected callables** → tests never hit the network.

- **Per-task unit tests** (tiny fixtures): exact_dedup collapses same-url; near_dedup collapses a
  cross-company near-dup and aggregates `company_ids`; generate_candidates includes an in-window
  cross-company pair and excludes an out-of-window one; gate_decide boundary cases with a synthetic
  gate; assemble_stories resolves a known edge list **and** splits an injected transitive-over-merge
  case; persist_outputs gives every original row a `story_id`.
- **DB test** against ephemeral Postgres (`pytest-postgresql`/testcontainers): idempotent by
  `run_id`; pointer flips on success; failed load leaves `is_current` untouched.
- **Golden flow test:** ~20-item synthetic corpus (known cross-company event + a near-dup),
  end-to-end with stub embedder + stub judge + ephemeral DB → asserts final stories and DB rows.
- **Determinism test:** run twice → identical partitions (story_id UUIDs aside).

## Out of Scope

- Gate calibration / labeling (separate run-once flow; reuses the Sonnet-subagent labeling +
  representative-eval method already built in `scripts/build_repr_eval_pairs.py` + `score_repr_eval.py`).
- Incremental/streaming assignment and persistent story state.
- ANN indexing / sharding for >20k corpora (design degrades to this but it's not built now).
- Body chunking / `chunk_pair` judge (measured worse; excluded).

## Open Questions / Placeholders

- **Real ERD** for the Postgres sink (current schema is a placeholder; only task 12 changes).
- Source connector for `load_normalized` (DB table vs parquet vs API) — to confirm at plan time.
- `split_threshold`, `cosine_floor`, `window_hours`, escalation band — initial values from the
  Bloomberg run; tune against the first representative eval of the real corpus.
