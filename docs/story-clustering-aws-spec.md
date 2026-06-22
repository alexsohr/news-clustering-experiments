# Story Clustering for Banker News & Research on AWS — Deployment Spec

The production specification. Lineage: `research.md` (foundational news-clustering research) → `story-clustering-aws-jpm.md` (first AWS adaptation) → `banker_client_clustering_pipeline.md` (Glue refinement) → this document. Where this spec diverges from the predecessors, the divergence is intentional and is called out inline. The companion `story-clustering-poc-spec.md` defines the experimentation notebook that must produce calibrated thresholds and a labeled eval set before any code in this spec is built.

---

## TL;DR

- **Architecture:** the classic Topic Detection & Tracking single-pass-nearest-cluster assignment (Miranda et al. 2018), with 2025-era dense embeddings, an LLM-judge gray-zone gate, and HDBSCAN on the residual. Same shape as `research.md` recommended and as Meltwater / NewsCatcher / GDELT have converged on.
- **Stack:** AWS Glue Workflow with **two** Glue jobs (Job 1 ingest + embed; Job 2 cluster + label + persist), Aurora PostgreSQL + pgvector + HNSW as the system of record, S3 as the handoff between jobs, EventBridge for the nightly cron, CloudWatch for observability. Bedrock for everything sensitive (Claude Haiku as gray-zone judge and entity extractor, Claude Sonnet for metadata generation). OpenAI Batch API for embeddings (with Cohere Embed v3 on Bedrock as the in-VPC fallback).
- **Stories are global objects, not partitioned.** Each story carries an `affected_clients` set as metadata via the `story_client` M:N table. The Apple-vs-Samsung lawsuit is one story tagged with both clients, not two parallel stories. This deliberately diverges from `research.md`'s Caveat #7 — that caveat is about multi-*tenant* SaaS isolation; your "clients" are corporate entities that are *subjects* of news, not tenants of the platform.
- **Always-on entity extraction.** A consequence of the global-stories choice: items don't arrive partitioned by entity, so a Bedrock Haiku entity-extraction call per item is required, not optional.
- **Multi-vector retrieval with contextual chunking.** Items above ~2000 tokens (research notes especially) are split into 800-token chunks with 100-token overlap; each chunk is embedded with a doc-level Haiku-generated context summary prepended at embed time (Anthropic Contextual Retrieval, doc-level variant). Assignment-side embedding remains a single vector from `title + lede`.
- **Inline metadata generation.** Story titles and summaries are produced in the same Glue job that performs assignment, before commit. Bankers never see stories with placeholder titles. This is one of the few deliberate divergences from the predecessor Doc 2.
- **Pluggable source connectors from day 1.** A `source` registry table plus a `SourceConnector` ABC let new sources (next research feed, next news API) be added by inserting a registry row and shipping a Python wheel — no Glue job redeploy.
- **The decisive simplification in your use case is entity grounding.** Every Perplexity result arrives tagged with the banker client it was searched for, and JPM research can be tagged at ingest. Combined with the entity-overlap gate, this collapses the assignment step from "ANN across all active stories" to "ANN across active stories that share at least one client with this item."
- **B-cubed F1 ≥ 0.85 is the go-live target.** Calibrate `τ_high` and `τ_low` on the POC's labeled eval set before flipping the EventBridge schedule on. Re-calibrate whenever the embedding model changes.

---

## Prerequisites: outputs from the POC

This deployment spec assumes the POC defined in `story-clustering-poc-spec.md` has been run and produced:

1. **`config/calibration.json`** — `{tau_high, tau_low, minhash_threshold, residual_threshold, hdbscan_min_cluster_size}` calibrated against the labeled set. These values are read at Job 2 startup; do not hard-code.
2. **`eval/labeled_set_v1.csv`** — the ~500–1000 hand-labeled item pairs, checked into the production repo. This becomes a CI regression test: any change to the embedding model, thresholds, or LLM prompts re-runs the labeled set and fails the build if B-cubed F1 drops by >0.03.
3. **`poc_findings.md`** — narrative of what the POC learned. Inform Stage 2 and 3 priorities below.

If any of these are missing, run the POC first. The thresholds in this spec are starting points from the literature, not calibrated optimums; shipping with priors is acceptable only if labeled-set calibration is on the immediate roadmap.

---

## How your use case differs from generic news clustering

`research.md` assumes a meaningfully harder problem: a stream of arbitrary news, no prior entity tagging, the clusterer has to discover entity identity from the embedding alone. That is the problem Meltwater and GDELT solve. Your problem is structurally easier in three ways:

**Items arrive entity-tagged.** Perplexity is invoked per banker client with a query like `news about <Banker Client Name>`. The returned items are already known to be about that entity. JPM internal research is tagged at ingest with the banker clients it covers (either via existing metadata or a Haiku extraction pass). Clustering does not need to discover the entity — only the *event involving that entity*.

**Candidate sets are small.** For a typical banker client (Microsoft, Tesla), there are 0–30 active stories at any time within the 72-hour window. The active-story ANN search is a per-entity-filtered query, not a corpus-wide search. pgvector HNSW is overkill at this scale — a sequential scan over 30 centroids is fast — but the index costs nothing and protects against unexpected story growth during active news cycles.

**Cluster sizes are small.** A banker client like Microsoft might have 5–30 items in a 72-hour window across all sources. A specific story ("Microsoft Q3 earnings beat") might have 2–8 items. We are not clustering thousands of items per entity; we are clustering tens. This makes LLM verification of every borderline match affordable, and makes HDBSCAN on the residual a quick step (typically <200 items, often <50).

The hard part of your problem is not the clustering algorithm. It is the operational plumbing: scheduling nightly per-source jobs, handling Perplexity rate limits and quirks, deduplicating cross-entity items (Apple-vs-Samsung lawsuits, sector research notes covering multiple clients), keeping JPM internal research in sync, and generating useful metadata bankers actually trust. The architecture below focuses on those.

---

## Architectural decision: global stories with `affected_clients` metadata

You have three plausible scopes for a story:

**Option A: scope stories to a single `banker_client`.** Story belongs to one entity. The Apple-vs-Samsung lawsuit becomes two stories — one in Apple's pool, one in Samsung's, potentially with diverging metadata. Doc 1 recommended this; it matches `research.md`'s per-client isolation caveat directly.

**Option B: scope stories to `(banker, banker_client)`.** Each banker has a private story space per client. Identical stories exist in parallel across bankers. Strongly discouraged by Doc 1.

**Option C (chosen): stories are global objects with an `affected_clients` set as metadata.** A story is a *real-world event*. The Apple-vs-Samsung lawsuit is one story tagged with both Apple and Samsung. A sector research note covering five mega-cap tech firms is one story tagged with all five. Bankers read through `banker_subscription → client → story_client → story`.

**Why Option C is correct for this product.**

- **News doesn't change based on who reads it.** Banker A and Banker B should not see slightly different titles/summaries for the same Microsoft earnings story.
- **Cross-entity events are first-class.** M&A, lawsuits, joint ventures, sector reports — these are the events bankers care about most, and they intrinsically involve multiple clients. Option A duplicates them; Option C represents them naturally.
- **Storage and LLM cost are roughly N× lower than Option A**, where N is the average number of clients per cross-entity story.
- **Subscriptions are a separate concern.** A banker's view is derived at read time: `SELECT s.* FROM story s JOIN story_client sc ON s.story_id = sc.story_id JOIN banker_client bc ON sc.client_id = bc.client_id WHERE bc.banker_id = :b AND s.closed_at IS NULL`.

**Why this doesn't violate `research.md` Caveat #7.** That caveat ("Cross-client isolation matters. Do not pool stories across clients.") is written for a multi-*tenant* SaaS where "client" means a paying customer with their own data, relevance criteria, and privacy expectations. In your domain, `client` is a banker_client — a *subject of news*, not a *tenant of the platform*. There is one real-world Apple. Pooling Apple stories across all bankers who cover Apple is correct, not a privacy leak. The caveat doesn't translate; the architecture is.

**Implication: entity extraction is required, not optional.** Under Option A, you knew the entity by construction (it was the search query). Under Option C, the source layer still gives you the *seed* clients (Perplexity returns Apple-tagged items for the Apple query), but an item about Apple may also mention Samsung as a counterparty, Tesla as a competitor, and three regulators as actors. To find that the same Apple-vs-Samsung article should join the Samsung story too, you must extract entities at ingest. This is one Bedrock Haiku call per canonical item, ~$0.0001 each, ~$10/day at 100k items. Cheap. Mandatory.

---

## Reference architecture (AWS-native, Glue Workflow)

```
EventBridge schedule (cron: nightly, e.g. 02:00 UTC)
   │
   ▼
Glue Workflow: daily_clustering_pipeline
   │
   ├── Job 1: gather_and_embed_job          (Glue Spark, 10-20 G.1X workers, ~45 min)
   │     Source connectors → normalize → MinHash dedupe →
   │     entity extraction → contextual chunking → assignment+retrieval embeddings →
   │     write enriched Parquet to S3
   │
   ├── Job 2: cluster_assign_label_job      (Glue Spark, 1 G.4X worker, ~60 min)
   │     Read enriched Parquet from S3 →
   │     load active stories from Aurora →
   │     single-pass assignment with gates (auto / gray-zone / residual) →
   │     HDBSCAN on residual → recompute centroids →
   │     inline Sonnet metadata generation →
   │     close stale stories →
   │     bulk write to Aurora
   │
   ▼
CloudWatch metrics + SNS alarms
   │
   ▼
Downstream banker UI (reads from Aurora with story_client M:N join)
```

**Why two jobs, not three.** Doc 2 used a three-job structure where metadata generation was a separate async Job 3, with stories briefly visible in the database with placeholder titles. This spec deliberately collapses metadata into Job 2: stories must not exist in queryable form with placeholder titles, even transiently. The cost is that Job 2 does two compute-shape-incompatible things (driver-bound NumPy clustering, then Spark-parallel Sonnet calls), so we size for the more demanding profile (G.4X driver memory) and let the metadata phase parallelize across the worker's 16 vCPUs. The benefit is operational simplicity and a stronger product guarantee.

**Why Glue Workflow over Step Functions or EventBridge Pipes.** Glue Workflow renders the DAG natively in the Glue console with per-job state, runtime, and failure history. Step Functions is the right escalation if the DAG ever needs branching, parallel sub-jobs, or finer error-handling logic; for a linear two-job pipeline, Glue Workflow has lower configuration overhead and one less service to wire alarms to.

**Concrete sizing.** Assuming ~500 active banker clients, ~100k items per nightly run after Perplexity + JPM research ingest (this is generous):

- Job 1: 15 G.1X workers (4 vCPU, 16 GB each). Wall-clock ~45 min. Spark parallelizes ingest, entity extraction, and embedding API calls naturally. Aurora is touched only for the small client+source registry reads.
- Job 2: 1 G.4X worker (16 vCPU, 64 GB driver). Wall-clock ~60 min. Clustering algorithm on driver (single-threaded NumPy); metadata generation parallelizes across 16 vCPU UDF slots; bulk write to Aurora via S3 staging + `aws_s3.table_import_from_s3`.
- Aurora: `db.r6g.large` is plenty. HNSW index on active stories stays well under 1 GB.
- Glue compute cost: ~$7/day. API cost (Bedrock + OpenAI + Perplexity): ~$60/day. Total ~$2,000/month at this scale.

You can scale this 10× without architectural changes by raising Job 1's worker count.

---

## Aurora schema

```sql
-- Reference tables, populated outside this pipeline

CREATE TABLE banker (
  banker_id        uuid PRIMARY KEY,
  display_name     text NOT NULL,
  email            text UNIQUE NOT NULL,
  active           bool DEFAULT true,
  created_at       timestamptz DEFAULT now()
);

-- Clients as first-class deduplicated entities (corporate entities the bankers cover)
CREATE TABLE client (
  client_id        uuid PRIMARY KEY,
  display_name     text NOT NULL,
  legal_name       text,
  aliases          text[],          -- for entity normalization
  primary_ticker   text,
  industry_codes   text[],
  active           bool DEFAULT true,
  created_at       timestamptz DEFAULT now()
);
CREATE INDEX client_aliases_gin ON client USING gin (aliases);

-- M:N: which bankers cover which clients
CREATE TABLE banker_client (
  banker_id        uuid REFERENCES banker NOT NULL,
  client_id        uuid REFERENCES client NOT NULL,
  PRIMARY KEY (banker_id, client_id)
);
CREATE INDEX banker_client_by_client ON banker_client (client_id);

-- Source registry (driven by config; Job 1 reads this at startup)
CREATE TABLE source (
  source_id        text PRIMARY KEY,                   -- 'perplexity', 'jpm_research', future sources
  display_name     text NOT NULL,
  connector_class  text NOT NULL,                      -- Python class path, resolved from a wheel
  config_json      jsonb NOT NULL,
  source_rank      int NOT NULL DEFAULT 50,            -- for medoid selection; Bloomberg=10, Reuters=15, WSJ=20, FT=20, AP=25, ...; lower is higher
  enabled          bool DEFAULT true,
  added_at         timestamptz DEFAULT now()
);

-- Items, globally deduped on url_hash
CREATE TABLE item (
  item_id           uuid PRIMARY KEY,
  source_id         text REFERENCES source NOT NULL,
  url               text,
  url_hash          bytea NOT NULL UNIQUE,
  title             text,
  body              text,
  lede              text,                              -- first 400-600 chars of body
  published_at      timestamptz,
  ingested_at       timestamptz DEFAULT now(),
  assignment_vec    vector(1024),                      -- single vector from title + lede
  entities          jsonb,                             -- {people, orgs, tickers, locations} from Haiku
  minhash           bytea,
  is_duplicate      bool DEFAULT false,
  duplicate_of      uuid REFERENCES item,
  story_id          uuid,                              -- nullable; assigned by Job 2
  embedding_model   text NOT NULL                      -- pinned per row, e.g. 'text-embedding-3-large@1024'
);
CREATE INDEX item_by_story  ON item (story_id) WHERE story_id IS NOT NULL;
CREATE INDEX item_by_ingest ON item (ingested_at);

-- M:N: which clients are mentioned in an item (seed from source query, plus extracted entities)
CREATE TABLE item_client (
  item_id          uuid REFERENCES item NOT NULL,
  client_id        uuid REFERENCES client NOT NULL,
  origin           text NOT NULL,        -- 'seed' (from source query) | 'extracted' (entity match)
  PRIMARY KEY (item_id, client_id)
);
CREATE INDEX item_client_by_client ON item_client (client_id);

-- Multi-vector retrieval chunks (for downstream RAG and search; not used by clustering)
CREATE TABLE item_chunk (
  chunk_id          uuid PRIMARY KEY,
  item_id           uuid REFERENCES item NOT NULL,
  chunk_index       int NOT NULL,
  chunk_text        text NOT NULL,                   -- the raw chunk text, displayed in search results
  start_char        int NOT NULL,
  end_char          int NOT NULL,
  retrieval_vec     vector(1024) NOT NULL,
  context_summary   text,                            -- the Haiku doc-level prepend (embedded-into, then stored for audit)
  embedding_model   text NOT NULL,
  UNIQUE (item_id, chunk_index)
);
CREATE INDEX item_chunk_ann ON item_chunk USING hnsw (retrieval_vec vector_cosine_ops);
CREATE INDEX item_chunk_by_item ON item_chunk (item_id);

-- Stories: global, with affected_clients via story_client
CREATE TABLE story (
  story_id          uuid PRIMARY KEY,
  centroid          vector(1024) NOT NULL,            -- recomputed from members on each update (idempotent)
  medoid_item_id    uuid REFERENCES item,             -- highest-rank source × closest-to-centroid
  n_items           int NOT NULL DEFAULT 1,
  first_seen_at     timestamptz NOT NULL,
  last_seen_at      timestamptz NOT NULL,
  closed_at         timestamptz,
  title             text NOT NULL,                    -- generated inline by Sonnet before commit
  summary           text NOT NULL,
  entities          jsonb,                            -- merged from member items
  topic             text,                             -- earnings | m&a | regulatory | product | litigation | research | other
  embedding_model   text NOT NULL,
  schema_version    int NOT NULL DEFAULT 1,
  merged_into       uuid REFERENCES story,            -- set by the weekly merge pass
  last_summary_n    int NOT NULL DEFAULT 0            -- n_items at last full Sonnet regeneration
);
CREATE INDEX story_active_ann
  ON story USING hnsw (centroid vector_cosine_ops)
  WHERE closed_at IS NULL AND merged_into IS NULL;
CREATE INDEX story_active_by_recency
  ON story (last_seen_at DESC)
  WHERE closed_at IS NULL AND merged_into IS NULL;

-- Story → affected clients
CREATE TABLE story_client (
  story_id          uuid REFERENCES story NOT NULL,
  client_id         uuid REFERENCES client NOT NULL,
  added_at          timestamptz DEFAULT now(),
  PRIMARY KEY (story_id, client_id)
);
CREATE INDEX story_client_by_client ON story_client (client_id);

-- Pipeline run tracking (one row per Glue Workflow execution)
CREATE TABLE pipeline_run (
  run_id            uuid PRIMARY KEY,
  run_date          date NOT NULL UNIQUE,
  status            text NOT NULL,     -- 'running' | 'job1_done' | 'succeeded' | 'failed'
  started_at        timestamptz NOT NULL,
  job1_finished_at  timestamptz,
  job2_finished_at  timestamptz,
  finished_at       timestamptz,
  stats             jsonb,             -- {items_ingested, items_assigned, stories_created, stories_closed, ...}
  error             text
);
```

A few notes on the choices:

- **`vector(1024)`** because text-embedding-3-large reduced via Matryoshka to 1024 dims is the sweet spot; full 3072 dims doubles index memory for ~1% retrieval gain. Cohere Embed v3 (the in-VPC fallback) is natively 1024 dims, so the schema doesn't change if you swap.
- **Partial HNSW index `WHERE closed_at IS NULL AND merged_into IS NULL`** is critical. The index stays small because closed and merged stories drop out automatically.
- **`url_hash` is a `bytea` SHA-256** of the canonicalized URL, not text. Bytea indexes faster and is more storage-efficient than hex text.
- **Two embeddings per item across different tables.** `item.assignment_vec` is from `title + lede` and drives clustering — one vector per item. `item_chunk.retrieval_vec` is per-chunk and powers downstream search/RAG — one vector per chunk, one item may have many. Assignment-side stays single-vector even for long items; only the retrieval path goes multi-vector.
- **`item_client.origin = 'seed'` vs `'extracted'`** distinguishes clients that came from the source query (Perplexity's per-client search) from clients that were found by entity extraction. Both contribute to `story_client`, but `'seed'` is treated as higher-confidence in the gray-zone gate.
- **`story.merged_into`** supports the weekly merge pass. Active queries always filter `merged_into IS NULL`; merged stories remain in the table for audit.

---

## The pipeline, step by step

### Job 1: `gather_and_embed_job`

Glue Spark, 10–20 G.1X workers, ~45 min wall-clock.

**Inputs.** Aurora `source` + `client` + `banker_client` reference tables.
**Outputs.** Enriched Parquet on S3 at `s3://<bucket>/processed/items/run_date=YYYY-MM-DD/`, plus a `_manifest.json` containing run metadata.

**Steps.**

1. **Resolve the client universe.** Read the deduplicated union of clients covered by any active banker:
   ```sql
   SELECT DISTINCT c.client_id, c.display_name, c.aliases, c.primary_ticker
     FROM banker b
     JOIN banker_client bc ON b.banker_id = bc.banker_id
     JOIN client c ON bc.client_id = c.client_id
    WHERE b.active AND c.active;
   ```
   Broadcast as a Spark broadcast variable.

2. **Per-source ingest via the plugin registry.** Read enabled rows from `source`. For each, dynamically load the `connector_class` (a `SourceConnector` subclass) from a Python wheel bundled into the Glue job. Each connector returns a Spark DataFrame with columns `(source_id, url, title, body, published_at, seed_client_id)`. Union all source DataFrames.

   *Source-specific notes are deferred to "Handling Perplexity-specific concerns" and "Handling JPM internal research" below.*

3. **Normalize and dedupe URLs.** Strip `utm_*`, `fbclid`, `gclid`, `mc_eid`, fragment identifiers. Compute `url_hash = sha256(canonical_url)`. Drop intra-batch duplicates via `dropDuplicates(["url_hash"])`. Cross-reference with existing `item.url_hash` in Aurora:
   - For already-seen items, *do not re-embed*. But still upsert any new `item_client` rows (a Bloomberg article already in the database for Apple may now also be seeded by a Samsung search — write the new `(item_id, samsung_client_id, 'seed')` row).
   - For new items, proceed to step 4.

4. **Extract main content.** Apply `trafilatura` (or equivalent readability extractor) to the raw body to strip boilerplate. Compute `lede = body_clean[:600]`.

5. **MinHash + LSH near-dup detection.** Compute 128-permutation MinHash on 5-gram shingles of `body_clean`. Insert into a `datasketch` LSH index at threshold 0.85 (calibrated value from POC). For each item, query the LSH; build duplicate clusters via union-find. Within each cluster, pick the canonical item by the rule: lowest `source.source_rank` (Bloomberg > Reuters > WSJ > FT > AP > aggregators > blogs), tiebreak by earliest `published_at`. Non-canonical items get `is_duplicate=true, duplicate_of=<canonical_id>`. **Critically: duplicates inherit the canonical's `story_id` in Job 2 — every URL ends up findable in a story, even if it's a wire copy.**

6. **Entity extraction (always-on, Bedrock Haiku).** For each canonical (non-duplicate) item, one Haiku call with a structured-output prompt:
   ```
   Extract entities from this financial news item. Return JSON:
   {"people": [...], "orgs": [...], "tickers": [...], "locations": [...]}
   TITLE: ...
   LEDE:  ...
   ```
   Force the schema via Bedrock `tool_use`. Resolve `tickers` and `orgs` against the broadcast client alias matcher to derive `extracted_client_ids`. Drop items whose final `item_clients` set (seeded + extracted) is empty — they are noise.

7. **Assignment-side embedding (OpenAI Batch API).** Build a JSONL file with one line per canonical item:
   ```json
   {"custom_id": "<item_id>", "method": "POST", "url": "/v1/embeddings",
    "body": {"model": "text-embedding-3-large", "dimensions": 1024,
             "input": "<title>\n\n<lede>"}}
   ```
   Submit via `POST /v1/batches`. Poll for completion. Typical wall-clock is minutes to a few hours; the 24-hour SLA matters only at very high volume. **50% cost discount** versus the synchronous Embeddings API.

   *In-VPC fallback:* replace this step with synchronous Bedrock calls to `cohere.embed-english-v3` with `input_type=clustering`. The Cohere model has a 512-token context, which `title + lede` fits comfortably; you cannot embed full bodies via Cohere on Bedrock.

8. **Retrieval-side chunking with doc-level context.** For each item where `token_count(body_clean) > 2000` (covers most JPM research, some long news features):
   - One Bedrock Haiku call: "Summarize this document in 1–2 sentences focusing on its overall topic and the primary entities involved." Output is `doc_context`.
   - Split `body_clean` into 800-token chunks with 100-token overlap.
   - For each chunk, embed `f"DOCUMENT CONTEXT: {doc_context}\n\nCHUNK: {chunk_text}"` via OpenAI Batch. The prepend is embed-time only; `item_chunk.chunk_text` stores the raw chunk for display.
   For items with `token_count(body_clean) ≤ 2000`, create a single `item_chunk` row covering the full body, no prepend needed. This unifies the retrieval-side schema: every item has at least one chunk row.

9. **Write enriched Parquet to S3.** Partitioned by `run_date`. Include all the fields Job 2 needs: `item_id, url_hash, title, lede, published_at, assignment_vec, entities, extracted_client_ids, seed_client_ids, is_duplicate, duplicate_of, embedding_model`. Write the `_manifest.json` with item counts, embedding-model identity, and run timestamps.

**Cold-start and idempotency.** Glue job bookmarks are configured on S3-input paths used by S3-based connectors so re-runs are cheap. API-based connectors (Perplexity, JPM research API) get idempotency from `item.url_hash UNIQUE` downstream — re-running is safe but does the duplicate API work. To bound the cost of a re-run, each connector writes its raw responses to `s3://<bucket>/raw/source=<id>/run_date=<date>/` first; Job 1 then processes from S3 with bookmarks. Re-runs become cheap *and* idempotent.

---

### Job 2: `cluster_assign_label_job`

Glue Spark, 1 G.4X worker, ~60 min wall-clock. Clustering on the driver (NumPy); metadata generation parallelized across worker vCPUs via Spark UDF.

**Inputs.** Enriched Parquet from S3 (Job 1's output). Active stories + their member clients from Aurora.
**Outputs.** Updated and new `story`, `story_client`, `item`, `item_client`, `item_chunk` rows in Aurora. Closed stale stories.

**Steps.**

1. **Load today's items from S3 → driver as a pandas DataFrame.** Sort by `published_at` (single-pass assignment requires temporal order).

2. **Load active stories + member tags from Aurora.**
   ```sql
   SELECT s.story_id, s.centroid, s.medoid_item_id, s.n_items, s.last_seen_at,
          s.entities, s.embedding_model
     FROM story s
    WHERE s.closed_at IS NULL
      AND s.merged_into IS NULL
      AND s.last_seen_at > now() - interval '72 hours';
   ```
   Also load `story_client` for these stories. Build a `client_to_story_idxs: dict[client_id, list[int]]` lookup mapping each client to the indices of stories tagged with that client.

3. **Run the single-pass assignment loop on the driver.** Pure NumPy:
   ```python
   TAU_HIGH = calibration.tau_high     # from POC, e.g. 0.75
   TAU_LOW  = calibration.tau_low      # from POC, e.g. 0.55

   gray_zone_pairs = []                # batched LLM judge calls
   assignments = []
   residual = []

   for item in canonical_items_sorted:
       candidate_idxs = set()
       for cid in item.item_clients:
           candidate_idxs |= client_to_story_idxs.get(cid, set())
       if not candidate_idxs:
           residual.append(item); continue

       sims = story_vecs[list(candidate_idxs)] @ item.assignment_vec
       best_local = int(np.argmax(sims))
       best_global = list(candidate_idxs)[best_local]
       best_sim = float(sims[best_local])

       if best_sim >= TAU_HIGH:
           assignments.append((item, best_global, best_sim))
           _running_mean_update(story_vecs, story_n, best_global, item.assignment_vec)
           # entity-overlap gate is bypassed for the high-confidence band
       elif best_sim >= TAU_LOW:
           if item.entities & stories[best_global].entities:    # entity-overlap gate
               gray_zone_pairs.append((item, best_global, best_sim))
           else:
               residual.append(item)
       else:
           residual.append(item)

   # Batch the Haiku gray-zone judge calls concurrently
   verdicts = await asyncio.gather(*[
       bedrock_haiku_judge(item, stories[g])
       for (item, g, _) in gray_zone_pairs
   ])

   for (item, g, sim), verdict in zip(gray_zone_pairs, verdicts):
       if verdict == "SAME":
           assignments.append((item, g, sim))
           _running_mean_update(story_vecs, story_n, g, item.assignment_vec)
       else:
           residual.append(item)
   ```

   The running-mean update lets later items in the same batch see the updated centroid. The *persisted* centroid is recomputed from the full member set in step 8 — this is the idempotency property that lets Job 2 be safely retried.

4. **Gray-zone LLM judge prompt (Bedrock Haiku).**
   ```
   You are deciding whether a news item belongs to an existing story.

   EXISTING STORY:
     title:   <story.title>
     summary: <story.summary>
     key entities: <story.entities, top 10 by frequency>

   NEW ITEM:
     title: <item.title>
     lede:  <item.lede>

   Are these about the same news event? Respond with one JSON object:
   {"verdict": "SAME" | "DIFFERENT", "reason": "<one sentence>"}
   ```
   Force schema via Bedrock `tool_use`. Haiku returns this in ~500ms per call; at ~20% gray-zone rate on 100k items, that's ~20k calls/night, costing ~$3/day.

5. **HDBSCAN on the residual.** Spawn new stories from residual items that share at least one client cluster-wide:
   ```python
   if len(residual) >= 2:
       vecs = np.vstack([it.assignment_vec for it in residual])
       clusterer = hdbscan.HDBSCAN(
           min_cluster_size=2,
           min_samples=2,
           metric="cosine",
           cluster_selection_method="eom",
       )
       labels = clusterer.fit_predict(vecs)
       # Enforce client overlap within each cluster: if cluster members
       # have no shared client, split along client lines.
       split_clusters_by_client_overlap(residual, labels)
       create_new_stories_from_clusters(residual, labels)
   ```
   Singleton residuals (HDBSCAN label `-1`) become 1-member stories on their own. The product appetite for 1-member stories should be reviewed in calibration: if singletons dominate, raise `min_cluster_size` to 2 and let single items wait for a sibling.

6. **Compute medoid for each new and grown story.** For each story:
   ```python
   def medoid(members, source_rank):
       # Among members, rank by source_rank ascending (Bloomberg > Reuters > ...).
       # Tiebreak by cosine similarity to the (recomputed) centroid.
       best = sorted(members, key=lambda m: (m.source_rank, -cosine(m.vec, centroid)))[0]
       return best.item_id
   ```
   Update the medoid every 5 added items, not every item — recomputation is O(n) member-scan, cheap but unnecessary at item granularity.

7. **Generate metadata via Bedrock Claude Sonnet (inline, before commit).** This is the divergence from Doc 2 that the inline-metadata decision requires. For each *new* story or story that grew this run:
   ```
   SYSTEM:
   You maintain structured records of news stories for an investment banking
   intelligence product. Return ONLY valid JSON matching the schema.

   USER (for new story):
   Given these items belonging to one story, produce:
   { "title": ≤80 chars, neutral, factual headline,
     "summary": ≤400 chars, 2-3 sentences "who/what/where/when",
     "entities": {"people":[], "orgs":[], "tickers":[], "locations":[]},
     "topic": one of {earnings, m_and_a, regulatory, product, litigation, research, other} }
   ITEMS:
   - <title 1>: <lede 1>
   - <title 2>: <lede 2>
   ...

   USER (for grown story — Chain-of-Key update):
   CURRENT_STORY: <existing title/summary/entities/topic JSON>
   NEW_ITEMS:    <new items list>
   Update the story JSON. Preserve title unless new items materially change the
   event. Extend summary to ≤400 chars covering all members. Merge entities.
   Topic stays the same unless materially shifted.
   ```
   Force schema via Bedrock `tool_use`. Parallelize across the worker's 16 vCPUs via Spark UDF.

   **Full regen safeguard.** If `n_items - last_summary_n >= 10`, regenerate from scratch instead of incremental update. Bounds the compounding drift Doc 1 warned about.

8. **Recompute centroids from full member sets.** This is what makes Job 2 idempotent on retry:
   ```python
   for story_id, members in updated_stories.items():
       full_member_vecs = np.vstack([m.assignment_vec for m in members])
       story.centroid = full_member_vecs.mean(axis=0)
       story.n_items  = len(members)
       story.last_seen_at = max(m.published_at for m in members)
   ```
   The running-mean update in step 3 is a within-batch optimization to let later items see updated centroids; the persisted centroid is always a recomputation. On retry, the same input items produce the same persisted centroid.

9. **Bulk write to Aurora.** Single Glue commit using `aws_s3.table_import_from_s3` for large batches (>10k rows) — 5–10× faster than JDBC `bulkCopyToSqlDB`. In order:
   - Insert new `item`, `item_client`, `item_chunk` rows.
   - Insert new `story`, `story_client` rows (with their generated titles — never with placeholders).
   - Update existing `story` rows that received items (centroid, n_items, last_seen_at, entities, title, summary, last_summary_n).
   - Set `item.story_id` for newly assigned items, including propagating canonical's `story_id` to all `is_duplicate=true` items.

10. **Close stale stories.**
    ```sql
    UPDATE story SET closed_at = now()
     WHERE closed_at IS NULL
       AND merged_into IS NULL
       AND last_seen_at <= now() - interval '72 hours';
    ```
    Single statement across the whole table — no per-entity iteration.

11. **Update `pipeline_run`** with `status='succeeded'`, `finished_at`, and the populated `stats` JSON for that day's run.

**Idempotency.** Job 2 is fully retryable. Inputs are immutable S3 Parquet from Job 1 and Aurora state. Repeated execution produces identical centroids (full-member recompute), identical story assignments (deterministic single-pass given sorted input), identical metadata (Sonnet with `temperature=0`). The only non-idempotent surface is the LLM judge verdicts in the gray zone — pin `temperature=0` there too, accept the rare flake, and cache call responses on retry via a `(prompt_hash, run_id)` key in S3.

---

## Source plugin pattern

Future sources are added without touching the Glue job code.

```python
# connectors/base.py
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class SourceConnector(ABC):
    """Subclasses must return a DataFrame with columns:
       source_id, url, title, body, published_at, seed_client_id (nullable)
    """
    @abstractmethod
    def fetch(self, spark, clients_df, config: dict, run_date: str) -> DataFrame: ...

# connectors/perplexity.py
class PerplexityConnector(SourceConnector):
    """One Perplexity News call per (run, client). Parallelized by repartition."""
    def fetch(self, spark, clients_df, config, run_date):
        rdd = clients_df.repartition(config.get("concurrency", 50)) \
            .rdd.mapPartitions(lambda partition: [
                self._call_perplexity(c, config, run_date) for c in partition
            ])
        return spark.createDataFrame(rdd, schema=ITEM_SCHEMA)

# connectors/jpm_research.py
class JpmResearchConnector(SourceConnector):
    """Reads from internal research API or an S3 drop zone."""
    def fetch(self, spark, clients_df, config, run_date):
        if config["mode"] == "s3":
            return spark.read.json(config["s3_input_glob"]).filter(...)
        else:
            return spark.read.format("jdbc").options(**config["jdbc"]).load()
```

**Adding a new source.** Write a connector class, bundle it into a Python wheel, upload to S3, reference via `--additional-python-modules s3://bucket/wheels/connectors-x.y.z-py3-none-any.whl`, and insert a row into `source` with `connector_class='connectors.newssource.NewsSourceConnector'`, config_json with API keys / endpoints, and `enabled=true`. No Glue job redeploy. Next nightly run picks it up.

**Wheel deployment.** Re-deploy the wheel for every connector addition. CI builds the wheel from a `connectors/` package, runs unit tests against mocked source responses, and publishes versioned wheels to S3.

---

## Long-document handling: contextual chunking

Articulated separately because it's the algorithmic addition over Doc 2 and the major novelty of this spec.

**The problem.** News items are usually ≤2,000 tokens — fit in any embedding model's context. JPM internal research notes can be 5–50 pages, well past any embedding model's context. Doc 2 silently truncated body to 2,000 characters before embedding, losing most of the document's signal. Truncation is not acceptable for research that bankers will retrieve from later.

**The pattern.** Multi-vector retrieval with **doc-level Anthropic-style contextual prepend** (a simplified variant of Anthropic's Contextual Retrieval, September 2024). For each long item:

1. One Bedrock Haiku call: "Summarize this document in 1–2 sentences focusing on overall topic and primary entities." Output: `doc_context` (~50 tokens).
2. Split body into 800-token chunks with 100-token overlap (`langchain.text_splitter.RecursiveCharacterTextSplitter` or equivalent).
3. For each chunk, embed `f"DOCUMENT CONTEXT: {doc_context}\n\nCHUNK: {chunk_text}"`. The prepend exists *only at embed time*; `item_chunk.chunk_text` stores the raw chunk for display.
4. Each chunk is one row in `item_chunk` with `retrieval_vec`, `chunk_text`, `start_char`, `end_char`, and `context_summary = doc_context` for audit.

**Cost.** At ~5,000 long docs/day (research is the dominant source of long items): one Haiku call per doc = $0.50/day. Then ~10–30 chunks per long doc averaged = ~50,000–150,000 retrieval embeddings/day. Via OpenAI Batch API at 1024 dims, ~$10–20/day.

**Why this and not Jina-style late chunking.** Late chunking (Jina embeddings v3, Sep 2024) is the academically purer approach: run the whole document through a long-context encoder first, then pool token spans into chunk vectors. It's elegant but requires a Jina-class model that isn't natively available on Bedrock; choosing it would force a model dependency outside your stack. The contextual-prepend pattern is model-agnostic, works with OpenAI or Cohere on Bedrock, and captures most of the same signal — doc-level context grounds each chunk's embedding without requiring a long-context encoder. If you ever move to Jina (via Bedrock Marketplace), revisit.

**Validation.** Section 10 of the POC notebook tests this approach against plain truncation on a small synthetic retrieval benchmark. Adopt the pattern in production if the POC's hits@5 improves by ≥5% over truncation.

**Assignment-side stays single-vector.** Clustering does *not* use `item_chunk` — it uses `item.assignment_vec` (one vector per item from title + lede). Long documents have a single assignment vector built from their title + abstract/executive summary, exactly as research.md recommends. Multi-vector retrieval is solely a downstream search/RAG concern.

---

## Model selection: what to use where

| Where | Recommended | In-VPC fallback | Rationale |
|---|---|---|---|
| Assignment embedding | OpenAI `text-embedding-3-large` (1024 dims via Matryoshka), Batch API | Bedrock `cohere.embed-english-v3` with `input_type=clustering` | OpenAI wins on raw retrieval quality and supports 8K context. Cohere on Bedrock wins on data residency; its 512-token limit fits `title + lede` only. |
| Retrieval embedding (per chunk) | OpenAI `text-embedding-3-large` (1024 dims), Batch API | Bedrock `cohere.embed-english-v3` (no `clustering` flag for retrieval) | Same trade-off. The contextual prepend is independent of the embedding model. |
| Gray-zone LLM judge | Bedrock `anthropic.claude-haiku-4-5-20251001-v1:0` | OpenAI `gpt-4o-mini` | Haiku is fast (~500ms), cheap (~$0.0001/call), in-VPC. The judge is high-volume, low-stakes — Bedrock keeps the data on AWS. |
| Doc-level chunk context summary | Bedrock Haiku | OpenAI `gpt-4o-mini` | Same reasoning. One call per long doc. |
| Story metadata generation | Bedrock `anthropic.claude-sonnet-4-6-20251001-v1:0` | OpenAI `gpt-4o` | Sonnet's structured output via `tool_use` is excellent and stays on AWS. Sonnet is the largest single LLM line item in the cost estimate; if budget pressure shows up here, downgrade incremental updates to Haiku while keeping Sonnet for new-story creation. |
| Entity extraction | Bedrock Haiku | OpenAI `gpt-4o-mini` | An LLM beats spaCy for ambiguous tickers and disambiguating people. Cost is negligible at this volume. |

**Pin every model ID in the database.** `item.embedding_model`, `story.embedding_model`, and `item_chunk.embedding_model` each store the exact pinned model name (e.g. `text-embedding-3-large@1024`). When you change models, you must re-embed every active story member or re-bootstrap clusters — embeddings from different models are not comparable. Plan a "model swap" runbook that re-embeds in place over a maintenance window, doing it incrementally per client_id to bound the work.

---

## Handling Perplexity-specific concerns

- **Result snippets are short.** The body content Perplexity returns is closer to a lede than a full article. That's fine for `title + lede` embedding — you may not need to fetch the underlying URL at all in v1.
- **Same URL surfaces in multiple per-client searches.** If "Apple sues Samsung over chip patents" appears in both the Apple search and the Samsung search, that's handled correctly by the global-stories design: one `item` row (deduped on `url_hash`), two `item_client` rows seeded from the two searches, one story that joins Apple's and Samsung's pools through `story_client`. Job 1 explicitly upserts new `item_client` rows for already-seen items so the second search's seed_client is recorded.
- **Source ranking is uneven.** Perplexity returns a mix of Bloomberg, Reuters, regional press, blogs. The `source.source_rank` column (set per registry row, lower is higher quality) drives medoid selection.
- **Rate limits.** Per-client API calls in parallel can hit Perplexity tier limits. The Spark `repartition` in the connector gives you a knob: `repartition(50)` caps concurrent calls at 50. If you exceed tier limits, lower this number — better than failing the whole job.
- **Citation hallucinations.** Perplexity occasionally returns confident-sounding items that don't fully match the query intent. The global-stories design handles this naturally: such items either fail entity extraction (no client matches) and get dropped as noise, or they reach the gray zone and the Haiku judge rejects them.

---

## Handling JPM internal research

- **Long documents.** Research notes are 5–50 pages. The contextual-chunking path is built for this case. Use the **abstract or executive summary as the lede** for `title + lede` embedding; if the document lacks a structured abstract, generate one via Bedrock Haiku at ingest.
- **Multi-entity coverage.** A sector note "Mega-Cap Tech Earnings Preview" covers Apple, Microsoft, Google, Meta, Amazon. Under the global-stories design this is one item with five `item_client` rows; the resulting story is tagged with all five clients via `story_client`. Every banker covering any of those clients sees this story in their feed.
- **Forward-looking content.** Research is "we expect" / "we estimate" while news is event-driven. Mixing them in the same story is usually correct — bankers want the JPM view tied to the news event. Tag items with `source_id` so downstream UI can render the distinction (e.g. "JPM Research" badge).
- **Don't embed the whole PDF as one vector.** Per research.md and the long-doc chunking section above, embed `title + abstract` for the assignment-side single vector; the full body lives in `item_chunk` rows for retrieval.

---

## Starting thresholds (calibrate against the POC's labeled set)

| Parameter | Starting value | Source | Notes |
|---|---|---|---|
| `τ_high` (auto-assign) | 0.75 | research.md / POC | For text-embedding-3-large at 1024 dims, cosine. For Cohere Embed v3, expect to lower by 0.05–0.10. |
| `τ_low` (skip LLM, go to residual) | 0.55 | research.md / POC | Below this, the Haiku judge call isn't worth making. |
| Residual HDBSCAN params | `min_cluster_size=2, min_samples=2, metric='cosine', cluster_selection_method='eom'` | research.md | |
| Story expiry | 72 hours since `last_seen_at` | spec / Miranda et al. | Matches Miranda et al.'s empirically-tuned σ=72h Gaussian decay; hard cutoff is the simpler implementation. |
| MinHash near-dup Jaccard | 0.85 | Doc 2 (stricter than research.md's 0.80) | At banker scale, false dupes are worse than missed dupes. Stricter is safer. |
| Body-cosine duplicate (if used) | 0.95 | research.md | Alternative dup signal; currently unused — MinHash on body shingles is the production rule. |
| Weekly merge: centroid cosine + shared entities | 0.85 cosine + ≥2 shared entities + ≥2 shared clients | research.md / Doc 1 | Conservative — bad merges are worse than missed merges. |
| B-cubed F1 go-live target | ≥ 0.85 | research.md | Calibrate on the POC's labeled set; recompute on every model or prompt change. |

**Calibration discipline.** These are *priors*, not optimums. The POC produces calibrated values; the production system reads from `config/calibration.json` at Job 2 startup. The labeled set in `eval/labeled_set_v1.csv` is the regression test — gates threshold or model changes in CI.

---

## Evaluation and observability

**Headline metric.** B-cubed precision / recall / F1 against the labeled set, run as a CI regression gate. Fail any PR that drops F1 by >0.03 on the labeled set.

**Per-run operational metrics (CloudWatch dashboards via Glue + Aurora).**

- Items ingested, deduped, assigned (by gate: auto / gray-zone-SAME / gray-zone-DIFFERENT / residual-clustered / singleton), and embedded.
- Story counts: new, grown, closed.
- Wall-clock per Job 1 / Job 2 phase. Tail latencies — one slow Perplexity call must not block the run.
- Gray-zone judge call rate. If this climbs above 25% of items, the gray zone is too wide; investigate (raise `τ_low`, or embeddings are weaker than expected).
- LLM cost per run, tagged by `run_id` and `stage`.
- Aurora pgvector HNSW build time. Watch trends.

**Story-quality metrics (weekly aggregation).**

- Median and p95 story size at end of run. Spikes signal active news cycles; sustained drift signals threshold drift toward over-clustering.
- Singleton rate (stories with `n_items == 1`). Persistently high signals tight thresholds or noisy items.
- Centroid drift: weekly mean of `cosine(centroid_at_first_member, centroid_now)` for stories ≥5 members. Catches the case where stories grow but their topic shifts away from origin.

**Spot-check sampling.** 50 stories per week sampled stratified by size, manually reviewed by analysts. Catches drift the metrics will miss.

---

## Cost estimate

At 100k canonical items/day (after dedup) and ~5k stories updated per day:

| Line item | Calls / units | Unit cost | Daily |
|---|---|---|---|
| **Job 1 — Glue Spark, 15 G.1X for 45 min** | 15 × 0.75h × $0.44 | | $4.95 |
| Perplexity News (1 call per unique client × 500) | 500 | ~$0.005 | $2.50 |
| Bedrock Haiku — entity extraction (per canonical item) | 100k | ~$0.0001 | $10 |
| Bedrock Haiku — long-doc context summary | ~5k | ~$0.0001 | $0.50 |
| OpenAI text-embedding-3-large — assignment vectors (batched) | ~100k @ ~500 tok avg | $0.065/M tok (Batch) | $3.25 |
| OpenAI text-embedding-3-large — retrieval-side chunks (batched) | ~150k @ ~900 tok avg | $0.065/M tok (Batch) | $8.75 |
| **Job 2 — Glue Spark, 1 G.4X for 60 min** | 4 × 1h × $0.44 | | $1.76 |
| Bedrock Haiku — gray-zone judge (~20% of items) | ~20k | ~$0.00015 | $3 |
| Bedrock Sonnet — story metadata, new (~500) | 500 | ~$0.005 | $2.50 |
| Bedrock Sonnet — story metadata, grown (~4500) | 4500 | ~$0.005 | $22.50 |
| **Glue compute total** | | | **$6.71** |
| **API total (Bedrock + OpenAI + Perplexity)** | | | **$52.50** |
| **Daily total** | | | **~$59/day** |
| **Monthly total** | | | **~$1,800/month** |

Notes:

- **API costs dominate Glue compute by ~8×.** This is the correct shape — paying for value (LLM reasoning, embeddings), not infrastructure.
- Sonnet metadata updates on grown stories are the largest single line item. If budget pressure shows up, downgrade incremental updates to Haiku (saving ~$15/day, at some cost in summary polish).
- Adding new sources increases costs only by the net-new items they generate after dedup. New bankers covering existing clients add ~zero cost.
- Compute scales linearly with data volume. At 10× scale (1M items/day), expect ~$50/day Glue and ~$500/day APIs.

---

## Recommended rollout

**Weeks 1–2 — minimum viable.**
- Run the POC notebook end-to-end. Hand-label the eval set. Produce `config/calibration.json` and `eval/labeled_set_v1.csv`.
- Stand up Aurora with the schema above, including HNSW indexes.
- Build the Perplexity and JPM-research connectors against the `SourceConnector` ABC. Ship the wheel.
- Implement Job 1 with one source (Perplexity) end-to-end: ingest, dedupe, entity extraction, assignment embedding. Skip the contextual-chunking long-doc path; treat all items as single-chunk.
- Implement Job 2 with single-threshold assignment (no gray-zone Haiku yet), HDBSCAN residual, inline Sonnet metadata. B-cubed F1 should already approach the POC's number; if it doesn't, debug the integration before continuing.
- EventBridge schedule, two-job Glue Workflow, daily `pipeline_run` row.
- One CloudWatch dashboard with the operational metrics above.

**Weeks 3–6 — quality.**
- Add Bedrock Haiku gray-zone judge with the entity-overlap gate. Split single threshold into `τ_high` / `τ_low`.
- Add the JPM research connector. Add long-doc contextual chunking (Job 1 step 8). Populate `item_chunk` rows.
- Add the source-rank table; implement medoid selection by source rank + centroid distance.
- Add CI regression gate on `eval/labeled_set_v1.csv`.
- Add the weekly merge pass as a separate scheduled Glue job (cron on Sunday). Use the merge thresholds from the table above; Haiku verifies each candidate merge.
- Add the `model swap` runbook: how to re-embed all active stories when text-embedding-3-large is replaced.

**Month 2+ — refinement.**
- A/B test Cohere Embed v3 on Bedrock vs OpenAI on the labeled set. If Cohere closes the gap, the all-in-AWS configuration becomes viable.
- Replace fixed `τ_high` / `τ_low` with the **trained merge classifier** (`research.md` Stage 3 recommendation): logistic regression or gradient-boosted trees over `(centroid_cosine, medoid_cosine, entity_overlap_count, time_delta_hours, source_rank)`. This is what gave Miranda et al. their best B³-F1 of 94.1. Use the labeled set as training data.
- Optional: USTORY-style adaptive threshold per active window. Worth the complexity only if calibration data shows the fixed thresholds drift week-to-week.
- Optional: contextual-retrieval upgrade — try chunk-specific context (Anthropic's original Contextual Retrieval) on the retrieval-side embeddings if banker-facing search starts to matter as a feature.
- Optional: re-evaluate `text-embedding-3-large` vs `voyage-3-large`. `voyage-3-large` reportedly leads by ~9–10% on MTEB but requires a model swap.

---

## What we're deliberately leaving out

- **Multi-member gray-zone verification.** `research.md` recommends verifying against medoid + 3 recent member embeddings as a pre-LLM gate. Superseded here by the combination of (a) inline metadata keeping `story.summary` always fresh, so the Haiku judge sees a current, semantically rich description; (b) cheap Haiku judge cost so pre-LLM filtering isn't a cost driver; (c) the entity-overlap gate already filtering most embedding false-positives. Reintroduce if you ever decouple metadata generation to a separate async job.
- **Story splits.** When a story's internal cohesion drops, optionally split it. `research.md` itself notes "most teams skip splits"; the 72-hour closure naturally limits drift. Revisit only if you observe specific failure modes where stories grow but topically diverge.
- **Late chunking with a Jina-class model.** The cleaner academic technique for long-doc retrieval, but requires a model dependency outside the Bedrock-native set. The doc-level contextual prepend captures most of the signal at no model cost.
- **Adaptive thresholds (USTORY).** Sensible at higher complexity, but fixed thresholds calibrated on a labeled set are simpler and sufficient for v1.
- **Real-time / streaming layer.** Nightly cadence matches the spec; don't build streaming until a real product reason demands it.
- **LLM-as-clusterer.** Cost is 10–50× embeddings + Haiku verification, no quality lift over the gated architecture at this scale.
- **Glue Spark distribution for the assignment algorithm itself.** Single-pass nearest-cluster is sequential by design; the algorithm runs on the Glue driver in NumPy. Spark is the vehicle for I/O and metadata parallelization, not for clustering.
- **`pgvectorscale` (Timescale DiskANN).** Not available as a managed extension on Aurora PostgreSQL. Plain pgvector with HNSW is sufficient at this scale.
- **Self-hosted models (BGE-M3, Qwen3, Voyage).** Out of scope per stack constraints.
- **3-job structure with separate async metadata.** The Doc 2 design. Rejected because of the "no placeholder titles ever" product guarantee. If that guarantee is ever relaxed (for graceful degradation reasons), revisit — the 3-job split has real operational benefits.

---

## Caveats specific to this stack

1. **OpenAI data egress requires a security review.** If compliance forbids any item content leaving the AWS account, you are locked to Bedrock-only — Cohere Embed v3 for embeddings (with the 512-token context limit constraining the retrieval path), Bedrock Claude for LLMs. The Cohere fallback is workable but materially constrained; resolve this in week 1, before building the embedding path.
2. **OpenAI Batch API has a 24-hour SLA, not a guarantee.** Batches usually complete within minutes to a few hours; design Job 1 to tolerate slower returns. Glue Workflow + Glue's job-level retry semantics handle this; ensure the Job 1 logic checkpoints batch IDs to S3 so a re-run picks up in-flight batches rather than resubmitting.
3. **Bedrock model availability varies by region.** Confirm Claude Haiku and Sonnet versions are available in your AWS region. Cross-region inference adds latency and may have data-residency implications.
4. **Aurora pgvector HNSW build time** is proportional to row count. The partial index `WHERE closed_at IS NULL AND merged_into IS NULL` keeps active-set size bounded — typically <10k rows — but the first populate from history wants a maintenance window. Schedule it.
5. **`pgvectorscale` is not available on Aurora.** Plain pgvector. If active stories ever exceed ~1M (very unlikely at your cadence), consider migrating Aurora → RDS PostgreSQL (which can install pgvectorscale) or to self-managed Postgres on EC2. Don't preempt this — the gap is several orders of magnitude away.
6. **Per-banker-client Perplexity quota.** At several hundred banker clients × nightly cadence × 36-hour overlap window, you may exceed Perplexity tier limits. Confirm quota before scaling Job 1's concurrency.
7. **Spark UDFs that call external APIs need careful sizing.** Use `mapPartitions` (Python-native iteration over a partition with one HTTP session reused) rather than per-row UDFs for the Bedrock and OpenAI calls. Configure partition count to match the API's rate limit, not the worker's CPU count.
8. **Glue cold-start is 1–3 minutes per job.** Two jobs in sequence = up to 6 minutes of pure cold-start overhead per day. Negligible compared to ~2-hour total runtime, but visible in CloudWatch latency.
9. **JDBC writes to Aurora are slow at scale.** Use `aws_s3.table_import_from_s3` (write to S3 staging first, then Aurora bulk import) for inserts >10k rows. Standard JDBC works but is 5–10× slower for large batches.
10. **Aurora connection limit.** Spark executors opening many concurrent JDBC connections will saturate `max_connections`. Use RDS Proxy in front of Aurora, or cap executor concurrency for JDBC-heavy stages.
11. **Bedrock model versions drift.** Pin model IDs (`anthropic.claude-haiku-4-5-20251001-v1:0`, `anthropic.claude-sonnet-4-6-20251001-v1:0`); never use `latest`. Capture the model ID in `story.embedding_model`, `item.embedding_model`, `item_chunk.embedding_model` so a future swap can be planned.
12. **Glue Workflow can't express complex DAGs.** A linear two-job sequence is fine. If you ever need branching ("if Job 2 succeeds, run Job 3a in parallel with Job 3b") or finer error handling, switch the orchestrator to Step Functions while keeping the jobs as Glue jobs.
13. **The two-job structure couples clustering and metadata.** A Bedrock Sonnet outage during the metadata phase of Job 2 fails the whole pipeline. The acceptable failure mode is: the nightly run fails, gets re-run on backoff, eventually succeeds — no story rows ever appear with placeholder titles. If Bedrock outages become a recurring operational pain, revisit the 3-job-with-summary_dirty design from Doc 2; the trade-off is well-understood.
14. **Threshold values are priors, not optimums.** Calibrate against the POC's labeled set before going live. Re-calibrate whenever the embedding model changes. There is no embedding model in 2026 for which a single fixed cosine threshold has been peer-reviewed as optimal.
15. **The labeled eval set is a maintenance commitment.** When you add a new source, when you change embedding models, when production data shifts — re-label, recompute, recalibrate. Plan ~half a day of analyst time per recalibration.
