# Story Clustering for Banker News & Research on AWS

Adapted from the generic news-clustering research report for your stack: Aurora RDS, Prefect on ECS, EventBridge, SQS, Lambda, Bedrock, OpenAI. Sources are Perplexity News (per banker client) and JPM internal research. No self-hosted models.

---

## TL;DR

- **Use single-pass nearest-cluster assignment** (the classic TDT algorithm, Miranda et al. 2018) implemented in Python on ECS, with Aurora PostgreSQL + pgvector + HNSW as the story store. This is the same architecture the generic report recommends; your stack does not change the algorithm, only the implementation.
- **The decisive simplification in your use case is entity grounding.** Every item arrives tagged with the banker client it was searched for. Stories are scoped to the banker client. You never cluster a "Tesla" item against an "Apple" story. This collapses the assignment step from "ANN search across all active stories" to "ANN search across active stories *for this banker client*" — usually a handful of candidates.
- **Embedding model: `text-embedding-3-large` via OpenAI Batch API (50% discount, 24h SLA — perfect for nightly).** Reduce to 1024 dims via Matryoshka truncation to fit Aurora storage and HNSW index budgets. If OpenAI egress is a compliance concern, fall back to **Cohere Embed English v3 on Bedrock** (1024 dims, supports a `clustering` input type, but a 512-token context limit means you must embed `title + lede` only, never long bodies).
- **LLM judge (gray-zone verifier): Claude Haiku on Bedrock** — cheap, fast, in-VPC, no data egress. **LLM metadata generator (titles, summaries, timelines): Claude Sonnet on Bedrock or GPT-4o.** Use Bedrock when the data must stay on the AWS account; use OpenAI when you want best-quality structured output and egress is acceptable.
- **Scope stories at the `banker_client` level, not the `banker` level.** Two bankers covering the same client see the same stories. This deduplicates work, simplifies the schema, and matches how news actually behaves (the Tesla Q3 earnings story is the same story regardless of which banker is reading it).
- **AWS Glue is not the right tool here.** Glue is for Spark-based ETL on data lakes; this workflow is row-oriented database work with API calls and small-vector math. Prefect on ECS is the right shape.

---

## How your use case differs from generic news clustering

The original research report assumes a fairly hard problem: a stream of arbitrary news articles, no prior entity tagging, you have to figure out from the embedding alone whether two articles are about the same event. That's the problem Meltwater, NewsCatcher, and GDELT solve.

Your problem is meaningfully easier in three ways:

**1. Items arrive entity-tagged.** Perplexity is invoked per banker client with a query like `news about <Banker Client Name>`. The returned items are already known to be about that entity. JPM internal research, similarly, can be tagged at ingest with the banker clients it covers (either via existing metadata or a one-shot LLM extraction). This means your clustering doesn't need to discover which entity an item is about — only which *event involving that entity* it belongs to.

**2. Clustering is partitioned by entity.** Tesla items only ever cluster against other Tesla items. The active-story ANN search becomes a small per-entity query (typically 0–10 active stories per banker client at any time) rather than a corpus-wide search. The "entity overlap gate" that the generic report recommends as a quality filter is enforced by construction in your design.

**3. Clusters are typically small.** A banker client like Microsoft might have 5–30 news items in a 72-hour window across all sources. A specific story ("Microsoft Q3 earnings beat") might have 2–8 items. You're not clustering thousands of items per entity; you're clustering tens. This means:
- pgvector HNSW indexes are overkill for the per-entity query; a sequential scan over a dozen story centroids is fast enough.
- LLM verification of every borderline match becomes affordable.
- HDBSCAN/Leiden on the residual is unnecessary in most cases — a simple pairwise similarity threshold over the unassigned items will do.

The hard part of your problem is not the clustering algorithm. It's the operational plumbing: scheduling per-banker-client jobs at scale, handling Perplexity rate limits and quirks, deduping cross-entity items (Apple-vs-Samsung lawsuits), keeping JPM internal research in sync, and generating useful metadata. The architecture below focuses on those.

---

## Scope decision: story.banker_client_id, not story.banker_id

You have a choice. Either:

**Option A (recommended): scope stories to `banker_client`.** A story is "about" a corporate entity. Multiple bankers can be subscribed to that entity and they all see the same story. Items from Perplexity searches for "Tesla" run by different bankers all flow into the same Tesla story pool.

**Option B: scope stories to `(banker, banker_client)`.** Each banker has a private story space for each of their clients. Identical stories exist in parallel across bankers, with potentially diverging metadata.

**Option A is correct for almost any version of this product** because:
- News doesn't change based on who's reading it.
- Banker A and Banker B should not see slightly different titles/summaries for the same Tesla earnings story.
- Aurora storage and Bedrock/OpenAI cost are cut roughly N× where N is the average number of bankers per client.
- Subscriptions are a separate concern: `banker_subscription(banker_id, banker_client_id)` joins bankers to the stories they should see at read time.

Pick Option B only if bankers genuinely need to maintain private annotations, custom story merges, or different summary lenses. In that case implement Option A as the storage layer and put per-banker overlays in a separate `banker_story_view` table — do not duplicate the underlying stories.

The rest of this document assumes Option A.

---

## Reference architecture (AWS-native)

```
EventBridge schedule (cron: nightly)
   │
   ▼
Lambda: kickoff_run
   │   creates run_id, snapshots banker_client list,
   │   enqueues one SQS message per banker_client
   ▼
SQS queue: story_jobs
   │
   ▼  (consumed by ECS service that scales on queue depth)
Prefect Flow on ECS (Fargate task per banker_client)
   │
   ├── 1. Pull Perplexity News results for this banker_client
   ├── 2. Pull JPM internal research tagged with this banker_client
   ├── 3. Normalize, URL-canonicalize, MinHash near-dup
   ├── 4. (Optional) Cheap-LLM entity extraction beyond banker_client itself
   ├── 5. Submit embedding batch → OpenAI Batch API (or Bedrock sync)
   ├── 6. ANN search Aurora pgvector for active stories of this banker_client
   ├── 7. Assign / verify / spawn loop
   ├── 8. Update story centroids, last_seen_at, member lists
   ├── 9. Chain-of-Key metadata update via Bedrock Claude / OpenAI GPT
   └── 10. Mark message complete
   ▼
Lambda: finalize_run
   │   closes stories where last_seen_at > 72h
   │   emits run summary to EventBridge / SNS
   ▼
Downstream consumers (banker UI, alerts)
```

**Why this shape:**

- **EventBridge** triggers the nightly cron. One rule, fires at e.g. 02:00 ET.
- **Lambda for kickoff and finalize.** These are short, single-purpose, no need to pay for an ECS task.
- **SQS fan-out per banker_client.** This is the natural unit of parallelism. SQS gives you retries, dead-letter queue for poison messages, and visibility timeout to avoid double-processing. ECS service auto-scaling on `ApproximateNumberOfMessagesVisible` lets you tune throughput without coding parallelism inside the flow.
- **Prefect Flow on ECS Fargate**, one task per banker_client. Prefect provides retries, observability, and per-task logging — much nicer than rolling your own. Fargate means no EC2 management; auto-scale on SQS depth via ECS service capacity provider or App Auto Scaling.
- **Aurora PostgreSQL with pgvector** as the only database. Stories, items, embeddings, run history all live here. Single transactional surface, single backup story, no separate vector DB to operate.
- **Bedrock for everything sensitive; OpenAI for batch embeddings and metadata where allowed.** Bedrock calls don't leave your AWS account. OpenAI gets you the Batch API discount and arguably better embeddings, but data transits to OpenAI infrastructure — confirm with your security review.

**Concrete sizing.** If you have, say, 500 active banker clients and 3000 items per nightly run, expect:
- 500 SQS messages
- ~25 concurrent ECS Fargate tasks at 1 vCPU / 2 GB
- ~3000 OpenAI Batch API embedding calls (one batch submission, completes inside 24h)
- ~200–400 Bedrock LLM calls (mostly Claude Haiku for gray-zone verification, Claude Sonnet for metadata)
- Aurora db.r6g.large is plenty for this scale; pgvector HNSW indexes well under 1GB
- Total nightly job runtime: 30–90 minutes wall-clock

You can scale this to 10x without architectural change by raising ECS task concurrency.

---

## Aurora schema

```sql
-- Banker clients (corporate entities the bankers cover)
CREATE TABLE banker_client (
  banker_client_id   uuid PRIMARY KEY,
  name               text NOT NULL,
  aliases            text[],                  -- for entity normalization
  primary_ticker     text,
  industry_codes     text[],
  created_at         timestamptz DEFAULT now()
);

-- Bankers (users of the platform)
CREATE TABLE banker (
  banker_id          uuid PRIMARY KEY,
  email              text UNIQUE NOT NULL,
  created_at         timestamptz DEFAULT now()
);

-- Subscription: which bankers care about which clients
CREATE TABLE banker_subscription (
  banker_id          uuid REFERENCES banker,
  banker_client_id   uuid REFERENCES banker_client,
  PRIMARY KEY (banker_id, banker_client_id)
);

-- Stories — scoped to banker_client
CREATE TABLE story (
  story_id           uuid PRIMARY KEY,
  banker_client_id   uuid REFERENCES banker_client NOT NULL,
  centroid           vector(1024) NOT NULL,
  medoid_item_id     uuid,
  n_items            int NOT NULL DEFAULT 1,
  first_seen_at      timestamptz NOT NULL,
  last_seen_at       timestamptz NOT NULL,
  closed_at          timestamptz,
  title              text,
  summary            text,
  entities           jsonb,           -- co-mentioned entities, tickers, locations
  topic              text,            -- earnings, m&a, regulatory, product, litigation, …
  embedding_model    text NOT NULL,
  schema_version     int NOT NULL DEFAULT 1
);

-- Indexes
CREATE INDEX story_active_ann
  ON story USING hnsw (centroid vector_cosine_ops)
  WHERE closed_at IS NULL;

CREATE INDEX story_active_by_client
  ON story (banker_client_id, last_seen_at DESC)
  WHERE closed_at IS NULL;

-- Items
CREATE TABLE story_item (
  item_id            uuid PRIMARY KEY,
  story_id           uuid REFERENCES story,
  banker_client_id   uuid REFERENCES banker_client NOT NULL,
  source             text NOT NULL,   -- 'perplexity' | 'jpm_research'
  source_ref         text,            -- perplexity result id, research doc id
  url                text,
  url_hash           bytea,
  title              text,
  lede               text,
  body               text,
  published_at       timestamptz,
  ingested_at        timestamptz DEFAULT now(),
  assignment_vector  vector(1024),    -- title + lede embedding
  retrieval_vector   vector(1024),    -- full-body embedding (optional, for search)
  is_near_duplicate  bool DEFAULT false,
  duplicate_of       uuid,
  added_to_story_at  timestamptz
);

CREATE UNIQUE INDEX story_item_dedup
  ON story_item (banker_client_id, url_hash);

CREATE INDEX story_item_by_story
  ON story_item (story_id);

-- Run tracking for idempotency and audit
CREATE TABLE clustering_run (
  run_id             uuid PRIMARY KEY,
  started_at         timestamptz NOT NULL,
  finished_at        timestamptz,
  banker_client_id   uuid,            -- null for the overall run row
  items_ingested     int,
  items_assigned     int,
  stories_created    int,
  stories_closed     int,
  embedding_model    text,
  status             text             -- running | succeeded | failed
);
```

A few notes on the choices:

- **`vector(1024)`** because text-embedding-3-large reduced via Matryoshka to 1024 dims is the sweet spot. Full 3072 dims doubles index memory for ~1% retrieval gain. Cohere Embed English v3 is natively 1024 dims, so this width works for either choice.
- **Partial HNSW index `WHERE closed_at IS NULL`** is critical. You only ever query active stories. The index stays small because closed stories drop out.
- **`url_hash` is a SHA-256 of the canonicalized URL.** Bytea, not text, for index efficiency and ordering stability.
- **Two embeddings per item.** `assignment_vector` is from `title + lede` and drives clustering. `retrieval_vector` is from full body and powers downstream search/RAG. You can defer adding the second column until you actually need it.

---

## The nightly job, step by step

Each ECS Fargate task processes one banker_client. Inside the task:

### Step 1 — Pull items

```
items = perplexity.search(
    query=f"news about {banker_client.name}",
    last_n_hours=36,            # overlap window to catch late-publishing items
    sources=["news"]
) + jpm_research.fetch_for_client(banker_client.id, since=yesterday_noon)
```

A 36-hour overlap (versus the natural 24-hour cadence) guards against items that arrive late in source feeds. Idempotency on `url_hash` ensures duplicates from the previous run are skipped.

### Step 2 — Normalize and dedupe

- Canonicalize URLs: strip `utm_*`, `fbclid`, `gclid`, fragment identifiers.
- Compute `url_hash`. Insert items into `story_item` with `ON CONFLICT (banker_client_id, url_hash) DO NOTHING`. Drop items that were already ingested.
- Extract article body. For Perplexity results, the API returns title + snippet; if you want full body, follow the URL and run a readability extractor (`trafilatura` is the Python standard). For JPM research, body comes from the document itself.
- Compute the lede: first 400–600 characters of the body, or the abstract if it's research.
- MinHash on body 5-grams; cluster items with Jaccard > 0.8 as near-duplicates (e.g., wire syndication of the same Reuters article on five different outlets). Keep one canonical, mark the rest `is_near_duplicate = true, duplicate_of = canonical_id`. Use the `datasketch` Python library — it's the standard.

### Step 3 — (Optional) entity extraction

You already know the primary entity (`banker_client_id`). But items may mention other entities you care about (deal counterparties, regulators, related tickers). One cheap Bedrock Claude Haiku call per item gives you a clean entity set for the `entities` jsonb column:

```
PROMPT:
Extract entities from this news item. Return JSON:
{
  "people":     [...],
  "orgs":       [...],
  "tickers":    [...],
  "locations":  [...]
}
TITLE: ...
LEDE:  ...
```

Skip this step in v1 if cost is a concern. You can backfill later.

### Step 4 — Embed via OpenAI Batch API

Build a JSONL file with one line per item:

```json
{"custom_id": "<item_id>", "method": "POST", "url": "/v1/embeddings",
 "body": {"model": "text-embedding-3-large", "input": "<title>\n<lede>",
          "dimensions": 1024}}
```

Submit via `POST /v1/batches`. Poll for completion. Typical completion time is minutes to a few hours; the 24-hour SLA only matters at very high volume. You get a 50% discount versus the synchronous Embeddings API.

If you need to stay in-AWS, replace this with synchronous Bedrock calls to `cohere.embed-english-v3` with `input_type=clustering`. The `clustering` input type is a real feature — Cohere tunes the embedding head differently for clustering vs. retrieval. The catch is the 512-token context limit, which means you cannot embed full bodies. Title + lede works fine within that limit.

### Step 5 — Per-client ANN search for candidate stories

```sql
SELECT story_id, centroid, last_seen_at, title, summary, entities, medoid_item_id
  FROM story
 WHERE banker_client_id = :bc
   AND closed_at IS NULL
   AND last_seen_at > now() - interval '72 hours'
 ORDER BY centroid <=> :item_vec
 LIMIT 10;
```

At this scale (likely <30 active stories per banker_client), HNSW is overkill — but it costs nothing to have it, and protects you against unexpected story growth (an entity in an active news cycle could spike to hundreds of active stories briefly).

### Step 6 — Assign / verify / spawn

```python
TAU_HIGH = 0.75    # auto-assign threshold; calibrate on labeled set
TAU_LOW  = 0.55    # below this, no LLM call, route to residual

for item in new_items:
    candidates = ann_search(item.assignment_vector, item.banker_client_id, k=10)
    best = candidates[0] if candidates else None
    sim  = 1 - best.centroid_distance if best else 0

    if best and sim >= TAU_HIGH:
        assign(item, best)
    elif best and sim >= TAU_LOW:
        # Gray zone — verify with cheap LLM
        verdict = bedrock_haiku_judge(item, best)
        if verdict == "SAME":
            assign(item, best)
        else:
            residual.append(item)
    else:
        residual.append(item)
```

The LLM judge prompt is tiny:

```
You are deciding whether a news item belongs to an existing story.
Existing story title:   <story.title>
Existing story summary: <story.summary>
New item title:         <item.title>
New item lede:          <item.lede>

Are these about the same news event? Respond with one JSON object:
{"verdict": "SAME" | "DIFFERENT", "reason": "<one sentence>"}
```

Claude Haiku on Bedrock returns this in ~500ms. At your scale this might be 5–15% of items, so well under a thousand judge calls per night.

### Step 7 — Cluster the residual (if any)

If the residual has 2+ items, run a simple pairwise check:

```python
from sklearn.cluster import AgglomerativeClustering

# For small residuals (<200 items), agglomerative on cosine is fine and deterministic
if len(residual) >= 2:
    X = np.stack([it.assignment_vector for it in residual])
    clf = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - 0.65,     # cosine distance cutoff
        metric="cosine",
        linkage="average"
    )
    labels = clf.fit_predict(X)
    for label in set(labels):
        members = [residual[i] for i, l in enumerate(labels) if l == label]
        create_story(members, banker_client_id)
```

Skip HDBSCAN here — the typical residual size doesn't justify its complexity, and agglomerative is more predictable at small N. If your residuals routinely exceed a few hundred items per banker client, switch to HDBSCAN with `min_cluster_size=2, metric='cosine'`.

### Step 8 — Update story state

For each story that absorbed new items in this run:

```sql
UPDATE story SET
  centroid    = (centroid * n_items + :new_sum_vec) / (n_items + :n_new),
  n_items     = n_items + :n_new,
  last_seen_at = greatest(last_seen_at, :new_max_pubtime),
  entities    = entities || :merged_entities,
  schema_version = schema_version + 1
WHERE story_id = :story_id;
```

The centroid math is a single weighted-mean update; you don't need to re-read all members. Recompute the medoid only periodically (every 5 added items, say) — it requires a scan of members but doesn't need to be live.

### Step 9 — Update metadata via LLM (Chain-of-Key)

For each updated story, one Bedrock Claude Sonnet or OpenAI GPT-4o call:

```
SYSTEM:
You maintain structured records of news stories for an investment banking
intelligence product. Return ONLY valid JSON matching the schema.

USER:
CURRENT_STORY:
{
  "title":   "<current title>",
  "summary": "<current summary>",
  "entities": { ... },
  "topic":   "<current topic>"
}

NEW_ITEMS:
- <title 1>: <lede 1>
- <title 2>: <lede 2>

Update the story JSON. Preserve the title unless the new items change the
core event. Extend summary to ≤400 chars covering all members. Merge
entities. Topic stays the same unless materially shifted.

Return JSON only.
```

Use structured output / tool-use to enforce the schema. Anthropic on Bedrock supports tool-use; OpenAI supports `response_format={"type": "json_schema", ...}`.

For brand-new stories, the prompt is just "given these N items, produce the story JSON" — same schema.

Regenerate from scratch (no current state) every 10 added items as a drift safeguard.

### Step 10 — Close stale stories

In the finalize Lambda, after all per-banker-client tasks finish:

```sql
UPDATE story SET closed_at = now()
 WHERE closed_at IS NULL
   AND last_seen_at <= now() - interval '72 hours';
```

This is one SQL statement across the whole table — no per-banker iteration needed.

---

## Model selection: what to use where

| Where | Recommended | Alternative | Rationale |
|---|---|---|---|
| Embedding for clustering | OpenAI `text-embedding-3-large` (1024 dims via Matryoshka), Batch API | Bedrock `cohere.embed-english-v3` with `input_type=clustering` | OpenAI wins on raw retrieval quality and supports 8K context; Cohere on Bedrock wins on data residency and has a clustering-optimized variant |
| Gray-zone LLM judge | Bedrock Claude Haiku | OpenAI `gpt-4o-mini` | Haiku is fast, cheap, in-VPC. The judge is a high-volume low-stakes call — Bedrock keeps the data on AWS |
| Story metadata generation | Bedrock Claude Sonnet or OpenAI `gpt-4o` | Bedrock Amazon Nova Pro | Sonnet's structured output is excellent and stays on AWS; GPT-4o is a fine alternative if egress is allowed |
| (Optional) entity extraction | Bedrock Claude Haiku | spaCy on the same ECS task | An LLM is more accurate than spaCy for ambiguous tickers and people names, and the cost is negligible at this volume |

A note on the embedding choice: **whichever you pick, pin the model name in the `story.embedding_model` column.** When you change models, you must re-embed every active story member or re-bootstrap clusters; embeddings from different models are not comparable. Plan a "model swap" runbook that re-embeds in place over a maintenance window.

---

## Handling Perplexity-specific concerns

Perplexity News results have a few quirks worth designing for:

- **Result snippets are short.** The body content Perplexity returns is closer to a lede than a full article. That's actually fine for your `title + lede` embedding strategy — you may not need to fetch the underlying URL at all in v1.
- **Same URL surfaces in multiple per-client searches.** If "Apple sues Samsung over chip patents" appears in both the Apple search and the Samsung search, you'll get two `story_item` rows (different `banker_client_id`), which is the correct behavior under Option A scoping. The same news event becomes a separate story under each entity, with potentially different framing.
- **Source ranking is uneven.** Perplexity returns a mix of Bloomberg, Reuters, regional press, blogs. Maintain a `source_rank` table (Bloomberg/Reuters/WSJ/FT ranked high; aggregators and blogs ranked low) and prefer high-rank items as the medoid.
- **Rate limits.** Per-banker-client API calls in parallel can hit Perplexity rate limits. Use a token bucket in Python (`aiolimiter` or similar) at the worker level, or — cleaner — set ECS service concurrency conservatively and let SQS buffer.
- **Citation hallucinations.** Perplexity occasionally returns confident-sounding items that don't fully match the query intent. Your entity-grounded design handles this naturally: if the item's content doesn't really match the banker_client, it'll fall into the residual and either form a singleton story (which you can suppress by setting `min_cluster_size=2` on residual clustering) or get pruned.

---

## Handling JPM internal research

Internal research differs from Perplexity News in important ways:

- **Longer documents.** A research note can be 5–50 pages. Use the **abstract or executive summary as the lede** for embedding; if missing, generate a 3-sentence summary via Bedrock Claude Haiku at ingest time.
- **Multi-entity coverage.** A sector note on "Mega-Cap Tech Earnings" might cover Apple, Microsoft, Google, Meta, Amazon. Insert one `story_item` row per (research_doc, banker_client) pair if the doc covers that client. The same underlying document can join different stories under different entities — that's by design under Option A.
- **Different time semantics.** Research is forward-looking ("we expect..."), while news is event-driven. Mixing these in the same story is usually correct — bankers want the news event AND the JPM view tied together. Tag items with a `source` field so downstream UI can distinguish.
- **Don't embed the whole PDF.** Per the original report's discussion of late chunking, embedding a long document as one vector loses signal. Either embed the abstract only, or embed the abstract and store the full-text retrieval vector separately for downstream search.

---

## Starting thresholds (calibrate against a labeled set)

| Parameter | Starting value | Notes |
|---|---|---|
| `TAU_HIGH` (auto-assign) | 0.75 | text-embedding-3-large, cosine. For Cohere Embed v3 expect to lower by 0.05–0.10 |
| `TAU_LOW` (skip LLM) | 0.55 | Below this, almost certainly a different story |
| Residual clustering threshold | 0.65 | Agglomerative cosine distance < 0.35 |
| Story expiry | 72 hours | Per your spec |
| Wire-dup MinHash Jaccard | 0.80 | Standard |
| Body-cosine duplicate | 0.95 | If you keep retrieval vectors |
| Merge threshold (weekly pass) | 0.85 + ≥2 shared entities | Be conservative — bad merges are worse than missed merges |

**You must calibrate `TAU_HIGH` and `TAU_LOW` on your own labeled data before going live.** Sample 500–1000 item-pairs from a recent week (mix of same-story and different-story pairs), have analysts label them, plot the cosine-similarity distribution per class, and pick thresholds that hit your target B-cubed F1 (aim for ≥0.85 on this entity-grounded version of the problem; the entity scoping should make it easier than the open benchmarks). Re-calibrate whenever you change embedding models.

---

## Evaluation and observability

Beyond B-cubed F1 on a labeled set, track these per nightly run via Prefect / CloudWatch dashboards:

- **Items ingested, items assigned, items in residual, items deduped.** Watch the assignment rate (assigned / total). If it drops suddenly, your threshold or embedding model has drifted.
- **Median and p95 story size at end of run.** Spikes signal an active news cycle; sustained increases signal threshold drift toward over-clustering.
- **Singleton rate** (stories with n_items == 1). Persistent high singleton rate means thresholds are too tight or items are too noisy.
- **LLM judge call rate.** If this climbs above 25%, your gray zone is too wide or embeddings are weaker than you thought.
- **Per-banker_client wall-clock time.** Tail latency is the operational concern — one slow Perplexity call shouldn't block the whole run.
- **Bedrock and OpenAI cost per run.** Tag every call with `run_id` and per-stage labels.

Sample 50 stories per week and have analysts spot-check for false-positive merges and false-negative splits. This catches drift the metrics will miss.

---

## What I'm deliberately leaving out

- **Glue.** This isn't ETL on a data lake; it's row-oriented database operations with API integrations. Prefect on ECS is the right shape and Glue would add Spark overhead with no benefit.
- **Pgvectorscale.** It's not available as a managed extension on Aurora PostgreSQL. The performance benchmarks cited in the original report don't apply to your stack. Plain pgvector with HNSW is fine at your scale.
- **Self-hosted models** (BGE-M3, Qwen3, Voyage). Out of scope per your constraints.
- **Real-time / streaming layer.** Your nightly cadence matches your spec; don't build streaming until a real product reason demands it.
- **LLM-as-clusterer.** The cost would not justify the marginal quality lift over embeddings + Haiku verification, especially given how much entity scoping already does for you.

---

## Recommended rollout

**Weeks 1–2.** Build the schema and a single-banker-client end-to-end happy path. Just OpenAI embeddings + a single threshold, no LLM judge yet, no metadata generation. Verify clustering quality by eyeballing 20 stories.

**Weeks 3–4.** Add Bedrock Claude Haiku as the gray-zone judge. Add Chain-of-Key metadata via Bedrock Claude Sonnet. Build a labeled eval set of 500 pairs. Calibrate thresholds.

**Weeks 5–6.** Productionize: EventBridge schedule, SQS fan-out, Prefect orchestration on ECS Fargate, finalize Lambda. Add CloudWatch metrics and Prefect deployment.

**Weeks 7+.** Add the weekly merge pass. Add a "model swap" runbook. Add per-banker subscription / read-time joining. Start collecting feedback from real bankers and iterate on metadata prompts — that's where the user-perceived quality lives.

---

## Caveats specific to your stack

1. **OpenAI data egress requires a security review.** If JPM compliance forbids any item content leaving the AWS account, you're locked to Bedrock-only — meaning Cohere Embed English v3 for embeddings (512-token limit) and Bedrock Claude for LLMs. This is a workable but materially constrained configuration; flag it as a decision needed in week 1 before you build the embedding path.
2. **OpenAI Batch API has a 24-hour SLA, not a guarantee.** In practice batches complete much faster, but design the orchestration to tolerate slower-than-expected returns. Prefect's resume-from-checkpoint behavior handles this cleanly.
3. **Bedrock model availability varies by region.** Confirm the Claude version you want is available in your AWS region; cross-region inference adds latency and may have data-residency implications.
4. **Aurora pgvector HNSW build time** is proportional to row count. The partial index `WHERE closed_at IS NULL` keeps this bounded — but the first time you populate it from history, schedule a maintenance window.
5. **Per-banker-client Perplexity quota.** At several hundred banker clients × nightly cadence × 36-hour overlap window, you may exceed Perplexity tier limits. Confirm quota before scaling.
6. **The threshold values above are priors, not optimums.** They're consistent with NewsCatcher's documented default of 0.7, the Miranda et al. 72-hour finding, and community-reported behavior of OpenAI's text-embedding-3 generation. They will need tuning on your data. There is no embedding model in 2026 for which a single fixed cosine threshold has been peer-reviewed as optimal.
