# Daily story clustering pipeline on AWS Glue
## v3: three-job Glue Workflow, S3 + Aurora pgvector, plugin-friendly sources

---

## TL;DR

Three Glue jobs wired together by a Glue Workflow, triggered daily by EventBridge:

1. **`gather_and_embed_job`** — Spark. Reads source connectors (Perplexity, JPM research, future sources via a plugin registry), normalizes and deduplicates items in memory, calls OpenAI for embeddings and Bedrock Haiku for entity extraction and summarize-before-embed. Output: enriched Parquet on S3.
2. **`cluster_and_persist_job`** — Spark with all clustering computation on the driver. Reads today's enriched Parquet from S3, loads active (last-72h) story centroids from Aurora, runs global assignment with client-overlap candidate filtering, spawns new stories from the residual via HDBSCAN, writes story rows and item-story assignments back to Aurora. Stories are written with placeholder titles/summaries — they're queryable immediately.
3. **`summarize_job`** — Spark. Reads new and updated stories from Aurora, calls Bedrock Sonnet to generate titles and summaries (full generation for new stories, Chain-of-Key incremental update for grown stories), writes back to Aurora. This can run late or fail without blocking the rest of the pipeline.

S3 is the handoff between Job 1 and Job 2 (with Glue job bookmarks for incremental processing). Aurora is the system of record. No Prefect, no ECS, no SQS. Glue Workflows handles orchestration; CloudWatch handles observability and alerting.

The cluster scope is global: one pool of stories across all bankers' deduplicated client universe. Stories carry `affected_clients` as a many-to-many, and a banker's read-time view derives through `banker_client → story_client → story`.

Starting thresholds (calibrate against a labeled set before going live): assignment τ_high = 0.75, gray-zone τ_low = 0.55, client-overlap required, story-merge cosine 0.85 plus ≥2 shared clients, near-dupe MinHash Jaccard 0.85.

---

## Why three jobs, in this order

The split isn't arbitrary — it follows what dominates each stage's wall clock and what kind of parallelism actually helps:

**Job 1 is genuinely data-parallel.** Every item is independent: normalize URL, MinHash, embed, summarize, extract entities. These are all per-row operations with no cross-row dependencies. Spark workers can churn through items in parallel and the API calls (OpenAI, Bedrock Haiku) batch beautifully across executor partitions. This is the job where Glue's Spark distribution earns its money.

**Job 2 is sequential by algorithm design.** The single-pass assignment loop has to process items in `published_at` order, updating the running-mean centroid as it goes, so item N+1 can find the story that item N just joined. You cannot partition this work across Spark workers without breaking correctness. We use Spark mainly as the vehicle for S3 reads, JDBC connectors to Aurora, and bulk writes — the actual clustering algorithm runs on the driver in NumPy. Spark adds little here over a Python Shell job, but gives us headroom for when data volumes outgrow 16 GB.

**Job 3 is again data-parallel but I/O-bound.** Each story needs one or two Bedrock Sonnet calls. Spark UDFs over story rows parallelize the API calls naturally. Splitting summarization into its own job has two virtues: (a) stories are queryable as soon as Job 2 finishes — placeholder titles like "New story (Apple, AAPL)" are fine until Job 3 enriches them; (b) if Job 3 takes longer than expected or fails, the upstream data is intact and Job 3 can be re-run independently.

---

## The data model

```sql
-- Reference tables, populated outside this pipeline
CREATE TABLE banker (
  banker_id    uuid PRIMARY KEY,
  display_name text NOT NULL,
  active       bool DEFAULT true
);

-- Clients as first-class entities, deduplicated across bankers
CREATE TABLE client (
  client_id     uuid PRIMARY KEY,
  display_name  text NOT NULL,
  legal_name    text,
  aliases       text[],
  active        bool DEFAULT true
);
CREATE INDEX ON client USING gin (aliases);

-- M:N: which bankers cover which clients
CREATE TABLE banker_client (
  banker_id   uuid REFERENCES banker NOT NULL,
  client_id   uuid REFERENCES client NOT NULL,
  PRIMARY KEY (banker_id, client_id)
);
CREATE INDEX ON banker_client (client_id);

-- Source registry (driven by config; Job 1 reads this)
CREATE TABLE source (
  source_id   text PRIMARY KEY,                   -- 'perplexity', 'jpm_research'
  display_name text NOT NULL,
  connector_class text NOT NULL,                  -- Python class path
  config_json jsonb NOT NULL,
  enabled     bool DEFAULT true,
  added_at    timestamptz DEFAULT now()
);

-- Raw items, globally deduped on url_hash
CREATE TABLE item (
  item_id         uuid PRIMARY KEY,
  source_id       text REFERENCES source NOT NULL,
  url             text,
  url_hash        bytea NOT NULL UNIQUE,
  title           text,
  body            text,
  published_at    timestamptz,
  ingested_at     timestamptz DEFAULT now(),
  assignment_vec  vector(1024),
  retrieval_vec   vector(1024),
  entities        jsonb,
  minhash         bytea,
  is_duplicate    bool DEFAULT false,
  duplicate_of    uuid REFERENCES item,
  story_id        uuid,
  embedding_model text NOT NULL
);
CREATE INDEX ON item (story_id) WHERE story_id IS NOT NULL;
CREATE INDEX ON item (ingested_at);

CREATE TABLE item_client (
  item_id    uuid REFERENCES item NOT NULL,
  client_id  uuid REFERENCES client NOT NULL,
  origin     text NOT NULL,                       -- 'seed' (from connector query) | 'extracted' (entity match)
  PRIMARY KEY (item_id, client_id)
);
CREATE INDEX ON item_client (client_id);

-- Global stories
CREATE TABLE story (
  story_id        uuid PRIMARY KEY,
  centroid        vector(1024) NOT NULL,
  medoid_item_id  uuid REFERENCES item,
  n_items         int NOT NULL DEFAULT 1,
  first_seen_at   timestamptz NOT NULL,
  last_seen_at    timestamptz NOT NULL,
  closed_at       timestamptz,
  title           text,                            -- placeholder until Job 3 fills it
  summary         text,
  entities        jsonb,
  topic           text,
  embedding_model text NOT NULL,
  schema_version  int NOT NULL DEFAULT 1,
  merged_into     uuid REFERENCES story,
  summary_dirty   bool NOT NULL DEFAULT true,     -- true → Job 3 should regenerate
  last_summary_n  int NOT NULL DEFAULT 0          -- n_items at last full summary regen
);
CREATE INDEX ON story USING hnsw (centroid vector_cosine_ops)
  WHERE closed_at IS NULL AND merged_into IS NULL;
CREATE INDEX ON story (last_seen_at) WHERE closed_at IS NULL;
CREATE INDEX ON story (summary_dirty) WHERE summary_dirty;  -- Job 3's work queue

CREATE TABLE story_client (
  story_id   uuid REFERENCES story NOT NULL,
  client_id  uuid REFERENCES client NOT NULL,
  added_at   timestamptz DEFAULT now(),
  PRIMARY KEY (story_id, client_id)
);
CREATE INDEX ON story_client (client_id);

-- Glue workflow run anchor
CREATE TABLE pipeline_run (
  run_id        uuid PRIMARY KEY,
  run_date      date NOT NULL UNIQUE,
  status        text NOT NULL,                    -- 'running', 'job1_done', 'job2_done', 'succeeded', 'failed'
  started_at    timestamptz NOT NULL,
  job1_finished_at timestamptz,
  job2_finished_at timestamptz,
  job3_finished_at timestamptz,
  stats         jsonb,
  error         text
);
```

Two fields worth highlighting that didn't exist in earlier versions:

- **`story.summary_dirty`** — Job 2 sets this `true` whenever a story is created or grows; Job 3 reads `WHERE summary_dirty = true` as its work queue, generates the summary, sets `false`. This decouples the two jobs cleanly. If Job 3 fails halfway, restarting it picks up exactly the unfinished work.
- **`story.last_summary_n`** — tracks how many members existed when the summary was last fully regenerated. Job 3 uses this to decide between Chain-of-Key incremental update vs. full regeneration (full regen every 10 items to bound drift).

---

## Job 1: `gather_and_embed_job`

**Runtime profile:** Glue Spark, 10–20 G.1X workers, ~30–60 min at moderate scale.

```python
# glue/jobs/gather_and_embed_job.py
import sys
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.sql import functions as F, types as T

from connectors import load_connector_registry
from text_ops import canonicalize_url, extract_main_content, sha256_bytes
from llm_ops import bedrock_haiku_summarize_udf, bedrock_haiku_entities_udf
from embed_ops import openai_embed_udf
from dedup_ops import compute_minhash_udf, lsh_mark_duplicates
from client_match import build_alias_matcher, resolve_client_mentions_udf

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'RUN_ID', 'RUN_DATE'])
sc = GlueContext(SparkContext.getOrCreate()).spark_session
run_id, run_date = args['RUN_ID'], args['RUN_DATE']

# 1. Resolve client universe (deduped union across bankers)
clients_df = sc.read.format("jdbc").options(**aurora_jdbc_opts(query="""
    SELECT DISTINCT c.client_id, c.display_name, c.aliases
      FROM banker b JOIN banker_client bc ON b.banker_id = bc.banker_id
      JOIN client c ON bc.client_id = c.client_id
     WHERE b.active AND c.active
""")).load().cache()

# 2. Per-source ingest via plugin registry
sources = sc.read.format("jdbc").options(**aurora_jdbc_opts(table="source")).load() \
    .filter(F.col("enabled")).collect()

raw_dfs = []
for src in sources:
    connector = load_connector_registry()[src.connector_class]
    # Each connector returns a DataFrame of (source_id, url, title, body, published_at, seed_client_id|null)
    raw_dfs.append(connector(sc, clients_df, src.config_json, run_date))

raw = raw_dfs[0]
for d in raw_dfs[1:]:
    raw = raw.unionByName(d, allowMissingColumns=True)

# 3. Normalize + URL hash + intra-batch dedup
norm = (raw
    .withColumn("url_canon", canonicalize_url(F.col("url")))
    .withColumn("body_clean", extract_main_content(F.col("body")))
    .withColumn("url_hash", sha256_bytes(F.col("url_canon")))
    .dropDuplicates(["url_hash"])
)

# 4. Cross-reference with existing items: skip embedding for already-seen URLs,
#    but still merge new seed client mappings
existing = sc.read.format("jdbc").options(**aurora_jdbc_opts(table="item")).load() \
    .select("item_id", "url_hash").withColumnRenamed("item_id", "existing_item_id")
joined = norm.join(existing, on="url_hash", how="left")
new = joined.filter(F.col("existing_item_id").isNull())
seen = joined.filter(F.col("existing_item_id").isNotNull())

# Write new seed mappings for already-seen items (one row per (existing_item_id, seed_client_id))
seen.filter(F.col("seed_client_id").isNotNull()) \
    .select("existing_item_id", "seed_client_id") \
    .withColumn("origin", F.lit("seed")) \
    .write.format("jdbc") \
    .option("dbtable", "item_client_staging_seen") \
    .mode("append").save()

# 5. MinHash + LSH near-dup detection (within today's new items)
new = compute_minhash_udf(new, num_perm=128, shingle_size=5)
new = lsh_mark_duplicates(new, threshold=0.85)  # adds is_duplicate, duplicate_of

# 6. Embed + summarize + entities only for canonical (non-duplicate) items
canonical = new.filter(~F.col("is_duplicate"))

canonical = canonical.withColumn(
    "summary_for_embedding",
    bedrock_haiku_summarize_udf(F.col("title"), F.col("body_clean"))
)
canonical = canonical.withColumn("entities", bedrock_haiku_entities_udf(F.col("title"), F.col("body_clean")))

# Build alias matcher from clients_df, broadcast it, resolve mentions
alias_bc = sc.sparkContext.broadcast(build_alias_matcher(clients_df.collect()))
canonical = canonical.withColumn(
    "extracted_client_ids",
    resolve_client_mentions_udf(F.col("body_clean"), F.col("entities"), alias_bc)
)

# Two embedding columns
canonical = (canonical
    .withColumn("embed_input", F.concat_ws("\n\n", F.col("title"), F.col("summary_for_embedding")))
    .withColumn("retrieval_input", F.concat_ws("\n\n", F.col("title"), F.substring(F.col("body_clean"), 1, 2000)))
    .withColumn("assignment_vec", openai_embed_udf(F.col("embed_input"), F.lit(1024)))
    .withColumn("retrieval_vec", openai_embed_udf(F.col("retrieval_input"), F.lit(1024)))
    .withColumn("embedding_model", F.lit("text-embedding-3-large@1024"))
)

# Drop items with no client matches (noise)
canonical = canonical.filter(F.size(F.col("extracted_client_ids")) > 0)

# 7. Write enriched items to S3 as Parquet, partitioned by run_date
canonical.write.mode("overwrite") \
    .partitionBy("run_date") \
    .parquet(f"s3://{bucket}/processed/items/run_date={run_date}/")

# Also write a small manifest for Job 2
manifest = {
    "run_id": run_id,
    "run_date": run_date,
    "n_new_items": canonical.count(),
    "embedding_model": "text-embedding-3-large@1024"
}
write_manifest(f"s3://{bucket}/processed/items/run_date={run_date}/_manifest.json", manifest)

job.commit()
```

### Source connectors (the plugin pattern)

Future sources are added without touching the job code. Each connector is a Python class registered in the `source` table:

```python
# connectors/base.py
from abc import ABC, abstractmethod

class SourceConnector(ABC):
    @abstractmethod
    def fetch(self, spark, clients_df, config: dict, run_date: str):
        """Return a Spark DataFrame with columns:
           source_id, url, title, body, published_at, seed_client_id (nullable)
        """
        ...

# connectors/perplexity.py
class PerplexityConnector(SourceConnector):
    """One Perplexity call per (run, client). Parallelized by repartition."""
    def fetch(self, spark, clients_df, config, run_date):
        # Repartition by number of concurrent slots to control rate
        rdd = clients_df.repartition(config.get("concurrency", 50)) \
            .rdd.mapPartitions(lambda partition: [
                self._call_perplexity(c, config, run_date) for c in partition
            ])
        return spark.createDataFrame(rdd, schema=ITEM_SCHEMA)

    def _call_perplexity(self, client_row, config, run_date):
        # HTTP call to Perplexity, returns one or more items
        ...

# connectors/jpm_research.py
class JpmResearchConnector(SourceConnector):
    """Reads from internal research API or S3 drop zone."""
    def fetch(self, spark, clients_df, config, run_date):
        if config["mode"] == "s3":
            return spark.read.json(config["s3_input_glob"]).filter(...)
        else:
            return spark.read.format("jdbc").options(**config["jdbc"]).load()
```

Adding a new source: write a connector class, insert a row into `source` with the class path and config JSON. No Glue job redeploy needed.

### Job bookmarks

Configure Glue job bookmarks on the S3 raw input paths used by S3-based connectors. This means re-running Job 1 for the same date skips already-processed source files at the connector layer. For API-based connectors (Perplexity, JPM research API), idempotency is enforced at the `item.url_hash` UNIQUE constraint downstream — re-running is safe but does the duplicate API work.

If you want bookmark-style behavior for API connectors, write each connector's raw responses to S3 first (cheap), then let Job 1 process from S3 with bookmarks. This makes re-runs cheap *and* idempotent.

---

## Job 2: `cluster_and_persist_job`

**Runtime profile:** Glue Spark, **1 G.4X worker** (64 GB driver memory, 16 vCPU), 30–60 min at moderate scale. The clustering algorithm runs on the Spark driver, not distributed — Spark is the vehicle, not the engine.

Alternative: **Glue Python Shell job, 1 DPU (16 GB)** for small-to-moderate scale. Cleaner code, no Spark overhead. Switch to Spark G.4X when the working set exceeds ~10 GB. Use Spark from day one to avoid the migration later.

```python
# glue/jobs/cluster_and_persist_job.py
import sys, numpy as np, hdbscan
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext

from clustering import build_client_to_story_index, assign_loop, spawn_new_stories
from persistence import bulk_upsert_stories, bulk_upsert_item_assignments

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'RUN_ID', 'RUN_DATE'])
sc = GlueContext(SparkContext.getOrCreate()).spark_session
run_date = args['RUN_DATE']

# 1. Read today's enriched items from S3 → collect to driver as pandas/polars
items_pdf = (sc.read.parquet(f"s3://{bucket}/processed/items/run_date={run_date}/")
    .select("item_id", "url_hash", "title", "body_clean", "published_at",
            "assignment_vec", "retrieval_vec", "entities",
            "extracted_client_ids", "seed_client_id",
            "is_duplicate", "duplicate_of")
    .toPandas()                              # collect to driver
    .sort_values("published_at")
)

# 2. Load active stories + their client tags from Aurora
stories_pdf = sc.read.format("jdbc").options(**aurora_jdbc_opts(query="""
    SELECT s.story_id, s.centroid, s.medoid_item_id, s.n_items, s.last_seen_at,
           s.entities, s.embedding_model
      FROM story s
     WHERE s.closed_at IS NULL
       AND s.merged_into IS NULL
       AND s.last_seen_at > now() - interval '72 hours'
""")).load().toPandas()

story_clients_pdf = sc.read.format("jdbc").options(**aurora_jdbc_opts(query="""
    SELECT sc.story_id, sc.client_id
      FROM story_client sc
      JOIN story s ON sc.story_id = s.story_id
     WHERE s.closed_at IS NULL AND s.merged_into IS NULL
       AND s.last_seen_at > now() - interval '72 hours'
""")).load().toPandas()

# 3. Build the client → story-index lookup
client_to_story_idxs = build_client_to_story_index(stories_pdf, story_clients_pdf)

# 4. Run the assignment loop (pure NumPy on the driver)
canonical = items_pdf[~items_pdf["is_duplicate"]]
assignments_df, residual_df = assign_loop(
    new_items=canonical,
    stories=stories_pdf,
    client_to_story_idxs=client_to_story_idxs,
    tau_high=0.75,
    tau_low=0.55,
    bedrock_judge=bedrock_haiku_judge,        # callable for gray-zone calls
)

# 5. HDBSCAN on the residual to spawn new stories
new_stories_df, residual_to_story = spawn_new_stories(
    residual_df,
    min_cluster_size=2,
    enforce_client_overlap=True,
)

# 6. Propagate duplicates: dupes inherit their canonical's story_id
duplicates = items_pdf[items_pdf["is_duplicate"]]
duplicate_assignments = propagate_duplicates_to_stories(
    duplicates, canonical_assignments=assignments_df
)

# 7. Bulk write to Aurora in a single transaction
with aurora_transaction() as conn:
    # Insert new items (canonical + duplicates)
    bulk_insert_items(conn, items_pdf)

    # Update item_client (seed + extracted, plus the merged-from-existing ones from Job 1)
    bulk_insert_item_clients(conn, items_pdf)

    # Insert new stories with placeholder titles (Job 3 will fill them)
    bulk_insert_new_stories(conn, new_stories_df, summary_dirty=True)
    bulk_insert_story_clients(conn, new_stories_df)

    # Update existing stories that received items: centroid, last_seen_at, n_items
    # Critically: recompute centroid from the full member set (not from running mean)
    # so this task is idempotent on retry
    bulk_update_existing_stories(conn, assignments_df, items_pdf, summary_dirty=True)
    bulk_insert_story_clients_for_grown_stories(conn, assignments_df, items_pdf)

    # Set item.story_id for all newly assigned items (canonical + duplicates)
    bulk_update_item_story_ids(conn, assignments_df, duplicate_assignments, residual_to_story)

    # Close stale stories (last_seen_at > 72h, didn't receive items today)
    conn.execute("""UPDATE story SET closed_at = now()
                     WHERE closed_at IS NULL
                       AND last_seen_at <= now() - interval '72 hours'""")
```

### The assignment loop (driver-side NumPy)

```python
# clustering/assign.py
import numpy as np

def assign_loop(new_items, stories, client_to_story_idxs,
                tau_high, tau_low, bedrock_judge):
    if stories.empty:
        return pd.DataFrame(columns=["item_id", "story_id", "similarity"]), new_items

    story_vecs = np.vstack(stories["centroid"].values)
    story_n = stories["n_items"].values.copy()
    assignments, residual = [], []

    # Pre-collect gray-zone pairs to batch the Bedrock judge calls.
    # We don't know which pairs are gray-zone until we look at sims,
    # so we do two passes: pre-scan to identify gray-zone, then judge in batch.
    pending = []  # (idx, item, best_story_global_idx, sim)
    for i, item in enumerate(new_items.itertuples(index=False)):
        candidate_idxs = set()
        for cid in item.extracted_client_ids:
            candidate_idxs.update(client_to_story_idxs.get(cid, []))
        if not candidate_idxs:
            residual.append(item); continue
        candidate_idxs = list(candidate_idxs)

        item_vec = item.assignment_vec
        sims = story_vecs[candidate_idxs] @ item_vec
        best_local = int(np.argmax(sims))
        best_global = candidate_idxs[best_local]
        best_sim = float(sims[best_local])

        if best_sim >= tau_high:
            _commit_assignment(item, best_global, best_sim, story_vecs, story_n,
                              client_to_story_idxs, stories, assignments)
        elif best_sim >= tau_low:
            pending.append((i, item, best_global, best_sim))
        else:
            residual.append(item)

    # Batch the gray-zone judge calls concurrently (asyncio)
    verdicts = bedrock_judge.batch([
        (item, stories.iloc[g]) for _, item, g, _ in pending
    ])
    for (i, item, best_global, best_sim), verdict in zip(pending, verdicts):
        if verdict == "SAME_STORY":
            _commit_assignment(item, best_global, best_sim, story_vecs, story_n,
                              client_to_story_idxs, stories, assignments)
        else:
            residual.append(item)

    return pd.DataFrame(assignments), pd.DataFrame(residual)
```

Two performance moves worth noting:

- **Pre-scan + batch the gray-zone judge calls.** The first pass through the loop identifies which items need a Bedrock judge call. We then send all of them concurrently via `asyncio.gather` rather than one at a time inside the loop. Wall-clock saving: 5–10× on the gray-zone slice.
- **Running-mean centroid update inside `_commit_assignment`** so later items in the same batch see the updated story centroid. The persisted centroid (written in step 7) is *recomputed from all members* rather than from the running mean — this makes the entire Job 2 idempotent on retry.

### Residual clustering

```python
# clustering/spawn.py
import hdbscan

def spawn_new_stories(residual_df, min_cluster_size=2, enforce_client_overlap=True):
    if len(residual_df) < 2:
        return create_singletons(residual_df), {}

    vecs = np.vstack(residual_df["assignment_vec"].values)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="cosine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vecs)
    residual_df = residual_df.copy()
    residual_df["cluster_label"] = labels

    if enforce_client_overlap:
        residual_df = split_clusters_with_empty_client_overlap(residual_df)

    new_stories, residual_to_story = build_story_rows_from_clusters(residual_df)
    return new_stories, residual_to_story
```

---

## Job 3: `summarize_job`

**Runtime profile:** Glue Spark, 5–10 G.1X workers, 20–40 min. Each story → one or two Bedrock Sonnet calls. Embarassingly parallel.

```python
# glue/jobs/summarize_job.py
import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.sql import functions as F

from llm_ops import bedrock_sonnet_init_story_udf, bedrock_sonnet_update_story_udf

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'RUN_ID'])
sc = GlueContext(SparkContext.getOrCreate()).spark_session

# 1. Pull dirty stories from Aurora (work queue)
dirty = sc.read.format("jdbc").options(**aurora_jdbc_opts(query="""
    SELECT s.story_id, s.title, s.summary, s.entities, s.topic,
           s.n_items, s.last_summary_n,
           s.first_seen_at IS NOT DISTINCT FROM s.last_seen_at AS is_new
      FROM story s
     WHERE s.summary_dirty = true
       AND s.closed_at IS NULL
""")).load()

# 2. Join with member items (need title + lede for context)
items_for_stories = sc.read.format("jdbc").options(**aurora_jdbc_opts(query="""
    SELECT i.story_id, i.item_id, i.title, substring(i.body, 1, 800) as lede,
           i.published_at,
           row_number() OVER (PARTITION BY i.story_id ORDER BY i.published_at DESC) as rn
      FROM item i
      JOIN story s ON s.story_id = i.story_id
     WHERE s.summary_dirty = true
       AND s.closed_at IS NULL
""")).load().filter(F.col("rn") <= 10)  # last 10 items per story for context

story_with_members = (dirty.join(
    items_for_stories.groupBy("story_id").agg(
        F.collect_list(F.struct("title", "lede", "published_at")).alias("members")
    ),
    on="story_id", how="left"
))

# 3. UDF dispatch: full generation for new stories, incremental for grown stories,
#    full regen if (n_items - last_summary_n) >= 10
def updated_story_udf(story_id, current_title, current_summary, current_entities,
                       n_items, last_summary_n, is_new, members):
    if is_new or (n_items - last_summary_n) >= 10:
        result = bedrock_sonnet_init_story(members)
        return (result["title"], result["summary"], result["entities"], result["topic"], n_items)
    else:
        result = bedrock_sonnet_update_story(
            current={"title": current_title, "summary": current_summary, "entities": current_entities},
            new_items=members,
        )
        return (result["title"], result["summary"], result["entities"], result["topic"], last_summary_n)

result = story_with_members.withColumn(
    "updated",
    updated_story_udf_struct(
        F.col("story_id"), F.col("title"), F.col("summary"), F.col("entities"),
        F.col("n_items"), F.col("last_summary_n"), F.col("is_new"), F.col("members"),
    )
)

# 4. Write back to Aurora; clear summary_dirty
final = result.select(
    "story_id",
    F.col("updated.title").alias("title"),
    F.col("updated.summary").alias("summary"),
    F.col("updated.entities").alias("entities"),
    F.col("updated.topic").alias("topic"),
    F.col("updated.last_summary_n").alias("last_summary_n"),
    F.lit(False).alias("summary_dirty"),
)

final.write.format("jdbc").options(**aurora_jdbc_opts(table="story_update_staging")).mode("overwrite").save()

# Merge staging into story (UPDATE ... FROM staging)
run_aurora_sql("""
    UPDATE story s
       SET title = u.title, summary = u.summary, entities = u.entities,
           topic = u.topic, last_summary_n = u.last_summary_n,
           summary_dirty = false
      FROM story_update_staging u
     WHERE s.story_id = u.story_id
""")
```

### Why the work queue is a column, not a separate table

`story.summary_dirty` doubles as a checkpoint. Job 2 sets `true` when it touches a story; Job 3 finds work by querying `WHERE summary_dirty = true`; Job 3 clears `false` on completion. If Job 3 crashes halfway through, restarting it picks up exactly the unfinished work — no reasoning about "where was I". A separate queue table would add operational complexity for no benefit at this cadence.

### Prompts

```python
# llm_ops/prompts.py

INIT_STORY_PROMPT = """You are creating the canonical summary for a news story
spanning multiple items.

ITEMS:
{members_json}

Return JSON matching this schema:
{
  "title":   string, ≤80 chars, neutral, factual headline
  "summary": string, ≤400 chars, 2-3 sentences, "who did what, where, when"
  "entities": {"people": [], "orgs": [], "locations": [], "tickers": []}
  "topic":   string, one of {politics, business, technology, science, finance, regulatory, other}
}
"""

UPDATE_STORY_PROMPT = """You maintain a JSON summary of an ongoing story.

CURRENT STORY:
{current_json}

NEW ITEMS TO INCORPORATE:
{new_items_json}

Return the updated JSON. Add new entities to existing lists. Update title/summary
ONLY if new items materially change the story; otherwise return unchanged.
Respect schema: title ≤80 chars, summary ≤400 chars.
"""
```

Both use Bedrock Sonnet 4.6 with structured output enforced via tool_use. Pin the model ID (`anthropic.claude-sonnet-4-6-20251001-v1:0` or equivalent) and store it on `story.embedding_model` analogue for auditability.

---

## Glue Workflow orchestration

```python
# infrastructure/glue_workflow.py (CDK / Terraform / boto3 equivalent)
workflow = {
    "Name": "daily_clustering_pipeline",
    "Description": "Daily ingest, cluster, summarize",
}

triggers = [
    {
        "Name": "trigger_job1_daily",
        "Type": "SCHEDULED",
        "Schedule": "cron(0 2 * * ? *)",  # 02:00 UTC daily
        "Actions": [{"JobName": "gather_and_embed_job", "Arguments": {"--RUN_DATE": "{date}"}}],
        "WorkflowName": "daily_clustering_pipeline",
    },
    {
        "Name": "trigger_job2_after_job1",
        "Type": "CONDITIONAL",
        "Predicate": {
            "Logical": "ANY",
            "Conditions": [{"LogicalOperator": "EQUALS", "JobName": "gather_and_embed_job", "State": "SUCCEEDED"}],
        },
        "Actions": [{"JobName": "cluster_and_persist_job"}],
        "WorkflowName": "daily_clustering_pipeline",
    },
    {
        "Name": "trigger_job3_after_job2",
        "Type": "CONDITIONAL",
        "Predicate": {
            "Logical": "ANY",
            "Conditions": [{"LogicalOperator": "EQUALS", "JobName": "cluster_and_persist_job", "State": "SUCCEEDED"}],
        },
        "Actions": [{"JobName": "summarize_job"}],
        "WorkflowName": "daily_clustering_pipeline",
    },
]
```

Three triggers form a linear DAG. EventBridge would also work but Glue Workflows is the more native fit — the workflow itself shows up in the Glue console with a graph view of job states, runtime, and failure history, no extra service needed.

**Failure handling:**

- If Job 1 fails: workflow stops. Job 2 and Job 3 don't run. Manual or scheduled retry of the workflow on the same `RUN_DATE` is idempotent (url_hash UNIQUE, upserts everywhere).
- If Job 2 fails: workflow stops. Stories aren't created today, but yesterday's stories are intact. Re-running Job 2 alone (via Workflow's "resume" or a manual trigger) picks up from S3 and processes idempotently.
- If Job 3 fails: stories from Job 2 are queryable with placeholder titles. Re-run Job 3 alone (via the `summary_dirty = true` work queue). This is the most graceful failure mode — read-side functionality is preserved.

CloudWatch alarms on each job's `glue.driver.aggregate.numFailedTasks` and on the workflow itself for end-to-end SLA monitoring. Send to SNS → Slack/PagerDuty.

---

## S3 layout

```
s3://your-clustering-bucket/
  config/
    sources.yaml                                # local override of source table for testing
    client_alias_cache.json                     # serialized Aho-Corasick matcher, regenerated nightly
  
  raw/                                           # connector dumps (optional; recommended for replay)
    source=perplexity/run_date=2026-05-27/client_id={uuid}.json
    source=jpm_research/run_date=2026-05-27/part-*.json
    source={new_source}/run_date=2026-05-27/...
  
  processed/items/                              # output of Job 1, input to Job 2
    run_date=2026-05-27/
      part-00000.snappy.parquet
      part-00001.snappy.parquet
      _manifest.json
  
  intermediate/                                  # Job 2 staging if needed for debug
    run_date=2026-05-27/
      assignments.parquet
      new_stories.parquet
  
  archive/                                       # lifecycle policy moves old processed/ here after 14 days
    ...
  
  job_artifacts/                                 # Glue script versions, dependencies wheels
    gather_and_embed/v1/
    cluster_and_persist/v1/
    summarize/v1/
```

S3 lifecycle policies: move `processed/` to Glacier after 30 days, delete after 1 year. Keep `raw/` for 90 days as a replay buffer.

---

## Cost estimate

At a notional 100k items / 30k active stories / 5k stories updated per day:

| Component | Calls | Unit | Total |
|---|---|---|---|
| **Job 1 — Glue Spark, 15 G.1X for 45 min** | | 15 × 0.75h × $0.44 = | $4.95 |
| Perplexity (1 per unique client × 500) | 500 | ~$0.005 | $2.50 |
| Bedrock Haiku summarize | 100k | ~$0.0001 | $10 |
| Bedrock Haiku entities | 100k | ~$0.0001 | $10 |
| OpenAI text-embedding-3-large (1024 dim, batched) | ~200k vec | $0.13/M tok × 500 tok avg | $13 |
| **Job 2 — Glue Spark, 1 G.4X for 30 min** | | 4 × 0.5h × $0.44 = | $0.88 |
| Bedrock Haiku judge (gray zone, ~20%) | ~20k | ~$0.00015 | $3 |
| **Job 3 — Glue Spark, 8 G.1X for 30 min** | | 8 × 0.5h × $0.44 = | $1.76 |
| Bedrock Sonnet init (new stories ~500) | 500 | ~$0.005 | $2.50 |
| Bedrock Sonnet update (grown stories ~4500) | 4500 | ~$0.005 | $22.50 |
| **Glue compute total** | | | **$7.59** |
| **API total (Bedrock + OpenAI + Perplexity)** | | | **$63.50** |
| **Daily run total** | | | **~$71/day, ~$2,100/month** |

Notes:

- **API costs dominate Glue compute by ~10×.** This is the right shape — you're paying for value (LLM reasoning, embeddings), not for infrastructure.
- Story-update Sonnet calls are the largest single line. If you need to cut, downgrade incremental updates to Haiku at the cost of summary polish — saves ~$15/day.
- Adding new sources only increases costs if they generate net-new items (after dedup). Adding new bankers covering existing clients adds ~zero cost since their stories already exist.
- Glue compute scales linearly with data volume. At 10× scale (~1M items/day), expect ~$50/day in Glue (still small) and ~$500/day in APIs.

Compared to Prefect+ECS for the equivalent workload, Glue is roughly 5–10× more expensive on compute alone, but compute is <15% of total daily spend. The cost premium buys you: managed scaling, no instance management, AWS-native observability, Spark distribution for the parts that benefit, and a single tool for both the runtime pipeline and offline analytics. At this cost profile it's the right call.

---

## What to build first vs. later

### Week 1-2 — minimum viable
- Aurora schema, HNSW index
- One Glue Python Shell job that does all three phases inline (for fast iteration); split into 3 jobs once the algorithm stabilizes
- Just the Perplexity connector via the plugin pattern
- Single threshold `tau_assign = 0.7`, no gray-zone judge
- New stories spawned 1:1 from any unassigned item (no HDBSCAN yet)
- Title/summary on creation only, via Bedrock Haiku to keep cost low
- EventBridge cron + manual workflow trigger
- 100-pair labeled sanity-check set

### Week 3-6 — quality
- Split into the three named Glue jobs
- Add JPM research connector + alias-matcher
- HDBSCAN residual clustering
- Bedrock Haiku gray-zone judge (split single threshold into tau_high / tau_low)
- Chain-of-Key incremental summary updates in Job 3
- 1,000-pair labeled set with B-cubed evaluation
- CloudWatch dashboards + alarms

### Month 2+ — scale and refinement
- OpenAI Batch API for embeddings (50% saving) once item counts justify the 24h SLA
- New source connectors as the data team onboards them
- Weekly merge pass: pairwise centroid cosine + LLM verify; run as a 4th Glue job on Sundays
- A/B test Cohere Embed v3 on Bedrock vs OpenAI on the labeled set
- Promote Job 2 to G.4X Spark from Python Shell when working set exceeds 10 GB
- Glue jobs for offline analytics: weekly B-cubed metrics, drift detection, threshold-tuning regression
- Optional: replace the simple TAU threshold with a small trained classifier over `(cosine, client_overlap_count, entity_overlap_count, time_delta_hours, source_rank)`

---

## Caveats and known risks

1. **Spark UDFs that call external APIs need careful sizing.** PySpark UDFs incur serialization overhead per call; the API call itself is dominantly network-bound. Use `mapPartitions` (Python-native iteration over a partition with one HTTP session reused) rather than per-row UDFs for the Bedrock and OpenAI calls. Configure executor concurrency to match the API's rate limit, not the worker's CPU count.

2. **Glue Spark cold start is 1–3 minutes per job.** Three jobs in sequence = up to 9 minutes of pure cold-start overhead per day. Negligible compared to the ~2-hour total runtime, but it shows up in CloudWatch latency metrics. Not a real problem; just expect it.

3. **JDBC writes to Aurora are the slowest bulk operation.** Use Glue's `s3 to postgres` connector via COPY for inserts > 10k rows (write to S3 staging, then Aurora `aws_s3.table_import_from_s3`). Standard JDBC `bulkCopyToSqlDB` works but is 5–10× slower for large batches.

4. **Glue Workflow can't express complex DAGs.** Linear sequences are fine; branching ("if Job 2 succeeds, run Job 3a in parallel with Job 3b") gets clunky. If you ever need a richer DAG, switch the orchestrator to Step Functions while keeping the jobs as Glue jobs.

5. **The 16 GB ceiling on Python Shell is the most likely surprise.** At 100k items it's comfortable; at 500k+ it gets tight; at 1M+ you need Spark. Start with Spark to avoid the migration; the cost difference is small.

6. **`pgvectorscale` (Timescale DiskANN) is not available on Aurora.** You're using plain pgvector HNSW. At <100k active stories this is fine. If active story count grows past ~1M, consider migrating Aurora to RDS Postgres (which can install pgvectorscale) or to a self-managed Postgres on EC2. Don't preempt this — the gap between "fine" and "not fine" is several orders of magnitude away.

7. **Cosine thresholds are starting points, not optima.** Tune against a labeled set. Re-tune any time you change the embedding model. Embedding-model-specific calibration is non-negotiable.

8. **Perplexity quality is the upstream bottleneck.** No amount of downstream clustering rescues low-quality search results. Invest in per-client Perplexity query templates, recency filters, source whitelisting. Treat the source layer as a product.

9. **Bedrock model versions drift.** Pin model IDs (`anthropic.claude-haiku-4-5-20251001-v1:0`); never use `latest`. Capture the model ID in `story.embedding_model` so you can re-embed when models change.

10. **Glue job bookmarks have edge cases.** They track S3 objects by `(path, last_modified, etag)`. If a connector overwrites a key with new content but the bookmark thinks the old version was processed, the new content is skipped. Always write connector output to date-partitioned keys (`run_date=…`) so each day's input is genuinely new keys, never overwrites.

11. **Source plugin loading from Python class paths needs care in Glue.** The Spark executor JVM doesn't see your `connectors/` package by default. Bundle connectors into a Python wheel uploaded to S3, and reference it via `--additional-python-modules s3://bucket/wheels/connectors-1.0.0-py3-none-any.whl` in the Glue job. Re-deploy the wheel for every connector addition.

12. **Glue Workflow's "resume from failed job" works at the job level, not at the row level.** If Job 1 fails at item 80,000 of 100,000, you re-run the whole Job 1 with the same RUN_DATE. The url_hash UNIQUE constraint makes this idempotent for already-processed items, but you pay the API cost again for any items whose embeddings weren't persisted. To bound this, checkpoint embedding results to S3 incrementally (write partial Parquet every 10k items) rather than only on full completion.

13. **Aurora connection limit is finite.** Spark executors opening 100 concurrent JDBC connections will saturate Aurora's max_connections fast. Use a connection pool sidecar (RDS Proxy) or limit executor count for JDBC-heavy stages.
