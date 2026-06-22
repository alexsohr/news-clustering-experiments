# A Practical Technical Guide to Mass Batch Clustering of News and Research Items into Persistent "Story" Objects

## TL;DR
- **For your specific use case (nightly per-client batch, persistent 72-hour stories, LLM + embedding APIs available), do not use BERTopic/HDBSCAN from scratch each night.** The right architecture is the classic Topic Detection and Tracking (TDT) "single-pass nearest-cluster-assignment" loop — embedding the new items, performing an ANN search against the *active* (last 72h) story centroids, assigning if cosine similarity passes a tuned threshold (~0.55–0.75 with a modern dense model, plus an entity-overlap or LLM-judge gate for borderline cases), otherwise running HDBSCAN/Leiden on the unassigned residual to spawn new stories. This is essentially the architecture of Miranda et al. (EMNLP 2018) and USTORY/SCStory (SIGIR 2023), updated with 2025-era embeddings.
- **Best embedding choice in 2025–2026:** Voyage-3-large or OpenAI text-embedding-3-large as the API default; BGE-M3 or Qwen3-Embedding-8B for self-hosted. Embed `title + first 2–3 paragraphs (lede)` or an LLM-generated 2-sentence summary — *not* the full body — for the assignment vector; store a separate full-document embedding for retrieval. Truncate articles longer than the context window; use late chunking for research PDFs.
- **Store a story as a row containing: centroid (running mean of member embeddings), medoid (the embedding closest to centroid), last_seen_at, key entities set, canonical title/summary, and member list.** Close stories when `now() − last_seen_at > 72h`. Use a hybrid pipeline: cheap embedding shortlist → LLM verifier only for the gray-zone (similarity in 0.55–0.75 band) → assign-or-create. This keeps LLM cost to a few percent of items while catching the cases where embeddings alone are unreliable.

---

## Key Findings

1. **There is a well-established academic literature ("Topic Detection and Tracking" / TDT) that solves exactly this problem.** Allan et al. (1998) and follow-ons established that **single-pass nearest-neighbor clustering with a similarity threshold and a time window is the canonical online news-story algorithm**; offline/retrospective settings favor agglomerative clustering. The 2018 Miranda et al. EMNLP paper is the modern reference implementation and is still cited as the state-of-the-art baseline architecture for online news clustering. It reports best **B-cubed F1 = 94.1 on English news** (SVM-merge with timestamp features), using TF-IDF features, a per-language tuned similarity threshold τ, an SVM-merge or-create classifier, and three Gaussian-decayed timestamp features with the authors stating: *"we fixed µ = 0 and tuned σ on the development set, yielding σ = 72 hours (3 days)."*
2. **The 72-hour active-window in your spec aligns exactly with the empirically-tuned time decay constant in Miranda et al.** That is not a coincidence — news stories typically die out on a ~3-day half-life. USTORY/SCStory (SIGIR 2023) defaults to a 7-day expiry window and hits B-cubed F1 = 0.789 on the Miranda News2013/Newsfeed benchmark using `sentence-transformers/all-roberta-large-v1` thematic embeddings, with an *adaptive* per-window threshold `γ = 1 − (1 − 1/|C_W|)^T` (default T=2) rather than a fixed cosine cutoff.
3. **HDBSCAN is the right *batch* clusterer, not the right *incremental* clusterer.** HDBSCAN cannot natively absorb new points into existing clusters — that is a known and documented limitation. BERTopic maintainer Maarten Grootendorst, in GitHub Discussion #2119, states: *"Merging models is what I would typically advise currently as it seems to me as the most stable approach. Although you could use the partial_fit method, it has some issues with stability as the underlying algorithm for dimensionality reduction and clustering generally aren't as powerful as the default alternatives."* Use HDBSCAN only on the residual of unassigned new items each night to *bootstrap* new stories.
4. **Embedding choice matters less than you'd think above a quality floor.** MTEB scores cluster within ~4 points across all top models. Voyage-3-large currently leads (MTEB ~65.1; "outperforms OpenAI-v3-large and Cohere-v3-English by an average of 9.74% and 20.71%, respectively, across 100 datasets, spanning eight diverse domains, including law, finance, and code" — Voyage AI blog, Jan 2025; supports Matryoshka and int8/binary quantization). OpenAI text-embedding-3-large is the safest commodity default. BGE-M3 is the best open-source quality-per-dollar at production scale; Qwen3-Embedding-8B and Jina v3 (with late chunking) are competitive open alternatives.
5. **LLMs add real value in two places, not at the clustering step itself**: (a) as a *verifier* on borderline cluster-assignment decisions, and (b) as a *story-state generator* (title, summary, timeline). Pure "LLM-as-clusterer" approaches (ClusterLLM, "Text Clustering as Classification with LLMs") are interesting research but **10×–50× the cost of embedding clustering and do not scale to the volumes implied by "many clients × nightly batch."**
6. **Storage choice is dominated by operational fit, not raw performance.** For a nightly batch, multi-tenant system, **pgvector with HNSW (and ideally pgvectorscale) is the recommended default**. Per Timescale's published benchmark: *"On a benchmark dataset of 50 million Cohere embeddings with 768 dimensions each, PostgreSQL with pgvector and pgvectorscale achieves 28x lower p95 latency and 16x higher query throughput compared to Pinecone's storage optimized (s1) index for approximate nearest neighbor queries at 99% recall, all at 75% less cost when self-hosted on AWS EC2"* (~$835/mo vs Pinecone s1's $3,241/mo). Qdrant is the right escalation path if you outgrow that; Pinecone/Weaviate are unnecessary expense for a batch workload.
7. **Production news-monitoring vendors (Meltwater, NewsCatcher, GDELT Cloud) converge on the same architecture.** Meltwater's "Content Clusters" feature uses a proprietary embedding clusterer with LLM-generated cluster overviews; NewsCatcher exposes a `clustering_enabled=true` parameter with `clustering_threshold` default 0.7 over Qwen3-Embedding-0.6B 1024-dim vectors (post-2026-01-01), and uses Leiden graph community detection at query time; GDELT Cloud uses Leiden community detection plus a Gemini-class LLM as a validation/extraction step. **Convergent design across vendors is a strong signal the reference architecture below is correct.**

---

## Details

### 1. Clustering algorithm comparison for unstructured text

| Algorithm | Strengths | Weaknesses for *news story* clustering | Verdict |
|---|---|---|---|
| **k-means / MiniBatchKMeans** | Fast, simple, online variants exist | Requires `k` in advance — fatal for story discovery where the number of stories is unbounded and changes nightly | ❌ |
| **DBSCAN** | Density-based, handles arbitrary shapes, no `k` | Single global `eps`; news has wide variance in story density (one breaking story = 1000 articles; a niche research note = 2) | ❌ |
| **HDBSCAN** | Density-based with variable density, robust noise label `-1`, great defaults; GDELT uses it on the Global Similarity Graph after UMAP-reducing 512-dim USEv4 embeddings to 10 dims | Static (no `partial_fit`); requires UMAP preprocessing for >1M vectors; can be slow at scale; outlier label `-1` sometimes dominates | ✅ for the *residual* (unassigned-new-items) clustering step each night, not for full-corpus re-cluster |
| **Agglomerative / hierarchical** | Yang et al. found it best for *offline* retrospective news event detection (F1 ≈ 82%); naturally produces a dendrogram you can cut at multiple granularities | O(n²) memory; not online | ✅ as an alternative to HDBSCAN for the residual step, especially when you want hierarchical topics |
| **Single-pass / nearest-cluster threshold (classic TDT)** | Pure online, O(N·k) per batch, no `k` needed, handles unbounded labels; Allan et al. showed *"Single-Link and Single-Pass clustering, which assigns the cluster label of the nearest neighbor (1-NN), achieved the best online topic detection performance"* | Quality is sensitive to threshold τ; can drift; cannot reorganize past assignments | ✅✅ **Best fit for your nightly incremental assignment step.** This is what Miranda et al. and the Google News patent (US 8,832,105) describe. |
| **Louvain / Leiden community detection on a kNN similarity graph** | No `k`, produces well-connected communities, scales well; NewsCatcher and GDELT Cloud both use Leiden | Requires graph construction (kNN); harder to incrementalize cleanly | ✅ as an *alternative* batch clusterer to HDBSCAN; Leiden is provably better than Louvain (avoids disconnected communities, "From Louvain to Leiden: guaranteeing well-connected communities") |
| **BERTopic** | Excellent UX, modular (UMAP + HDBSCAN + cTF-IDF), great topic labels | Documented as "inherently static and computationally heavy"; the partial_fit path uses weaker MiniBatchKMeans + IncrementalPCA; outlier handling weak in online mode (River's DBSTREAM lacks a `-1` outlier class) | ✅ Use offline for exploratory analysis and threshold tuning; ⚠️ do not use as the production runtime clusterer |
| **Top2Vec** | Joint topic + document embedding | Older (pre-2022) approach; superseded by BERTopic and dense-embedding pipelines | ❌ Skip |
| **LDA / probabilistic topic models** | Interpretable, well-understood | Bag-of-words, very weak on news similarity; not event-level | ❌ |
| **LLM-as-clusterer** (ClusterLLM EMNLP 2023, "Text Clustering as Classification with LLMs" 2024) | Best semantic quality on hard cases, produces labels for free | 10×–50× the cost; latency-prohibitive for nightly batches of 10k+ items per client; APIs lack embedding access for vector DB use | ⚠️ Use only as a *judge* on borderline cases or for cluster *labeling*, not the primary clusterer |
| **CluStream / DenStream micro-clustering** | Designed for streams; constant-memory | Originally for low-dim numeric data; Miranda et al. Table 2 shows their method beats CluStream on B³-F1 | ❌ Skip |

### 2. Incremental / online / streaming clustering for persistent stories

The canonical streaming approach is **centroid-based nearest-cluster assignment** (Allan et al. 1998; Miranda et al. EMNLP 2018), modernized with dense embeddings (Saravanakumar et al. 2021 "Time-Aware Document Embeddings"; Yoon et al. SIGIR 2023 "USTORY/SCStory"). The control flow is:

```
for each new item d in tonight's batch (in arrival order):
    if d.embedding has nearest active-story centroid c with sim(d, c) > τ_assign:
        assign d to c
        update c.centroid (running mean), c.last_seen_at, c.entities
    else:
        d goes to "unassigned" residual
# After processing all new items:
cluster the residual with HDBSCAN/Leiden → spawn new stories
# Sweep all stories: close any with last_seen_at older than 72h
```

Why this works for your spec specifically:

- **Stories naturally persist** because the centroid (or medoid) is the state; you only need to keep the centroid vector and a small entity set in your DB.
- **The 72-hour active window maps directly to a SQL filter** (`WHERE last_seen_at > now() - interval '72 hours'`) on the candidate-story ANN search — no special data structure needed.
- **Idempotency** is straightforward: hash `(client_id, item_url)` as a primary key so re-running a batch is a no-op.
- **Drift is controlled** by the centroid being a running mean: a story that drifts semantically will have lower internal similarity over time and will naturally stop matching new items (which then form a new sibling story).
- **Miranda et al. (2018) achieved B-cubed F1 = 94.1 on English news** with this architecture using only TF-IDF features and three Gaussian-decayed timestamp features (μ=0, σ=72 hours). Replacing TF-IDF with a 2024-era dense embedding model is the obvious modern upgrade. Their merge decision is verbatim: *"If the largest similarity exceeds a threshold τ for cluster index j, then we set C(d) = j. ... If none of the similarity values exceed a threshold τ, we find the first i such that H(i, L(d)) = ⊥ ... and set C(d) = i, therefore creating a new cluster."* Their best result actually uses a LIBLINEAR binary SVM merge classifier on per-feature max similarities rather than a single τ.
- **USTORY (SIGIR 2023) used `all-roberta-large-v1`** and an *adaptive* threshold `γ = 1 − (1 − 1/|C_W|)^T` (default temperature T=2) within a 7-day window, hitting B³-F1 = 0.789 on the Miranda Newsfeed benchmark — a useful, less-magic-number alternative to a fixed cosine cutoff. The thematic similarity is `sim_theme(a, C) = max(0, cos(E_{a|C}, E_{C|a})) · JSD(P_{a,K_C} ∥ P_{C,K_C})`.

**Online HDBSCAN variants exist** (`prediction_data=True` in the original `hdbscan` library gives you `approximate_predict()` for new points against an already-fit tree), but they do not let the *tree* grow with new points; for true incrementality you'd refit. Don't use this for primary assignment; it is fine for sanity-checking outlier labels.

**Micro-clustering (CluStream, DenStream)** is overkill for a nightly batch — designed for second-by-second streams. Skip.

### 3. Embedding strategies for news and research items (2025–2026 landscape)

**Recommended primary embedding models (April 2026):**

| Model | Provider | Dims | Strengths | When to pick it |
|---|---|---|---|---|
| **voyage-3-large** | Voyage AI (MongoDB) | up to 2048 (Matryoshka) | Best retrieval scores; +9.74% over OpenAI text-embedding-3-large averaged across 100 datasets in eight domains; native int8/binary quantization | Best quality if budget allows; pair with `voyage-context-3` for chunk-aware embedding |
| **text-embedding-3-large** | OpenAI | up to 3072 (Matryoshka) | Universally supported; mature; safe default; stable since January 2024 | Default when you want one vendor, simple API |
| **embed-v4** | Cohere | 256/512/1024/1536 (Matryoshka) | Native binary/int8 quantization; `clustering` input-type flag tunes embeddings for grouping | Pick if you want a `clustering` task hint and binary quantization out-of-the-box |
| **BGE-M3 / Qwen3-Embedding-8B** | BAAI / Alibaba | 1024 / variable | Best open-source quality; multilingual; self-hostable | Pick when API cost or data residency dominates |
| **Jina v3** with late chunking | Jina AI | 1024 | 8K context; late chunking preserves cross-chunk context | Pick for long research PDFs |
| **Gemini Embedding 2** | Google | 3072 (multimodal) | Single model embeds text/images/video/audio/PDF into a shared space; MTEB retrieval 67.71 | Pick if you have mixed-media research items |

**What text to actually feed the embedder.** This is the most-overlooked design choice and matters more than the model. For news *story* clustering, do **not** embed full articles — boilerplate, ads, and tangents dilute the signal. The standard choices, ranked:

1. **Best: an LLM-generated 1–2-sentence canonical event summary** ("WHO did WHAT, WHERE, WHEN"). Costs ~$0.0001/article with a cheap model (GPT-4o-mini, Claude Haiku) and dramatically tightens cluster cohesion. This is what the Stanford "Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings" (2025) and several production systems do.
2. **Good: `title + dek/lede` (first 1–2 paragraphs).** Cheap, no LLM hop. Most production vendors (NewsCatcher embeds `title + content`; GDELT) use a variant of this.
3. **Adequate: title only.** Used by Google News-era systems and the classic TDT literature. Loses too much for nuanced research items.
4. **Worst for clustering: full article.** Use only as a *retrieval* index, not for cluster assignment.

For research PDFs >8K tokens: chunk into ~512-token windows, use **late chunking** (Jina v2/v3 supports it natively — embed the *whole document* first via long-context model, then pool token spans into chunk vectors) so each chunk retains global context. Jina's published evaluation: *"In all cases, late chunking improved the scores compared to the naive approach. ... the longer the document, the more effective the late chunking strategy becomes."* For cluster assignment, take a `[summary]` vector; for retrieval, index all chunks.

### 4. Story persistence and state management

**Recommended story-state schema (Postgres + pgvector):**

```sql
CREATE TABLE story (
  story_id         uuid PRIMARY KEY,
  client_id        uuid NOT NULL,
  centroid         vector(1024) NOT NULL,    -- running mean of member embeddings
  medoid_item_id   uuid         NOT NULL,    -- canonical/lead item
  n_items          int          NOT NULL DEFAULT 1,
  first_seen_at    timestamptz  NOT NULL,
  last_seen_at     timestamptz  NOT NULL,
  closed_at        timestamptz,               -- null = active
  title            text,                       -- LLM-generated, updated incrementally
  summary          text,                       -- LLM-generated, updated incrementally
  entities         jsonb,                      -- {people:[…], orgs:[…], locations:[…], tickers:[…]}
  embedding_model  text         NOT NULL,     -- pin so you can re-embed on model swap
  schema_version   int          NOT NULL DEFAULT 1
);
CREATE INDEX ON story USING hnsw (centroid vector_cosine_ops) WHERE closed_at IS NULL;
CREATE INDEX ON story (client_id, last_seen_at);

CREATE TABLE story_item (
  item_id      uuid PRIMARY KEY,
  story_id     uuid REFERENCES story,
  client_id    uuid NOT NULL,
  url          text,
  url_hash     bytea,
  source       text,
  title        text,
  published_at timestamptz,
  embedding    vector(1024),
  is_duplicate bool DEFAULT false,
  duplicate_of uuid,
  added_at     timestamptz DEFAULT now()
);
CREATE UNIQUE INDEX ON story_item (client_id, url_hash);  -- idempotency
```

**Centroid vs medoid vs all-member retrieval.** Three options:

- **Centroid only** (running mean of member embeddings): cheapest; what Miranda et al. use; works well for stories with <50 members but drifts on long-running threads.
- **Medoid** (the actual member embedding closest to the centroid): more stable identity; recommended *in addition to* the centroid as the "canonical exemplar" for LLM prompting.
- **k-NN over all members** ("nearest *member*, not nearest centroid"): higher recall on stories that have evolved; this is the Google News patent approach: *"for each transient article, finding one or more nearest neighbor articles … determining whether a ratio of nearest neighbors that are fixed articles to nearest neighbors that are transient articles is greater than a predetermined threshold."* More expensive but more robust to drift.

**Recommendation:** store the centroid as the indexed vector for fast ANN, but on a candidate match in the gray zone (similarity 0.55–0.75) verify against the medoid AND the 3 most recent member embeddings; only assign if 2 of 3 also exceed threshold. This catches drift without re-indexing every member.

**Story merges and splits.**
- **Merges:** Once a week (or when a story exceeds N=20 items), run pairwise cosine on active-story centroids; if two stories have centroid similarity > 0.85 AND share ≥2 named entities AND a shared LLM-judge verdict, merge them. Keep a `merged_into` pointer for auditability.
- **Splits:** Optionally, when a story's internal cohesion (avg pairwise sim of members) drops below e.g. 0.60, run HDBSCAN on the member set; if it splits into ≥2 dense subclusters with mean intra-similarity > 0.70, materialize the split. In practice, **most teams skip splits**; the cost/risk usually exceeds the benefit. The 72-hour closure naturally limits how badly a story can drift.

**The 72-hour freshness window:**

```sql
-- Active stories eligible to absorb new items in tonight's batch
SELECT * FROM story
 WHERE client_id = :c
   AND closed_at IS NULL
   AND last_seen_at > now() - interval '72 hours';

-- At end of nightly job, close stale stories
UPDATE story
   SET closed_at = now()
 WHERE client_id = :c
   AND closed_at IS NULL
   AND last_seen_at <= now() - interval '72 hours';
```

This is cleaner than the time-decayed Gaussian in Miranda et al. and matches your business definition exactly.

### 5. Hybrid LLM + embedding pipelines

The dominant production pattern is **embed → ANN-search candidates → cheap-LLM verify (only on borderline cases) → assign-or-create**. Concretely:

```
sim = cosine(item.embedding, candidate_story.centroid)
if sim ≥ τ_high (≈0.75):       # high-confidence: pure embedding decision
    assign
elif sim ≥ τ_low (≈0.55):       # gray zone: call LLM judge
    verdict = LLM("Are these two items about the same news story? <item_title>, <item_lede> vs <story_title>, <story_summary>. Answer YES/NO and one reason.")
    if verdict == "YES":
        assign
    else:
        send to residual
else:                            # low confidence: skip the LLM, go to residual
    residual.add(item)
```

**Why this is the right shape:**
- The embedding step is essentially free (OpenAI text-embedding-3-large $0.13/M tokens; text-embedding-3-small $0.02/M tokens).
- The LLM judge is invoked on perhaps 10–25% of items (those in the gray zone) at ~$0.0001–0.001 per call — small compared to the embedding bill once you scale.
- **The verifier should use a cheap, fast model** (GPT-4o-mini, Claude Haiku, Gemini Flash). Reserve the frontier model for title/summary generation only.
- **Always use the batched embedding API.** Per OpenAI's official Batch API documentation: *"Better cost efficiency: 50% cost discount compared to synchronous APIs"* in exchange for a 24-hour completion window — perfect for a nightly batch.

This matches the "reasoning-based cluster refinement" framework recently formalized (arXiv 2604.07562): *"LLMs are used not as embedding generators, but as semantic judges that accept, reject, or revise structural hypotheses produced by unsupervised methods."*

When is pure embedding clustering sufficient (skip LLM verification entirely)?
- High homogeneity domain (a single market vertical, single language).
- Near-duplicate detection (sim > 0.90; embeddings are essentially unambiguous).
- When you can afford a slightly looser threshold and accept some over-clustering.

When does LLM verification meaningfully improve quality?
- Stories with named-entity ambiguity ("Apple" the company vs. the fruit; two CEOs with the same name).
- Multilingual or cross-source paraphrasing where embedding similarity drops but semantic identity holds.
- Research documents where one is a follow-on / methodology paper of another — embeddings often miss this; an LLM with the title+abstract catches it.

**Entity overlap as a cheap pre-LLM gate.** Run a NER pass on each new item (cheap LLM call or spaCy/Stanza) and require that to be assigned to story S, the item must share at least 1 named entity (person/org/location/ticker) with S's entity set. This single rule eliminates most embedding-driven false positives at zero LLM cost.

### 6. Story metadata generation with LLMs

**Initial creation (first item of a new story).** Single LLM call with structured output:

```
SYSTEM: You produce JSON describing news stories for a media-monitoring product.
USER: Given this item, return: {
  "title":   "≤80-char neutral headline",
  "summary": "≤400-char canonical 2-sentence summary, who/what/where/when",
  "entities": {"people": [...], "orgs": [...], "locations": [...], "tickers": [...]},
  "topic":   "one of {politics, business, technology, science, …}"
}
ITEM: <title>\n<lede>
```

Force structured output via JSON schema (OpenAI Structured Outputs, Anthropic tool-use, or Outlines/Instructor on open models). Validates schema, prevents drift.

**Incremental update (new item joins existing story).** Two viable patterns:

1. **Chain-of-Key / structured update (Google DeepMind 2024).** Pass the current story JSON plus the new item; the LLM emits the *updated* JSON. The DeepMind paper reports: *"structured knowledge representations (GUjson) … significantly improve summarization performance by 40% and 14% across two public datasets. … we propose the Chain-of-Key strategy (CoKjson) that dynamically updates or augments these representations with new information, rather than recreating the structured memory for each new source. This method further enhances performance by 7% and 4% on the datasets."* Format:
   ```
   CURRENT_STORY: <JSON>
   NEW_ITEM: <title>\n<lede>
   Update the story JSON. Add to entities. If the new item materially changes the story, update title/summary. Otherwise return the same JSON.
   ```
2. **Refine / progressive summarization.** Maintain `summary_v(n+1) = LLM(summary_v(n), new_item.lede)`. Cheaper but more drift-prone.

**Recommendation:** Pattern 1 (Chain-of-Key) for structured fields (title, summary, entities); skip incremental update if no member has been added in >24h (idempotent reads); regenerate from scratch every N=10 items as a "ground truth refresh" to prevent compounding drift.

**Identifying the canonical / lead item.** Several reasonable definitions:
- **Earliest** member (the breaker). Good for "who scooped this?"
- **Most authoritative source** (use a `source_rank` lookup: Reuters/AP/Bloomberg/FT/NYT rank highest).
- **Centroid-nearest** (the medoid). Most representative.
- **Longest article body** (most comprehensive).

Pick one and pin it as `medoid_item_id`. The simplest defensible rule is **highest-rank-source × recency**, broken ties by closeness to centroid.

**Near-duplicate detection inside a story.** Wire syndication means a single AP story will appear at 50 outlets. Detect with: (a) MinHash/LSH on shingled article body — the standard pretraining-data dedup threshold is Jaccard 0.8 (per Preferred Networks tech blog: *"We set the threshold at T=0.8. This means that whenever the Jaccard similarity between two documents exceeds 0.8, they are regarded as duplicates"*); or (b) cosine similarity on body embedding > 0.95 (per SemHash documentation: *"Default threshold is often ~0.9 but depends on data domain"*). Mark `is_duplicate=true, duplicate_of=<canonical_item_id>` rather than deleting — preserves count metrics but keeps the LLM context clean.

### 7. Production architecture

**Recommended reference architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                       NIGHTLY BATCH (per client)                     │
└─────────────────────────────────────────────────────────────────────┘

[1] INGEST
    Source connectors (RSS, news APIs, research feeds, search APIs)
    → raw_item table, partitioned by client_id, deduped on url_hash

[2] NORMALIZE & DEDUPE
    - Canonicalize URL (strip utm_*, fbclid, …)
    - Article body extraction (trafilatura / readability)
    - MinHash LSH on body shingles (Jaccard 0.8) → tag wire duplicates
    - NER (spaCy/stanza or cheap LLM) → entities column

[3] EMBED
    - assignment_vector = embed(title + lede)                  [for cluster assignment]
    - retrieval_vector  = embed(full body, possibly late-chunked) [for search]
    - Use the Batch API for a 50% discount over synchronous calls

[4] ASSIGN (single-pass nearest-cluster, per client)
    For each new item (sorted by published_at):
       candidates = pgvector ANN search on story.centroid
                    WHERE client_id = c
                      AND closed_at IS NULL
                      AND last_seen_at > now() - interval '72 hours'
                    ORDER BY centroid <=> item.assignment_vector
                    LIMIT 10
       best = candidates[0]
       sim  = 1 - (best.centroid <=> item.assignment_vector)
       if sim >= τ_high (≈0.75) AND entities ∩ best.entities ≠ ∅:
           assign(item, best)
       elif sim >= τ_low (≈0.55):
           if LLM_judge(item, best) == SAME_STORY:
               assign(item, best)
           else:
               residual.add(item)
       else:
           residual.add(item)

[5] SPAWN NEW STORIES (residual clustering)
    Run HDBSCAN (min_cluster_size=2, min_samples=2, metric='cosine')
      OR Leiden on a kNN(k=10) graph filtered to similarity > 0.55
    on residual.assignment_vector  → new stories
    Each new story: LLM call → title, summary, entities, topic

[6] UPDATE EXISTING STORIES
    For each story that received new items:
       centroid   = (centroid*n + Σ new_vectors) / (n + |new|)   # running mean
       last_seen  = max(last_seen, max(new.published_at))
       entities   = entities ∪ new.entities  (cap at top-50 by frequency)
       title/summary: Chain-of-Key incremental LLM update
       medoid: recompute as member closest to new centroid

[7] CLOSE STALE STORIES
    UPDATE story SET closed_at = now()
     WHERE client_id = c AND closed_at IS NULL
       AND last_seen_at <= now() - interval '72 hours'

[8] MERGE PASS (weekly, not nightly)
    For each pair of active stories with cosine > 0.85 AND ≥2 shared entities:
       LLM verifier: SAME or DIFFERENT?
       If SAME: merge (keep older story_id, repoint items, recompute centroid)

[9] EMIT
    - story_snapshot table for downstream dashboards
    - webhook / Kafka topic for real-time consumers
```

**Vector database recommendation.**

| Choice | Verdict for this use case |
|---|---|
| **pgvector (with HNSW) + pgvectorscale** | ✅ **Primary recommendation.** Co-locates story state and embeddings; transactional update of centroid+last_seen; per-client row filter is trivial; one operational system. Per Timescale's benchmark on 50M Cohere 768-dim embeddings, pgvectorscale's StreamingDiskANN achieves *"28x lower p95 latency and 16x higher query throughput compared to Pinecone's storage optimized (s1) index for approximate nearest neighbor queries at 99% recall, all at 75% less cost when self-hosted on AWS EC2"* (~$835/mo vs $3,241/mo). |
| **Qdrant** | ✅ Best if you outgrow pgvector or want strong filtered search. Rust-based, fast tail latency, great per-tenant collections, server-side BM25 since v1.15.2. |
| **Pinecone** | ⚠️ Managed convenience; expensive for nightly batch where you only query once per night per item; no HNSW tuning. |
| **Weaviate** | ⚠️ Strong hybrid search; overkill for cluster assignment. |
| **Milvus** | ⚠️ Use only if you're at 100M+ vectors per tenant; mature sharding. |
| **LanceDB** | ✅ Good embedded option for development or small clients; less mature for multi-tenant scale. |

**Idempotency.** Use `(client_id, url_hash)` as the natural primary key for items; use `(client_id, batch_id)` to record batch runs. Re-running tonight's job should be a no-op on `INSERT ... ON CONFLICT DO NOTHING`. Make centroid updates additive only on insert, not on re-processing.

**Evaluation metrics.**
- **B-cubed precision / recall / F1** — the standard TDT metric introduced by Bagga & Baldwin (1998); element-wise, handles unequal cluster sizes. Amigó et al. (2009) demonstrate *"it is the only metric that fulfills a set of constraints that they deem useful."* **Use this as your headline metric.**
- **V-measure / NMI / ARI** — Information-theoretic alternatives. Report alongside B-cubed; do not rely on alone (NMI can be inflated by over-clustering).
- **Business-relevant measures:** (a) median story size at day-3, (b) % items assigned vs. % residual, (c) % stories that ever absorb a 2nd item ("singletons"), (d) human spot-check precision on N=50 sampled stories per week, (e) per-client SLO for runtime.

**Building a labeled eval set.** This is the highest-leverage investment. Sample 1,000 article pairs spanning easy/medium/hard buckets; have 2 humans label "same story or not." Use this to tune τ_high, τ_low, and the merge threshold. Re-tune whenever you change the embedding model — thresholds are model-specific. Per OpenAI community forum (relevant when text-embedding-3-large was released): the original poster noted *"I have used 0.79 as the cosine similarity threshold for text-embedding-ada-002. … However, upon utilizing text-embedding-3-large, the same threshold no longer seems effective. Initial tests indicate that a lower threshold number should be chosen."* — i.e., cosine distributions shift across embedding generations and require recalibration.

**Debugging clustering quality in practice.** The high-yield workflow used by production teams:
1. UMAP-project all this-week's items to 2D, color by story_id, eyeball for obviously-wrong clusters.
2. For each suspicious cluster, run LLM judge across pairs → flag false-positive merges.
3. For low-recall stories (split apart), check entity overlap and time gaps.
4. Plot the cosine-similarity distribution of "same-story" vs. "different-story" pairs from your labeled set → the threshold should sit at the crossing point (or biased toward precision if downstream readers are humans).

### 8. Real-world references and how vendors do it

- **Google News** ("Cluster-based identification of news stories", US Patent 9,116,995 and "System for incrementally clustering news stories", US Patent 8,832,105): the latter explicitly describes the nearest-neighbor incremental algorithm with an age cutoff dividing "transient" (recent) from "fixed" (old) articles. The patent: *"the news articles that are less than the predetermined age limit are defined as transient articles and the remaining news articles are defined as fixed articles, wherein the incremental clustering is withheld from being performed on the fixed articles so that the fixed articles retain their initial clusters."* This is *exactly* your 72-hour active-vs-archived design.
- **Meltwater "Content Clusters"**: from their help center, *"Using a proprietary AI, Meltwater clusters relate search results based on topic similarity and generate overviews for the cluster content. … Each cluster can consist of any combination of News and Blog content. If social content on X or Reddit shares a link to one of the documents contained within the cluster, we retrieve it as additional context to the story."* Confirms the embedding + LLM-summary pattern is standard.
- **NewsCatcher**: exposes `clustering_enabled=true` with `clustering_threshold` default 0.7. Per docs, from 2026-01-01 onward the embedding model is `Qwen3-Embedding-0.6B` (fields: title + content, fixed; pre-2026 used `multilingual-e5-large`); cluster formation at query time uses the **Leiden** graph community detection algorithm. **The default 0.7 is a useful starting point for your τ_high.**
- **GDELT Cloud**: uses Leiden community detection (graph-based) over a candidate set with a Gemini-class LLM as a final validation/extraction step: *"Intelligent Clustering: Graph-based clustering (Leiden algorithm) groups near-duplicate stories, reducing the dataset to distinct events. Validation: A Gemini-class model evaluates web pages to determine whether they meet your criteria—keeping only relevant, credible items."* Confirms the "Leiden as alternative to HDBSCAN, LLM as gate" pattern.
- **Signal AI / Cision**: PR-monitoring; cluster around brand mentions and entity overlap rather than pure semantics.
- **TDT academic corpus**: TDT-2 through TDT-5 evaluations (1998–2004) established B-cubed and time-windowed single-pass clustering as the standard. Yang et al. found agglomerative best for offline retrospective event detection (F1 ≈ 82%); single-pass with time windowing best for online alerting.
- **Miranda et al. (EMNLP 2018, arXiv 1809.00540)** is still the dominant cited modern baseline for online clustering.
- **USTORY/SCStory (Yoon et al., SIGIR 2023, arXiv 2304.04099)**: dynamic thematic embedding with adaptive threshold; B³-F1 = 0.789 on Miranda Newsfeed with `all-roberta-large-v1`.
- **BERTopic**: best-in-class for *offline* topic modeling; use for exploration, not production runtime.

---

## Recommendations

**Stage 1 (week 1–2 — minimum viable pipeline):**
1. Stand up Postgres + pgvector + HNSW index. Schema as above.
2. Pick embedding model: **start with OpenAI text-embedding-3-large** (3072 → reduce to 1024 via Matryoshka) for fastest integration; A/B test against voyage-3-large after 4 weeks of data.
3. Embed `title + first 500 chars of body` as the assignment vector. Use the OpenAI Batch API for the 50% discount.
4. Implement the single-pass assignment loop with **τ_assign = 0.65** as a single threshold (no gray zone yet). NewsCatcher uses 0.7 default; Miranda et al. tune τ on a dev set per language. Start at 0.65 and tune up after Stage 2 calibration.
5. Spawn-new-story residual step: HDBSCAN with `min_cluster_size=2, metric='cosine', cluster_selection_method='eom'`. UMAP-reduce to 10 dims if you exceed ~500k residual items.
6. LLM title/summary on story creation only (no incremental update yet).
7. Close stories where `last_seen_at < now() - 72h`.

**Stage 2 (week 3–6 — quality):**
1. Build a labeled eval set of 1,000 article pairs. Compute B-cubed; **target ≥ 0.85 F1 before going live.**
2. Split into τ_high / τ_low gray zone; add the LLM judge for the gray zone using GPT-4o-mini, Claude Haiku, or Gemini Flash.
3. Add entity-overlap gate (must share ≥1 entity to assign).
4. Implement incremental Chain-of-Key summary updates; full regeneration every N=10 items.
5. Add weekly merge pass.
6. Add per-client dashboards: items/night, stories created, mean story size, residual rate, singleton rate.

**Stage 3 (month 2+ — scale):**
1. Move to batched embedding APIs (50% cost saving).
2. Add pgvectorscale if any client exceeds ~10M vectors (75% cost saving vs Pinecone).
3. A/B test voyage-3-large vs. OpenAI text-embedding-3-large on your labeled set.
4. Consider Matryoshka dimensionality reduction (3072 → 512) once you have enough data to verify quality is preserved; cuts vector storage ~6×.
5. Build a "model swap" job: when you change embedding model, you must re-embed all members of all active stories — bake this into a Lambda-architecture-style ability to re-run from raw_item.
6. **Optional advanced:** replace the single τ threshold with a small trained classifier (logistic regression or gradient-boosted trees) over features like `(centroid_cosine, medoid_cosine, entity_overlap_count, time_delta_hours, source_rank)` — this is what gave Miranda et al. their best B³-F1 = 94.1.

**Starting thresholds (tune on your labeled set; these are *priors*, not optimums):**
- `τ_high` (auto-assign): **0.75** with text-embedding-3-large or voyage-3-large
- `τ_low` (skip, no LLM): **0.55**
- Gray-zone LLM judge: between τ_low and τ_high
- Story merge threshold: centroid-pair cosine **0.85** plus ≥2 shared entities
- Wire-duplicate threshold (mark `is_duplicate`): body-embedding cosine **0.95** OR MinHash Jaccard **0.80**
- Story expiry: **72 hours** since `last_seen_at` (per your spec)
- HDBSCAN residual: `min_cluster_size=2, min_samples=2`, metric `cosine`

**When to revisit these choices:**
- B-cubed F1 below 0.80 on your eval set → tune `τ_assign` and consider adding the entity gate.
- Stories ballooning past 50 items in 72h → either you have a huge breaking event (fine) or `τ_assign` is too low.
- Singletons (1-item stories) dominate (>60% of stories) → `τ_assign` is too high or your embedding is too noisy (try the LLM-summary embedding).
- Cross-client cost dominated by LLM judge calls → tighten the gray zone (raise `τ_low` to 0.65).
- Embedding model is updated by vendor → re-tune all thresholds; do not trust the old ones.

---

## Caveats

1. **No vendor or peer-reviewed source publishes a definitive "same-story" cosine threshold for text-embedding-3-large or voyage-3.** The numbers above (`τ_high=0.75`, `τ_low=0.55`) are reasoned starting points consistent with NewsCatcher's documented default of 0.7, Miranda et al.'s dev-set-tuned τ, and community rules of thumb. **You must tune them on your own labeled set** — embedding-model-specific calibration is non-negotiable. The OpenAI dev community thread on threshold migration from ada-002 to text-embedding-3-large explicitly observed *"similarities are much lower across the board"* — a single fixed number does not survive a model change.
2. **The published B-cubed F1 numbers (Miranda 0.94 English, USTORY 0.789 Newsfeed) are on *clean*, single-language benchmarks**. Expect 0.10–0.15 lower on your real, multi-source feed unless you invest in source-level normalization and entity disambiguation. Miranda et al.'s 94.1 used an SVM-merge classifier with TF-IDF + timestamp features; replicating it requires the classifier, not just a threshold.
3. **HDBSCAN does not natively support new points growing the tree.** Approximate prediction (`hdbscan.approximate_predict`) is available but does not refit; for true incremental absorption you must use the centroid-threshold path described, not HDBSCAN's prediction API.
4. **BERTopic's partial_fit / online mode is documented by its maintainer as less stable than the static path.** Do not assume it will give you the same quality as offline BERTopic.
5. **Embedding model deprecation/version drift is a real operational risk.** OpenAI's text-embedding-3-large has been stable since January 2024 but is not guaranteed to remain unchanged. Pin `embedding_model` in your story row so you can detect and re-embed when a vendor changes weights.
6. **LLM cost can balloon if you put the LLM in the inner loop of cluster assignment.** Always run the cheap embedding shortlist first; only invoke the LLM on the gray zone. Production teams report LLM judge call rates of 10–25% of items at most.
7. **Cross-client isolation matters.** Do not pool stories across clients (different clients have different relevance criteria; pooling leaks signal). Index per-client; benchmark per-client. The single-pass-assignment loop in step [4] above must be per-client; do *not* share `story` rows across clients.
8. **Sentiment/NLP enrichment is out of scope of clustering quality** but the same architecture supports it. Don't conflate "cluster sentiment" with "story sentiment" — story sentiment should be computed once per story update from the canonical summary, not aggregated from item-level sentiments which can be noisy.
9. **Some sources cited above (vendor MTEB leaderboards, threshold defaults) are dated 2025–2026 snapshots.** Treat exact MTEB rankings as moving targets; the *ordering* of voyage-3-large vs text-embedding-3-large vs BGE-M3 is more durable than any single benchmark score.
10. **The Miranda et al. "best F1 = 94.1" number used an SVM-merge binary classifier, not pure threshold-based assignment.** If you want the highest possible quality, train a small classifier over features like `(centroid_cosine, medoid_cosine, entity_overlap_count, time_delta_hours, source_rank)` rather than relying on a single τ threshold. This is a Stage-3 optimization.
11. **Vendors do not publish their algorithms in detail.** Meltwater says "proprietary AI"; Google News' patents describe the 2010-era algorithm, not whatever LLM-enhanced approach they use today. Treat vendor-architecture claims in this report as informed inference, not internal documentation.