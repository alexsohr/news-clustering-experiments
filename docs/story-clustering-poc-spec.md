# Story Clustering — POC Experimentation Notebook Spec

This document specifies a Jupyter notebook that validates the story-clustering algorithm on real financial news data before any AWS infrastructure work begins. It is a companion to `story-clustering-aws-spec.md`, which consumes this POC's outputs (calibrated thresholds, labeled eval set, validated algorithm) and turns them into production infrastructure.

---

## TL;DR

- **One self-contained Jupyter notebook** that runs end-to-end on a laptop in roughly two hours and ~$30–80 of API spend.
- **Dataset:** [`Brianferrell787/financial-news-multisource`](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource) on HuggingFace. Same domain (financial news, multi-source) as your production use case, with real publication dates and source labels.
- **Algorithm under test:** the same single-pass-nearest-cluster + LLM-judge gray-zone + HDBSCAN residual + contextual-chunked retrieval architecture defined in the AWS spec. Lighter-weight implementation: pandas + numpy + sklearn + direct OpenAI/Anthropic SDKs. **No Aurora, no Glue, no Bedrock** — the *algorithm* matters here, not the infrastructure.
- **Headline deliverables:** (1) calibrated `τ_high` and `τ_low` backed by ~600 labeled item pairs (hybrid LLM-ensemble + targeted human spot-check, see Section 8), (2) the labeled eval set as a CSV for production regression testing, (3) B-cubed F1 on the labeled set with a go/no-go signal at ≥0.85, (4) evidence for or against each major architectural choice (entity-overlap gate, contextual chunking, HDBSCAN residual).
- **Every step ends in a chart or table** that makes a design decision visible — cosine distribution overlays for threshold picking, precision/recall curves, UMAP projections, B-cubed scores per configuration, cost breakdowns.
- **Out of scope for the POC:** production-scale throughput, Aurora-specific operations, Glue cold-start dynamics, multi-day cumulative behavior. Those belong to the AWS deployment spec.

---

## Why this POC exists

Three questions a research report cannot answer:

1. **Does the algorithm actually produce sensible clusters on real financial news?** The architecture is well-supported by literature (Miranda et al. 2018, USTORY 2023) and vendor convergence (NewsCatcher, GDELT, Meltwater), but published B-cubed F1 numbers come from clean English-news benchmarks. Financial news has its own quirks — earnings calendars, ticker disambiguation, wire-syndication patterns specific to financial outlets. We need to see the algorithm work on data that looks like yours before committing to Glue jobs.

2. **What threshold values actually achieve ≥0.85 B-cubed F1?** `τ_high = 0.75` and `τ_low = 0.55` are priors from the literature, not optimums. The OpenAI community has documented that cosine similarity distributions shift across embedding-model generations — what worked for `text-embedding-ada-002` does not transfer to `text-embedding-3-large`. The only way to pick correct thresholds is to label pairs and look at the distribution.

3. **Does the contextual-chunking-with-doc-summary-prepend approach actually help?** This is the major algorithmic divergence from research.md's Jina-style late chunking. The pattern is borrowed from Anthropic's Contextual Retrieval (Sep 2024), simplified to a doc-level prepend. It needs empirical validation on financial research-style items before being committed to production.

The POC is the cheapest way to answer all three.

---

## Deliverables

The notebook produces six artifacts, in order of importance:

1. **Labeled eval set CSV.** ~600 item pairs labeled via LLM ensemble (Claude Sonnet 4.6 + GPT-5.2 + Gemini 3.5 Flash) with human spot-check on the threshold-critical subset. Headline column is `final_label ∈ {SAME, DIFFERENT}`; per-model verdicts and human override are preserved for audit. See Section 8 for the full column schema. This becomes a regression test for the production system, run in CI on any candidate model or threshold change.
2. **Calibrated thresholds.** `τ_high` and `τ_low` values that maximize B-cubed F1 on the labeled set, with confidence intervals.
3. **B-cubed F1 scorecard.** Headline F1 with breakdown into precision and recall. Sanity baselines (pure embedding clustering, pure HDBSCAN, naive title-Jaccard).
4. **Cost projection.** Token consumption per pipeline stage, daily cost estimate at production scale (100k items/day), breakdown by model and step.
5. **Algorithmic findings document.** Any unexpected behavior on financial news — failure modes, missed clusters, false merges, drift patterns — and proposed fixes before going to production.
6. **Decision recommendation.** Ship to AWS as specified, or iterate further on the algorithm.

---

## Dataset

### Source

[`Brianferrell787/financial-news-multisource`](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource) on HuggingFace. Loaded via `datasets.load_dataset("Brianferrell787/financial-news-multisource")`.

The exact column schema will be inspected in Section 2 of the notebook. The POC adapts to whatever columns are present — minimum required signals are: a text body of some kind, a publication date, a source label, and ideally either ticker/entity tags or enough body text to extract them.

### Why this dataset is a reasonable proxy for production

- **Domain match.** Financial news, not generic news. Banker clients are corporate entities; this dataset's items are about corporate entities.
- **Multi-source.** Mirrors production's mix of Perplexity-aggregated outlets plus internal research.
- **Real time signal.** Items have real publication dates, so the 72-hour active-window logic gets exercised on realistic event timelines.
- **Likely contains wire-syndication patterns.** Multi-source financial news inevitably includes Reuters/AP syndication across outlets — exactly what MinHash near-dup detection must catch.

### What the dataset is *not* — and how the POC handles each gap

- **No "banker client" labels.** Production gets `banker_client_id` from Perplexity (each query is per-client). The POC simulates this by extracting tickers/companies as the "client" tag for each item (Section 3).
- **No ground-truth story IDs.** Production wants to ask "did the algorithm cluster these correctly?" We get this by *manually labeling* a sampled subset of item pairs (Section 8). This is the most labor-intensive step of the POC and the highest-leverage one.
- **Possibly no long research documents.** The dataset is news, not research notes. To test the contextual-chunking path the POC will optionally augment with one or two long synthetic items (concatenated news from the same week per ticker) or skip that section entirely with a note. Decision deferred to Section 10.

---

## Stack for the POC

| Concern | Production (AWS spec) | POC notebook |
|---|---|---|
| Storage | Aurora PostgreSQL + pgvector HNSW | In-memory pandas DataFrames |
| ANN search | pgvector HNSW per-client filtered | `sklearn.neighbors.NearestNeighbors` or `numpy.dot` for ≤10k stories |
| Embeddings | OpenAI `text-embedding-3-large` (1024 dims via Matryoshka), Batch API | **Same model and dims**, synchronous API (POC volume too small to justify Batch) |
| Gray-zone LLM judge | Claude Haiku via Bedrock | **Same model**, Anthropic API direct |
| Story metadata generation | Claude Sonnet via Bedrock | **Same model**, Anthropic API direct |
| Entity extraction | Claude Haiku via Bedrock | **Same model**, Anthropic API direct |
| Doc-level chunk context | Claude Haiku via Bedrock | **Same model**, Anthropic API direct |
| Source ingest | Perplexity + JPM research via plugin connectors | HuggingFace dataset load — no source connectors |
| Orchestration | EventBridge + Glue Workflow | The notebook itself, top to bottom |
| Dedup primary key | `item.url_hash` global UNIQUE | DataFrame `drop_duplicates` on `url_hash` |
| Near-dup detection | `datasketch` MinHash + LSH | **Same library**, same thresholds |
| Residual clustering | `hdbscan` on driver | **Same library** |

The model choices are deliberately identical to production. Anthropic on Bedrock uses the same model weights as the Anthropic API; OpenAI is identical regardless of route. Swapping to Bedrock in production gives us VPC residency without changing algorithmic behavior. If the POC works, the AWS port is mechanical.

---

## Notebook structure

Seventeen sections. Each section ends with at least one visualization or table that informs a downstream decision. Don't skip the visualizations — they are how the POC produces evidence rather than vibes.

### Section 1 — Setup & configuration

**Purpose.** Pin versions, load API keys, set thresholds-as-variables in one place so the whole notebook re-runs deterministically.

**Operations.**
- Install: `datasets`, `pandas`, `numpy`, `scikit-learn`, `hdbscan`, `umap-learn`, `matplotlib`, `seaborn`, `plotly`, `datasketch`, `tiktoken`, `openai`, `anthropic`, `google-generativeai`, `trafilatura`, `tqdm`.
- Load API keys from environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Fail fast with a clear message if missing.
- Define `CONFIG` dict in one cell with all thresholds, model IDs, sample sizes, and random seeds. Every downstream cell reads from `CONFIG`.

**Outputs.** A printed config table and a `pip list` confirmation. No charts.

---

### Section 2 — Dataset loading & exploration

**Purpose.** Understand what you have before deciding what to do with it.

**Operations.**
- `dataset = load_dataset("Brianferrell787/financial-news-multisource", split="train")` (adjust split per the dataset card).
- Convert to a pandas DataFrame.
- `df.info()`, `df.head(10)`, `df.describe(include="all")`.
- Map dataset columns to the canonical schema the notebook uses internally: `item_id`, `title`, `body`, `source`, `published_at`, `url`. The mapping is whatever the dataset provides; if a field is missing, document it here.

**Tables.**
- Column inventory: dataset column name, mapped internal name, dtype, null %, example value.
- Source breakdown: count, % of total, earliest and latest published_at per source.
- Token length histogram bucket counts for `body` (computed via `tiktoken` with the embedding model's tokenizer).

**Charts.**
- Bar chart: item count per source.
- Time series: items per day stacked by source.
- Histogram: body token length, log-scaled y, with vertical lines at 512 / 2000 / 8000 (the embedding-model context boundaries).
- Histogram: title length in characters.

**Decision this section enables.** Are there enough items per day to exercise the 72-hour-window logic? Is body length distribution heavy enough at the long tail to warrant the contextual-chunking section? If the dataset is unexpectedly small or uniform, scope down later sections.

---

### Section 3 — Adapt the dataset to the production schema

**Purpose.** Make the POC data look enough like production data that the algorithm code is identical.

**Operations.**
- Generate `item_id` as a deterministic UUID5 from URL (so re-runs are stable).
- Build the **client universe**: identify the top N (default 20) most-mentioned tickers or companies across all items. If the dataset has explicit ticker columns, use them. Otherwise, run a single Claude Haiku pass over titles + first 200 chars of bodies to extract ticker/org entities, then take the top N.
- Build a `client_aliases` lookup: `{"Microsoft": ["Microsoft", "MSFT", "msft", "Microsoft Corp", "Microsoft Corporation"], ...}`. For top-20 tickers, this is fast to curate by hand from extracted entities — do not automate it for v1.
- Build an Aho-Corasick or simple regex-based matcher that maps a body to the set of clients it mentions.
- Filter the dataset to items that mention at least one client in the universe. Items with zero matches go to a separate "noise" partition and are excluded from downstream processing (matching the production Job 1 behavior).

**Tables.**
- Client universe table: rank, canonical name, ticker, alias list, item count.
- Items-per-client distribution (long tail expected).
- Multi-client item examples: titles where ≥2 clients are mentioned (these are the cross-entity stories the global-stories design is built for).

**Charts.**
- Horizontal bar chart: top-20 clients by item count.
- Heatmap: client × day, cell colored by item count for the last N=30 days. Reveals event spikes per client.
- Stacked histogram: # of clients mentioned per item (1, 2, 3+). Multi-client items will be a sizable minority — this is the empirical justification for the global-stories-with-affected-clients schema choice.

**Decision this section enables.** Verifies the global-stories choice empirically: if a meaningful fraction of items mention multiple clients, per-client story scoping (Doc 1 Option A) would have duplicated those items across pools.

---

### Section 4 — URL canonicalization & exact-duplicate drop

**Purpose.** First-pass dedup before any expensive operations.

**Operations.**
- Strip `utm_*`, `fbclid`, `gclid`, `mc_eid`, and fragment identifiers from URLs.
- Compute `url_hash = sha256(canonical_url).digest()`.
- `df.drop_duplicates(subset="url_hash")` — keep first occurrence by published_at.

**Tables.**
- Counts: input rows, post-canonicalize rows, exact duplicates dropped, % reduction.
- 10 randomly sampled before/after URL pairs that collapsed onto the same hash.

**Charts.**
- Bar chart: dropped duplicates per source. Wire syndication outlets dominate.

**Decision this section enables.** Sanity check that URL canonicalization is doing real work on this dataset. If <2% drop, the rule set may need extending; if >20% drop, the dataset has heavy duplication that will affect downstream embedding cost estimates.

---

### Section 5 — Near-duplicate detection (MinHash + LSH)

**Purpose.** Catch wire-syndicated copies where URLs differ but body content is essentially identical.

**Operations.**
- For each item, generate body shingles (5-grams of normalized tokens) and compute a 128-permutation MinHash via `datasketch.MinHash`.
- Insert into a `MinHashLSH` index with `threshold=0.85` (the AWS spec value).
- For each item, query the LSH for near-duplicates. Build duplicate clusters via union-find.
- Within each cluster, pick the canonical item: the one from the highest-ranked source (Bloomberg > Reuters > WSJ > FT > AP > others), tiebreak by earliest `published_at`.
- Mark non-canonical items: `is_duplicate=True`, `duplicate_of=<canonical_id>`.

**Tables.**
- Near-duplicate cluster size distribution (cluster_size, n_clusters).
- 5 randomly sampled dupe clusters: show titles + sources + published_at side-by-side so the reader can verify these really are syndications.

**Charts.**
- MinHash threshold sweep: re-run LSH at thresholds 0.70, 0.75, 0.80, 0.85, 0.90 and plot # duplicate items detected. Annotate the production choice (0.85). The curve helps justify the threshold.
- Bar chart: duplicates per source (which outlets are wire syndicators in this dataset).

**Decision this section enables.** Confirms 0.85 is sensible for financial news. If the curve shows 0.80 captures meaningfully more real syndications without adding obvious false merges (visually inspect the sampled clusters), this is evidence to consider loosening the threshold in production.

---

### Section 6 — Entity extraction

**Purpose.** Build the entity sets that drive (a) the entity-overlap gate in the gray zone, and (b) the `entities` field on each story for downstream display.

**Operations.**
- For each canonical (non-duplicate) item, one Claude Haiku call with a structured-output prompt:
  ```
  Extract entities from this financial news item. Return JSON:
  {"people": [...], "orgs": [...], "tickers": [...], "locations": [...]}
  TITLE: ...
  LEDE:  ...
  ```
- Use Anthropic's `tool_use` to enforce the JSON schema.
- Persist results to the DataFrame as a nested column.
- Resolve `tickers` to the client universe using the alias lookup from Section 3. Items whose ticker set intersects the client universe get those clients added to `item_clients`.

**Tables.**
- Entity counts per type across the whole dataset.
- Top-30 entities per type (top orgs, top tickers, top people).
- 10 sampled items with their extracted entity sets and the inferred `item_clients` derived from ticker resolution. Manually verify a few for correctness.

**Charts.**
- Bar chart: items by `len(entities.orgs)` bucket. Most items will have 1–3; a few will have many.
- Confusion-style table: items where the entity extraction's tickers disagree with the regex-matcher's client mentions from Section 3 (e.g., extractor found "MSFT" but regex didn't, or vice versa). Sample of 10. Lets you eyeball entity-extraction accuracy.

**Decision this section enables.** Calibrates how aggressive the entity-overlap gate should be. If 90% of items share at least one entity with at least one other item in the same week, the gate is permissive enough not to break clustering. If <50%, the gate may be too tight for this domain.

**Cost note.** This is the dominant Haiku spend in the POC. At ~5,000 items × ~$0.0001 = $0.50. Negligible.

---

### Section 7 — Assignment embeddings (title + lede)

**Purpose.** Produce the vector each item is clustered on.

**Operations.**
- Build `embed_input = title + "\n\n" + body[:lede_chars]` where `lede_chars = 600`.
- Call OpenAI `text-embedding-3-large` with `dimensions=1024` for each canonical item. Use synchronous endpoint — the POC's volume doesn't justify Batch API setup overhead.
- Store as `np.ndarray` of shape `(n_items, 1024)`, dtype `float32`, L2-normalized so cosine similarity equals dot product.

**Tables.**
- Cost summary: tokens consumed, dollars spent, per-item average.
- Norm distribution: confirm post-normalization all vectors have unit norm (sanity).

**Charts.**
- **[Stakeholder demo chart] Interactive 3D UMAP scatter of all assignment vectors, colored by top-1 client tag** (Plotly `scatter_3d`). Project with `umap.UMAP(n_components=3, metric="cosine", random_state=42).fit(assignment_vecs)`. Each point hover-shows `(title, source, client_tag, published_at)`. Stakeholders can rotate to see that MSFT items form a tight blob separate from AAPL items. This is the chart that proves "embeddings are doing real work" without needing to read F1 scores.
- **Interactive 3D UMAP scatter colored by source** (Plotly). Same UMAP coords, different coloring. Should *not* show clean separation — if Reuters items cluster separately from Bloomberg items regardless of topic, you have a source-bias problem in embeddings (most often: HTML or boilerplate in the lede).
- Token length histogram of `embed_input`. Confirm the lede truncation is doing what you expect; very long titles or weird HTML in the lede can blow this up.

**3D chart labeling convention (applies to all 3D scatter plots in this notebook).** The 3D space is **not** the actual embedding space — that lives in 1024 dimensions, which can't be rendered. The 3D coords are a UMAP projection that preserves neighborhood structure but distorts absolute distances. Label charts honestly so stakeholders aren't misled:
- **Title:** *"Item clusters in 1024-dim embedding space (projected to 3D via UMAP for visualization)"*
- **Axes:** *"UMAP component 1 / 2 / 3"* — never "x / y / z" and never anything semantic like "sentiment" or "industry"
- **Subtitle or caption:** *"Distances are topology-preserving, not absolute. Clustering algorithm operates in the full 1024-dim space."*
- **Persist the fitted UMAP reducer** (`reducer.pickle` to disk after `fit`). Later sections re-use the same reducer via `reducer.transform(new_vecs)` so points across sections sit in a consistent space — Section 11 story centroids land in the same 3D coordinate system as Section 7 items.

**Decision this section enables.** Sanity check that embeddings separate by topic, not by source. If UMAP shows obvious source-clustering, the embedding pipeline has a problem that must be fixed before continuing (most often: HTML or boilerplate in the lede).

---

### Section 8 — Build the labeled eval set (THE critical section)

**Purpose.** Produce the ground truth that calibrates every threshold in the system.

**Methodology — hybrid LLM-ensemble labeling with gray-zone human spot-check.** Pure human labeling is the conservative baseline, but at current frontier-model capability it is the wrong cost-quality trade-off. Modern frontier models agree with human judgment ~85% of the time on pairwise same-story decisions, with disagreements concentrated at the threshold boundary. The optimal approach: LLM-label everything, route only the threshold-zone pairs to a human reviewer. Roughly 3× faster and ~5× cheaper than pure human labeling while staying anchored to human ground truth where it matters most.

**Critical anti-bias guardrail.** The production gray-zone judge is **Claude Haiku**. The labeler models must NOT be from the same family — otherwise the labels are calibrating Haiku against its own architectural worldview (circular calibration). The recommended ensemble uses three different vendors.

**Step 1 — Sample the pairs.**
- Stratified random sampling across cosine bins: `[0.20, 0.40), [0.40, 0.50), [0.50, 0.55), [0.55, 0.60), [0.60, 0.65), [0.65, 0.70), [0.70, 0.75), [0.75, 0.80), [0.80, 0.85), [0.85, 1.00)`. ~60 pairs per bin. Total ~600 pairs.
- Both items in each pair must share at least one entity (otherwise the production gate excludes them and they don't inform calibration).
- Both items must be published within 72 hours of each other (matches production active window).

**Step 2 — LLM ensemble labeling.** Send each pair to three frontier models, independently, with `temperature=0` for reproducibility:
- **Anthropic Claude Sonnet 4.6** (via Anthropic API)
- **OpenAI GPT-5.2** (via OpenAI API)
- **Google Gemini 3.5 Flash** (via Google AI Studio / Vertex API)

Three different vendors so no single model's blind spot dominates the labels, and none of them is the production gray-zone judge (Haiku) so calibration is not circular.

Identical structured-output prompt across all three:

```
You are determining whether two financial news items describe the SAME news event.

ITEM A:
  title: <item_a.title>
  lede:  <item_a.lede>
  published_at: <item_a.published_at>

ITEM B:
  title: <item_b.title>
  lede:  <item_b.lede>
  published_at: <item_b.published_at>

Two items are the SAME story if they describe the same underlying news event involving
the same primary entities — even if framing, source, or details differ. They are
DIFFERENT if the primary event differs (e.g., "Apple Q3 earnings beat" vs "Apple Q3
earnings preview" are different events despite same topic). If the comparison is
genuinely ambiguous, return UNCLEAR.

Respond with one JSON object:
{"verdict": "SAME" | "DIFFERENT" | "UNCLEAR", "reason": "<one sentence>"}
```

Force schema via each provider's structured-output mechanism (Anthropic `tool_use`, OpenAI `response_format`, Gemini `responseSchema`). Persist all three verdicts and reasoning strings per pair.

**Step 3 — Compute ensemble label.**
- 3-of-3 SAME or 3-of-3 DIFFERENT → high-confidence ensemble label.
- 2-of-3 majority → working ensemble label, flagged as "split ensemble" for review.
- 1-1-1 disagreement OR ≥2 UNCLEAR → routed to human regardless of cosine bin.

**Step 4 — Human spot-check on the threshold-critical subset.** Build the human-review subset:
- All pairs with cosine in [0.55, 0.75] (the threshold-critical zone where calibration matters most — ~3 bins × 60 = ~180 pairs).
- All pairs from Step 3 routed to human (1-1-1 disagreement, ≥2 UNCLEAR — typically a handful).

Expect ~175–200 pairs total. One analyst labels these by hand using the same rubric (SAME / DIFFERENT / UNCLEAR). Estimated ~1.5–2 person-hours.

**Step 5 — Compute LLM-vs-human agreement on the spot-checked subset.**
- Compute: of the ~175 spot-checked pairs, what fraction's human label matches the LLM ensemble majority?
- **≥85% agreement:** trust LLM labels at scale. Use ensemble majority as `final_label` across all 600 pairs.
- **70–85% agreement:** mixed-trust mode. Use human labels for the spot-checked subset (inside [0.55, 0.75]); LLM ensemble majority for everything outside that band.
- **<70% agreement:** stop. Frontier models aren't matching human judgment well enough on this domain. Either escalate to full human labeling, or revisit labeler model choices.

**Step 6 — Disagreement analysis.** For spot-checked pairs where LLM ensemble disagreed with human: tabulate by pair shape (cosine bin, source pair, entity overlap count, time delta). Are failures systematic (e.g., always over-merging earnings-adjacent items, always under-merging cross-source paraphrases)? Document patterns in `poc_findings.md` — useful signal for Stage 2 prompt tuning of the production Haiku judge.

**Outputs to CSV.** `labeled_eval_set.csv` columns:
- `pair_id, item_a_id, item_b_id, item_a_title, item_a_lede, item_b_title, item_b_lede`
- `cosine_sim, sim_bucket, item_a_published_at, item_b_published_at, shared_entities`
- `verdict_sonnet, verdict_gpt52, verdict_gemini` — ensemble verdicts
- `reason_sonnet, reason_gpt52, reason_gemini` — ensemble reasoning, for later audit
- `ensemble_majority` (SAME / DIFFERENT / UNCLEAR / SPLIT)
- `ensemble_confidence` (3-of-3 / 2-of-3 / split)
- `human_label` (nullable; populated for spot-checked subset)
- `final_label` (SAME / DIFFERENT — the calibration ground truth, defined per Step 5)
- `disagreement_flag` (true where `human_label` was provided and differs from `ensemble_majority`)

**Tables.**
- Sample distribution: pairs per cosine bin, target vs. actual.
- Per-LLM label distribution: SAME / DIFFERENT / UNCLEAR counts per model.
- Inter-LLM pairwise agreement: 3×3 matrix of % agreement between Sonnet 4.6, GPT-5.2, and Gemini 3.5 Flash.
- LLM-vs-human agreement on the spot-checked subset, with 95% Wilson confidence interval.
- Disagreement examples: 10 sampled pairs where ensemble and human diverged, with the LLM reasons and the human's rationale.

**Charts.**
- Stacked bar per cosine bin: fraction of `final_label` that's SAME vs DIFFERENT. The bin where this ratio crosses 50% is the natural threshold center.
- Bar chart: per-LLM agreement with the ensemble majority (each LLM's "how often did I match the consensus?" score).
- Bar chart: LLM-vs-human agreement on spot-checked subset, broken out by cosine bin within [0.55, 0.75].

**Decision this section enables.** Everything downstream. The `final_label` column is the regression test from now until the embedding model is changed.

**Estimated cost & time.**
- LLM ensemble: ~50 min wall-clock + ~$45 API spend (600 pairs × 3 models × ~$0.025 avg per call at frontier-tier prices)
- Human spot-check on ~175 pairs: ~1.5–2 person-hours
- Total: ~3 hours wall-clock + ~2 person-hours labor + ~$45 API

Compare to pure-human baseline: ~6–8 person-hours + $0 API (≈$300–400 labor cost). Hybrid is roughly 3× faster and 5× cheaper while preserving human anchoring on the threshold-critical pairs.

**Caveats.**
- **Use `temperature=0` everywhere** for reproducibility. Re-running the labeling pipeline tomorrow should produce identical labels (modulo any model-version drift on the provider side).
- **Pin exact model IDs** in the labeling script (e.g. `claude-sonnet-4-6-<release-tag>`, `gpt-5.2-<release-tag>`, `gemini-3.5-flash-<release-tag>`). Capture these as a header comment row in `labeled_eval_set.csv` so labels are tied to specific model versions for future audit and reproducibility.
- **Re-label when something changes.** If the embedding model is upgraded, re-sample pairs against the new embeddings and re-run the labeling pipeline. The LLM labels are coupled to *items* (titles + ledes), not to the embedding model, so the labels themselves remain valid as long as the items don't change — but the cosine bins shift, so re-bin and re-sample.
- **Respect the <70% agreement gate.** Don't lower it to make the LLM-only path "work" — that defeats the purpose of having a guardrail. If the gate fails, full human labeling is the correct fallback.

---

### Section 9 — Threshold calibration

**Purpose.** Pick `τ_high` and `τ_low` from the labeled data.

**Operations.**
- Plot the cosine-similarity distribution of `SAME` pairs vs `DIFFERENT` pairs as overlapping KDEs or histograms.
- Compute precision/recall/F1 of "predict SAME if cosine ≥ τ" across a fine grid of τ values from 0.3 to 0.95 in steps of 0.01.
- `τ_high` = the τ where precision crosses 0.95 (or higher, if business needs are stricter). This is the auto-assign threshold.
- `τ_low` = the τ where recall crosses 0.95. Below this, the LLM judge is not even worth calling because the pair is almost certainly different. This is the residual cutoff.
- Compute F1 across all τ for a single-threshold baseline (no gray zone). Report the maximum.

**Tables.**
- Calibration table: τ, precision, recall, F1, n_SAME_predicted, n_SAME_actual.
- The two picked thresholds with their precision/recall.
- Comparison to research.md priors: published value (0.75 / 0.55) vs POC-calibrated value, delta.

**Charts.**
- KDE overlay: SAME (green) and DIFFERENT (red) cosine distributions on the same axes. Annotate `τ_high` and `τ_low` as vertical lines. This single chart is the most important visual in the entire POC.
- Precision-recall curve as τ sweeps from 0.3 → 0.95.
- F1 vs τ curve. Mark the maximum.
- ROC curve (true-positive vs false-positive rate).

**Decision this section enables.** Are the priors right for this dataset and embedding model? If POC `τ_high` lands within ±0.05 of 0.75 and `τ_low` within ±0.05 of 0.55, ship with priors and re-calibrate quarterly. If further off, the calibrated values must go into production from day one.

---

### Section 10 — Long-document contextual chunking

**Purpose.** Validate the contextual-retrieval pattern (doc-level Haiku summary prepended to each chunk) for items too long to embed whole.

**Operations.**
- Identify long items: token count > 2000 (rare for news; synthetic if the dataset has none).
- For each long item:
  - Call Haiku to produce a 1–2 sentence doc-level context summary.
  - Split body into 800-token chunks with 100-token overlap.
  - Embed two versions of each chunk: (a) plain chunk text; (b) `"DOCUMENT CONTEXT: {summary}\n\nCHUNK: {chunk}"`.
- Construct a small synthetic retrieval benchmark: 20 hand-crafted questions about specific facts buried deep in long items. For each question:
  - Retrieve top-5 chunks via cosine similarity, version (a).
  - Retrieve top-5 chunks via cosine similarity, version (b).
  - Score: is the correct chunk in top-5? In top-1?

**Tables.**
- Retrieval@k metric: hits@1, hits@3, hits@5 for both versions.
- 5 sampled queries showing top-1 chunk side-by-side (with vs without prepend).

**Charts.**
- Grouped bar: hits@k for k=1,3,5 with and without contextual prepend.

**Decision this section enables.** Does the prepend pattern actually help on financial-research-style long items? If hits@5 goes from 0.6 → 0.85+, the pattern earns its place in production. If the lift is <5%, drop it and rely on truncation.

**If the dataset has no long items.** Document this section as "deferred until long internal research data is available" and skip. The decision can wait until production data exists.

---

### Section 11 — Single-pass assignment loop (the main algorithm)

**Purpose.** Run the production clustering algorithm on the dataset.

**Operations.**
- Sort canonical items by `published_at`.
- Initialize empty story state: `stories = []` (each story = `{story_id, centroid, n_items, last_seen_at, member_ids, entities, affected_clients, title=None, summary=None}`).
- For each item:
  - **Candidate selection:** stories whose `affected_clients ∩ item.item_clients ≠ ∅` and `last_seen_at within 72h of item.published_at` and `closed_at is None`.
  - If no candidates: residual. Continue.
  - Compute cosine similarity to each candidate's centroid. Pick best.
  - **Apply gates:**
    - If `best_sim ≥ τ_high`: auto-assign.
    - Else if `best_sim ≥ τ_low` and `item.entities ∩ best_story.entities ≠ ∅` (entity-overlap gate): call Haiku judge. If SAME, assign; else residual.
    - Else: residual.
  - On assign: update story centroid (recompute from full member set), `n_items += 1`, `last_seen_at = max(last_seen_at, item.published_at)`, merge entities and affected_clients.
- After the pass, close all stories where `last_seen_at < (max_seen_published_at - 72h)`.

**Tables.**
- Items by outcome: auto-assigned, gray-zone-judged-SAME, gray-zone-judged-DIFFERENT, no-candidate-no-entity-match, below-τ_low. Counts and percentages.
- Distribution of `n_items` per story (active stories only).
- Top-20 stories by `n_items` with their constituent items' titles (manual quality inspection).

**Charts.**
- **[Stakeholder demo chart — THE money chart] Interactive 3D UMAP scatter of top stories with projected centroids** (Plotly `scatter_3d`). Pick the top 8–12 stories by `n_items`. Re-use the UMAP reducer fitted in Section 7: project each member item's 1024-dim assignment vector via `reducer.transform()`, and project each story's 1024-dim centroid via `reducer.transform()` too. Render:
  - Member items as small dots, colored by `story_id`, alpha ≈ 0.7. Hover tooltip: item title, source, published_at.
  - Story centroids as large diamonds (`symbol="diamond"`, `size=14`), same color as their members. Hover tooltip: story title, n_items, first_seen_at, last_seen_at.
  - Text annotations at each centroid showing the Sonnet-generated story title. If the top-12 titles render too cluttered, annotate only the top 4–5 and let the rest of the titles live in the legend.

  Why centroids matter: the diamond is the *algorithmic anchor* of the cluster in the projected space. Stakeholders see "the algorithm found this story's center here, and these items belong to it" rather than just a cloud of equally-weighted points. Use the labeling convention from Section 7 — axes are "UMAP component 1/2/3", title makes clear the space is 1024-dim. Export to HTML for sharing outside the notebook (e.g. emailed to leadership).
- Stacked area: items processed per day, colored by outcome bucket (auto-assigned / gray-judged-SAME / gray-judged-DIFFERENT / no-candidate / below-τ_low). Visualizes how the gate population evolves over time.
- Histogram: story sizes (`n_items` per story). Expect a long tail with most stories at 1–3 items and a few at 10+.
- Line: number of active stories over time, with `last_seen_at`-aging visible (stories appear, accumulate, close).
- Scatter: for each gray-zone item, plot `best_sim` on x and color by Haiku verdict (SAME=green, DIFFERENT=red). Reveals whether Haiku's decisions track sim within the gray zone or are essentially independent.

**Decision this section enables.** Sanity check that the algorithm produces a sensible story population — not all-singletons, not one mega-story, with growth that tracks news cycles. Spot-check the top-20 stories: do the constituent items actually belong together?

---

### Section 12 — Residual clustering (HDBSCAN)

**Purpose.** Spawn new stories from items that didn't match anything existing.

**Operations.**
- On the residual pool (items not assigned in Section 11), run `hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="cosine", cluster_selection_method="eom")`.
- Items with label `-1` are HDBSCAN noise — singletons; they become 1-member stories on their own.
- For each non-noise cluster, instantiate a new story with the cluster members.
- Enforce client-overlap within each cluster: if a cluster's members don't share at least one client across all of them, *split* the cluster along client lines before story creation. (This is the equivalent of the production "split clusters with empty client overlap" step.)

**Tables.**
- Residual cluster sizes (cluster_size, n_clusters).
- Noise rate: fraction of residual items labeled -1.
- 5 sampled multi-member residual clusters with member titles and shared clients.

**Charts.**
- Bar chart: residual cluster size distribution.
- **[Stakeholder demo chart] Interactive 3D UMAP scatter of residual items, colored by HDBSCAN label, with projected cluster centroids** (Plotly `scatter_3d`). Re-use the Section 7 UMAP reducer via `reducer.transform()`. Render:
  - Noise items (label `-1`) as gray dots, alpha ≈ 0.3, no annotation. The visual de-emphasis matches their algorithmic status as "not part of any cluster."
  - Non-noise residual cluster members as colored dots, alpha ≈ 0.7. One color per HDBSCAN label.
  - Cluster centroids (mean of member 1024-dim vectors, projected through `reducer.transform()`) as large diamonds, same color as their members. Hover: cluster_id, member count, shared clients.

  Stakeholders see "items that didn't match existing stories still got organized into coherent new ones" — and can visually distinguish real clusters (compact, colorful, diamond-anchored) from noise (gray haze). Use the same axis labeling and title convention from Section 7.

**Decision this section enables.** Does HDBSCAN find legitimate stories that the single-pass missed, or does it mostly produce noise? If noise rate >60%, the residual contains a lot of singletons; consider tightening τ_low so fewer items reach residual. If clusters are large and ill-defined, HDBSCAN might need tuning (lower `min_samples`, different selection method).

---

### Section 13 — Story metadata generation (inline)

**Purpose.** Test the Sonnet-based title + summary generation that runs in the same job as clustering.

**Operations.**
- For each new story (from Sections 11 and 12) and each grown story (≥1 new member added in this run): one Claude Sonnet call with structured output.
- Use the Chain-of-Key incremental update for grown stories, full generation for new.
- Force JSON schema via Anthropic `tool_use`.

**Tables.**
- 20 sampled stories with their generated `(title, summary, topic, entities)`. Read them; do they fairly describe the constituent items?
- Cost: total Sonnet tokens consumed, dollars spent, average per story.
- Length compliance: % of titles ≤80 chars, % of summaries ≤400 chars.

**Charts.**
- Histogram of generated title lengths and summary lengths.

**Decision this section enables.** Are Sonnet's outputs production-quality? If titles routinely run over the cap, the prompt's `≤80 chars` directive isn't being honored — tighten the prompt or post-process with truncation. If summaries are bland or wrong, you have a prompt problem to fix here, before AWS.

---

### Section 14 — Story merge & expiry pass

**Purpose.** Apply the weekly merge logic on the simulated full corpus.

**Operations.**
- Find all pairs of active stories with `centroid_cosine_similarity > 0.85` and `|affected_clients ∩| ≥ 2` and `|entities ∩| ≥ 2`.
- For each candidate merge, call Haiku judge: "Are these the same story?" If yes, merge: union members, recompute centroid, set `merged_into` on the smaller story.
- Apply final 72-hour expiry: stories whose newest member is older than 72h before the dataset's end-date are closed.

**Tables.**
- Merge candidates considered, merges applied, merge precision (sampled human verify).
- Active vs closed story counts before and after the expiry sweep.

**Charts.**
- Story age distribution at the end of the run (active stories' age = `now - first_seen_at`).

**Decision this section enables.** Validates that the merge pass isn't over-aggressive (high false-positive merge rate) and that the 72-hour expiry actually closes stories on a financial-news cadence (financial news often has multi-day arcs; if everything is closing immediately, lengthen the window for production).

---

### Section 15 — End-to-end evaluation against the labeled eval set

**Purpose.** Compute the headline B-cubed F1 and decide go/no-go.

**Operations.**
- For every labeled pair from Section 8, look up which stories the two items ended up in.
- Predict `SAME` if both items are in the same story (or one is a duplicate of the other), `DIFFERENT` otherwise.
- Compute B-cubed precision, recall, F1 over the labeled set.
- Compute baselines:
  - **Baseline 1:** pure single-threshold. Assign if `best_sim ≥ 0.65`, no LLM, no entity gate. Bare minimum.
  - **Baseline 2:** pure HDBSCAN on the full dataset (no single-pass). Reflects "what if we just clustered offline from scratch each night."
  - **Baseline 3:** title-Jaccard. Assign if title-token Jaccard > 0.5. Floor.
- Compute B-cubed for each baseline. Tabulate.

**Tables.**
- B-cubed scorecard: configuration, precision, recall, F1.
- Confusion matrix at the chosen thresholds: TP, FP, FN, TN.
- Worst false-positive merges (10 sampled story IDs with constituent items that the labels say *shouldn't* be together) and worst false-negative splits (10 cases the labels say should have merged but didn't).

**Charts.**
- Bar chart: F1 across configurations (baseline 1, 2, 3, full POC algorithm). The headline result.
- Confusion matrix heatmap.
- Threshold sensitivity: F1 as `τ_high` varies by ±0.05, holding `τ_low` fixed. Lets you see how brittle the choice is.

**Decision this section enables.** Ship to AWS (F1 ≥ 0.85), or iterate (F1 < 0.85). If iterating, the worst FP/FN samples tell you which direction to push — false merges → tighten thresholds; false splits → loosen thresholds or revisit the entity-overlap gate.

---

### Section 16 — Cost accounting & production projection

**Purpose.** Estimate the daily API bill at production scale.

**Operations.**
- Sum tokens consumed per stage: embeddings, entity extraction (Haiku), gray-zone judge (Haiku), metadata generation (Sonnet), doc-level chunk context (Haiku).
- Convert to dollars at current API list prices (note: production uses Bedrock pricing, which is typically the same or slightly lower than direct Anthropic; the POC uses direct API prices for simplicity).
- Compute per-item cost across the full POC run.
- Scale to production: 100k items/day, 5k stories updated/day → daily dollar estimate per stage.

**Tables.**
- Tokens & dollars per stage: stage, model, token count, dollars, % of total.
- Projected production cost per day, broken down by API.

**Charts.**
- Pie or stacked bar: cost breakdown by stage at POC scale.
- Same chart at projected production scale, with annotation of the POC ratio.

**Decision this section enables.** Is the production API budget plausible? If projected daily cost is wildly above expectations, identify the dominant line item and consider downgrading (e.g., Haiku instead of Sonnet for metadata updates, embed only title for assignment, etc.).

---

### Section 17 — Findings, recommendations, and handoff

**Purpose.** Capture what we learned in one digestible place.

**Operations.**
- Write a markdown cell with:
  - **Decision:** ship / iterate / abandon.
  - **Calibrated thresholds:** the final `τ_high`, `τ_low`, MinHash threshold, residual threshold, with justification from the labeled set.
  - **Algorithmic findings:** anything surprising about how the algorithm behaved on financial news that the production team needs to know.
  - **Failure modes observed:** specific cases that broke (with example item pairs).
  - **Open questions:** anything the POC couldn't answer that needs production data.
- Export `labeled_eval_set.csv` and the threshold/calibration JSON as artifacts.

**Tables.**
- Final threshold table (POC values vs research.md priors).
- Top-3 algorithmic findings.

**Charts.**
- One summary panel: 2×2 grid of the four most decision-driving charts from earlier sections (the KDE overlay, the precision-recall curve, the B-cubed scorecard bar chart, the cost breakdown).

**Outputs to disk.**
- `labeled_eval_set.csv` — for production regression testing.
- `pos_calibration.json` — `{tau_high, tau_low, minhash_threshold, residual_threshold, hdbscan_min_cluster_size, b3_f1, eval_set_size}`.
- `poc_findings.md` — the narrative of what we learned.

---

## Threshold calibration methodology (detail)

The most important methodology in the POC. Documented separately because production will repeat this every time the embedding model changes.

**Sampling.** Stratified random pair sampling across cosine-similarity bins (Section 8). Stratification is critical — uniform random sampling oversamples low-cosine pairs (which are obviously different and uninformative) and undersamples the threshold zone.

**Filtering.** Both items must share at least one entity (otherwise the production gate would never even consider them as candidates), and they must be within a 72-hour window of each other.

**Labeling (hybrid LLM-ensemble + targeted human spot-check).** Three frontier models from three different vendors — Claude Sonnet 4.6, GPT-5.2, Gemini 3.5 Flash — label all sampled pairs independently at `temperature=0`. Take ensemble majority. For pairs in the threshold-critical cosine band [0.55, 0.75] plus any pairs with ensemble disagreement, one human analyst provides ground-truth labels. Trust LLM labels at scale only if LLM-vs-human agreement on the spot-checked subset is ≥85%; otherwise fall back to human labels inside the band (70–85%) or to full human labeling (<70%). Crucial guardrail: none of the labeler models is Claude Haiku, the production gray-zone judge — otherwise the calibration is circular. See Section 8 for the full methodology.

**Threshold picking.** From the labeled set, compute precision and recall as functions of τ. `τ_high` = the smallest τ where precision ≥ 0.95. `τ_low` = the largest τ where recall ≥ 0.95. If those constraints can't be simultaneously satisfied with a meaningful gray-zone gap, the embedding model is not strong enough for this domain and you need to upgrade (try `voyage-3-large`).

**B-cubed F1.** Computed on the full simulated run (Section 15), not just on the labeled pairs. The labeled pairs calibrate τ; B-cubed evaluates the end-to-end story membership.

**Re-calibration triggers in production.**
- Embedding model version changes.
- Domain shift (e.g., expanding from financial news to general news).
- F1 on the labeled-set regression test drops by >0.03.

---

## What this POC does *not* validate

State these explicitly so reviewers don't expect more than the POC delivers.

- **Production-scale throughput.** A laptop running 5–10k items doesn't tell you how Glue will behave on 100k items.
- **Aurora-specific operations.** HNSW build time, JDBC bulk write throughput, RDS Proxy connection management — none of this is exercised in pandas.
- **Glue cold-start dynamics.** Wall-clock per job in production includes ~1–3 min cold start per job; the POC has none.
- **Multi-day cumulative behavior.** Stories that span the 72-hour window edge, re-emergence of dormant stories, weekly merge dynamics over multiple weeks — only partially testable with one batch run. Plan a longitudinal POC if the algorithm appears sensitive here.
- **Bedrock-specific behavior.** The POC uses Anthropic API directly; Bedrock has its own throttling, model availability per region, and structured-output behavior. Test these once in AWS, not now.
- **Source plugin abstraction.** The dataset comes pre-bundled; the production plugin registry is irrelevant in the notebook.
- **Failure modes under provider outage.** Retry semantics, circuit breaking, partial-run resumption — production concerns, not POC concerns.

---

## Dependencies & environment

**Python.** 3.11 or newer.

**Required packages.** `datasets`, `pandas`, `numpy`, `scikit-learn`, `hdbscan`, `umap-learn`, `matplotlib`, `seaborn`, `plotly>=5.0`, `datasketch`, `tiktoken`, `openai>=1.0`, `anthropic`, `google-generativeai`, `trafilatura`, `tqdm`. Pin versions in a `requirements.txt` checked in alongside the notebook. Plotly is required for the 3D interactive cluster visualizations used in Sections 7, 11, and 12 (the "stakeholder demo" charts).

**API keys.** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (for Gemini 3.5 Flash labeling in Section 8). Load from `.env` via `python-dotenv` or shell env. Never commit them.

**Resource footprint.** Laptop sufficient. Peak memory ~4 GB at 10k items with 1024-dim embeddings. If the dataset is much larger, sample down to 10k items rather than upgrading the machine — POC purpose is algorithm validation, not scale testing.

**Determinism.** Set seeds for: HDBSCAN (`random_state=42`), UMAP (`random_state=42`), pandas sampling (`random_state=42`), MinHash (`seed=42` in `MinHash` constructor). Embeddings and LLM calls are intrinsically non-deterministic; for re-runs, cache responses to disk by input hash.

---

## Estimated runtime and cost

Assuming ~5,000 canonical items after dedup:

| Step | Time | Cost |
|---|---|---|
| Dataset load + exploration | ~5 min | $0 |
| URL canonicalize + MinHash dedup | ~10 min | $0 |
| Entity extraction (Haiku) | ~30 min | ~$0.50 |
| Assignment embeddings (OpenAI) | ~10 min | ~$2 |
| Labeled-set sample preparation | ~5 min | $0 |
| LLM-ensemble labeling (Sonnet 4.6 + GPT-5.2 + Gemini 3.5 Flash) | ~50 min wall-clock | ~$45 |
| Human spot-check on threshold-zone subset (~175 pairs) | ~1.5–2 person-hours | (labor) |
| Threshold calibration | ~2 min | $0 |
| Long-doc chunking experiment | ~15 min | ~$1 (Haiku + embeddings) |
| Single-pass assignment loop | ~30 min | ~$3 (Haiku judge for gray zone) |
| HDBSCAN residual | ~2 min | $0 |
| Metadata generation (Sonnet) | ~20 min | ~$15 (Sonnet is the largest line item) |
| Merge pass | ~5 min | ~$0.50 (Haiku) |
| Evaluation + cost accounting | ~5 min | $0 |
| **Total wall-clock** | **~3 hours** machine + ~2 person-hours spot-check | **~$70** API |

Scale costs roughly linearly to dataset size. At 50k canonical items the API bill grows to ~$200–300.

---

## Handoff to the AWS deployment spec

When the POC's `decision = SHIP`, hand the following to the production build:

1. `labeled_eval_set.csv` — checked into the production repo at `eval/labeled_set_v1.csv`; runs as a regression test in CI before any model/threshold change is merged.
2. `pos_calibration.json` — checked into the production repo at `config/calibration.json`; loaded by Job 2's clustering code.
3. `poc_findings.md` — appended to the AWS deployment spec's "Evaluation and observability" section as the v1 calibration record.

The AWS spec (`story-clustering-aws-spec.md`) assumes these three files exist before any Glue job is built.

---

## Caveats

1. **The dataset is a proxy, not your data.** Financial news from a public dataset has different source mix, different entity distribution, and possibly different language register than Perplexity-aggregated banker news + internal JPM research. Thresholds calibrated here are a *starting point* for production calibration on real Perplexity output, not a final answer. Plan a second calibration run within the first month of production data.
2. **No internal research notes in the POC.** JPM internal research has different document structure (long, sections, sometimes tables) and likely different embedding behavior than news. The contextual-chunking section will be incomplete without it. If at all possible, augment with a sample of internal research before the AWS spec is implemented; otherwise, plan the first production calibration to specifically test chunking on real research.
3. **API non-determinism.** LLM judge verdicts can change on re-run for borderline pairs. Cache responses by `(prompt_hash, model_id)` to disk on first call; re-runs are then deterministic.
4. **The labeled set is the bottleneck.** Plan the ~6 hours of human labeling realistically. Don't rush it; bad labels poison the calibration.
5. **The POC is not a one-shot effort.** Treat the notebook as living artifact: when you change embedding models, when you observe a failure mode in production, when you onboard a new source, re-run the relevant sections and update the calibration.
