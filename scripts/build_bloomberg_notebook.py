"""Build story_clustering_bloomberg.ipynb — a clean, top-to-bottom thin driver over
scripts/bloomberg_clustering.py. Bloomberg-only, core clustering, runtime-only calibration.
Run: .venv/bin/python scripts/build_bloomberg_notebook.py
"""
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Story Clustering — Bloomberg (production reference)

Clean, top-to-bottom clustering pipeline for a **body-rich** corpus (Bloomberg = proxy for the
production research-artifact corpus). This is the *winning* v4 configuration:
**body-paragraph chunks + max-pool nearest-chunk similarity + a two-tier `chunk_pair` LLM judge.**

The clustering logic lives in **`scripts/bloomberg_clustering.py`** (importable, provider-agnostic);
this notebook is a thin driver. To productionize, lift that module + the calibration artifact.

Pipeline: ingest → URL+MinHash dedup → body-chunk embeddings → single-pass assignment loop
(max-pool + fusion gate + judge) → HDBSCAN residual clustering → final stories.""")

# ---------------- §1 setup ----------------
md("## §1 — Setup & config")
code('''import sys, os, json, asyncio, hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.cwd() / "scripts"))
import v4_chunking as v4c
import bloomberg_clustering as bc

# Load API keys from a project-local .env (OPENAI_API_KEY required for embeddings + the judge).
try:
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY (e.g. in a project .env) before running."
print("modules loaded:", [n for n in dir(bc) if not n.startswith("_")][:8], "...")''')

md("### §1.1 — CONFIG (all production knobs)")
code('''CONFIG = {
    # --- data ---
    "items_path":   Path.cwd() / "artifacts" / "v4" / "bloomberg_items.parquet",  # 🔌 SWAP for prod
    "max_items":    int(os.environ["BB_MAX_ITEMS"]) if os.environ.get("BB_MAX_ITEMS") else None,  # None=full corpus; int=quick demo
    "cache_dir":    Path.cwd() / ".cache",
    "fusion_gate_path": Path.cwd() / "artifacts" / "v4" / "fusion_model_chunk.json",  # runtime-only

    # --- embeddings ---
    "embed_model": "text-embedding-3-large", "embed_dims": 1024,

    # --- chunking (body-only paragraph chunks; title-only items fall back to title) ---
    "chunk_min_tokens": 25, "chunk_max_tokens": 400, "max_chunks_per_item": 12,

    # --- dedup ---
    "minhash_num_perm": 128, "minhash_threshold": 0.85,

    # --- clustering ---
    "active_window_hours": 72,
    "hdbscan_min_cluster_size": 2, "hdbscan_min_samples": 2, "hdbscan_cluster_selection_method": "eom",

    # --- two-tier judge (OpenAI Responses API: structured verdict + reasoning effort) ---
    "judge_base_model": "gpt-4.1-mini",          # base: handles every gray-zone call
    "judge_base_effort": None,                   # gpt-4.1-mini is not a reasoning model -> no effort
    "judge_escalation_model": "gpt-5.4-mini",    # escalate the uncertain band to this reasoning model
    "judge_escalation_effort": "none",           # reasoning effort: none | low | medium | high
    "judge_escalation_band": (0.62, 0.86),       # escalate when max-pool similarity is in this band
    "judge_verbosity": "high",                   # text.verbosity (reasoning model)
    "judge_summary": "concise",                  # reasoning.summary
    "judge_text_mode": "chunk_pair",             # title + date + matched body chunk (no lede)
    "judge_prompt_version": "v4_chunkpair",
    "judge_cache_dir": Path.cwd() / ".cache" / "judge",
}
print(f"Bloomberg pipeline | base judge={CONFIG['judge_base_model']} -> escalate "
      f"{CONFIG['judge_escalation_band']} to {CONFIG['judge_escalation_model']}")''')

# ---------------- §2 ingest ----------------
md("""## §2 — Ingest Bloomberg items

🔌 **SWAP POINT:** in production replace this load with your own source. The pipeline needs a
DataFrame with columns: `item_id` (stable str), `title`, `body` (the full text — this is what gets
chunked), `url`, `published_at` (tz-aware datetime), and `item_clients` (a `set` of entity tags used
to scope merge candidates — your entity-extraction output).""")
code('''items_df = pd.read_parquet(CONFIG["items_path"])
items_df["item_clients"] = items_df["item_clients"].apply(set)   # parquet stores it as a list
items_df["published_at"] = pd.to_datetime(items_df["published_at"], utc=True)
if CONFIG["max_items"]:   # quick-demo subset (earliest by time); set max_items=None for full corpus
    items_df = items_df.sort_values("published_at").head(CONFIG["max_items"]).reset_index(drop=True)
    print(f"⏱️  demo subset: first {len(items_df):,} items by time (max_items=None => full corpus)")
print(f"{len(items_df):,} items | columns: {list(items_df.columns)}")
print(f"body present: {(items_df['body'].str.len() > 0).mean()*100:.0f}%  (body-rich corpus)")
items_df.head(2)[["item_id", "title", "published_at", "item_clients"]]''')

# ---------------- §3 dedup ----------------
md("## §3 — Dedup (URL canonicalization + MinHash/LSH near-duplicates)")
code('''items_df = bc.dedup_pipeline(
    items_df, num_perm=CONFIG["minhash_num_perm"], minhash_threshold=CONFIG["minhash_threshold"])
canonical = items_df[~items_df["is_duplicate"]].reset_index(drop=True)
print(f"canonical (non-duplicate) items: {len(canonical):,} / {len(items_df):,} "
      f"({items_df['is_duplicate'].sum():,} near-dups marked)")''')

# ---------------- §4 chunk + embed ----------------
md("""## §4 — Body-paragraph chunks + embeddings

Each item's body is split into paragraph chunks (title-only items fall back to the title). Chunks are
embedded with a small on-disk cache. Production: swap `embed_texts` for your embedding service.""")
code('''import tiktoken
_enc = tiktoken.encoding_for_model(CONFIG["embed_model"])
count_tokens = lambda s: len(_enc.encode(s))
chunks_df = bc.build_chunks(canonical, count_tokens=count_tokens,
                            min_tokens=CONFIG["chunk_min_tokens"], max_tokens=CONFIG["chunk_max_tokens"],
                            max_chunks=CONFIG["max_chunks_per_item"])
print(f"{len(canonical):,} items -> {len(chunks_df):,} chunks (mean {len(chunks_df)/len(canonical):.2f}/item)")''')

code('''# Cached async embedder (cache key matches the POC so existing chunk vectors are reused).
from openai import AsyncOpenAI
_embed_client = AsyncOpenAI()
_EMBED_CACHE = CONFIG["cache_dir"] / "embeddings.pkl"

def _embed_key(text):
    h = hashlib.sha256(); h.update(f"{CONFIG['embed_model']}|{CONFIG['embed_dims']}|chunk_v1".encode())
    h.update(text.encode("utf-8")); return h.hexdigest()[:16]

async def embed_texts(texts, batch=100):
    cache = pickle.loads(_EMBED_CACHE.read_bytes()) if _EMBED_CACHE.exists() else {}
    miss = [t for t in dict.fromkeys(texts) if _embed_key(t) not in cache]
    for i in range(0, len(miss), batch):
        b = miss[i:i + batch]
        r = await _embed_client.embeddings.create(model=CONFIG["embed_model"], input=b, dimensions=CONFIG["embed_dims"])
        for t, d in zip(b, r.data): cache[_embed_key(t)] = np.asarray(d.embedding, dtype=np.float32)
    if miss: _EMBED_CACHE.write_bytes(pickle.dumps(cache))
    print(f"chunk embed: {len(texts)-len(miss):,} cached, {len(miss):,} fetched")
    return np.stack([cache[_embed_key(t)] for t in texts])

chunk_vecs = await embed_texts(chunks_df["chunk_text"].tolist())
index = bc.ChunkIndex(chunks_df, chunk_vecs)
print(f"chunk_vecs: {chunk_vecs.shape}")''')

# ---------------- §5 runtime artifacts + judge ----------------
md("""## §5 — Runtime artifacts: fusion gate + two-tier judge

Calibration is **runtime-only**: load the pre-built fusion gate. (Recalibrate offline on your own
labeled pairs and regenerate `fusion_model_chunk.json`.) The judge wraps any LLM via `complete_fn`.""")
code('''gate = bc.FusionGate.load(CONFIG["fusion_gate_path"])
minhashes = bc.compute_minhashes(canonical, num_perm=CONFIG["minhash_num_perm"])  # for the lexical features
print(f"fusion gate: p_high={gate.p_high:.3f} p_low={gate.p_low:.3f} | features={gate.features}")''')

code('''# OpenAI Responses API: structured verdict (bc.VERDICT_SCHEMA) + reasoning effort for gpt-5.x.
# Swap the body for your provider; respond_fn must return {"verdict","reason"}.
from openai import AsyncOpenAI, RateLimitError
_judge_client = AsyncOpenAI()

async def respond_fn(model, prompt, effort):
    text_cfg = {"format": {"type": "json_schema", "name": "comparison_verdict",
                           "strict": True, "schema": bc.VERDICT_SCHEMA}}
    kwargs = dict(model=model, input=[{"role": "user", "content": prompt}], store=False)
    if effort is not None:                       # reasoning model (gpt-5.x): add effort + verbosity
        text_cfg["verbosity"] = CONFIG["judge_verbosity"]
        kwargs["reasoning"] = {"effort": effort, "summary": CONFIG["judge_summary"]}
    kwargs["text"] = text_cfg
    for attempt in range(6):
        try:
            r = await _judge_client.responses.create(**kwargs)
            try:
                return json.loads(r.output_text)
            except Exception:
                return {"verdict": "UNCLEAR", "reason": "unparseable: " + (r.output_text or "")[:120]}
        except RateLimitError:
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError("judge 429 retries exhausted")

judge = bc.TwoTierJudge(
    respond_fn, base_model=CONFIG["judge_base_model"], base_effort=CONFIG["judge_base_effort"],
    escalation_model=CONFIG["judge_escalation_model"], escalation_effort=CONFIG["judge_escalation_effort"],
    escalation_band=CONFIG["judge_escalation_band"], cache_dir=CONFIG["judge_cache_dir"],
    judge_text_mode=CONFIG["judge_text_mode"], prompt_version=CONFIG["judge_prompt_version"])
print(f"judge ready: {CONFIG['judge_base_model']} -> {CONFIG['judge_escalation_model']} "
      f"(effort={CONFIG['judge_escalation_effort']}, band={CONFIG['judge_escalation_band']})")''')

# ---------------- §6 assignment loop ----------------
md("""## §6 — Single-pass assignment loop (max-pool + gate + judge)

Walks items in time order; for each, finds open same-entity stories within the 72h window, scores by
**max-pool nearest-chunk** similarity, applies the fusion gate (auto-merge / gray-zone / new), and
sends gray-zone pairs to the two-tier judge with the matched chunk pair.""")
code('''canonical = canonical.sort_values("published_at", kind="stable").reset_index(drop=True)
canonical["__pos__"] = np.arange(len(canonical))
from tqdm.auto import tqdm
stories = await bc.assignment_loop(
    canonical, index, gate, judge, window_hours=CONFIG["active_window_hours"],
    minhashes=minhashes, progress=lambda xs: tqdm(xs, desc="assign"))
bc.expire_stories(stories, window_hours=CONFIG["active_window_hours"])
print(f"{len(stories):,} stories | multi-item: {sum(s['n_items']>1 if 'n_items' in s else len(s['member_ids'])>1 for s in stories)} "
      f"| judge {judge.stats}")''')

# ---------------- §7 residual ----------------
md("## §7 — Residual clustering (HDBSCAN over chunk vectors → union items)")
code('''singletons = [s for s in stories if len(s["member_ids"]) == 1]
multi = [s for s in stories if len(s["member_ids"]) > 1]
new_stories, absorbed = await bc.residual_cluster(
    singletons, index, judge, min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
    min_samples=CONFIG["hdbscan_min_samples"], cluster_selection_method=CONFIG["hdbscan_cluster_selection_method"],
    minhashes=minhashes, items_df=canonical)
final_stories = multi + new_stories + [s for s in singletons if s["member_ids"][0] not in absorbed]
print(f"residual: +{len(new_stories)} multi-stories, {len(absorbed)} singletons absorbed")
print(f"FINAL: {len(final_stories):,} stories | multi-item: {sum(len(s['member_ids'])>1 for s in final_stories):,}")''')

# ---------------- §8 output ----------------
md("## §8 — Output: final stories")
code('''rows = [{"story_id": s.get("story_id") or f"f{i:05d}", "n_items": len(s["member_ids"]),
         "clients": sorted(s["item_clients"]), "first_seen_at": s["first_seen_at"],
         "last_seen_at": s["last_seen_at"], "member_ids": s["member_ids"]}
        for i, s in enumerate(final_stories)]
stories_df = pd.DataFrame(rows).sort_values("n_items", ascending=False).reset_index(drop=True)
stories_df.to_parquet(Path.cwd() / "artifacts" / "v4" / "bloomberg_stories.parquet", index=False)
print(f"saved {len(stories_df):,} stories -> artifacts/v4/bloomberg_stories.parquet")
stories_df["n_items"].value_counts().sort_index()''')

code('''# Sanity: largest stories' member titles.
import matplotlib.pyplot as plt
_title = canonical.set_index("item_id")["title"]
for _, s in stories_df[stories_df.n_items > 1].head(8).iterrows():
    print(f"\\n[{s.story_id}] n={s.n_items} clients={s.clients}")
    for mid in s.member_ids[:4]:
        print("   •", _title.get(mid, "?")[:90])
ax = stories_df["n_items"].clip(upper=10).value_counts().sort_index().plot.bar(
    title="Story-size distribution (capped at 10)", figsize=(8, 3))
ax.set_xlabel("items per story"); ax.set_ylabel("# stories"); plt.tight_layout()''')

# ---------------- write ----------------
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("story_clustering_bloomberg.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote story_clustering_bloomberg.ipynb: {len(CELLS)} cells "
      f"({sum(t=='code' for t,_ in CELLS)} code, {sum(t=='markdown' for t,_ in CELLS)} md)")
