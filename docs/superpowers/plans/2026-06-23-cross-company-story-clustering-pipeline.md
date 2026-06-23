# Cross-Company Story Clustering Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic batch pipeline that clusters a normalized news corpus into cross-company *stories* and loads each run as a versioned batch into PostgreSQL, orchestrated by Prefect with each stage an atomic, independently-testable task.

**Architecture:** A new `pipeline/` package holds one focused module per stage (pure functions). `pipeline/flow.py` wraps each pure function in a Prefect `@task` and wires them into a `@flow`; tasks pass artifact *paths* under `runs/<run_id>/`. Pure clustering helpers (URL canonicalization, MinHash dedup, fusion gate, LLM judge) are reused from the existing `scripts/bloomberg_clustering.py` via a small import shim — not re-derived.

**Tech Stack:** Python 3.12, Prefect, pandas/pyarrow, numpy, scikit-learn (pre-fit gate), datasketch (MinHash), hdbscan (optional residual), OpenAI (embeddings + judge), SQLAlchemy + psycopg (Postgres), pytest + pytest-postgresql (tests).

## Global Constraints

- **Run model:** batch one-shot — every run reprocesses the whole corpus and rebuilds all stories. No persistent story state.
- **Story scope:** cross-company. `company_id` is metadata/feature, never a candidate filter; the same event under different `company_id`s must be able to merge.
- **Scale:** ≤ ~20k items/batch. Candidate generation is exact brute-force cosine kNN within a publish-date window — no ANN index.
- **Gate:** pure inference — load a pre-fit `fusion_model.json`. No calibration in this pipeline.
- **Baked-in config:** single-vector embeddings + **full-body** judge + union-find clustering. No body chunking, no `chunk_pair` judge.
- **Determinism:** same input + `config_hash` ⇒ identical story partitions.
- **Column mapping:** source `content`→internal `body`, source `publish_date`→internal `published_at`; `company_id`, `ingestion_date`, `url`, `title` kept. This lets us reuse `bloomberg_clustering.py` (which expects `body`/`published_at`) unchanged.
- **Secrets:** `DATABASE_URL` + `OPENAI_API_KEY` from env / Prefect Secret block; never committed.
- **Reuse shim:** all modules import the existing helpers via `from pipeline._bc import bc, v4c`.
- **Artifacts:** every stage writes exactly one primary artifact under `runs/<run_id>/`; filenames are fixed (`00_items.parquet` … `metrics.json`).

---

## File Structure

```
pipeline/
  __init__.py
  _bc.py            # import shim: re-exports bloomberg_clustering as bc, v4_chunking as v4c
  config.py         # RunConfig (frozen), config_hash, run_id, run_dir
  artifacts.py      # read/write parquet|npy|json, artifact_path, stage_cache_key
  load.py           # load_normalized: validate + column-map + item_id + quarantine
  dedup.py          # exact_dedup (group by item_id, aggregate company_ids) + near_dedup (MinHash)
  embed.py          # embed_items: single-vector embed (injected embed_fn) + content-hash cache
  candidates.py     # generate_candidates: cross-company sliding-window exact kNN
  features.py       # features_for_candidates: 7 fusion features per pair (wraps bc.pair_features)
  gate.py           # gate_decisions: auto_same | gray | auto_reject (wraps bc.FusionGate)
  judge.py          # judge_gray_pairs: full-body judge on gray pairs (wraps bc.TwoTierJudge)
  cluster.py        # connected_components + assemble_stories (cohesion split) + residual_cluster
  persist.py        # persist_outputs: expand reps -> items_scored, stories, metrics
  db.py             # load_to_postgres: versioned-batch transactional load
  schema.sql        # PLACEHOLDER DDL (clustering_runs, stories, story_items)
  evaluate.py       # evaluate_stories: pairwise P/R/F1 + chart (optional QA)
  flow.py           # Prefect @task wrappers + cluster_stories_flow(@flow) + real embed/respond fns
tests/pipeline/
  test_config.py test_artifacts.py test_load.py test_dedup.py test_embed.py
  test_candidates.py test_features.py test_gate.py test_judge.py test_cluster.py
  test_persist.py test_db.py test_evaluate.py test_flow_golden.py
  conftest.py       # shared fixtures: tiny corpus, stub embed_fn, stub respond_fn
```

---

## Task 1: Package scaffold, dependencies, reuse shim, RunConfig, artifacts

**Files:**
- Create: `pipeline/__init__.py`, `pipeline/_bc.py`, `pipeline/config.py`, `pipeline/artifacts.py`
- Create: `tests/pipeline/__init__.py`, `tests/pipeline/test_config.py`, `tests/pipeline/test_artifacts.py`
- Modify: `requirements.txt` (add deps)

**Interfaces:**
- Produces:
  - `pipeline._bc.bc`, `pipeline._bc.v4c` — the reused modules.
  - `RunConfig` frozen dataclass with fields: `source_uri:str, runs_dir:str="runs", window_hours:int=72, cosine_floor:float=0.45, gate_path:str, embed_model:str="text-embedding-3-large", embed_dims:int=1024, judge_base_model:str="gpt-4.1-mini", judge_escalation_model:str|None=None, judge_escalation_band:tuple[float,float]=(0.0,0.0), judge_budget:int|None=None, split_threshold:float=0.55, residual_enabled:bool=False, eval_path:str|None=None, eval_min_f1:float|None=None, database_url:str|None=None`. Methods: `config_hash()->str`, property `run_id->str` (`"run_"+config_hash()[:12]`), property `run_dir->Path`.
  - `artifacts.artifact_path(cfg, name)->Path`, `read_parquet/write_parquet/read_npy/write_npy/read_json/write_json`, `stage_cache_key(cfg, *input_paths)->str`.

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:
```
prefect>=2.19
SQLAlchemy>=2.0
psycopg[binary]>=3.1
pytest-postgresql>=6.0
```
Run: `.venv/bin/python -m pip install -q -r requirements.txt`

- [ ] **Step 2: Create the package skeleton + reuse shim**

`pipeline/__init__.py`: empty. `tests/pipeline/__init__.py`: empty.

`pipeline/_bc.py`:
```python
"""Import shim so pipeline modules reuse the existing clustering helpers (DRY)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import bloomberg_clustering as bc   # noqa: E402,F401
import v4_chunking as v4c           # noqa: E402,F401
```

- [ ] **Step 3: Write failing tests for config + artifacts**

`tests/pipeline/test_config.py`:
```python
from pipeline.config import RunConfig

def test_config_hash_is_deterministic_and_field_sensitive():
    a = RunConfig(source_uri="s", gate_path="g.json")
    b = RunConfig(source_uri="s", gate_path="g.json")
    c = RunConfig(source_uri="s", gate_path="g.json", window_hours=48)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
    assert a.run_id.startswith("run_")

def test_database_url_excluded_from_hash():
    a = RunConfig(source_uri="s", gate_path="g.json", database_url="postgres://x")
    b = RunConfig(source_uri="s", gate_path="g.json", database_url="postgres://y")
    assert a.config_hash() == b.config_hash()   # secret must not change run identity
```

`tests/pipeline/test_artifacts.py`:
```python
import numpy as np, pandas as pd
from pipeline.config import RunConfig
from pipeline import artifacts as A

def test_roundtrip_parquet_npy_json(tmp_path):
    cfg = RunConfig(source_uri="s", gate_path="g.json", runs_dir=str(tmp_path))
    df = pd.DataFrame({"x": [1, 2]})
    A.write_parquet(df, A.artifact_path(cfg, "t.parquet"))
    assert A.read_parquet(A.artifact_path(cfg, "t.parquet")).equals(df)
    A.write_npy(np.arange(3), A.artifact_path(cfg, "v.npy"))
    assert list(A.read_npy(A.artifact_path(cfg, "v.npy"))) == [0, 1, 2]
    A.write_json({"a": 1}, A.artifact_path(cfg, "m.json"))
    assert A.read_json(A.artifact_path(cfg, "m.json")) == {"a": 1}

def test_stage_cache_key_changes_with_input(tmp_path):
    cfg = RunConfig(source_uri="s", gate_path="g.json", runs_dir=str(tmp_path))
    p = A.artifact_path(cfg, "t.parquet"); A.write_parquet(pd.DataFrame({"x": [1]}), p)
    k1 = A.stage_cache_key(cfg, p)
    A.write_parquet(pd.DataFrame({"x": [9]}), p)
    assert A.stage_cache_key(cfg, p) != k1
```

- [ ] **Step 4: Run tests — verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_config.py tests/pipeline/test_artifacts.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.config`).

- [ ] **Step 5: Implement `pipeline/config.py`**

```python
import hashlib, json
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass(frozen=True)
class RunConfig:
    source_uri: str
    gate_path: str
    runs_dir: str = "runs"
    window_hours: int = 72
    cosine_floor: float = 0.45
    embed_model: str = "text-embedding-3-large"
    embed_dims: int = 1024
    judge_base_model: str = "gpt-4.1-mini"
    judge_escalation_model: str | None = None
    judge_escalation_band: tuple = (0.0, 0.0)
    judge_budget: int | None = None
    split_threshold: float = 0.55
    residual_enabled: bool = False
    eval_path: str | None = None
    eval_min_f1: float | None = None
    database_url: str | None = None   # secret: excluded from hash

    def config_hash(self) -> str:
        d = asdict(self)
        d.pop("database_url"); d.pop("runs_dir")
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def run_id(self) -> str:
        return "run_" + self.config_hash()[:12]

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_id
```

- [ ] **Step 6: Implement `pipeline/artifacts.py`**

```python
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

def artifact_path(cfg, name: str) -> Path:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg.run_dir / name

def write_parquet(df: pd.DataFrame, path: Path): df.to_parquet(path)
def read_parquet(path: Path) -> pd.DataFrame: return pd.read_parquet(path)
def write_npy(arr: np.ndarray, path: Path): np.save(path, arr)
def read_npy(path: Path) -> np.ndarray: return np.load(path, allow_pickle=True)
def write_json(obj, path: Path): Path(path).write_text(json.dumps(obj, indent=2, default=str))
def read_json(path: Path): return json.loads(Path(path).read_text())

def _digest(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]

def stage_cache_key(cfg, *input_paths) -> str:
    parts = [cfg.config_hash()] + [_digest(p) for p in input_paths]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
```

- [ ] **Step 7: Run tests — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_config.py tests/pipeline/test_artifacts.py -q`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
git add pipeline tests/pipeline requirements.txt
git commit -m "pipeline: scaffold, RunConfig, artifact IO, bc reuse shim"
```

---

## Task 2: `load_normalized` — validate, column-map, item_id, quarantine

**Files:**
- Create: `pipeline/load.py`, `tests/pipeline/conftest.py`, `tests/pipeline/test_load.py`

**Interfaces:**
- Consumes: `bc.canonicalize_url`, `bc.url_hash_hex`.
- Produces: `load_normalized(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(items, rejected)`. `items` columns: `item_id, title, body, url, canonical_url, published_at, company_id, ingestion_date`. `item_id = bc.url_hash_hex(bc.canonicalize_url(url))`. Rows missing `url`/empty canonical_url, or missing both `title` and `content`, go to `rejected` with a `reason` column.

- [ ] **Step 1: Shared fixtures**

`tests/pipeline/conftest.py`:
```python
import numpy as np, pandas as pd, pytest

@pytest.fixture
def raw_corpus():
    # 6 rows: a cross-company event (2 companies, near-identical text), a distinct event,
    # an exact-url duplicate under a different company, and a malformed row.
    rows = [
        dict(company_id="AAPL", title="Apple wins patent appeal", content="Apple defeated Kodak in the patent appeal ruling today in federal court.", url="http://x.com/a?utm=1", publish_date="2012-03-01T10:00:00Z", ingestion_date="2012-03-01T12:00:00Z"),
        dict(company_id="KODK", title="Kodak loses to Apple", content="Apple defeated Kodak in the patent appeal ruling today in federal court.", url="http://x.com/b", publish_date="2012-03-01T11:00:00Z", ingestion_date="2012-03-01T12:00:00Z"),
        dict(company_id="GOOG", title="FTC finishes Google probe", content="The FTC is poised to finish its antitrust probe of Google within weeks.", url="http://x.com/c", publish_date="2012-03-02T09:00:00Z", ingestion_date="2012-03-02T10:00:00Z"),
        dict(company_id="MSFT", title="FTC finishes Google probe (wire)", content="The FTC is poised to finish its antitrust probe of Google within weeks per sources.", url="http://x.com/c", publish_date="2012-03-02T09:30:00Z", ingestion_date="2012-03-02T10:00:00Z"),  # exact-url dup of /c under another company
        dict(company_id="TSLA", title="Unrelated EV story", content="Tesla announced a new battery chemistry unrelated to anything else here.", url="http://x.com/d", publish_date="2012-03-02T09:30:00Z", ingestion_date="2012-03-02T10:00:00Z"),
        dict(company_id="BAD", title=None, content=None, url=None, publish_date="2012-03-02T09:30:00Z", ingestion_date="2012-03-02T10:00:00Z"),  # malformed
    ]
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write failing test**

`tests/pipeline/test_load.py`:
```python
import pandas as pd
from pipeline.load import load_normalized

def test_load_maps_columns_assigns_item_id_and_quarantines(raw_corpus):
    items, rejected = load_normalized(raw_corpus)
    assert {"item_id", "title", "body", "url", "published_at", "company_id", "ingestion_date"} <= set(items.columns)
    assert len(rejected) == 1 and rejected.iloc[0]["reason"]            # the malformed row
    assert len(items) == 5
    assert pd.api.types.is_datetime64_any_dtype(items["published_at"])
    # exact-url rows (/c) share an item_id
    c = items[items["url"] == "http://x.com/c"]
    assert c["item_id"].nunique() == 1
```

- [ ] **Step 3: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_load.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.load`).

- [ ] **Step 4: Implement `pipeline/load.py`**

```python
import pandas as pd
from pipeline._bc import bc

REQUIRED = ["company_id", "title", "content", "url", "publish_date", "ingestion_date"]

def load_normalized(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError(f"source missing required columns: {missing}")
    df = df_raw.copy()
    df["body"] = df["content"].fillna("")
    df["title"] = df["title"].fillna("")
    df["published_at"] = pd.to_datetime(df["publish_date"], utc=True, errors="coerce")
    df["ingestion_date"] = pd.to_datetime(df["ingestion_date"], utc=True, errors="coerce")
    df["canonical_url"] = df["url"].apply(lambda u: bc.canonicalize_url(u) if isinstance(u, str) else "")

    bad_url = df["canonical_url"].str.len() == 0
    bad_text = (df["title"].str.len() == 0) & (df["body"].str.len() == 0)
    bad_time = df["published_at"].isna()
    reject_mask = bad_url | bad_text | bad_time
    rejected = df[reject_mask].copy()
    rejected["reason"] = (
        bad_url.map({True: "no_url;", False: ""}) + bad_text.map({True: "no_text;", False: ""})
        + bad_time.map({True: "bad_time;", False: ""}))[reject_mask]

    items = df[~reject_mask].copy()
    items["item_id"] = items["canonical_url"].apply(bc.url_hash_hex)
    cols = ["item_id", "title", "body", "url", "canonical_url", "published_at", "company_id", "ingestion_date"]
    return items[cols].reset_index(drop=True), rejected.reset_index(drop=True)
```

- [ ] **Step 5: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_load.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/load.py tests/pipeline/conftest.py tests/pipeline/test_load.py
git commit -m "pipeline: load_normalized (schema map + item_id + quarantine)"
```

---

## Task 3: `exact_dedup` + `near_dedup` — collapse and aggregate company_ids

**Files:**
- Create: `pipeline/dedup.py`, `tests/pipeline/test_dedup.py`

**Interfaces:**
- Consumes: items from Task 2; `bc.compute_minhashes`, `bc.minhash_near_dups`.
- Produces:
  - `exact_dedup(items) -> pd.DataFrame` — one row per `item_id`, new column `company_ids: list[str]` (sorted set across exact-url members), keep earliest `published_at`.
  - `near_dedup(canon, threshold=0.85) -> tuple[pd.DataFrame, pd.DataFrame]` — `(items_dedup, dup_map)`. `items_dedup` one row per representative with `company_ids` unioned across near-dup members; `dup_map` columns `member_id, rep_id`.

- [ ] **Step 1: Write failing test**

`tests/pipeline/test_dedup.py`:
```python
from pipeline.load import load_normalized
from pipeline.dedup import exact_dedup, near_dedup

def test_exact_dedup_aggregates_company_ids(raw_corpus):
    items, _ = load_normalized(raw_corpus)
    canon = exact_dedup(items)
    assert canon["item_id"].is_unique
    c = canon[canon["url"] == "http://x.com/c"].iloc[0]   # GOOG + MSFT shared the url
    assert set(c["company_ids"]) == {"GOOG", "MSFT"}

def test_near_dedup_collapses_cross_company_near_dups(raw_corpus):
    items, _ = load_normalized(raw_corpus)
    canon = exact_dedup(items)
    dedup, dup_map = near_dedup(canon, threshold=0.80)
    # the two patent rows (AAPL, KODK, near-identical body) collapse to one representative
    assert len(dedup) < len(canon)
    rep_companies = set().union(*dedup["company_ids"])
    assert {"AAPL", "KODK"} <= rep_companies
    assert set(dup_map.columns) == {"member_id", "rep_id"}
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_dedup.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.dedup`).

- [ ] **Step 3: Implement `pipeline/dedup.py`**

```python
import pandas as pd
from pipeline._bc import bc

def exact_dedup(items: pd.DataFrame) -> pd.DataFrame:
    df = items.sort_values("published_at", kind="stable")
    agg = (df.groupby("item_id", sort=False)
             .agg(company_ids=("company_id", lambda s: sorted(set(s))),
                  title=("title", "first"), body=("body", "first"), url=("url", "first"),
                  canonical_url=("canonical_url", "first"),
                  published_at=("published_at", "first"),
                  ingestion_date=("ingestion_date", "first"))
             .reset_index())
    return agg

def near_dedup(canon: pd.DataFrame, threshold: float = 0.85) -> tuple[pd.DataFrame, pd.DataFrame]:
    mh = bc.compute_minhashes(canon)
    marked = bc.minhash_near_dups(canon, mh, threshold=threshold)   # adds is_duplicate/duplicate_of
    dup_map = (marked.loc[marked["is_duplicate"], ["item_id", "duplicate_of"]]
                     .rename(columns={"item_id": "member_id", "duplicate_of": "rep_id"})
                     .reset_index(drop=True))
    member_to_rep = dict(zip(dup_map["member_id"], dup_map["rep_id"]))
    # union company_ids from each dropped member into its representative
    reps = marked[~marked["is_duplicate"]].copy()
    rep_companies = {r: set(cs) for r, cs in zip(reps["item_id"], reps["company_ids"])}
    for _, row in marked[marked["is_duplicate"]].iterrows():
        rep_companies.setdefault(member_to_rep[row["item_id"]], set()).update(row["company_ids"])
    reps["company_ids"] = reps["item_id"].map(lambda i: sorted(rep_companies.get(i, [])))
    keep = ["item_id", "title", "body", "url", "canonical_url", "published_at", "ingestion_date", "company_ids"]
    return reps[keep].reset_index(drop=True), dup_map
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_dedup.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/dedup.py tests/pipeline/test_dedup.py
git commit -m "pipeline: exact + near dedup with cross-company company_ids aggregation"
```

---

## Task 4: `embed_items` — single-vector embeddings with content-hash cache

**Files:**
- Create: `pipeline/embed.py`, `tests/pipeline/test_embed.py`

**Interfaces:**
- Produces: `embed_items(items_dedup, embed_fn, cache_path) -> tuple[np.ndarray, list[str]]` returning `(vectors, item_order)`. `vectors` is L2-normalized float32, row `k` aligned to `item_order[k] == items_dedup.item_id[k]`. `embed_fn(texts: list[str]) -> np.ndarray` is injected (real OpenAI in flow; deterministic stub in tests). Embeddings cached to `cache_path` (pickle dict keyed by `sha256(embed_model_unused|text)`); only cache-missing texts are sent to `embed_fn`.

- [ ] **Step 1: Write failing test**

`tests/pipeline/test_embed.py`:
```python
import numpy as np
from pipeline.load import load_normalized
from pipeline.dedup import exact_dedup, near_dedup
from pipeline.embed import embed_items

def stub_embed(texts):
    # deterministic 8-d vector from a hash of the text — no network
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(rng.standard_normal(8).astype(np.float32))
    return np.stack(out)

def test_embed_aligned_normalized_and_cached(raw_corpus, tmp_path):
    items, _ = load_normalized(raw_corpus)
    dedup, _ = near_dedup(exact_dedup(items), threshold=0.80)
    calls = {"n": 0}
    def counting(texts): calls["n"] += len(texts); return stub_embed(texts)
    vecs, order = embed_items(dedup, counting, tmp_path / "emb.pkl")
    assert vecs.shape == (len(dedup), 8) and order == list(dedup["item_id"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    first = calls["n"]
    embed_items(dedup, counting, tmp_path / "emb.pkl")   # second run: all cached
    assert calls["n"] == first
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_embed.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.embed`).

- [ ] **Step 3: Implement `pipeline/embed.py`**

```python
import hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd

def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def embed_items(items_dedup: pd.DataFrame, embed_fn, cache_path) -> tuple[np.ndarray, list[str]]:
    cache_path = Path(cache_path)
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    texts = [f"{t}\n{b}" for t, b in zip(items_dedup["title"], items_dedup["body"])]
    miss = [t for t in dict.fromkeys(texts) if _key(t) not in cache]
    if miss:
        vecs = embed_fn(miss)
        for t, v in zip(miss, vecs):
            cache[_key(t)] = np.asarray(v, dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))
    M = np.stack([cache[_key(t)] for t in texts]).astype(np.float32)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M, list(items_dedup["item_id"])
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_embed.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/embed.py tests/pipeline/test_embed.py
git commit -m "pipeline: embed_items single-vector with content-hash cache"
```

---

## Task 5: `generate_candidates` — cross-company sliding-window exact kNN

**Files:**
- Create: `pipeline/candidates.py`, `tests/pipeline/test_candidates.py`

**Interfaces:**
- Produces: `generate_candidates(items_dedup, vectors, item_order, *, window_hours, cosine_floor) -> pd.DataFrame` with columns `a_id, b_id, cosine` (one row per unordered pair, `a_id` is the earlier item). Pairs are within `±window_hours` of `published_at` and `cosine >= cosine_floor`. Cross-company: company is never filtered.

- [ ] **Step 1: Write failing test**

`tests/pipeline/test_candidates.py`:
```python
import numpy as np, pandas as pd
from pipeline.candidates import generate_candidates

def _items(ids, times):
    return pd.DataFrame({"item_id": ids,
                         "published_at": pd.to_datetime(times, utc=True)})

def test_window_and_floor_and_cross_company():
    ids = ["x", "y", "z"]
    items = _items(ids, ["2012-01-01T00:00Z", "2012-01-01T01:00Z", "2012-01-05T00:00Z"])
    v = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)   # x,y identical; z far in time
    cands = generate_candidates(items, v, ids, window_hours=72, cosine_floor=0.5)
    pairs = set(zip(cands.a_id, cands.b_id))
    assert ("x", "y") in pairs        # in-window, high cosine
    assert ("x", "z") not in pairs and ("y", "z") not in pairs   # out of window
    assert (cands["cosine"] >= 0.5).all()
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_candidates.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.candidates`).

- [ ] **Step 3: Implement `pipeline/candidates.py`**

```python
import numpy as np, pandas as pd

def generate_candidates(items_dedup, vectors, item_order, *, window_hours, cosine_floor) -> pd.DataFrame:
    order = pd.Series(range(len(item_order)), index=item_order)
    df = items_dedup.set_index("item_id").loc[item_order].reset_index()
    t = df["published_at"].values.astype("datetime64[ns]")
    sort = np.argsort(t, kind="stable")
    t_s = t[sort]
    V = vectors[sort]
    ids_s = [item_order[i] for i in sort]
    win = np.timedelta64(int(window_hours), "h")
    rows = []
    for i in range(len(ids_s)):
        hi = np.searchsorted(t_s, t_s[i] + win, side="right")
        if hi <= i + 1:
            continue
        block = V[i + 1:hi]
        cs = block @ V[i]
        for off, c in enumerate(cs):
            if c >= cosine_floor:
                rows.append((ids_s[i], ids_s[i + 1 + off], float(c)))
    return pd.DataFrame(rows, columns=["a_id", "b_id", "cosine"])
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_candidates.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/candidates.py tests/pipeline/test_candidates.py
git commit -m "pipeline: cross-company sliding-window exact kNN candidate generation"
```

---

## Task 6: `features_for_candidates` — 7 fusion features per pair

**Files:**
- Create: `pipeline/features.py`, `tests/pipeline/test_features.py`

**Interfaces:**
- Consumes: candidates from Task 5; `bc.pair_features`, `bc.compute_minhashes`.
- Produces: `features_for_candidates(items_dedup, candidates) -> pd.DataFrame` with columns `a_id, b_id` + the 7 named features `["cosine","title_jac","minhash_jac","dt_days","len_ratio","num_mismatch","caps_mismatch"]` (matching `bc.FusionGate.features` order).

- [ ] **Step 1: Write failing test**

`tests/pipeline/test_features.py`:
```python
import numpy as np, pandas as pd
from pipeline.features import features_for_candidates, FEATURE_NAMES

def test_features_shape_and_names():
    items = pd.DataFrame({
        "item_id": ["a", "b"], "title": ["Apple wins", "Apple wins appeal"],
        "body": ["body one text", "body two text"],
        "published_at": pd.to_datetime(["2012-01-01T00:00Z", "2012-01-01T02:00Z"], utc=True)})
    cands = pd.DataFrame({"a_id": ["a"], "b_id": ["b"], "cosine": [0.81]})
    feats = features_for_candidates(items, cands)
    assert list(feats.columns) == ["a_id", "b_id"] + FEATURE_NAMES
    assert abs(feats.iloc[0]["cosine"] - 0.81) < 1e-9
    assert 0.0 <= feats.iloc[0]["title_jac"] <= 1.0
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_features.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.features`).

- [ ] **Step 3: Implement `pipeline/features.py`**

```python
import pandas as pd
from pipeline._bc import bc

FEATURE_NAMES = ["cosine", "title_jac", "minhash_jac", "dt_days", "len_ratio", "num_mismatch", "caps_mismatch"]

def features_for_candidates(items_dedup: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    by_id = items_dedup.set_index("item_id")
    minhashes = bc.compute_minhashes(items_dedup)
    rows = []
    for a, b, cos in zip(candidates["a_id"], candidates["b_id"], candidates["cosine"]):
        ra, rb = by_id.loc[a], by_id.loc[b]
        ra_d = {"title": ra["title"], "body": ra["body"], "item_id": a, "published_at": ra["published_at"]}
        rb_d = {"title": rb["title"], "body": rb["body"], "item_id": b, "published_at": rb["published_at"]}
        feats = bc.pair_features(ra_d, rb_d, float(cos), minhashes=minhashes)
        rows.append([a, b, *feats])
    return pd.DataFrame(rows, columns=["a_id", "b_id"] + FEATURE_NAMES)
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_features.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/features.py tests/pipeline/test_features.py
git commit -m "pipeline: 7 fusion features per candidate pair (reuses bc.pair_features)"
```

---

## Task 7: `gate_decisions` — auto_same | gray | auto_reject

**Files:**
- Create: `pipeline/gate.py`, `tests/pipeline/test_gate.py`

**Interfaces:**
- Consumes: features from Task 6; `bc.FusionGate`.
- Produces: `gate_decisions(features, gate) -> pd.DataFrame` columns `a_id, b_id, cosine, prob, decision` where `decision ∈ {"auto_same","gray","auto_reject"}`. `gate` is a `bc.FusionGate`.

- [ ] **Step 1: Write failing test (synthetic gate)**

`tests/pipeline/test_gate.py`:
```python
import json, numpy as np, pandas as pd
from pipeline._bc import bc
from pipeline.features import FEATURE_NAMES
from pipeline.gate import gate_decisions

def _synthetic_gate(tmp_path):
    spec = {"features": FEATURE_NAMES,
            "coef": [6.0] + [0.0]*6, "intercept": -3.0,
            "scaler_mean": [0.0]*7, "scaler_scale": [1.0]*7,
            "gates": {"p_high": 0.9, "p_low": 0.1}}
    p = tmp_path / "gate.json"; p.write_text(json.dumps(spec))
    return bc.FusionGate.load(p)

def test_three_way_decision(tmp_path):
    gate = _synthetic_gate(tmp_path)
    feats = pd.DataFrame([["a","b",0.95]+[0]*6, ["c","d",0.50]+[0]*6, ["e","f",0.05]+[0]*6],
                         columns=["a_id","b_id"]+FEATURE_NAMES)
    out = gate_decisions(feats, gate)
    assert out.set_index("a_id").loc["a","decision"] == "auto_same"
    assert out.set_index("a_id").loc["c","decision"] == "gray"
    assert out.set_index("a_id").loc["e","decision"] == "auto_reject"
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_gate.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.gate`).

- [ ] **Step 3: Implement `pipeline/gate.py`**

```python
import pandas as pd
from pipeline.features import FEATURE_NAMES

def gate_decisions(features: pd.DataFrame, gate) -> pd.DataFrame:
    rows = []
    for _, r in features.iterrows():
        feats = [float(r[f]) for f in gate.features]
        p = float(gate.prob(feats))
        decision = "auto_same" if p >= gate.p_high else ("auto_reject" if p < gate.p_low else "gray")
        rows.append((r["a_id"], r["b_id"], float(r["cosine"]), p, decision))
    return pd.DataFrame(rows, columns=["a_id", "b_id", "cosine", "prob", "decision"])
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_gate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/gate.py tests/pipeline/test_gate.py
git commit -m "pipeline: gate_decisions three-way (auto_same/gray/auto_reject)"
```

---

## Task 8: `judge_gray_pairs` — full-body LLM judge on gray pairs

**Files:**
- Create: `pipeline/judge.py`, `tests/pipeline/test_judge.py`

**Interfaces:**
- Consumes: gated pairs from Task 7; `bc.TwoTierJudge`.
- Produces: `judge_gray_pairs(gated, items_dedup, respond_fn, *, base_model, escalation_model, escalation_band, budget, cache_dir) -> pd.DataFrame` columns `a_id, b_id, same` (bool). Only `decision == "gray"` rows are judged. `respond_fn(model, prompt, effort) -> {"verdict","reason"}` is injected (async). If `budget` is set and gray count exceeds it, the overflow (deterministic order) is set `same=False` and a warning logged. Full-body judge text (`judge_text_mode="full_body"`). Synchronous wrapper runs the async core via `asyncio.run`.

- [ ] **Step 1: Write failing test (stub judge, no network)**

`tests/pipeline/test_judge.py`:
```python
import pandas as pd
from pipeline.judge import judge_gray_pairs

def _items():
    return pd.DataFrame({"item_id": ["a","b","c","d"],
                         "title": ["Apple patent","Apple patent win","Tesla battery","Greek vote"],
                         "body": ["apple kodak ruling","apple kodak ruling appeal","tesla cells","greece election"],
                         "published_at": pd.to_datetime(["2012-01-01Z"]*4, utc=True, format="mixed")})

async def stub_respond(model, prompt, effort):
    same = "apple" in prompt.lower() and prompt.lower().count("apple") >= 2
    return {"verdict": "SAME" if same else "DIFFERENT", "reason": "stub"}

def test_only_gray_judged_and_full_body(tmp_path):
    gated = pd.DataFrame({"a_id": ["a","c"], "b_id": ["b","d"], "cosine": [0.7, 0.6],
                          "prob": [0.5, 0.5], "decision": ["gray", "gray"]})
    out = judge_gray_pairs(gated, _items(), stub_respond, base_model="m",
                           escalation_model=None, escalation_band=(0,0), budget=None,
                           cache_dir=str(tmp_path))
    res = dict(zip(zip(out.a_id, out.b_id), out.same))
    assert res[("a","b")] is True and res[("c","d")] is False

def test_budget_caps_judging(tmp_path):
    gated = pd.DataFrame({"a_id": ["a","c"], "b_id": ["b","d"], "cosine": [0.7,0.6],
                          "prob": [0.5,0.5], "decision": ["gray","gray"]})
    out = judge_gray_pairs(gated, _items(), stub_respond, base_model="m",
                           escalation_model=None, escalation_band=(0,0), budget=1,
                           cache_dir=str(tmp_path))
    assert int(out["same"].sum()) <= 1     # at most the budgeted pair can be SAME
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_judge.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.judge`).

- [ ] **Step 3: Implement `pipeline/judge.py`**

```python
import asyncio, logging
import pandas as pd
from pipeline._bc import bc

log = logging.getLogger("pipeline.judge")

def judge_gray_pairs(gated, items_dedup, respond_fn, *, base_model, escalation_model,
                     escalation_band, budget, cache_dir) -> pd.DataFrame:
    gray = gated[gated["decision"] == "gray"].sort_values(["a_id", "b_id"]).reset_index(drop=True)
    by_id = items_dedup.set_index("item_id")
    judge = bc.TwoTierJudge(respond_fn, base_model=base_model, escalation_model=escalation_model,
                            escalation_band=tuple(escalation_band), judge_text_mode="full_body",
                            prompt_version="v5_fullbody", cache_dir=cache_dir)

    capped = gray if budget is None else gray.iloc[:budget]
    overflow = gray.iloc[len(capped):]
    if len(overflow):
        log.warning("judge budget %s exceeded; %d gray pairs auto-rejected", budget, len(overflow))

    async def run():
        sem = asyncio.Semaphore(8)
        async def one(a, b, cos):
            ra, rb = by_id.loc[a], by_id.loc[b]
            ra_d = {"item_id": a, "title": ra["title"], "body": ra["body"], "published_at": ra["published_at"]}
            rb_d = {"item_id": b, "title": rb["title"], "body": rb["body"], "published_at": rb["published_at"]}
            async with sem:
                return await judge.judge_same(ra_d, rb_d, None, None, float(cos))
        return await asyncio.gather(*[one(a, b, c) for a, b, c in
                                      zip(capped.a_id, capped.b_id, capped.cosine)])

    sames = asyncio.run(run()) if len(capped) else []
    rows = list(zip(capped.a_id, capped.b_id, sames)) + \
           [(a, b, False) for a, b in zip(overflow.a_id, overflow.b_id)]
    return pd.DataFrame(rows, columns=["a_id", "b_id", "same"])
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_judge.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/judge.py tests/pipeline/test_judge.py
git commit -m "pipeline: judge_gray_pairs (full-body, disk-cached, budget cap)"
```

---

## Task 9: `assemble_stories` — union-find + cohesion split

**Files:**
- Create: `pipeline/cluster.py`, `tests/pipeline/test_cluster.py`

**Interfaces:**
- Produces:
  - `connected_components(item_ids: list[str], edges: list[tuple[str,str]]) -> dict[str,int]` — item_id → component index (deterministic: components sorted by min item_id).
  - `assemble_stories(gated, judged, item_order, vectors, *, split_threshold) -> pd.DataFrame` columns `item_id, story_idx`. Edges = `auto_same` pairs ∪ judged-SAME pairs. After union-find, components whose min member-to-centroid cosine `< split_threshold` are re-split using only edges with pair cosine `>= split_threshold`. All representative item_ids appear (singletons get their own `story_idx`).

- [ ] **Step 1: Write failing test (incl. transitive over-merge split)**

`tests/pipeline/test_cluster.py`:
```python
import numpy as np, pandas as pd
from pipeline.cluster import connected_components, assemble_stories

def test_connected_components_basic():
    cc = connected_components(["a","b","c","d"], [("a","b"),("b","c")])
    assert cc["a"] == cc["b"] == cc["c"] and cc["d"] != cc["a"]

def test_assemble_splits_weak_bridge():
    # a~b tight, c~d tight, b-c a weak bridge. Cohesion split must separate {a,b} from {c,d}.
    order = ["a","b","c","d"]
    V = np.array([[1,0],[1,0],[0,1],[0,1]], dtype=np.float32)
    gated = pd.DataFrame({"a_id":["a","b","c"],"b_id":["b","c","d"],
                          "cosine":[0.99,0.10,0.99],"prob":[1,1,1],
                          "decision":["auto_same","auto_same","auto_same"]})
    judged = pd.DataFrame(columns=["a_id","b_id","same"])
    out = assemble_stories(gated, judged, order, V, split_threshold=0.55)
    s = dict(zip(out.item_id, out.story_idx))
    assert s["a"] == s["b"] and s["c"] == s["d"] and s["a"] != s["c"]

def test_singletons_get_their_own_story():
    order = ["a","b","z"]
    V = np.array([[1,0],[1,0],[0,1]], dtype=np.float32)
    gated = pd.DataFrame({"a_id":["a"],"b_id":["b"],"cosine":[0.99],"prob":[1],"decision":["auto_same"]})
    out = assemble_stories(gated, pd.DataFrame(columns=["a_id","b_id","same"]), order, V, split_threshold=0.55)
    assert out["story_idx"].nunique() == 2 and len(out) == 3
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_cluster.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.cluster`).

- [ ] **Step 3: Implement `connected_components` + `assemble_stories` in `pipeline/cluster.py`**

```python
import numpy as np, pandas as pd

def connected_components(item_ids, edges) -> dict:
    parent = {i: i for i in item_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    comps = {}
    for i in item_ids:
        comps.setdefault(find(i), []).append(i)
    ordered = sorted(comps.values(), key=lambda g: min(g))
    return {iid: idx for idx, grp in enumerate(ordered) for iid in grp}

def _same_edges(gated, judged):
    edges = [(a, b) for a, b, d in zip(gated.a_id, gated.b_id, gated.decision) if d == "auto_same"]
    if len(judged):
        edges += [(a, b) for a, b, s in zip(judged.a_id, judged.b_id, judged.same) if s]
    return edges

def assemble_stories(gated, judged, item_order, vectors, *, split_threshold) -> pd.DataFrame:
    pos = {iid: k for k, iid in enumerate(item_order)}
    edges = _same_edges(gated, judged)
    comp = connected_components(list(item_order), edges)
    members = {}
    for iid, c in comp.items():
        members.setdefault(c, []).append(iid)

    # cohesion split for low-cohesion components, using only strong edges
    strong = [(a, b) for a, b, c in zip(gated.a_id, gated.b_id, gated.cosine) if c >= split_threshold]
    strong_set = {tuple(sorted(e)) for e in strong}
    final = {}
    next_idx = 0
    for c, ids in sorted(members.items()):
        if len(ids) >= 3:
            cen = vectors[[pos[i] for i in ids]].mean(axis=0)
            cen /= (np.linalg.norm(cen) + 1e-9)
            cohesion = min(float(vectors[pos[i]] @ cen) for i in ids)
            if cohesion < split_threshold:
                sub_edges = [(a, b) for a in ids for b in ids if a < b and tuple(sorted((a, b))) in strong_set]
                sub = connected_components(ids, sub_edges)
                for iid in ids:
                    final[iid] = next_idx + sub[iid]
                next_idx += (max(sub.values()) + 1) if sub else 1
                continue
        for iid in ids:
            final[iid] = next_idx
        next_idx += 1
    # renumber deterministically by min item_id
    groups = {}
    for iid, s in final.items():
        groups.setdefault(s, []).append(iid)
    relabel = {s: i for i, s in enumerate(sorted(groups, key=lambda s: min(groups[s])))}
    return pd.DataFrame([(iid, relabel[final[iid]]) for iid in item_order],
                        columns=["item_id", "story_idx"])
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_cluster.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/cluster.py tests/pipeline/test_cluster.py
git commit -m "pipeline: assemble_stories union-find + cohesion split"
```

---

## Task 10: `residual_cluster` — optional single-vector HDBSCAN over singletons

**Files:**
- Modify: `pipeline/cluster.py` (add function)
- Modify: `tests/pipeline/test_cluster.py` (add test)

**Interfaces:**
- Produces: `residual_cluster(assignments, item_order, vectors, *, min_cluster_size=2, enabled=True) -> pd.DataFrame` (same columns `item_id, story_idx`). When `enabled`, HDBSCAN (precomputed cosine distance) over the vectors of current singletons; items sharing an HDBSCAN cluster are merged into a new shared `story_idx`. When `enabled=False`, returns `assignments` unchanged.

- [ ] **Step 1: Add failing test**

Append to `tests/pipeline/test_cluster.py`:
```python
from pipeline.cluster import residual_cluster

def test_residual_disabled_is_noop():
    a = pd.DataFrame({"item_id": ["a","b"], "story_idx": [0, 1]})
    out = residual_cluster(a, ["a","b"], np.array([[1,0],[1,0]], dtype=np.float32), enabled=False)
    assert out.equals(a)

def test_residual_merges_tight_singletons():
    order = ["a","b","c"]
    V = np.array([[1,0],[1,0],[0,1]], dtype=np.float32)   # a,b identical singletons
    a = pd.DataFrame({"item_id": order, "story_idx": [0, 1, 2]})
    out = residual_cluster(a, order, V, min_cluster_size=2, enabled=True)
    s = dict(zip(out.item_id, out.story_idx))
    assert s["a"] == s["b"] and s["a"] != s["c"]
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_cluster.py -q`
Expected: FAIL (`cannot import name 'residual_cluster'`).

- [ ] **Step 3: Implement `residual_cluster`**

Append to `pipeline/cluster.py`:
```python
def residual_cluster(assignments, item_order, vectors, *, min_cluster_size=2, enabled=True):
    if not enabled:
        return assignments
    pos = {iid: k for k, iid in enumerate(item_order)}
    sizes = assignments.groupby("story_idx")["item_id"].transform("size")
    singles = list(assignments[sizes == 1]["item_id"])
    if len(singles) < min_cluster_size:
        return assignments
    import hdbscan
    from sklearn.metrics.pairwise import cosine_distances
    M = vectors[[pos[i] for i in singles]]
    dist = cosine_distances(M).astype(np.float64)
    labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size,
                             min_samples=1).fit_predict(dist)
    out = assignments.set_index("item_id")["story_idx"].to_dict()
    nxt = max(out.values()) + 1
    new_label = {}
    for iid, lab in zip(singles, labels):
        if lab < 0:
            continue
        new_label.setdefault(lab, nxt + lab)
        out[iid] = new_label[lab]
    res = pd.DataFrame([(i, out[i]) for i in item_order], columns=["item_id", "story_idx"])
    # renumber compactly + deterministically
    groups = {}
    for i, s in zip(res.item_id, res.story_idx):
        groups.setdefault(s, []).append(i)
    relabel = {s: k for k, s in enumerate(sorted(groups, key=lambda s: min(groups[s])))}
    res["story_idx"] = res["story_idx"].map(relabel)
    return res
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_cluster.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/cluster.py tests/pipeline/test_cluster.py
git commit -m "pipeline: optional single-vector HDBSCAN residual clustering"
```

---

## Task 11: `persist_outputs` — expand reps to every row, stories table, metrics

**Files:**
- Create: `pipeline/persist.py`, `tests/pipeline/test_persist.py`

**Interfaces:**
- Produces: `persist_outputs(assignments, items_loaded, items_dedup, dup_map) -> tuple[pd.DataFrame, pd.DataFrame, dict]` returning `(items_scored, stories, metrics)`.
  - `items_scored`: one row per ORIGINAL loaded item — columns `item_id, company_id, story_idx, title, url, published_at, ingestion_date, is_representative, near_dup_of`.
  - `stories`: columns `story_idx, size, company_ids, first_publish, last_publish, representative_item_id, representative_title`.
  - `metrics`: counts dict.
- Mapping: each loaded `item_id` → representative via `dup_map` (member_id→rep_id), default itself; `story_idx = assignments[rep]`.

- [ ] **Step 1: Write failing test**

`tests/pipeline/test_persist.py`:
```python
import pandas as pd
from pipeline.persist import persist_outputs

def test_every_row_scored_and_stories_aggregate():
    loaded = pd.DataFrame({
        "item_id": ["a","b","c"], "company_id": ["AAPL","KODK","TSLA"],
        "title": ["A","B","C"], "url": ["ua","ub","uc"],
        "published_at": pd.to_datetime(["2012-01-01Z","2012-01-01Z","2012-01-02Z"], utc=True, format="mixed"),
        "ingestion_date": pd.to_datetime(["2012-01-01Z"]*3, utc=True, format="mixed")})
    dedup = pd.DataFrame({"item_id": ["a","c"], "company_ids": [["AAPL","KODK"], ["TSLA"]],
                          "title": ["A","C"],
                          "published_at": pd.to_datetime(["2012-01-01Z","2012-01-02Z"], utc=True, format="mixed")})
    dup_map = pd.DataFrame({"member_id": ["b"], "rep_id": ["a"]})         # b collapsed into a
    assignments = pd.DataFrame({"item_id": ["a","c"], "story_idx": [0, 1]})
    scored, stories, metrics = persist_outputs(assignments, loaded, dedup, dup_map)
    assert len(scored) == 3 and scored["story_idx"].notna().all()
    assert dict(zip(scored.item_id, scored.story_idx))["b"] == 0          # inherits a's story
    s0 = stories.set_index("story_idx").loc[0]
    assert set(s0["company_ids"]) == {"AAPL","KODK"} and s0["size"] == 2
    assert metrics["n_items"] == 3 and metrics["n_stories"] == 2
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_persist.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.persist`).

- [ ] **Step 3: Implement `pipeline/persist.py`**

```python
import pandas as pd

def persist_outputs(assignments, items_loaded, items_dedup, dup_map):
    member_to_rep = dict(zip(dup_map["member_id"], dup_map["rep_id"])) if len(dup_map) else {}
    story_of = dict(zip(assignments["item_id"], assignments["story_idx"]))

    scored = items_loaded.copy()
    scored["rep_id"] = scored["item_id"].map(lambda i: member_to_rep.get(i, i))
    scored["story_idx"] = scored["rep_id"].map(story_of)
    scored["is_representative"] = scored["item_id"] == scored["rep_id"]
    scored["near_dup_of"] = scored.apply(lambda r: None if r["is_representative"] else r["rep_id"], axis=1)
    scored = scored[["item_id", "company_id", "story_idx", "title", "url",
                     "published_at", "ingestion_date", "is_representative", "near_dup_of"]]

    rep_title = dict(zip(items_dedup["item_id"], items_dedup["title"]))
    rows = []
    for sidx, grp in scored.groupby("story_idx"):
        reps = grp[grp["is_representative"]]
        rep_id = reps.sort_values("published_at").iloc[0]["item_id"] if len(reps) else grp.iloc[0]["item_id"]
        rows.append(dict(story_idx=int(sidx), size=int(len(grp)),
                         company_ids=sorted(set(grp["company_id"])),
                         first_publish=grp["published_at"].min(), last_publish=grp["published_at"].max(),
                         representative_item_id=rep_id, representative_title=rep_title.get(rep_id, "")))
    stories = pd.DataFrame(rows).sort_values("story_idx").reset_index(drop=True)
    metrics = dict(n_items=int(len(scored)), n_representatives=int(scored["is_representative"].sum()),
                   n_stories=int(stories["story_idx"].nunique()),
                   n_multi_item=int((stories["size"] > 1).sum()))
    return scored, stories, metrics
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_persist.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/persist.py tests/pipeline/test_persist.py
git commit -m "pipeline: persist_outputs (row expansion, stories table, metrics)"
```

---

## Task 12: Postgres schema + `load_to_postgres` (versioned batch)

**Files:**
- Create: `pipeline/schema.sql`, `pipeline/db.py`, `tests/pipeline/test_db.py`

**Interfaces:**
- Produces: `load_to_postgres(engine, items_scored, stories, run_meta) -> bool`. `engine` is a SQLAlchemy Engine. `run_meta` keys: `run_id, config_hash, source_snapshot, n_items, n_stories`. Behavior: idempotent (returns `False` without writing if `run_id` already `succeeded`); otherwise one transaction — insert `clustering_runs` (succeeded), generate deterministic `story_id = uuid5(run_id:story_idx)`, bulk insert `stories` + `story_items`, clear prior `is_current`, set this run `is_current=true`. Returns `True` on write.
- `ensure_schema(engine)` executes `schema.sql` (idempotent `CREATE TABLE IF NOT EXISTS`).

- [ ] **Step 1: Placeholder DDL `pipeline/schema.sql`**

```sql
-- PLACEHOLDER schema (replace with the real ERD when available)
CREATE TABLE IF NOT EXISTS clustering_runs (
  run_id          text PRIMARY KEY,
  config_hash     text,
  source_snapshot text,
  started_at      timestamptz DEFAULT now(),
  finished_at     timestamptz,
  n_items         int,
  n_stories       int,
  is_current      boolean DEFAULT false,
  status          text
);
CREATE TABLE IF NOT EXISTS stories (
  story_id        uuid PRIMARY KEY,
  run_id          text REFERENCES clustering_runs(run_id),
  size            int,
  company_ids     jsonb,
  first_publish   timestamptz,
  last_publish    timestamptz,
  representative_item_id text,
  representative_title   text
);
CREATE TABLE IF NOT EXISTS story_items (
  id              bigserial PRIMARY KEY,
  run_id          text REFERENCES clustering_runs(run_id),
  story_id        uuid REFERENCES stories(story_id),
  item_id         text,
  company_id      text,
  url             text,
  publish_date    timestamptz,
  ingestion_date  timestamptz,
  is_representative boolean,
  near_dup_of     text
);
CREATE INDEX IF NOT EXISTS ix_story_items_run_story ON story_items(run_id, story_id);
CREATE INDEX IF NOT EXISTS ix_story_items_item ON story_items(item_id);
```

- [ ] **Step 2: Write failing test (ephemeral Postgres)**

`tests/pipeline/test_db.py`:
```python
import uuid, pandas as pd, pytest
sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("pytest_postgresql")
from sqlalchemy import create_engine, text
from pytest_postgresql import factories
from pipeline.db import ensure_schema, load_to_postgres

postgresql_proc = factories.postgresql_proc()
postgresql = factories.postgresql("postgresql_proc")

def _engine(postgresql):
    p = postgresql.info
    return create_engine(f"postgresql+psycopg://{p.user}:@{p.host}:{p.port}/{p.dbname}")

def _frames():
    scored = pd.DataFrame({"item_id":["a","b"],"company_id":["AAPL","KODK"],"story_idx":[0,0],
        "title":["A","B"],"url":["ua","ub"],
        "published_at":pd.to_datetime(["2012-01-01Z","2012-01-01Z"],utc=True,format="mixed"),
        "ingestion_date":pd.to_datetime(["2012-01-01Z"]*2,utc=True,format="mixed"),
        "is_representative":[True,False],"near_dup_of":[None,"a"]})
    stories = pd.DataFrame({"story_idx":[0],"size":[2],"company_ids":[["AAPL","KODK"]],
        "first_publish":pd.to_datetime(["2012-01-01Z"],utc=True,format="mixed"),
        "last_publish":pd.to_datetime(["2012-01-01Z"],utc=True,format="mixed"),
        "representative_item_id":["a"],"representative_title":["A"]})
    return scored, stories

def test_versioned_load_and_idempotency(postgresql):
    eng = _engine(postgresql); ensure_schema(eng)
    scored, stories = _frames()
    meta = dict(run_id="run_x", config_hash="h", source_snapshot="s", n_items=2, n_stories=1)
    assert load_to_postgres(eng, scored, stories, meta) is True
    assert load_to_postgres(eng, scored, stories, meta) is False     # idempotent
    with eng.begin() as c:
        assert c.execute(text("select count(*) from story_items")).scalar() == 2
        assert c.execute(text("select is_current from clustering_runs where run_id='run_x'")).scalar() is True
```

- [ ] **Step 3: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_db.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.db`). (If `postgresql_proc` can't start a local PG, the test is skipped — note this in the commit and verify on a machine with PG.)

- [ ] **Step 4: Implement `pipeline/db.py`**

```python
import json, uuid
from pathlib import Path
from sqlalchemy import text

_NS = uuid.UUID("00000000-0000-0000-0000-0000000c1057")

def ensure_schema(engine):
    ddl = (Path(__file__).parent / "schema.sql").read_text()
    with engine.begin() as c:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            c.execute(text(stmt))

def _story_uuid(run_id, story_idx) -> str:
    return str(uuid.uuid5(_NS, f"{run_id}:{int(story_idx)}"))

def load_to_postgres(engine, items_scored, stories, run_meta) -> bool:
    run_id = run_meta["run_id"]
    with engine.begin() as c:
        done = c.execute(text("select status from clustering_runs where run_id=:r"),
                         {"r": run_id}).scalar()
        if done == "succeeded":
            return False
        c.execute(text("insert into clustering_runs(run_id,config_hash,source_snapshot,finished_at,"
                       "n_items,n_stories,is_current,status) values(:r,:h,:s,now(),:ni,:ns,false,'succeeded') "
                       "on conflict (run_id) do update set status='succeeded'"),
                  {"r": run_id, "h": run_meta["config_hash"], "s": run_meta["source_snapshot"],
                   "ni": run_meta["n_items"], "ns": run_meta["n_stories"]})
        for _, s in stories.iterrows():
            c.execute(text("insert into stories(story_id,run_id,size,company_ids,first_publish,"
                           "last_publish,representative_item_id,representative_title) values"
                           "(:id,:r,:sz,:cj,:fp,:lp,:ri,:rt)"),
                      {"id": _story_uuid(run_id, s["story_idx"]), "r": run_id, "sz": int(s["size"]),
                       "cj": json.dumps(list(s["company_ids"])), "fp": s["first_publish"],
                       "lp": s["last_publish"], "ri": s["representative_item_id"],
                       "rt": s["representative_title"]})
        for _, it in items_scored.iterrows():
            c.execute(text("insert into story_items(run_id,story_id,item_id,company_id,url,publish_date,"
                           "ingestion_date,is_representative,near_dup_of) values"
                           "(:r,:sid,:iid,:co,:u,:pd,:ing,:isr,:nd)"),
                      {"r": run_id, "sid": _story_uuid(run_id, it["story_idx"]), "iid": it["item_id"],
                       "co": it["company_id"], "u": it["url"], "pd": it["published_at"],
                       "ing": it["ingestion_date"], "isr": bool(it["is_representative"]),
                       "nd": it["near_dup_of"]})
        c.execute(text("update clustering_runs set is_current=false where is_current=true"))
        c.execute(text("update clustering_runs set is_current=true where run_id=:r"), {"r": run_id})
    return True
```

- [ ] **Step 5: Run — verify pass (or skip if no local PG)**

Run: `.venv/bin/python -m pytest tests/pipeline/test_db.py -q`
Expected: PASS (or SKIPPED if `pytest-postgresql` can't launch a server — then run on a PG-capable host before merge).

- [ ] **Step 6: Commit**

```bash
git add pipeline/schema.sql pipeline/db.py tests/pipeline/test_db.py
git commit -m "pipeline: versioned-batch Postgres loader + placeholder schema"
```

---

## Task 13: `evaluate_stories` — optional pairwise QA gate

**Files:**
- Create: `pipeline/evaluate.py`, `tests/pipeline/test_evaluate.py`

**Interfaces:**
- Produces: `evaluate_stories(items_scored, eval_df) -> dict` with `precision, recall, f1, tp, fp, fn, n_pairs, n_same`. `eval_df` has columns `item_a_id, item_b_id, final_label` (SAME/DIFFERENT). Prediction = both items share a `story_idx`. Pairs whose items are absent from `items_scored` are skipped.

- [ ] **Step 1: Write failing test**

`tests/pipeline/test_evaluate.py`:
```python
import pandas as pd
from pipeline.evaluate import evaluate_stories

def test_pairwise_scoring():
    scored = pd.DataFrame({"item_id":["a","b","c"], "story_idx":[0,0,1]})
    ev = pd.DataFrame({"item_a_id":["a","a"], "item_b_id":["b","c"], "final_label":["SAME","SAME"]})
    m = evaluate_stories(scored, ev)
    assert m["tp"] == 1 and m["fn"] == 1 and m["recall"] == 0.5 and m["f1"] == round(2*1*0.5/1.5, 4)
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_evaluate.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.evaluate`).

- [ ] **Step 3: Implement `pipeline/evaluate.py`**

```python
import pandas as pd

def evaluate_stories(items_scored: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    story = dict(zip(items_scored["item_id"], items_scored["story_idx"]))
    tp = fp = fn = 0
    n = 0
    for a, b, lab in zip(eval_df["item_a_id"], eval_df["item_b_id"], eval_df["final_label"]):
        if a not in story or b not in story:
            continue
        n += 1
        pred_same = story[a] == story[b]
        gold_same = str(lab).upper() == "SAME"
        if pred_same and gold_same: tp += 1
        elif pred_same and not gold_same: fp += 1
        elif not pred_same and gold_same: fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                tp=tp, fp=fp, fn=fn, n_pairs=n, n_same=tp + fn)
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/evaluate.py tests/pipeline/test_evaluate.py
git commit -m "pipeline: evaluate_stories pairwise QA"
```

---

## Task 14: Prefect flow wiring + golden end-to-end + determinism

**Files:**
- Create: `pipeline/flow.py`, `tests/pipeline/test_flow_golden.py`

**Interfaces:**
- Consumes: every pure function above + `RunConfig` + artifacts helpers.
- Produces:
  - Real adapters: `openai_embed_fn(cfg)` → returns `embed_fn(texts)->np.ndarray`; `openai_respond_fn(cfg)` → returns async `respond_fn(model,prompt,effort)->dict` (OpenAI Responses API with `bc.VERDICT_SCHEMA`).
  - `cluster_stories_flow(cfg, df_raw, *, embed_fn=None, respond_fn=None, engine=None) -> dict` — the Prefect `@flow` returning `metrics`. Each stage is a `@task`. When `embed_fn`/`respond_fn`/`engine` are passed they override the real adapters (used by tests). The DB load runs only if an engine (or `cfg.database_url`) is available.

- [ ] **Step 1: Write the golden flow test (stubs + ephemeral PG)**

`tests/pipeline/test_flow_golden.py`:
```python
import json, numpy as np, pandas as pd, pytest
from pipeline.config import RunConfig
from pipeline.features import FEATURE_NAMES
from pipeline.flow import cluster_stories_flow

def stub_embed(texts):
    # cluster by keyword so the golden corpus has a deterministic structure
    out = []
    for t in texts:
        low = t.lower()
        if "apple" in low or "kodak" in low: base = np.array([1, 0, 0], float)
        elif "google" in low or "ftc" in low: base = np.array([0, 1, 0], float)
        else: base = np.array([0, 0, 1], float)
        out.append((base + 1e-3 * np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(3)).astype(np.float32))
    return np.stack(out)

async def stub_respond(model, prompt, effort):
    low = prompt.lower()
    same = (low.count("apple") >= 2) or (low.count("google") >= 2) or (low.count("ftc") >= 2)
    return {"verdict": "SAME" if same else "DIFFERENT", "reason": "stub"}

def _gate(tmp_path):
    spec = {"features": FEATURE_NAMES, "coef": [5.0] + [0.0]*6, "intercept": -2.5,
            "scaler_mean": [0.0]*7, "scaler_scale": [1.0]*7,
            "gates": {"p_high": 0.95, "p_low": 0.05}}
    p = tmp_path / "gate.json"; p.write_text(json.dumps(spec)); return str(p)

def test_golden_flow_produces_expected_stories(raw_corpus, tmp_path):
    cfg = RunConfig(source_uri="mem", gate_path=_gate(tmp_path), runs_dir=str(tmp_path / "runs"),
                    window_hours=72, cosine_floor=0.4, split_threshold=0.5, residual_enabled=False)
    metrics = cluster_stories_flow(cfg, raw_corpus, embed_fn=stub_embed, respond_fn=stub_respond)
    assert metrics["n_items"] == 5            # malformed row dropped
    # Apple/Kodak (cross-company near-dup) form one multi-item story
    assert metrics["n_multi_item"] >= 1

def test_flow_is_deterministic(raw_corpus, tmp_path):
    cfg = RunConfig(source_uri="mem", gate_path=_gate(tmp_path), runs_dir=str(tmp_path / "runs"),
                    cosine_floor=0.4, split_threshold=0.5)
    m1 = cluster_stories_flow(cfg, raw_corpus, embed_fn=stub_embed, respond_fn=stub_respond)
    m2 = cluster_stories_flow(cfg, raw_corpus, embed_fn=stub_embed, respond_fn=stub_respond)
    assert m1 == m2
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_flow_golden.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.flow`).

- [ ] **Step 3: Implement `pipeline/flow.py`**

```python
import json
import numpy as np, pandas as pd
from prefect import flow, task
from pipeline import artifacts as A
from pipeline._bc import bc
from pipeline.load import load_normalized
from pipeline.dedup import exact_dedup, near_dedup
from pipeline.embed import embed_items
from pipeline.candidates import generate_candidates
from pipeline.features import features_for_candidates
from pipeline.gate import gate_decisions
from pipeline.judge import judge_gray_pairs
from pipeline.cluster import assemble_stories, residual_cluster
from pipeline.persist import persist_outputs
from pipeline.evaluate import evaluate_stories

# ---- real adapters (overridable in tests) ------------------------------------
def openai_embed_fn(cfg):
    from openai import OpenAI
    client = OpenAI()
    def embed(texts):
        out = []
        for i in range(0, len(texts), 100):
            r = client.embeddings.create(model=cfg.embed_model, input=texts[i:i+100], dimensions=cfg.embed_dims)
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in r.data)
        return np.stack(out)
    return embed

def openai_respond_fn(cfg):
    import asyncio
    from openai import AsyncOpenAI
    from openai import RateLimitError
    client = AsyncOpenAI()
    async def respond(model, prompt, effort):
        kw = dict(model=model, input=[{"role": "user", "content": prompt}], store=False,
                  text={"format": {"type": "json_schema", "name": "comparison_verdict",
                                   "strict": True, "schema": bc.VERDICT_SCHEMA}})
        if effort is not None:
            kw["reasoning"] = {"effort": effort}
        for attempt in range(6):
            try:
                r = await client.responses.create(**kw)
                try: return json.loads(r.output_text)
                except Exception: return {"verdict": "UNCLEAR", "reason": "unparseable"}
            except RateLimitError:
                await asyncio.sleep(2 * (attempt + 1))
        return {"verdict": "UNCLEAR", "reason": "rate-limited"}
    return respond

# ---- tasks -------------------------------------------------------------------
@task
def t_load(cfg, df_raw):
    items, rejected = load_normalized(df_raw)
    A.write_parquet(items, A.artifact_path(cfg, "00_items.parquet"))
    A.write_parquet(rejected, A.artifact_path(cfg, "rejected.parquet"))
    return items

@task
def t_dedup(cfg, items):
    canon = exact_dedup(items)
    dedup, dup_map = near_dedup(canon)
    A.write_parquet(dedup, A.artifact_path(cfg, "02_items_dedup.parquet"))
    A.write_parquet(dup_map, A.artifact_path(cfg, "02_dup_map.parquet"))
    return dedup, dup_map

@task
def t_embed(cfg, dedup, embed_fn):
    vecs, order = embed_items(dedup, embed_fn, A.artifact_path(cfg, "embeddings.pkl"))
    A.write_npy(vecs, A.artifact_path(cfg, "03_vectors.npy"))
    A.write_parquet(pd.DataFrame({"item_id": order}), A.artifact_path(cfg, "03_item_order.parquet"))
    return vecs, order

@task
def t_candidates(cfg, dedup, vecs, order):
    cands = generate_candidates(dedup, vecs, order, window_hours=cfg.window_hours, cosine_floor=cfg.cosine_floor)
    A.write_parquet(cands, A.artifact_path(cfg, "04_candidates.parquet"))
    return cands

@task
def t_features(cfg, dedup, cands):
    feats = features_for_candidates(dedup, cands)
    A.write_parquet(feats, A.artifact_path(cfg, "05_features.parquet"))
    return feats

@task
def t_gate(cfg, feats):
    gate = bc.FusionGate.load(cfg.gate_path)
    gated = gate_decisions(feats, gate)
    A.write_parquet(gated, A.artifact_path(cfg, "06_gated.parquet"))
    return gated

@task
def t_judge(cfg, gated, dedup, respond_fn):
    judged = judge_gray_pairs(gated, dedup, respond_fn, base_model=cfg.judge_base_model,
                              escalation_model=cfg.judge_escalation_model,
                              escalation_band=cfg.judge_escalation_band, budget=cfg.judge_budget,
                              cache_dir=str(A.artifact_path(cfg, "judge_cache")))
    A.write_parquet(judged, A.artifact_path(cfg, "07_judged.parquet"))
    return judged

@task
def t_cluster(cfg, gated, judged, order, vecs):
    asg = assemble_stories(gated, judged, order, vecs, split_threshold=cfg.split_threshold)
    asg = residual_cluster(asg, order, vecs, enabled=cfg.residual_enabled)
    A.write_parquet(asg, A.artifact_path(cfg, "08_assignments.parquet"))
    return asg

@task
def t_persist(cfg, asg, items, dedup, dup_map):
    scored, stories, metrics = persist_outputs(asg, items, dedup, dup_map)
    A.write_parquet(scored, A.artifact_path(cfg, "items_scored.parquet"))
    A.write_parquet(stories, A.artifact_path(cfg, "stories.parquet"))
    A.write_json(metrics, A.artifact_path(cfg, "metrics.json"))
    return scored, stories, metrics

@flow(name="cluster_stories")
def cluster_stories_flow(cfg, df_raw, *, embed_fn=None, respond_fn=None, engine=None) -> dict:
    embed_fn = embed_fn or openai_embed_fn(cfg)
    respond_fn = respond_fn or openai_respond_fn(cfg)
    items = t_load(cfg, df_raw)
    dedup, dup_map = t_dedup(cfg, items)
    vecs, order = t_embed(cfg, dedup, embed_fn)
    cands = t_candidates(cfg, dedup, vecs, order)
    feats = t_features(cfg, dedup, cands)
    gated = t_gate(cfg, feats)
    judged = t_judge(cfg, gated, dedup, respond_fn)
    asg = t_cluster(cfg, gated, judged, order, vecs)
    scored, stories, metrics = t_persist(cfg, asg, items, dedup, dup_map)

    if cfg.eval_path:
        ev = evaluate_stories(scored, pd.read_csv(cfg.eval_path, comment="#"))
        A.write_json(ev, A.artifact_path(cfg, "metrics_eval.json"))
        metrics["eval"] = ev
        if cfg.eval_min_f1 is not None and ev["f1"] < cfg.eval_min_f1:
            raise ValueError(f"eval F1 {ev['f1']} < gate {cfg.eval_min_f1} — refusing DB load")

    if engine is None and cfg.database_url:
        from sqlalchemy import create_engine
        engine = create_engine(cfg.database_url)
    if engine is not None:
        from pipeline.db import ensure_schema, load_to_postgres
        ensure_schema(engine)
        load_to_postgres(engine, scored, stories,
                         dict(run_id=cfg.run_id, config_hash=cfg.config_hash(),
                              source_snapshot=cfg.source_uri, n_items=metrics["n_items"],
                              n_stories=metrics["n_stories"]))
    return metrics
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_flow_golden.py -q`
Expected: PASS (2 passed). If Prefect emits a server/telemetry warning, set `PREFECT_LOGGING_LEVEL=ERROR` and `PREFECT_API_ENABLE_HTTP2=false`; it does not affect correctness.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/pipeline -q`
Expected: PASS (all tasks' tests; `test_db.py` may SKIP without local PG).

- [ ] **Step 6: Commit**

```bash
git add pipeline/flow.py tests/pipeline/test_flow_golden.py
git commit -m "pipeline: Prefect flow wiring + golden end-to-end + determinism tests"
```

---

## Task 15: CLI entrypoint + README

**Files:**
- Create: `pipeline/__main__.py`, `pipeline/README.md`

**Interfaces:**
- Produces: `python -m pipeline --source <parquet> --gate artifacts/v4/fusion_model_chunk.json [--database-url ...] [--eval ...] [--residual]` → runs `cluster_stories_flow`, prints `metrics.json`.

- [ ] **Step 1: Implement `pipeline/__main__.py`**

```python
import argparse, json
import pandas as pd
from pipeline.config import RunConfig
from pipeline.flow import cluster_stories_flow

def main():
    ap = argparse.ArgumentParser("cluster-stories")
    ap.add_argument("--source", required=True, help="parquet of normalized items")
    ap.add_argument("--gate", required=True)
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--eval-min-f1", type=float, default=None)
    ap.add_argument("--window-hours", type=int, default=72)
    ap.add_argument("--residual", action="store_true")
    a = ap.parse_args()
    cfg = RunConfig(source_uri=a.source, gate_path=a.gate, database_url=a.database_url,
                    eval_path=a.eval, eval_min_f1=a.eval_min_f1, window_hours=a.window_hours,
                    residual_enabled=a.residual)
    df = pd.read_parquet(a.source)
    metrics = cluster_stories_flow(cfg, df)
    print(json.dumps(metrics, indent=2, default=str))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `pipeline/README.md`**

Document: the schema contract, the 14 stages + artifacts, the `RunConfig` knobs, the column mapping, the placeholder ERD note, and the run command. (Prose — no code to test.)

- [ ] **Step 3: Smoke-run on the Bloomberg sample**

Run:
```bash
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet("artifacts/v4/bloomberg_items.parquet").rename(
    columns={"body":"content","published_at":"publish_date"})
df["company_id"] = df["item_clients"].apply(lambda a: list(a)[0] if len(a) else "NA")
df["ingestion_date"] = df["publish_date"]
df[["company_id","title","content","url","publish_date","ingestion_date"]].head(500).to_parquet("/tmp/smoke.parquet")
PY
OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python -m pipeline --source /tmp/smoke.parquet --gate artifacts/v4/fusion_model_chunk.json
```
Expected: prints a `metrics.json` with `n_items≈500`, `n_stories`, `n_multi_item` > 0. (Uses real embeddings + judge; ~$0.05.)

- [ ] **Step 4: Commit**

```bash
git add pipeline/__main__.py pipeline/README.md
git commit -m "pipeline: CLI entrypoint + README"
```

---

## Self-Review

**Spec coverage:** load(1·T2), exact+near dedup w/ company aggregation(T3), single-vec embed(T4), cross-company kNN(T5), features(T6), gate(T7), full-body judge(T8), union-find+cohesion split(T9), optional residual(T10), persist+rows expand+metrics(T11), versioned Postgres + placeholder ERD(T12), optional eval gate(T13), Prefect flow + golden/determinism(T14), CLI(T15). Error handling: quarantine(T2), judge cache+budget(T8), cohesion split(T9), DB idempotency/`is_current`(T12), eval gate refusing DB load(T14). All spec sections map to a task.

**Placeholder scan:** the only "placeholder" is the intentional `schema.sql` ERD (spec-required). README prose in T15 is the one non-code step (acceptable). No TBD/TODO in code steps.

**Type consistency:** `RunConfig` fields, `FEATURE_NAMES` order (T6) = `bc.FusionGate.features` (T7), `decision` enum strings shared T7→T9, `dup_map` columns `member_id/rep_id` (T3) consumed in T11, `assignments` columns `item_id/story_idx` (T9/T10/T11), `run_meta` keys (T12) produced in T14 — all aligned.

---

**Tech debt accepted (YAGNI):** ANN indexing, incremental state, gate calibration, and chunk pathways are intentionally out of scope (spec). The real ERD replaces only `schema.sql` + the column list in `db.py`.
