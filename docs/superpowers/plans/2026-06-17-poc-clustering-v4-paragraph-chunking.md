# POC Clustering v4 — Paragraph Chunking + Judge-Text A/B — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the v4 clustering notebook so the vector path uses body **paragraph chunks** (multi-vector per item) across §6/§8/§10/§11, and measure a judge-text **A/B** (`chunk_pair` vs `full_body`) on accuracy + cost to decide whether chunking beats the v3 single-vector pipeline.

**Architecture:** Pure, high-risk, deterministic logic (chunking, max-pool similarity, eval-seeded sampling, judge-text builders, chunk-cluster→item union) lives in an importable module `scripts/v4_chunking.py` with real pytest TDD. The notebook `story_clustering_poc_v4.ipynb` is authored fresh, imports those helpers, ports the validated v3 pipeline behind a `use_chunking=False` master switch (baseline), then wires the chunk path on top. Three end-to-end runs (`baseline`, `chunk+chunk_pair`, `chunk+full_body`) are recorded in the ledger and charted.

**Tech Stack:** Python 3, Jupyter (driven via Jupyter MCP), pandas, numpy, `text-embedding-3-large` (async cached embedder), `datasketch` MinHash/LSH, `hdbscan`, `umap`, OpenAI judge client, `pytest` for the module, matplotlib/seaborn/plotly for charts.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from `docs/superpowers/specs/2026-06-17-poc-clustering-v4-paragraph-chunking-design.md`.

- **Corpus:** `CONFIG["target_canonical_items"] = 3000`; sampling uses existing `CONFIG["random_seed"]`; corpus = **(all items referenced by `labeled_eval_set.csv`) ∪ (random fill to 3000)**.
- **Chunks are body-only.** Title excluded from vectors. **Title-only / empty-body → title-as-single-chunk fallback.**
- **Master switch** `use_chunking` (default `True`); `False` = v3 single-vector baseline (must stay runnable).
- **§10 similarity** = `max` over (item-chunk, story-chunk) cosine (nearest-chunk); the `argmax` pair is the matched chunk pair.
- **§11 HDBSCAN** clusters chunk vecs → **union items sharing any chunk-cluster**; add a transitive-component-size diagnostic; keep §6.1b boilerplate routing + §11.3 client-overlap gate + §11.3b judge gate.
- **§8 calibration** `τ_high`/`τ_low` + fusion model **re-fit on chunk max-pool cosine**; two profiles selected by `use_chunking`.
- **Judge** `judge_text_mode ∈ {chunk_pair, full_body}` applied to all gray-zone judges (§10.2, §11.3b, §13.0, §13.2). `chunk_pair` = `title + published_at + matched chunk` (**no lede**); `full_body` = `title + published_at + full body`. `judge_prompt_version` **bumped per arm** so cache arms never collide.
- **Chunking defaults:** `chunk_min_tokens=25`, `chunk_max_tokens=400`, `max_chunks_per_item=12`.
- **Embeddings:** `text-embedding-3-large`, unit-normalized; reuse on-disk `.cache/` (embed, judge, minhash) and `artifacts/`.
- **Two baselines:** **port-correctness gate** = 10k single-vector run with v3 ship `judge_prompt_version` must reproduce **≈0.870** (cache-reuse → near-free). **Experiment baseline** = 3k-seeded single-vector run (won't equal 0.870; that's expected). Chunking wins only if its best F1 **beats the 3k-seeded single-vector baseline**; judge arm chosen by F1, **cost as tiebreaker (~±0.01)**.
- **Notebook conventions:** markdown cell before every code cell; short single-concern cells (~10–25 lines); **a chart with every measurement**; build via Jupyter MCP.

## Plan Conventions (notebook + no-git adaptations)

This plan targets a **Jupyter notebook + a small Python module**, so the generic TDD/commit shape is adapted:

- **Module tasks (Phase 1)** are real pytest TDD: write failing test → run → implement → pass. Commands shown.
- **Notebook tasks** use **validation cells** as their "test": insert a markdown cell, insert a code cell, execute it via Jupyter MCP, and assert/observe the printed output or chart. "Expected" describes the assertion/observation that must hold.
- **Porting v3:** the validated pipeline already exists in `story_clustering_poc_v3.ipynb` (the v4 file is currently a copy of it). "Port §X" means **recreate that section's cells in the fresh v4 notebook**, reading the source of truth from `story_clustering_poc_v3.ipynb` §X, applying the listed edits. This is a precise instruction, not a placeholder — the code lives at the referenced section.
- **No git:** the repo is not a git repository. Replace "Commit" with **"Checkpoint"**: save the notebook and (where v3 did) persist the `.pkl`/`.json` artifact. If the user wants version control, run `git init` first (ask before doing so).
- **Jupyter MCP tools:** `mcp__jupyter__use_notebook`, `insert_cell` / `insert_execute_code_cell`, `overwrite_cell_source`, `execute_cell`, `read_cell`. Connect once at the start of a session.

---

## Visualization Requirement (every measurement gets a chart)

**Hard rule:** no notebook task that produces a number, distribution, or outcome is "done" until it ends with a rendered chart. A printed table alone does not satisfy this. Each chart must have a title, axis labels, and (where relevant) a reference line (ship line `F1=0.85`, baseline marker, `τ` lines). Pure-module tasks (Phase 1, pytest) are exempt — they have no data to plot.

This table is authoritative; the listed chart is a required step in that task even where the task body doesn't repeat it:

| Task | Required chart(s) |
|---|---|
| 0.2 CONFIG | CONFIG summary table (display) — no chart (no measurement) |
| 2.1 dataset/adapt | Ported v3 charts: title-len, body-len (tokens), items/day stacked by source, timestamp-precision, client-universe distribution |
| 2.2 sampling | **Corpus-composition bar:** eval-seeded items vs random-fill count making up the 3000; + per-source breakdown of the sampled corpus |
| 2.3 dedup | Ported: dropped-dups per source, MinHash cluster-size dist, threshold sweep |
| 2.4 §6 embed (single-vec) | Ported: embed_input token histogram, UMAP 3D scatter (by client, by source) |
| 2.5 §7/§8 calibration | Ported: SAME/DIFFERENT KDE, F1-vs-τ, ROC, PR, fusion coefficients + score dist |
| 2.6 port gate + baseline | **Port-gate bar:** reproduced 10k F1 vs the v3 0.870 reference line; then **3k baseline** P/R/F1 bar with ship line; ported §14 confusion matrix + FP-by-stage/FN-by-blocker |
| 3.1 chunk embed | Chunks-per-item histogram; chunk-token histogram; **chunk-vector UMAP 3D scatter** (parallels v3 §6.7) |
| 3.2 recalibration | Chunk KDE; ROC/PR; **side-by-side cosine-distribution overlay: chunk max-pool vs single-vector** (shows the τ shift); fusion coefficients |
| 3.3 §10 loop | Outcomes-count bar; **best max-pool sim distribution**; **matched-chunk-position histogram** (which `chunk_idx` wins — validates that deep-body matches actually fire) |
| 3.4 §11 HDBSCAN | **Component-size bar (over-merge diagnostic)**; residual cluster-size dist; **residual chunk-vector UMAP scatter colored by HDBSCAN label** (parallels v3 §11.7); gate-outcome bar |
| 4.1 judge wiring | **Prompt-token comparison:** chunk_pair vs full_body tokens across N sample pairs (box/bar) — visualizes the cost gap before the run |
| 4.2 judge-isolated A/B | Accuracy-by-cosine-bin grouped bars (per arm); $/call per arm; **per-arm confusion matrices** |
| 5.1 end-to-end runs | Each run renders the ported §14 P/R/F1 + confusion-matrix charts |
| 5.2 decision | Grouped P/R/F1 across the 3 runs (ship line + baseline marker); **cost-vs-F1 scatter** (3 points — the decision tradeoff made visual) |

---

## File Structure

| File | Responsibility |
|---|---|
| `story_clustering_poc_v4.ipynb` | **Create (fresh).** Orchestrates the whole pipeline; imports helpers from the module. Replaces the current v3-copy. |
| `scripts/v4_chunking.py` | **Create.** Pure helpers: `split_paragraphs`, `chunk_body`, `max_pool_sim`, `build_eval_seeded_sample`, `build_judge_block`, `union_items_from_chunk_clusters`, `component_sizes`. No I/O, no globals. |
| `scripts/test_v4_chunking.py` | **Create.** pytest unit tests for every function in the module. |
| `story_clustering_poc_v3.ipynb` | **Read-only.** Source of truth for ported sections. |
| `.cache/`, `artifacts/`, `dataset/`, `labeled_eval_set.csv` | **Reuse.** Embed/judge/minhash caches, eval set, parquet shards. |

---

## Phase 0 — Scaffolding

### Task 0.1: Module + pytest skeleton

**Files:**
- Create: `scripts/v4_chunking.py`
- Create: `scripts/test_v4_chunking.py`

**Interfaces:**
- Produces: an importable module the notebook and tests share. No functions yet — just the file + a smoke test that imports it.

- [ ] **Step 1: Write the failing test**

```python
# scripts/test_v4_chunking.py
import importlib

def test_module_imports():
    mod = importlib.import_module("v4_chunking")
    assert mod is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest test_v4_chunking.py::test_module_imports -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v4_chunking'`

- [ ] **Step 3: Create the module**

```python
# scripts/v4_chunking.py
"""Pure helpers for v4 paragraph-chunking clustering. No I/O, no notebook globals."""
from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest test_v4_chunking.py::test_module_imports -v`
Expected: PASS

- [ ] **Step 5: Checkpoint** — save both files.

---

### Task 0.2: Fresh notebook skeleton + §1 CONFIG

**Files:**
- Create: `story_clustering_poc_v4.ipynb` (replaces the v3-copy)

**Interfaces:**
- Produces: `CONFIG` dict (single source of truth) with all v4 keys; ledger helpers `record_run`, `config_fingerprint` (ported from v3 §1.4c–e).

- [ ] **Step 1:** Connect Jupyter MCP and create a fresh notebook. Overwrite the existing v4 file with an empty notebook (e.g. via the openai-jupyter-notebook `new_notebook.py` helper, or write a minimal `.ipynb` with one title markdown cell `# Story Clustering — POC v4 (paragraph chunking + judge A/B)`).

- [ ] **Step 2:** Insert §1 markdown + the imports cell (port v3 §1.2 imports verbatim) + the `.env` loader (port v3 §1.3).

- [ ] **Step 3:** Insert the **§1.4 CONFIG cell**. Start from v3 §1.4 + §1.4b overlay **collapsed into one clean dict**, then add the v4 keys:

```python
CONFIG.update({
    # --- v4 corpus ---
    "target_canonical_items": 3000,
    # --- v4 chunking ---
    "use_chunking": True,            # master switch; False = v3 single-vector baseline
    "chunk_min_tokens": 25,
    "chunk_max_tokens": 400,
    "max_chunks_per_item": 12,
    # --- v4 judge A/B ---
    "judge_text_mode": "chunk_pair", # "chunk_pair" | "full_body"
    # judge_prompt_version is set per arm in §10 (e.g. "v4_chunkpair", "v4_fullbody",
    # and "v2"/ship value for the single-vector port-correctness gate)
})
```

- [ ] **Step 4:** Port v3 §1.4c–e ledger helpers (serialization, `config_fingerprint`, `record_run`, stop-rule) verbatim, repointing artifact paths to a v4 ledger dir (`artifacts/v4/`).

- [ ] **Step 5:** Execute all §1 cells via Jupyter MCP.
Expected: `CONFIG` prints with the new keys; no exceptions.

- [ ] **Step 6: Checkpoint** — save notebook.

---

## Phase 1 — Pure helpers (module, TDD)

> Do this phase before the notebook chunk path so the notebook can import working, tested helpers. All commands run from `scripts/`.

### Task 1.1: `split_paragraphs`

**Files:**
- Modify: `scripts/v4_chunking.py`
- Test: `scripts/test_v4_chunking.py`

**Interfaces:**
- Produces: `split_paragraphs(body: str) -> list[str]` — paragraphs split on blank lines (`\n\s*\n`), fallback single `\n`, stripped, empties dropped; non-str/None → `[]`.

- [ ] **Step 1: Write the failing tests**

```python
from v4_chunking import split_paragraphs

def test_split_on_blank_lines():
    assert split_paragraphs("A para.\n\nB para.\n\nC.") == ["A para.", "B para.", "C."]

def test_split_fallback_single_newline():
    assert split_paragraphs("line one\nline two") == ["line one", "line two"]

def test_split_strips_and_drops_empty():
    assert split_paragraphs("  x  \n\n\n\n  y ") == ["x", "y"]

def test_split_none_and_nonstr():
    assert split_paragraphs(None) == []
    assert split_paragraphs(123) == []
    assert split_paragraphs("") == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest test_v4_chunking.py -k split -v`
Expected: FAIL — `ImportError: cannot import name 'split_paragraphs'`

- [ ] **Step 3: Implement**

```python
_PARA_SPLIT = re.compile(r"\n\s*\n")

def split_paragraphs(body):
    if not isinstance(body, str):
        return []
    parts = _PARA_SPLIT.split(body)
    if len(parts) == 1:                      # no blank-line breaks → fall back to single \n
        parts = body.split("\n")
    return [p.strip() for p in parts if p.strip()]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_v4_chunking.py -k split -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Checkpoint** — save.

---

### Task 1.2: `chunk_body` (tiny-merge, huge-split, cap, title-only fallback)

**Files:**
- Modify: `scripts/v4_chunking.py`
- Test: `scripts/test_v4_chunking.py`

**Interfaces:**
- Consumes: `split_paragraphs`.
- Produces: `chunk_body(title, body, *, count_tokens, min_tokens=25, max_tokens=400, max_chunks=12) -> list[str]` — **body-only** chunk texts. Tiny paragraphs (`< min_tokens`) merge into the previous chunk; oversize chunks (`> max_tokens`) sentence-split; result capped to first `max_chunks`. If the body yields no paragraphs, returns `[title.strip()]` (title-only fallback), or `[]` if title is also empty. `count_tokens` is a required callable `str -> int` (notebook passes a tiktoken counter; tests pass a word counter).

- [ ] **Step 1: Write the failing tests**

```python
from v4_chunking import chunk_body

WC = lambda s: len(s.split())   # word-count token proxy for tests

def test_chunk_empty_body_falls_back_to_title():
    assert chunk_body("Apple beats Q3", "", count_tokens=WC) == ["Apple beats Q3"]
    assert chunk_body("Apple beats Q3", None, count_tokens=WC) == ["Apple beats Q3"]

def test_chunk_empty_title_and_body_is_empty():
    assert chunk_body("", "", count_tokens=WC) == []

def test_chunk_body_only_excludes_title():
    out = chunk_body("HEADLINE", "Para one has several words here.\n\nPara two also has words.",
                     count_tokens=WC, min_tokens=2)
    assert out == ["Para one has several words here.", "Para two also has words."]
    assert all("HEADLINE" not in c for c in out)

def test_tiny_paragraph_merges_into_previous():
    body = "This first paragraph has more than three words total.\n\ntiny\n\nAnother real paragraph here now."
    out = chunk_body("T", body, count_tokens=WC, min_tokens=3)
    # "tiny" (1 word) merges into the first chunk
    assert out[0].endswith("tiny")
    assert len(out) == 2

def test_oversize_paragraph_is_split():
    body = " ".join(["word"] * 50)            # one 50-word paragraph
    out = chunk_body("T", body, count_tokens=WC, min_tokens=1, max_tokens=20)
    assert len(out) >= 3
    assert all(WC(c) <= 20 for c in out)

def test_cap_keeps_first_n():
    body = "\n\n".join([f"Paragraph number {i} with enough words." for i in range(20)])
    out = chunk_body("T", body, count_tokens=WC, min_tokens=1, max_chunks=12)
    assert len(out) == 12
    assert "number 0" in out[0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest test_v4_chunking.py -k chunk -v`
Expected: FAIL — `cannot import name 'chunk_body'`

- [ ] **Step 3: Implement**

```python
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _split_oversize(text, count_tokens, max_tokens):
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces, cur = [], []
    for sent in _SENT_SPLIT.split(text):
        cur.append(sent)
        if count_tokens(" ".join(cur)) >= max_tokens:
            pieces.append(" ".join(cur)); cur = []
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [text]

def chunk_body(title, body, *, count_tokens, min_tokens=25, max_tokens=400, max_chunks=12):
    title = (title or "").strip()
    paras = split_paragraphs(body)
    if not paras:                                  # title-only / empty-body fallback
        return [title] if title else []
    # merge tiny paragraphs into the previous chunk (or the next if there is no previous)
    merged = []
    for p in paras:
        if merged and count_tokens(p) < min_tokens:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    if len(merged) > 1 and count_tokens(merged[0]) < min_tokens:
        merged[1] = merged[0] + " " + merged[1]; merged = merged[1:]
    # sentence-split oversize chunks
    chunks = []
    for c in merged:
        chunks.extend(_split_oversize(c, count_tokens, max_tokens))
    return chunks[:max_chunks]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_v4_chunking.py -k chunk -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Checkpoint** — save.

---

### Task 1.3: `max_pool_sim` (nearest-chunk + matched pair)

**Files:**
- Modify: `scripts/v4_chunking.py`
- Test: `scripts/test_v4_chunking.py`

**Interfaces:**
- Produces: `max_pool_sim(vecs_a, vecs_b) -> tuple[float, int, int]` — for unit-normalized row matrices, returns `(max_cosine, argmax_a_row, argmax_b_row)`. Empty either side → `(-1.0, -1, -1)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from v4_chunking import max_pool_sim

def test_max_pool_identity_pair():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0]])               # matches row 1 of a
    sim, ia, ib = max_pool_sim(a, b)
    assert abs(sim - 1.0) < 1e-9 and ia == 1 and ib == 0

def test_max_pool_picks_best_pair():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.7071, 0.7071], [1.0, 0.0]])
    sim, ia, ib = max_pool_sim(a, b)
    assert ib == 1 and abs(sim - 1.0) < 1e-9

def test_max_pool_empty_side():
    assert max_pool_sim(np.zeros((0, 3)), np.ones((2, 3))) == (-1.0, -1, -1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest test_v4_chunking.py -k max_pool -v`
Expected: FAIL — `cannot import name 'max_pool_sim'`

- [ ] **Step 3: Implement**

```python
def max_pool_sim(vecs_a, vecs_b):
    a = np.asarray(vecs_a); b = np.asarray(vecs_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return (-1.0, -1, -1)
    sims = a @ b.T
    ia, ib = np.unravel_index(int(sims.argmax()), sims.shape)
    return (float(sims[ia, ib]), int(ia), int(ib))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_v4_chunking.py -k max_pool -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Checkpoint** — save.

---

### Task 1.4: `build_eval_seeded_sample`

**Files:**
- Modify: `scripts/v4_chunking.py`
- Test: `scripts/test_v4_chunking.py`

**Interfaces:**
- Produces: `build_eval_seeded_sample(working_df, eval_item_ids, target, seed, id_col="item_id") -> DataFrame` — all rows whose `id_col` is in `eval_item_ids` (deduped) plus a random fill (seeded) up to `target`. Repeatable. If eval items already ≥ target, returns just the eval rows (corpus may exceed target; eval measurability has priority).

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
from v4_chunking import build_eval_seeded_sample

def _df(n):
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(n)], "v": range(n)})

def test_seed_includes_all_eval_items():
    df = _df(100); evals = {"i3", "i7", "i42"}
    out = build_eval_seeded_sample(df, evals, target=10, seed=1)
    assert evals.issubset(set(out["item_id"]))
    assert len(out) == 10

def test_seed_is_repeatable():
    df = _df(100); evals = {"i1"}
    a = build_eval_seeded_sample(df, evals, target=20, seed=1)
    b = build_eval_seeded_sample(df, evals, target=20, seed=1)
    assert list(a["item_id"]) == list(b["item_id"])

def test_eval_larger_than_target_keeps_all_eval():
    df = _df(100); evals = {f"i{k}" for k in range(30)}
    out = build_eval_seeded_sample(df, evals, target=10, seed=1)
    assert set(out["item_id"]) == evals and len(out) == 30
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest test_v4_chunking.py -k seed -v`
Expected: FAIL — `cannot import name 'build_eval_seeded_sample'`

- [ ] **Step 3: Implement**

```python
def build_eval_seeded_sample(working_df, eval_item_ids, target, seed, id_col="item_id"):
    eval_set = set(eval_item_ids)
    present = working_df[working_df[id_col].isin(eval_set)].drop_duplicates(id_col)
    n_fill = max(0, target - len(present))
    remaining = working_df[~working_df[id_col].isin(eval_set)]
    fill = remaining.sample(n=min(n_fill, len(remaining)), random_state=seed) if n_fill else remaining.iloc[:0]
    import pandas as pd
    return pd.concat([present, fill]).reset_index(drop=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_v4_chunking.py -k seed -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Checkpoint** — save.

---

### Task 1.5: `build_judge_block` (chunk_pair / full_body)

**Files:**
- Modify: `scripts/v4_chunking.py`
- Test: `scripts/test_v4_chunking.py`

**Interfaces:**
- Produces: `build_judge_block(row, mode, matched_chunk=None) -> str` — a per-item block `"  title: ...\n  published_at: ...\n  text: ..."`. `mode="chunk_pair"` uses `matched_chunk` as the text (no lede, no body); `mode="full_body"` uses `row["body"]`. Unknown mode raises `ValueError`. `row` is a mapping with `title`, `published_at`, `body`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from v4_chunking import build_judge_block

ROW = {"title": "Apple beats Q3", "published_at": "2012-07-24", "body": "FULL BODY TEXT here."}

def test_chunk_pair_uses_matched_chunk_not_body():
    blk = build_judge_block(ROW, "chunk_pair", matched_chunk="the matched paragraph")
    assert "the matched paragraph" in blk
    assert "FULL BODY TEXT" not in blk
    assert "Apple beats Q3" in blk and "2012-07-24" in blk

def test_full_body_uses_body():
    blk = build_judge_block(ROW, "full_body")
    assert "FULL BODY TEXT here." in blk

def test_bad_mode_raises():
    with pytest.raises(ValueError):
        build_judge_block(ROW, "title_lede")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest test_v4_chunking.py -k judge_block -v`
Note: name the tests with `judge_block` or adjust `-k`. Expected: FAIL — `cannot import name 'build_judge_block'`

- [ ] **Step 3: Implement**

```python
def build_judge_block(row, mode, matched_chunk=None):
    title = row["title"] or ""
    pub = row["published_at"]
    if mode == "chunk_pair":
        text = matched_chunk or ""
    elif mode == "full_body":
        text = row["body"] or ""
    else:
        raise ValueError(f"unknown judge_text_mode: {mode!r}")
    return f"  title: {title}\n  published_at: {pub}\n  text: {text}"
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_v4_chunking.py -k judge_block -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Checkpoint** — save.

---

### Task 1.6: `union_items_from_chunk_clusters` + `component_sizes`

**Files:**
- Modify: `scripts/v4_chunking.py`
- Test: `scripts/test_v4_chunking.py`

**Interfaces:**
- Produces:
  - `union_items_from_chunk_clusters(chunk_item_ids, chunk_labels) -> dict[item_id, root]` — union-find: items owning chunks in the same non-noise (`!= -1`) HDBSCAN cluster share a root. Every distinct item_id appears as a key (noise-only items map to themselves).
  - `component_sizes(item_to_root) -> dict[root, int]` — the over-merge diagnostic (size of each transitive component).

- [ ] **Step 1: Write the failing tests**

```python
from v4_chunking import union_items_from_chunk_clusters, component_sizes

def test_items_sharing_cluster_are_unioned():
    ids    = ["A", "B", "C"]
    labels = [ 0,   0,  -1 ]          # A,B share cluster 0; C is noise
    m = union_items_from_chunk_clusters(ids, labels)
    assert m["A"] == m["B"] and m["C"] != m["A"]

def test_transitive_merge_across_clusters():
    # A,B share c0 ; B,C share c1 → A,B,C one component
    ids    = ["A", "B", "B", "C"]
    labels = [ 0,   0,   1,   1 ]
    m = union_items_from_chunk_clusters(ids, labels)
    assert m["A"] == m["B"] == m["C"]
    assert max(component_sizes(m).values()) == 3

def test_noise_only_items_are_singletons():
    ids = ["A", "B"]; labels = [-1, -1]
    m = union_items_from_chunk_clusters(ids, labels)
    assert m["A"] != m["B"]
    assert set(component_sizes(m).values()) == {1}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest test_v4_chunking.py -k union or -k component -v` (use `-k "union or component"`)
Expected: FAIL — `cannot import name 'union_items_from_chunk_clusters'`

- [ ] **Step 3: Implement**

```python
def union_items_from_chunk_clusters(chunk_item_ids, chunk_labels):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for iid in chunk_item_ids:               # ensure every item is a node
        find(iid)
    by_cluster = defaultdict(list)
    for iid, lab in zip(chunk_item_ids, chunk_labels):
        if lab != -1:
            by_cluster[lab].append(iid)
    for members in by_cluster.values():
        for m in members[1:]:
            union(members[0], m)
    return {iid: find(iid) for iid in parent}

def component_sizes(item_to_root):
    from collections import Counter
    return dict(Counter(item_to_root.values()))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest test_v4_chunking.py -k "union or component" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the whole module suite**

Run: `cd scripts && python -m pytest test_v4_chunking.py -v`
Expected: PASS (all ~22 tests)

- [ ] **Step 6: Checkpoint** — save.

---

## Phase 2 — Port the validated v3 pipeline (single-vector baseline)

> Goal of this phase: a fresh v4 notebook that runs the **v3 algorithm** end-to-end behind `use_chunking=False`, reuses on-disk caches, and passes the **port-correctness gate (10k ⇒ ≈0.870)**. Each task recreates a v3 section; source of truth is `story_clustering_poc_v3.ipynb`. The notebook must `import sys; sys.path.insert(0, "scripts")` and `import v4_chunking as v4c` near the top (do this in the §1 imports cell from Task 0.2).

### Task 2.1: §2 dataset load + §3 adapt + client universe

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Produces: `working_df` (full filtered pool, **before** sampling) with the canonical schema and `mentioned_clients`; reuses the v3 canonical cache.

- [ ] **Step 1:** Port v3 §2.1–§2.8b (parquet load, `extra_fields` parse, canonical schema + cache, inventories, the timestamp-precision diagnostic chart). Reuse the existing canonical cache path.
- [ ] **Step 2:** Port v3 §3.1–§3.5 (date filter, megacap candidates, regex matcher, top-20 client universe, mention filter). Stop **before** §3.6 sampling. Name the result `working_df`.
- [ ] **Step 3:** Execute via Jupyter MCP.
Expected: `working_df` row count printed (the full 2012–2013 universe-mention pool, ~tens of thousands); the §2 charts render.
- [ ] **Step 4: Checkpoint** — save.

### Task 2.2: §3.6 item_id-first + eval-seeded sampling

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `working_df`, `v4c.build_eval_seeded_sample`, `labeled_eval_set.csv`.
- Produces: `items_df` (sampled corpus with `item_id`).

- [ ] **Step 1:** Insert markdown explaining the change vs v3: **assign `item_id` to the full pool first**, then seed-sample so eval items survive.

- [ ] **Step 2:** Insert the sampling cell:

```python
import uuid, pandas as pd
# item_id BEFORE sampling (v4 change) so eval-set seeding can target real ids
def _iid(url):
    return None if (pd.isna(url) or not url) else str(uuid.uuid5(uuid.NAMESPACE_URL, str(url)))
working_df = working_df.copy()
working_df["item_id"] = working_df["url"].apply(_iid)
working_df = working_df[working_df["item_id"].notna()].reset_index(drop=True)

# eval item ids (both endpoints of every labeled pair)
_eval = pd.read_csv("labeled_eval_set.csv")
eval_item_ids = set(_eval["item_id_a"]).union(set(_eval["item_id_b"]))   # adjust col names to the CSV

items_df = v4c.build_eval_seeded_sample(
    working_df, eval_item_ids, CONFIG["target_canonical_items"], CONFIG["random_seed"])
present = items_df["item_id"].isin(eval_item_ids).sum()
print(f"items_df: {len(items_df):,} | eval items present: {present:,} / {len(eval_item_ids):,}")
```

- [ ] **Step 3:** Execute.
Expected: `len(items_df) ≈ 3000`; **`eval items present` equals the count of eval items that exist in `working_df`** (should be the full set, since eval items came from this pool in v3). If many eval items are missing, the CSV column names or the `item_id` derivation differ from v3 — fix before continuing.

- [ ] **Step 4:** Read `labeled_eval_set.csv` header first (`mcp__plugin_context-mode_context-mode__ctx_execute_file` on the CSV, print columns) to confirm the exact `item_id_a`/`item_id_b` column names; adjust Step 2 accordingly.

- [ ] **Step 5: Checkpoint** — save.

### Task 2.3: §4 URL dedup + §5 MinHash/LSH (verbatim)

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Produces: `items_df` with `url_hash`, `is_duplicate`; `minhashes` dict; reuses the minhash cache.

- [ ] **Step 1:** Port v3 §4.1–§4.5 verbatim (URL canonicalization, hash, exact-dup drop, charts).
- [ ] **Step 2:** Port v3 §5.1–§5.7 verbatim (shingle on `title + body`, MinHash cached, LSH, union-find dup clusters, charts). No chunking here — full body, unchanged.
- [ ] **Step 3:** Execute.
Expected: dedup counts + per-source duplicate charts render; `minhashes` populated.
- [ ] **Step 4: Checkpoint** — save.

### Task 2.4: §6 single-vector embedding path (port, behind switch)

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `canonical_items` (non-dup, non-boilerplate).
- Produces: `canonical_items`, `boilerplate_df`, `assignment_vecs` (single vector per item) when `use_chunking=False`.

- [ ] **Step 1:** Port v3 §6.1 (`build_embed_input` = title+lede), §6.1b boilerplate routing, §6.2 async cached embedder, §6.3 run, §6.4 stack `assignment_vecs`, §6.5–§6.8 charts/UMAP. Reuse the embed cache.
- [ ] **Step 2:** Wrap the single-vector stacking so it only runs when `not CONFIG["use_chunking"]` (the chunk path in Phase 3 produces the chunk equivalent). Keep `assignment_vecs` as the baseline artifact.
- [ ] **Step 3:** Execute with `use_chunking=False`.
Expected: `assignment_vecs.shape == (len(canonical_items), dims)`, norms ≈ 1.0.
- [ ] **Step 4: Checkpoint** — save.

### Task 2.5: §7 load eval set + §8 calibration (single-vector profile)

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `labeled_eval_set.csv`, `assignment_vecs`.
- Produces: `TAU_HIGH`, `TAU_LOW`, fusion model (single-vector profile), `pos_calibration.json` (v4 path).

- [ ] **Step 1:** Port v3 §7.0 + §7.10 **load-only** path (reuse `labeled_eval_set.csv` and the supplemental frozen slice). **Exclude** the entire `REUSE_EXISTING_EVAL=False` re-labeling branch (§7.1–§7.9, §7.10a–c generation) — delete those cells, keep only the loads.
- [ ] **Step 2:** Port v3 §8.1–§8.6 (sweep, KDE, ROC, pick τ, export calibration) and §8.7a–d (fusion features, train, gates, viz). This is the **single-vector profile**.
- [ ] **Step 3:** Execute.
Expected: τ_high/τ_low printed; calibration + fusion charts render; values match v3 within noise.
- [ ] **Step 4: Checkpoint** — save.

### Task 2.6: §10–§14 port (loop, HDBSCAN, metadata, merge, eval) + port-correctness gate

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: all of the above.
- Produces: `stories`, eval metrics; ledger record for the 10k single-vector run.

- [ ] **Step 1:** Port v3 §10 (judge infra §10.2 with the **ship `judge_prompt_version`**, the loop §10.3 centroid path, expiry, checkpoints) keeping only the **final ship config** — exclude superseded judge/escalation variant cells (keep the `judge_escalation_model`/`judge_escalation_band` ship values).
- [ ] **Step 2:** Port v3 §11 (HDBSCAN residual, client-overlap gate §11.3, judge gate §11.3b, assemble §11.4, charts) — single-vector path.
- [ ] **Step 3:** Port v3 §12 metadata generation, but gate it behind a flag `CONFIG.get("run_metadata", False)` defaulting **off** (metadata is not on the F1 path; skip it to save cost during the experiment). Populate trivial metadata so §13/§14 still run.
- [ ] **Step 4:** Port v3 §13 (second sweep, merge-pool repair, merge judge, expiry — ship config) and §14 (eval, hygiene resolution §14.1b–c, attribution §14.1d, supplemental §14.1e, baselines, charts, §14.7 ledger writer).
- [ ] **Step 5: Port-correctness gate.** Temporarily set `CONFIG["target_canonical_items"]=10000`, `CONFIG["use_chunking"]=False`, `CONFIG["judge_prompt_version"]` = the v3 ship value. Re-run §3.6 → §14.
Expected: end-to-end **F1 ≈ 0.870 (P ≈ 0.917)** within ±0.01–0.02; judge calls **mostly cached** (near-zero paid). If F1 is off, the port diverged — diff against v3 §-by-§ and fix.

- [ ] **Step 6:** Reset `CONFIG["target_canonical_items"]=3000`; re-run §3.6 → §14 to record the **3k-seeded single-vector experiment baseline** (still `use_chunking=False`). Note its F1/P/R/$.
- [ ] **Step 7:** Write the ledger record `baseline_single_vector_3k` via §14.7.
- [ ] **Step 8: Checkpoint** — save notebook + `artifacts/v4/checkpoints/`.

---

## Phase 3 — Chunk path

### Task 3.1: §6b chunk embedding

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `canonical_items`, `v4c.chunk_body`, the §6.2 cached embedder, a tiktoken counter.
- Produces: `chunks_df` (`item_id`, `chunk_idx`, `chunk_text`), `chunk_vecs` (np.ndarray, unit-norm), `item_chunk_ranges: dict[item_id, (start, end)]` into `chunk_vecs`, and a helper `chunks_for(item_id) -> np.ndarray`.

- [ ] **Step 1:** Insert markdown for §6b. Insert the chunk-building cell:

```python
import tiktoken, numpy as np, pandas as pd
_enc = tiktoken.encoding_for_model(CONFIG["embedding_model"])     # same model as v3 §6
_tok = lambda s: len(_enc.encode(s))

rows = []
for r in canonical_items.itertuples():
    chs = v4c.chunk_body(r.title, r.body, count_tokens=_tok,
                         min_tokens=CONFIG["chunk_min_tokens"],
                         max_tokens=CONFIG["chunk_max_tokens"],
                         max_chunks=CONFIG["max_chunks_per_item"])
    for j, c in enumerate(chs):
        rows.append({"item_id": r.item_id, "chunk_idx": j, "chunk_text": c})
chunks_df = pd.DataFrame(rows)
print(f"{len(canonical_items):,} items → {len(chunks_df):,} chunks "
      f"(mean {len(chunks_df)/len(canonical_items):.1f}/item)")
assert chunks_df.groupby("item_id").size().min() >= 1, "every item must yield ≥1 chunk"
```

- [ ] **Step 2:** Execute.
Expected: every item has ≥1 chunk (title-only items show exactly 1); mean chunks/item printed; **no item missing** (`chunks_df.item_id.nunique() == len(canonical_items)`).

- [ ] **Step 3:** Insert the chunk-embedding cell — embed `chunks_df.chunk_text` through the **same async cached embedder** from §6.2 (cache keyed by text, so re-runs are free), stack into `chunk_vecs`, build `item_chunk_ranges` (contiguous per item, since `chunks_df` is grouped by item order):

```python
chunk_texts = chunks_df["chunk_text"].tolist()
# reuse the §6.2 embedder (background-warm cache; no blocking batch — see memory)
chunk_vecs = await embed_texts(chunk_texts)          # returns (N,dims) unit-normalized
chunk_vecs = np.asarray(chunk_vecs)
norms = np.linalg.norm(chunk_vecs, axis=1)
assert np.allclose(norms, 1.0, atol=1e-3)

item_chunk_ranges = {}
start = 0
for iid, grp in chunks_df.groupby("item_id", sort=False):
    item_chunk_ranges[iid] = (start, start + len(grp)); start += len(grp)
def chunks_for(item_id):
    s, e = item_chunk_ranges[item_id]; return chunk_vecs[s:e]
print(f"chunk_vecs: {chunk_vecs.shape}")
```

- [ ] **Step 4:** Execute. Expected: `chunk_vecs.shape == (len(chunks_df), dims)`, norms ≈ 1.
- [ ] **Step 5:** Insert a small **chart** (per the conventions): histogram of chunks-per-item. Execute.
- [ ] **Step 6: Checkpoint** — save.

### Task 3.2: §8b recalibration on chunk max-pool cosine

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `labeled_eval_set.csv`, `chunks_for`, `v4c.max_pool_sim`.
- Produces: chunk-profile `TAU_HIGH`/`TAU_LOW` + fusion model when `use_chunking=True`; `pos_calibration_chunk.json`.

- [ ] **Step 1:** Insert a cell that, for each labeled pair, computes the **max-pool cosine** between the two items' chunk sets (skip pairs where either item isn't in `item_chunk_ranges`):

```python
pair_cos = []
for p in labeled_pairs.itertuples():           # labeled_pairs = loaded eval df
    if p.item_id_a in item_chunk_ranges and p.item_id_b in item_chunk_ranges:
        sim, _, _ = v4c.max_pool_sim(chunks_for(p.item_id_a), chunks_for(p.item_id_b))
        pair_cos.append((p.item_id_a, p.item_id_b, p.final_label, sim))
chunk_cos_df = pd.DataFrame(pair_cos, columns=["a", "b", "label", "cosine"])
print(chunk_cos_df.groupby("label")["cosine"].describe())
```

- [ ] **Step 2:** Re-run the §8.3 sweep / §8.4 τ-pick / §8.5 ROC and the §8.7 fusion train **using `chunk_cos_df.cosine`** as the cosine signal (the fusion `pair_features` cosine slot becomes max-pool cosine; other lexical features unchanged). Persist to `pos_calibration_chunk.json` and a chunk fusion model file. Select profile by `use_chunking`.
- [ ] **Step 3:** Execute. Expected: KDE shows SAME vs DIFFERENT separation; **τ values higher than the single-vector profile** (max-pool inflates cosine — sanity check this is true); ROC/PR + fusion charts render.
- [ ] **Step 4: Checkpoint** — save.

### Task 3.3: §10 max-pool main loop

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `chunks_for`, `v4c.max_pool_sim`, chunk-profile τ/fusion.
- Produces: `stories` (chunk path), `assignment_log` with `matched_chunk_text` per gray-zone decision.

- [ ] **Step 1:** Insert markdown describing the chunk story-state change: a story holds the **stacked chunk vectors of all members**; item↔story sim = `max_pool_sim(item_chunks, story_chunks)`.

- [ ] **Step 2:** Add chunk-aware story helpers (parallel to v3 §10.1, selected by `use_chunking`):

```python
def new_story_chunk(item_idx, row):
    cv = chunks_for(row["item_id"])
    return {"story_id": f"s{len(stories):05d}", "chunk_vecs": cv,
            "n_items": 1, "member_ids": [row["item_id"]], "member_idxs": [int(item_idx)],
            "item_clients": set(row["item_clients"]),
            "first_seen_at": row["published_at"], "last_seen_at": row["published_at"],
            "closed_at": None}

def assign_to_story_chunk(story, item_idx, row):
    story["chunk_vecs"] = np.vstack([story["chunk_vecs"], chunks_for(row["item_id"])])
    story["member_ids"].append(row["item_id"]); story["member_idxs"].append(int(item_idx))
    story["item_clients"].update(row["item_clients"]); story["n_items"] += 1
    if row["published_at"] > story["last_seen_at"]:
        story["last_seen_at"] = row["published_at"]
```

- [ ] **Step 3:** In the main loop (port v3 §10.3, branch on `use_chunking`), replace the centroid dot with max-pool and capture the matched chunk for the judge:

```python
sims = []
for s in candidates:
    sim, ia, ib = v4c.max_pool_sim(chunks_for(row["item_id"]), s["chunk_vecs"])
    sims.append((sim, ia, ib))
best_i = int(np.argmax([s[0] for s in sims]))
best_story = candidates[best_i]; best_sim, best_ia, best_ib = sims[best_i]
# matched chunk on the ITEM side (for chunk_pair judge); rep side handled in §10.2 judge
item_matched_chunk = chunks_df.iloc[item_chunk_ranges[row["item_id"]][0] + best_ia]["chunk_text"]
rec["matched_item_chunk_idx"] = int(best_ia)   # feeds the Step 5 matched-chunk-position chart
```

Gate on the chunk-profile τ/fusion exactly as v3, then if gray-zone call the judge (Task 4.1) passing `item_matched_chunk` and the rep's matched chunk.

- [ ] **Step 4:** Execute the loop with `use_chunking=True`, `judge_text_mode="chunk_pair"`.
Expected: outcomes table prints; `stories` formed; loop completes without shape errors.
- [ ] **Step 5: Charts (required).** Outcomes-count bar; best max-pool sim distribution; and the **matched-chunk-position histogram** — this validates the premise (if matches only ever come from `chunk_idx==0`, chunking adds nothing over title/lede):

```python
import matplotlib.pyplot as plt
# record best_ia per gray/auto decision into assignment_log as "matched_item_chunk_idx"
idxs = [r["matched_item_chunk_idx"] for r in assignment_log if r.get("matched_item_chunk_idx") is not None]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
pd.Series([r["outcome"] for r in assignment_log]).value_counts().plot.bar(ax=ax[0], title="§10 outcomes")
pd.Series([r["best_sim"] for r in assignment_log if r.get("best_sim")]).plot.hist(bins=40, ax=ax[1], title="best max-pool sim")
pd.Series(idxs).plot.hist(bins=range(0, CONFIG["max_chunks_per_item"]+1), ax=ax[2],
                          title="matched item chunk_idx (0 = first paragraph)")
ax[2].set_xlabel("chunk index that won the match"); plt.tight_layout()
```

Execute. Expected: a non-trivial share of matches come from `chunk_idx > 0` — otherwise note it (chunking is not contributing and the result will likely track the lede baseline).

- [ ] **Step 6: Checkpoint** — save + `artifacts/v4/checkpoints/post_s10_chunk.pkl`.

### Task 3.4: §11 HDBSCAN over chunks + item-union + over-merge diagnostic

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: residual singletons' chunk vectors, `v4c.union_items_from_chunk_clusters`, `v4c.component_sizes`.
- Produces: `new_residual_stories` (chunk path), passing through the existing client + judge gates.

- [ ] **Step 1:** Insert markdown for the chunk HDBSCAN approach.
- [ ] **Step 2:** Build the residual **chunk** matrix: for singleton items, gather their chunks + a parallel `chunk_owner_item_ids` list. Insert:

```python
res_item_ids = [s["member_ids"][0] for s in singleton_stories]
res_chunk_vecs, chunk_owner = [], []
for iid in res_item_ids:
    cv = chunks_for(iid)
    res_chunk_vecs.append(cv); chunk_owner.extend([iid] * len(cv))
res_chunk_vecs = np.vstack(res_chunk_vecs)
print(f"residual: {len(res_item_ids):,} items → {len(chunk_owner):,} chunks")
```

- [ ] **Step 3:** Run HDBSCAN on `cosine_distances(res_chunk_vecs)` (port v3 §11.2 params), then union items:

```python
from v4_chunking import union_items_from_chunk_clusters, component_sizes
item_to_root = union_items_from_chunk_clusters(chunk_owner, hdb_labels)
sizes = component_sizes(item_to_root)
big = sorted(sizes.values(), reverse=True)[:10]
print(f"transitive components: {len(sizes)} | largest sizes: {big}")
```

- [ ] **Step 4: Over-merge diagnostic chart** — bar chart of component sizes. **If the largest component is implausibly large (e.g. > 20 items), STOP and report** — that's the generic-chunk over-merge the spec warns about; the follow-up "generic-chunk filter" would be needed. Otherwise continue.
- [ ] **Step 5:** Build `new_residual_stories` from the components (group `item_to_root` by root), then pass them through the **existing** §11.3 client-overlap gate and §11.3b judge gate (ported, judge-mode-aware via Task 4.1). Port §11.4 assemble + charts.
- [ ] **Step 6:** Execute. Expected: residual multi-stories formed; gate-outcome chart renders.
- [ ] **Step 7: Checkpoint** — save + `artifacts/v4/checkpoints/post_s11_chunk.pkl`.

---

## Phase 4 — Judge A/B

### Task 4.1: Wire `judge_text_mode` into all judges

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: `v4c.build_judge_block`, `CONFIG["judge_text_mode"]`, matched chunks from §10/§11/§13.
- Produces: a single `build_judge_prompt(item_row, rep_row, item_chunk=None, rep_chunk=None)` used by `judge_same` everywhere; `judge_prompt_version` set per arm.

- [ ] **Step 1:** Replace v3 §10.2 `build_judge_prompt` with a chunk-aware version that uses the rubric body from v3's `pv="v2"` ship prompt but builds each side via `v4c.build_judge_block`:

```python
def build_judge_prompt(item_row, rep_row, item_chunk=None, rep_chunk=None):
    mode = CONFIG["judge_text_mode"]
    a = v4c.build_judge_block(item_row, mode, matched_chunk=item_chunk)
    b = v4c.build_judge_block(rep_row,  mode, matched_chunk=rep_chunk)
    return (
        "You are determining whether two financial news items describe the SAME news event.\n\n"
        f"ITEM A:\n{a}\n\nITEM B:\n{b}\n\n"
        "Two items are the SAME story if they describe the same underlying news event involving "
        "the same primary entities — even if framing, source, or details differ. They are "
        "DIFFERENT if the primary event differs (preview vs results; distinct follow-ups in an "
        "ongoing saga are DIFFERENT events).\n\nReply with a single word: SAME or DIFFERENT."
    )
```

- [ ] **Step 2:** Thread the matched chunks through `judge_same` → `_judge_once` (add `item_chunk`/`rep_chunk` params). For the §10 rep side, compute the rep's matched chunk from the `best_ib` index against that rep's chunk range; for §11/§13, compute the nearest chunk pair between the two stories with `v4c.max_pool_sim`.
- [ ] **Step 3:** Set `judge_prompt_version` per arm so caches never collide: `"v4_chunkpair"` when `judge_text_mode=="chunk_pair"`, `"v4_fullbody"` when `"full_body"`. (The single-vector port-correctness gate keeps the v3 ship value.)
- [ ] **Step 4:** Execute a tiny smoke check: call `build_judge_prompt` on two sample rows in each mode; print both prompts.
Expected: `chunk_pair` shows the matched chunk and **no full body / no lede**; `full_body` shows the whole body; titles+dates present in both.
- [ ] **Step 5: Checkpoint** — save.

### Task 4.2: Judge-isolated A/B measurement (543 pairs)

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: labeled pairs, `build_judge_prompt`, the judge client + token accounting.
- Produces: `judge_ab_df` (per-arm accuracy/P/R + tokens + $), a per-cosine-bin chart.

- [ ] **Step 1:** Insert a harness (extends v3 §10.2c) that, for each arm in `["chunk_pair", "full_body"]`, runs the judge across all labeled pairs (resolving each pair's text + matched chunk), records verdict vs `final_label`, and tallies tokens/$:

```python
results = {}
for mode in ["chunk_pair", "full_body"]:
    CONFIG["judge_text_mode"] = mode
    CONFIG["judge_prompt_version"] = "v4_chunkpair" if mode == "chunk_pair" else "v4_fullbody"
    verdicts, tokens = [], 0
    for p in labeled_pairs.itertuples():
        ra, rb = row_of(p.item_id_a), row_of(p.item_id_b)
        ic = rc = None
        if mode == "chunk_pair":
            _, ia, ib = v4c.max_pool_sim(chunks_for(p.item_id_a), chunks_for(p.item_id_b))
            ic, rc = chunk_text_at(p.item_id_a, ia), chunk_text_at(p.item_id_b, ib)
        v, t = judge_with_tokens(ra, rb, ic, rc)     # returns (bool, total_tokens)
        verdicts.append(v); tokens += t
    results[mode] = score_against_labels(verdicts, labeled_pairs["final_label"], tokens)
judge_ab_df = pd.DataFrame(results).T            # rows: modes; cols: acc/P/R/tokens/$
judge_ab_df
```

- [ ] **Step 2:** Execute (uses the rate-limited, cached judge — no blocking batch; background-warm per the memory note).
Expected: a 2-row table with accuracy, P, R, tokens/call, and $ per arm.
- [ ] **Step 3: Chart** — grouped bars: accuracy by cosine bin per arm (left) and $/call per arm (right).
- [ ] **Step 4: Checkpoint** — save.

---

## Phase 5 — End-to-end runs + decision

### Task 5.1: Three end-to-end ledger runs

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Produces: ledger records `baseline_single_vector_3k`, `chunk_chunk_pair`, `chunk_full_body` in `artifacts/v4/`.

- [ ] **Step 1:** Confirm `baseline_single_vector_3k` exists from Task 2.6 Step 7; if not, run `use_chunking=False` at 3k and record it.
- [ ] **Step 2:** Run `use_chunking=True, judge_text_mode="chunk_pair"` through §10→§14; write ledger record `chunk_chunk_pair` (F1/P/R/$ + config fingerprint).
- [ ] **Step 3:** Run `use_chunking=True, judge_text_mode="full_body"` through §10→§14; write ledger record `chunk_full_body`.
Expected: three ledger JSONs in `artifacts/v4/experiments/`.
- [ ] **Step 4: Checkpoint** — save.

### Task 5.2: Comparison chart + decision

**Files:**
- Modify: `story_clustering_poc_v4.ipynb`

**Interfaces:**
- Consumes: the three ledger records + `judge_ab_df`.
- Produces: §16 findings + the applied decision rule.

- [ ] **Step 1:** Port v3 §17.1 ledger-comparison loader; load the three v4 runs into one table.
- [ ] **Step 2: Chart** — grouped P/R/F1 bars for the three runs, with the **0.85 ship line** and a marker at the **3k single-vector baseline F1**.
- [ ] **Step 2b: Decision-tradeoff chart (required).** Cost-vs-F1 scatter — the decision rule made visual (one point per run; F1 on y, total $ on x; annotate each; draw the baseline-F1 and 0.85 lines):

```python
import matplotlib.pyplot as plt
runs = ledger_v4[["run", "f1", "total_cost_usd"]]   # 3 rows from Step 1
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(runs["total_cost_usd"], runs["f1"], s=120)
for _, r in runs.iterrows():
    ax.annotate(r["run"], (r["total_cost_usd"], r["f1"]), xytext=(6, 6), textcoords="offset points")
ax.axhline(0.85, ls="--", color="grey", label="ship line 0.85")
ax.axhline(baseline_3k_f1, ls=":", color="red", label="3k single-vec baseline")
ax.set_xlabel("total run cost (USD)"); ax.set_ylabel("F1"); ax.set_title("Decision: F1 vs cost"); ax.legend()
```

Execute. Expected: 3 labelled points; the winning arm is the highest-F1 point above the baseline line, cheaper one if within ±0.01.

- [ ] **Step 3:** Insert the decision cell applying the rule verbatim from Global Constraints: chunking wins only if its best F1 **beats** the 3k single-vector baseline; pick the judge arm by F1, **cost tiebreaker within ±0.01**. Print the verdict (`SHIP chunking + <arm>` / `KEEP single-vector`) with the numbers.
- [ ] **Step 4:** Port v3 §16.3 findings writer → `artifacts/v4/poc_findings.md`, including the judge-isolated A/B table, the end-to-end comparison, the over-merge diagnostic result, and the decision.
- [ ] **Step 5:** Execute all of §16/§17.
Expected: findings file written; decision printed with evidence.
- [ ] **Step 6: Checkpoint** — save notebook + findings.

---

## Self-Review

**1. Spec coverage**

| Spec section | Covered by |
|---|---|
| §3 scope-of-change table | Phases 2–4 (per-section tasks) |
| §4 dataset reduction + eval seeding | Task 1.4 + Task 2.2 |
| §5 fresh notebook, exclusions, two baselines | Task 0.2 + Phase 2 (port + exclusions) + Task 2.6 (10k gate, 3k baseline) |
| §6 body-only chunking + title-only fallback | Task 1.2 + Task 3.1 |
| §7.1 §10 max-pool | Task 1.3 + Task 3.3 |
| §7.2 §11 HDBSCAN union + over-merge diagnostic | Task 1.6 + Task 3.4 |
| §7.3 §8 recalibration (two profiles) | Task 2.5 (single-vec) + Task 3.2 (chunk) |
| §8 judge A/B + per-arm prompt_version | Task 1.5 + Task 4.1 |
| §9 measurement (isolated + end-to-end), charts | Task 4.2 + Task 5.1 |
| §10 decision rule | Task 5.2 |
| §11 risks (each mitigated) | title fallback 1.2/3.1; over-merge 3.4; recalibration 3.2; eval seeding 2.2; port gate 2.6; full_body cost — judge cached/non-blocking 4.2 |
| §12 CONFIG keys | Task 0.2 |

No spec section is left without a task.

**2. Placeholder scan:** New-logic steps carry full code; ported steps name the exact v3 section + edits + a validation/expected output (not placeholders — the code exists at the referenced section). Note any CSV column names (`item_id_a`/`item_id_b`, `final_label`) and the §6 embedder symbol (`embed_texts`/`embed_cache_key`) must be confirmed against v3 in Task 2.2 Step 4 / Task 2.4 — flagged inline.

**3. Type consistency:** `chunks_for(item_id) -> np.ndarray`, `max_pool_sim(...) -> (float,int,int)`, `build_judge_block(row, mode, matched_chunk=None) -> str`, `union_items_from_chunk_clusters(ids, labels) -> dict`, `component_sizes(map) -> dict`, `build_eval_seeded_sample(...) -> DataFrame` are used consistently across Phases 1, 3, 4, 5.
