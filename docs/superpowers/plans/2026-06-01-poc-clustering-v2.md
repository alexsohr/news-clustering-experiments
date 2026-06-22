# POC Story-Clustering v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un-break the clustering pipeline (fix the §10 vector-indexing bug), curate templated "boilerplate" wire items out of the corpus, swap the LLM judge from Anthropic Haiku to OpenAI `gpt-4.1`, and re-measure — all in a copy notebook so the original 0.32 run is preserved for comparison.

**Architecture:** Work entirely in `story_clustering_poc_v2.ipynb` (a copy of `story_clustering_poc.ipynb`). Changes are localized to a handful of cells; DataFrame-centric code is introduced only where these changes touch. Embeddings are reused from the content-hash-keyed disk cache (free). The eval uses the *existing* 543 labeled pairs (no re-labeling, $0). Outputs land in `artifacts/v2/`.

**Tech Stack:** Python 3.11 (`.venv`), pandas/numpy/scikit-learn, `hdbscan`, `openai` SDK (judge), Jupyter (live kernel via the `mcp__jupyter__*` MCP tools). API keys are in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).

**Reference spec:** `docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md` (diagnosis: `docs/poc-diagnosis-and-improvements.md`).

---

## Conventions for this plan (read first)

- **No git** in this repo. Wherever a normal plan says "commit," this plan says **"Checkpoint: save the notebook"** (`mcp__jupyter__*` autosaves; or `File > Save`). The v2-copy is the safety net.
- **Apply cell edits via the Jupyter MCP** against the running kernel: `mcp__jupyter__use_notebook` to open `story_clustering_poc_v2.ipynb`, `mcp__jupyter__read_cell` / `overwrite_cell_source` / `insert_cell` to edit, `mcp__jupyter__execute_cell` to run. **Reference cells by their leading comment/heading** (e.g. the cell starting `# 10.3 (cont.)`), NOT by absolute index — inserting cells renumbers everything.
- **Notebook style** (user memory): a short markdown cell before each new code cell; keep cells to one concern (~10–25 lines); prefer `insert_cell` over giant overwrites.
- **Authoritative verification is Task 8** (Restart Kernel + Run All). Per-task "run" steps run only the just-edited cells against current kernel state to catch errors early.
- **Numbers to beat / sanity anchors:** old pipeline pairwise-F1 **0.323**; `cosine≥0.65` baseline **0.405**; old outcome mix **91% residual / 8,300 singletons**; old `τ_high=0.88` (degenerate).

---

### Task 0: Create the v2 notebook and wire up config + inputs

**Files:**
- Create: `story_clustering_poc_v2.ipynb` (copy of `story_clustering_poc.ipynb`)
- Create: `artifacts/v2/` and copy the existing eval CSVs into it
- Modify: the `CONFIG` cell (starts `# Single source of truth for every tunable parameter`)
- Insert: a "v2 changelog" markdown cell after the title cell

- [ ] **Step 1: Copy the notebook and stage inputs**

Run:
```bash
cd /Users/alex/Projcts/news-clustering
cp story_clustering_poc.ipynb story_clustering_poc_v2.ipynb
mkdir -p artifacts/v2
cp artifacts/labeled_eval_set.csv artifacts/v2/labeled_eval_set.csv
cp artifacts/human_labels.csv artifacts/v2/human_labels.csv 2>/dev/null || true
ls -la story_clustering_poc_v2.ipynb artifacts/v2/
```
Expected: `story_clustering_poc_v2.ipynb` exists; `artifacts/v2/labeled_eval_set.csv` present.

- [ ] **Step 2: Open v2 in the kernel**

Use `mcp__jupyter__use_notebook` with path `story_clustering_poc_v2.ipynb` (kernel python = `.venv`). Confirm it loads (262 cells).

- [ ] **Step 3: Insert the v2 changelog markdown cell**

Insert immediately after the top title cell (`# Story Clustering — POC Notebook`):
```markdown
## v2 changes (see docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md)
1. **P0 — fixed the §10 vector-indexing bug** (unstable sort scrambled tied-timestamp embeddings).
2. **P2 — boilerplate curation**: templated wire items (REG-/Form 8.x/NAV/PR templates) set aside before clustering.
3. **Judge swap**: gray-zone (§10) and merge (§13) judges now use OpenAI `gpt-4.1` instead of Claude Haiku.
DataFrame changes are limited to the cells these touch. Outputs → `artifacts/v2/`. Eval reuses the existing 543 labeled pairs.
```

- [ ] **Step 4: Edit the `CONFIG` cell**

In the `CONFIG` dict cell, change the artifacts dir and add the judge model. Edit these two spots:
```python
    "artifacts_dir":    Path.cwd() / "artifacts" / "v2",   # was: Path.cwd() / "artifacts"
```
and add inside the `--- LLM models ---` block:
```python
    "judge_model":        "gpt-4.1",   # v2: OpenAI judge replaces Haiku at §10 + §13 call sites
```

- [ ] **Step 5: Run setup + verify**

Execute the `CONFIG` cell and the `# 1.5 Pretty-print the CONFIG summary` cell.
Expected: printed config shows `judge_model = gpt-4.1` and `artifacts_dir` ending in `/artifacts/v2`.

- [ ] **Step 6: Checkpoint** — save the notebook.

---

### Task 1: Boilerplate detector, partition, and hand-audit

**Files:**
- Modify: v2 notebook, inserting cells immediately AFTER `canonical_items` is created (the cell starting `canonical_items = items_df[~items_df["is_duplicate"]]`) and BEFORE the embeddings are stacked (`# 6.4 — Stack vectors in canonical_items order`).

- [ ] **Step 1: Insert a markdown cell**

After the `canonical_items = …` cell, insert:
```markdown
### 6.1b — Boilerplate curation (v2): set templated wire items aside before clustering
```

- [ ] **Step 2: Insert the detector + partition cell**

```python
# 6.1b — Detect templated/non-editorial wire items and route them out of the clustering corpus.
# STRUCTURAL templates only. Must NOT flag editorial wire tags (UPDATE-N/WRAPUP/FACTBOX) —
# 32% of true-SAME pairs carry those on legitimate same-story follow-ups.
import re
_BP_PATTERNS = [
    r"^\s*REG\s*-", r"\bForm\s*8(\.\d)?\b", r"\(EPT/", r"\(DD\)",
    r"Net Asset Value", r"\bNAV\b", r"Total Voting Rights",
    r"Transaction in Own Shares", r"\bDaily Share\b",
    r"to Participate", r"Invites You", r"to Present at", r"One-on-One",
    r"4G LTE.*Expand", r"Expand.*4G LTE", r"\bZacks\b",
]
_BP_RE = re.compile("|".join(_BP_PATTERNS), re.IGNORECASE)

def is_boilerplate_text(title, body=""):
    blob = f"{title or ''}  {(body or '')[:200]}"
    return bool(_BP_RE.search(blob))

canonical_items_all = canonical_items.copy()
_bp_mask = canonical_items_all.apply(
    lambda r: is_boilerplate_text(r["title"], r.get("body", "")), axis=1)
canonical_items_all["is_boilerplate"] = _bp_mask.to_numpy()

boilerplate_df = canonical_items_all[canonical_items_all["is_boilerplate"]].reset_index(drop=True)
canonical_items = canonical_items_all[~canonical_items_all["is_boilerplate"]].reset_index(drop=True)

print(f"Boilerplate flagged: {_bp_mask.sum():,} ({_bp_mask.mean()*100:.1f}%)")
print(f"Clustering corpus (non-boilerplate): {len(canonical_items):,}  |  set-aside: {len(boilerplate_df):,}")
```
Note: the `reset_index(drop=True)` is REQUIRED — it makes `canonical_items` positions 0..M-1, which `assignment_vecs` (built next) will match.

- [ ] **Step 3: Insert the hand-audit + guardrail cell**

```python
# 6.1b (cont.) — Hand-audit: eyeball what got flagged, and assert wire tags are NOT flagged.
print("Sample FLAGGED (should be templates/filings/PRs):")
for t in boilerplate_df["title"].head(15).tolist():
    print("  •", str(t)[:90])

import numpy as _np
_rng = _np.random.default_rng(CONFIG["random_seed"])
_keep = canonical_items["title"].dropna()
print("\nSample NOT-flagged (should be real stories):")
for t in _keep.sample(min(15, len(_keep)), random_state=CONFIG["random_seed"]).tolist():
    print("  •", str(t)[:90])

# Guardrail: a pure editorial wire-tag headline must NOT be flagged.
assert not is_boilerplate_text("UPDATE 2-Boeing 787's dimmable windows not dark enough, says ANA"), \
    "Detector wrongly flags an UPDATE wire item"
assert not is_boilerplate_text("WRAPUP 6-Boeing Dreamliners grounded worldwide on battery checks"), \
    "Detector wrongly flags a WRAPUP wire item"
print("\nGuardrail OK: UPDATE/WRAPUP headlines are not treated as boilerplate.")
```

- [ ] **Step 4: Run the three cells**

Execute them. Expected: a non-trivial flagged % (roughly 10–25% of titles), flagged samples are clearly templates (REG-/NAV/PR), the not-flagged samples are real stories, and the guardrail asserts pass.

- [ ] **Step 5: Checkpoint** — save the notebook.

---

### Task 2: Reuse the existing labeled eval set (skip §7 re-sampling/re-labeling)

**Files:**
- Modify: §7 cells — guard the sample/label/export cells so the run reuses the existing 543 pairs instead of re-sampling from the now-filtered corpus.

Why: filtering boilerplate changed `canonical_items`, so re-sampling (§7) would pick different pairs and overwrite the eval set, destroying comparability with the old 0.32. §8 loads `artifacts/v2/labeled_eval_set.csv` (staged in Task 0), so §7's sampling/labeling is unnecessary.

- [ ] **Step 1: Insert a flag cell at the top of §7**

Right after the `## Section 7 — Labeled eval set` heading, insert:
```python
# 7.0 (v2) — Reuse the existing labeled eval set; do NOT re-sample/re-label (preserves the 543 pairs).
REUSE_EXISTING_EVAL = True
print("REUSE_EXISTING_EVAL =", REUSE_EXISTING_EVAL, "→ §7 sampling/labeling/export are skipped.")
```

- [ ] **Step 2: Guard the stratified-sampling cell**

In the cell starting `# 7.2 — Stratified sample, one bin at a time.`, wrap the whole body:
```python
if not REUSE_EXISTING_EVAL:
    # ... existing sampling code, indented one level ...
else:
    print("Skipped §7.2 sampling (reusing existing eval set).")
```

- [ ] **Step 3: Guard the ensemble-run cell**

In the cell starting `# 7.4` (`Run the 3-vendor ensemble across all sampled pairs`), wrap the body the same way:
```python
if not REUSE_EXISTING_EVAL:
    # ... existing ensemble-run code ...
else:
    print("Skipped §7.4 ensemble labeling (reusing existing eval set).")
```

- [ ] **Step 4: Guard the export cell**

In the cell starting `# 7.9` (`export labeled_eval_set.csv`), wrap the body the same way so it does NOT overwrite the staged CSV:
```python
if not REUSE_EXISTING_EVAL:
    # ... existing export code ...
else:
    print("Skipped §7.9 export (artifacts/v2/labeled_eval_set.csv already staged).")
```

- [ ] **Step 5: Verify §8 loads the staged set**

Execute the §7 flag cell, then the `# 8.1 — Load the eval set` cell.
Expected: `Loaded 543 labeled pairs from .../artifacts/v2/labeled_eval_set.csv`.

- [ ] **Step 6: Checkpoint** — save the notebook.

---

### Task 3: GPT-4.1 judge helper + swap both call sites

**Files:**
- Modify: §10 judge cells (the cell defining `haiku_judge_same`, and the limiter/`_retry_on_429` cell before it) and the §13 merge-judge cell (`# 13.2 — Pairwise Haiku merge judge`).

- [ ] **Step 1: Add the OpenAI judge helper**

In the cell that defines `haiku_judge_same` (starts `# 10.2 (cont.) — The judge function itself.`), append a new OpenAI-backed judge (keep `haiku_judge_same` for reference, but add and use `judge_same`):
```python
# 10.2 (v2) — OpenAI gpt-4.1 judge (replaces Haiku). Same prompt; cache keyed by model id.
from openai import AsyncOpenAI
_openai_judge_client = AsyncOpenAI()
try:
    _openai_judge_limiter = _openai_limiter            # reuse §7's 50-RPM limiter if present
except NameError:
    _openai_judge_limiter = AsyncRateLimiter(CONFIG["vendor_rate_limits_rpm"].get("openai", 50))

async def judge_same(item_row, rep_row) -> bool:
    """Ask the configured judge model whether two items are the same story. Cached by (model, id, id)."""
    model = CONFIG["judge_model"]
    a_id, b_id = sorted([item_row["item_id"], rep_row["item_id"]])
    fpath = JUDGE_CACHE_DIR / f"{_judge_key(model, a_id, b_id)}.json"
    if fpath.exists():
        return json.loads(fpath.read_text())["verdict"] == "SAME"
    await _openai_judge_limiter.acquire()
    prompt = (
        "Two financial news items — same story or different?\n\n"
        f"ITEM A: {item_row['title']}\n  {(item_row['body'] or '')[:200]}\n\n"
        f"ITEM B: {rep_row['title']}\n  {(rep_row['body'] or '')[:200]}\n\n"
        "Reply with a single word: SAME or DIFFERENT."
    )
    async def _call():
        resp = await _openai_judge_client.chat.completions.create(
            model=model, temperature=0, max_completion_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip().upper()
    text = await _retry_on_429(_call)
    verdict = "SAME" if text.startswith("SAME") else "DIFFERENT"
    fpath.write_text(json.dumps({"verdict": verdict}))
    return verdict == "SAME"

print("judge_same (gpt-4.1) ready.")
```
Note: `_judge_key`, `JUDGE_CACHE_DIR`, `_retry_on_429`, `AsyncRateLimiter` are all already defined in the §10.2 cells above this one.

- [ ] **Step 2: Point the §10 loop at `judge_same`**

In the loop cell (`# 10.3 (cont.) — The actual single-pass loop.`), change:
```python
            is_same = await haiku_judge_same(row, rep_row)
```
to:
```python
            is_same = await judge_same(row, rep_row)
```

- [ ] **Step 3: Point the §13 merge judge at gpt-4.1**

In the cell `# 13.2 — Pairwise Haiku merge judge with cache + 429 retry.`, change the cache-key line and the model line. Replace the Anthropic call with an OpenAI call:
- cache key: change `…|{CONFIG['haiku_model']}` → `…|{CONFIG['judge_model']}`
- the API call: replace
  ```python
  resp = await _anthropic_10.messages.create(
      model=CONFIG["haiku_model"], max_tokens=10, temperature=0, messages=[...])
  text = resp.content[0].text.strip().upper()
  ```
  with
  ```python
  await _openai_judge_limiter.acquire()
  resp = await _openai_judge_client.chat.completions.create(
      model=CONFIG["judge_model"], temperature=0, max_completion_tokens=5,
      messages=[{"role": "user", "content": prompt}])
  text = (resp.choices[0].message.content or "").strip().upper()
  ```
  (Keep that cell's existing `prompt`, cache-read/write, and `_retry_on_429` wrapper.)

- [ ] **Step 4: Smoke-test the judge**

In a scratch cell, run:
```python
_a = {"item_id": "x1", "title": "Apple cuts iPhone 5 orders on weak demand", "body": ""}
_b = {"item_id": "x2", "title": "Apple reduces iPhone 5 component orders, report says", "body": ""}
print("judge_same →", await judge_same(_a, _b))   # expect True (SAME)
```
Expected: prints `True`; a new cache file appears under `.cache/judge/` (model-namespaced). Delete the scratch cell after.

- [ ] **Step 5: Checkpoint** — save the notebook.

---

### Task 4: Recalibrate τ on the non-boilerplate subset (§8)

**Files:**
- Modify: §8 cells — add `is_boilerplate` to the eval frame and calibrate on the non-boilerplate subset; report old vs new τ.

- [ ] **Step 1: Add an `is_boilerplate` column to the eval frame**

Right after `# 8.1 — Load the eval set`, insert:
```python
# 8.1b (v2) — Flag boilerplate pairs (either item templated) so calibration uses the clean subset.
eval_df["is_boilerplate_pair"] = [
    is_boilerplate_text(ta) or is_boilerplate_text(tb)
    for ta, tb in zip(eval_df["item_a_title"], eval_df["item_b_title"])
]
print(f"Boilerplate pairs in eval set: {eval_df['is_boilerplate_pair'].sum()} / {len(eval_df)}")
```

- [ ] **Step 2: Calibrate on the clean subset**

In the `# 8.3 — Sweep τ` cell, change the source arrays to the non-boilerplate subset:
```python
_clean = eval_df[~eval_df["is_boilerplate_pair"]]
y_true = (_clean["final_label"] == "SAME").astype(int).to_numpy()
sim    = _clean["cosine_sim"].to_numpy()
```
(Leave the sweep/threshold-pick logic in `# 8.4` unchanged — it consumes `calib_df`.)

- [ ] **Step 3: Report old vs new τ**

The `# 8.4` cell already updates `CONFIG["tau_high"]`/`tau_low` and prints them. Confirm it prints the new values and the priors (0.88/0.54 from the old run are recorded in `pos_calibration.json` for reference).

- [ ] **Step 4: Run §8 and verify**

Execute §8 cells. Expected: a recalibrated `τ_high` (likely **non-degenerate, ~0.78–0.85**, i.e. a real precision≥0.95 point now exists on the clean subset) and `τ_low`. If `τ_high` is still 0.95+ with a "no τ reached precision≥0.95" warning, note it in findings (means even de-boilerplated cosine is weak — expected per the diagnosis).

- [ ] **Step 5: Checkpoint** — save the notebook.

---

### Task 5: Fix the §10 indexing bug + DataFrame touches + behavioral test

**Files:**
- Modify: the §10 loop-setup cell (`# 10.3 — Main loop. Reset state and run end-to-end.`) and the loop cell (`# 10.3 (cont.)`); add a `stories_df` materialization cell and an `outcomes` column.

- [ ] **Step 1: Add a "prove the bug" cell (the regression test)**

Insert a markdown cell `### 10.3a — Verify the item↔vector mapping (regression guard)`, then a code cell:
```python
# 10.3a — Show how many items the OLD (unstable-sort + arange) approach would MIS-MAP,
# then build the CORRECT id-keyed mapping and assert it.
pos_of_id = pd.Series(np.arange(len(canonical_items)), index=canonical_items["item_id"])

_old = canonical_items.sort_values("published_at").copy()          # default = unstable quicksort (the bug)
_old["pos_buggy"] = np.arange(len(_old))
_old["pos_true"]  = _old["item_id"].map(pos_of_id).to_numpy()
_n_wrong = int((_old["pos_buggy"].to_numpy() != _old["pos_true"].to_numpy()).sum())
print(f"OLD approach would mis-map {_n_wrong:,} / {len(_old):,} items "
      f"({_n_wrong/len(_old)*100:.1f}%) to the wrong embedding.")
assert _n_wrong > 0, "Expected the old approach to mis-map at least some tied-timestamp items"
```
Expected: a large `_n_wrong` (hundreds–thousands), documenting the bug was real.

- [ ] **Step 2: Fix the loop-setup cell**

Replace the body of `# 10.3 — Main loop. Reset state and run end-to-end.` with:
```python
# 10.3 — Main loop setup (v2: stable sort + id-keyed positions, with a regression guard).
from tqdm import tqdm
TAU_HIGH = CONFIG["tau_high"]; TAU_LOW = CONFIG["tau_low"]
WINDOW   = pd.Timedelta(hours=CONFIG["active_window_hours"])

sorted_items = canonical_items.sort_values("published_at", kind="stable").copy()   # STABLE
sorted_items["pos"] = sorted_items["item_id"].map(pos_of_id).to_numpy()            # TRUE position
assert (canonical_items["item_id"].to_numpy()[sorted_items["pos"].to_numpy()]
        == sorted_items["item_id"].to_numpy()).all(), "item↔vector mapping is broken"

def vec_for(item_id):
    return assignment_vecs[pos_of_id[item_id]]

stories = []; outcomes = []
print(f"Processing {len(sorted_items):,} items (τ_high={TAU_HIGH}, τ_low={TAU_LOW}) ...")
```

- [ ] **Step 3: Fix the vector access in the loop cell**

In `# 10.3 (cont.)`, change:
```python
        item_pos = int(row["pos"])
        item_vec = assignment_vecs[item_pos]
```
to:
```python
        item_pos = int(row["pos"])          # now the TRUE canonical position
        item_vec = vec_for(row["item_id"])  # id-keyed lookup (== assignment_vecs[item_pos])
```
(Leave the rest of the loop unchanged — `member_idxs` now hold correct positions, so §11/§13 work as-is.)

- [ ] **Step 4: Run §10 and verify the assert + behavioral change**

Execute 10.3a, 10.3 setup, and the loop cell. Expected: regression assert passes; the loop completes; the `Done. N stories created` line shows **materially fewer stories than 8,489** and more multi-item stories than before.

- [ ] **Step 5: Materialize `stories_df` + outcomes column**

After the `# 10.5 — Per-outcome counts` cell, insert:
```python
# 10.5b (v2) — Materialize the story accumulator as a DataFrame for inspection/portability.
sorted_items = sorted_items.reset_index(drop=True)
sorted_items["outcome"] = outcomes                      # outcomes is per-item, in loop order
stories_df = pd.DataFrame([{
    "story_id": s["story_id"], "n_items": s["n_items"],
    "member_ids": s["member_ids"], "item_clients": sorted(s["item_clients"]),
    "first_seen_at": s["first_seen_at"], "last_seen_at": s["last_seen_at"],
    "closed_at": s["closed_at"],
} for s in stories])
print(f"stories_df: {len(stories_df):,} rows | singletons={int((stories_df.n_items==1).sum()):,} "
      f"| multi-item={int((stories_df.n_items>1).sum()):,}")
stories_df.head()
```
Run it. Expected: residual/singleton counts notably better than the old 91% / 8,300.

- [ ] **Step 6: Checkpoint** — save the notebook.

---

### Task 6: §14 eval — two-number reporting + baselines + behavioral check

**Files:**
- Modify: §14 cells (`# 14.1 — Build item_id → story_id lookup; predict; …` and `# 14.2 — Compute baseline predictions`).

- [ ] **Step 1: Predict + report on full set AND non-boilerplate subset**

Replace the metrics-printing tail of `# 14.1` with (keep the existing `item_to_story`, `pred_for_pair`, `pr_f1` definitions):
```python
# 14.1 (v2) — two-number reporting on the SAME pairwise metric (NOT B-cubed; labeled accordingly).
eval_df["pred_poc"] = [pred_for_pair(a, b) for a, b in zip(eval_df["item_a_id"], eval_df["item_b_id"])]
# Boilerplate items were set aside → not in any story → treat as DIFFERENT (correctly not merged).
eval_df["pred_poc"] = eval_df["pred_poc"].fillna("DIFFERENT")

m_full  = pr_f1(eval_df["final_label"], eval_df["pred_poc"])
_clean  = eval_df[~eval_df["is_boilerplate_pair"]]
m_clean = pr_f1(_clean["final_label"], _clean["pred_poc"])
print("PAIRWISE F1 on the stratified eval set (NOT corpus B-cubed):")
print(f"  full 543 pairs   (vs old 0.323): P={m_full['precision']:.3f} R={m_full['recall']:.3f} F1={m_full['f1']:.3f}")
print(f"  non-boilerplate  ({len(_clean)} pairs): P={m_clean['precision']:.3f} R={m_clean['recall']:.3f} F1={m_clean['f1']:.3f}")
```

- [ ] **Step 2: Baselines on both sets**

In `# 14.2`, after computing `pred_b1`/`pred_b3`, add:
```python
print("\nBaselines (cosine≥0.65 / title-Jaccard≥0.5):")
for name, col in [("cosine≥0.65", "pred_b1"), ("title-Jaccard", "pred_b3")]:
    bf = pr_f1(eval_df["final_label"], eval_df[col])
    bc = pr_f1(_clean["final_label"], _clean[col])
    print(f"  {name:14s} full F1={bf['f1']:.3f}  | non-boilerplate F1={bc['f1']:.3f}")
```

- [ ] **Step 3: Behavioral check — easy near-dups now merge**

Insert a cell:
```python
# 14.1b (v2) — the bug used to split trivial near-dups; confirm they now co-cluster.
_hi = eval_df[(eval_df.final_label == "SAME") & (eval_df.cosine_sim >= 0.90)]
_tp = (_hi["pred_poc"] == "SAME").sum()
print(f"High-cosine (≥0.90) SAME pairs co-clustered: {_tp}/{len(_hi)} "
      f"(was ~0 in the buggy run)")
```
Expected: most of these are now `SAME` (TP) — direct evidence the bug fix worked.

- [ ] **Step 4: Run §14 and verify**

Execute §14 cells. Expected: full-set F1 **> 0.323** (target ~0.45–0.55 region with the gpt-4.1 judge); non-boilerplate F1 higher still; high-cosine SAME pairs mostly co-clustered.

- [ ] **Step 5: Checkpoint** — save the notebook.

---

### Task 7: §15 cost update + §16 findings to `artifacts/v2/`

**Files:**
- Modify: the §15 pricing cell (`# 15.1 — Pricing constants`) and the §16 findings-write cell (`# 16.3 — Assemble + write …`).

- [ ] **Step 1: Update judge pricing to gpt-4.1**

In `# 15.1`, add gpt-4.1 pricing and point the judge lines at it:
```python
PRICES["gpt41-input"]  = 2.00 / 1_000_000     # gpt-4.1 list price (USD/token) — verify current rate
PRICES["gpt41-output"] = 8.00 / 1_000_000
```
Then in the cost-table cell change the `gray_judge` and `merge_judge` rows to use `"gpt41-input"`/`"gpt41-output"` instead of `"haiku-input"`/`"haiku-output"`. (Leave `doc_context` on Haiku unless §9 is run with the new judge.)

- [ ] **Step 2: Write v2 findings**

In `# 16.3`, change the output path/content to write `artifacts/v2/v2_findings.md`, including the explicit metric caveat. Minimal version:
```python
_v2 = CONFIG["artifacts_dir"] / "v2_findings.md"
_v2.write_text(
    "# POC v2 findings (minimal un-break + gpt-4.1 judge)\n\n"
    f"- Pairwise F1 (full 543 pairs, NOT B-cubed): {m_full['f1']:.3f}  (old buggy run: 0.323)\n"
    f"- Pairwise F1 (non-boilerplate subset, {len(_clean)} pairs): {m_clean['f1']:.3f}\n"
    f"- Outcome mix: {dict(pd.Series(outcomes).value_counts())}\n"
    f"- Stories: {len(stories_df)} (singletons={int((stories_df.n_items==1).sum())})  vs old 8,489/8,300\n"
    f"- τ_high/τ_low (recalibrated on non-boilerplate): {CONFIG['tau_high']}/{CONFIG['tau_low']}  vs prior 0.88/0.54\n"
    f"- Judge model: {CONFIG['judge_model']}\n\n"
    "Metric caveat: pairwise F1 on a cosine-stratified eval set — NOT the spec's corpus B-cubed. "
    "Deferred work (classifier, dense corpus, B-cubed eval, independent gold set) is in "
    "docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md §6.\n"
)
print("Wrote", _v2)
```

- [ ] **Step 3: Run + verify** — execute the cells; confirm `artifacts/v2/v2_findings.md` exists and reads sensibly.

- [ ] **Step 4: Checkpoint** — save the notebook.

---

### Task 8: Clean full re-run + acceptance verification

**Files:** none (verification only).

- [ ] **Step 1: Restart kernel and run all**

Use `mcp__jupyter__restart_notebook` then execute every cell top-to-bottom (skip §9 long-doc unless you want to validate it — it's optional and not part of the number). This guarantees correct ordering (boilerplate → embeddings → load-eval → recalibrate τ → fixed loop → HDBSCAN → merge → eval).

- [ ] **Step 2: Verify acceptance criteria**

Confirm, from the executed outputs:
- (a) the §10.3 regression `assert` passed and `_n_wrong > 0` was reported;
- (b) full-set pairwise F1 printed and **> 0.323**;
- (c) non-boilerplate-subset F1 printed; baselines printed on both sets;
- (d) new outcome mix (residual %/singletons) reported and **better than 91% / 8,300**;
- (e) old vs new τ reported;
- (f) boilerplate hand-audit printed + guardrail asserts passed;
- (g) `artifacts/v2/v2_findings.md` written.

- [ ] **Step 3: Final checkpoint** — save `story_clustering_poc_v2.ipynb`.

---

## Self-Review (completed against the spec)

**Spec coverage:** §3.1 layout → Task 0. §3.2 P0 id-keyed fix + assert → Task 5. §3.3 DataFrame touches (`pos_of_id`, `stories_df`, `outcomes` column, `is_boilerplate`) → Tasks 1/4/5/6. §3.4 P2 boilerplate (detector w/ UPDATE guardrail, partition, hand-audit, recalibration, two-number eval) → Tasks 1/4/6. §3.5 judge swap both call sites + cache namespacing + cost → Tasks 3/7. §3.6 acceptance criteria → Task 8. §3.7 error handling/determinism → reused `_retry_on_429`/limiter (Task 3), stable sort (Task 5). Deferred §6 → explicitly out of scope, untouched. **No gaps.**

**Placeholder scan:** all code steps contain real code; commands have expected output; no TBD/TODO. (One value to confirm at runtime: gpt-4.1 list price in Task 7 Step 1 — marked "verify current rate".)

**Type/name consistency:** `judge_same`, `vec_for`, `pos_of_id`, `is_boilerplate_text`, `is_boilerplate_pair`, `stories_df`, `canonical_items_all`, `boilerplate_df`, `_clean`, `m_full`/`m_clean` are used consistently across tasks. `pos_of_id` is defined in Task 5 Step 2's loop-setup cell and also (independently) in Task 5 Step 1's guard cell — both are inside §10 and run in order, so the name resolves; the loop-setup definition is the authoritative one.

---

## Execution Handoff
(filled in by the writing-plans flow after the plan is saved)
