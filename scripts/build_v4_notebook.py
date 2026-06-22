"""Build a fresh, clean story_clustering_poc_v4.ipynb from the validated v3 notebook.

Strategy (per docs/superpowers/plans/2026-06-17-poc-clustering-v4-paragraph-chunking.md):
- Reuse v3's *validated* cell bodies for ported sections (byte-faithful -> reproduces 0.870).
- DROP dead-experiment cells (v2-changes md, overlay archaeology, stale-audit, the §7
  re-labeling branch + §7.10 generation, the §9 removed-chunking stub).
- REPLACE the title, collapse §1.4 + §1.4b overlay into one clean CONFIG (ship defaults baked),
  and replace §3.6 with eval-seeded sampling (flag-gated so the v3-exact port gate still works).
- EDIT the imports cell to load the unit-tested scripts/v4_chunking.py helpers.
- Clear all outputs/execution_count -> a clean, unexecuted notebook.

Chunk-path cells (Phase 3+) are added later via the Jupyter MCP.
Run: .venv/bin/python scripts/build_v4_notebook.py
"""
import json
from pathlib import Path

SRC = Path("story_clustering_poc_v3.ipynb")
DST = Path("story_clustering_poc_v4.ipynb")

# --- dead-experiment cells to drop (v3 indices) ------------------------------
DROP = {
    1,            # "v2 changes" archaeology markdown
    11, 12,       # §1.4b overlay md+code (collapsed into CONFIG)
    17,           # §1.4e stale-code audit markdown
    163,          # §9 removed-chunking stub
}
DROP |= set(range(112, 137))   # §7.1–§7.10c generation / re-labeling branch (keep §7.0 loader=111)

# --- replacement cell sources ------------------------------------------------
TITLE_MD = (
    "# Story Clustering — POC v4 (paragraph chunking + judge A/B)\n\n"
    "Fresh notebook for the v4 experiment: body **paragraph-chunk** vector representation "
    "across §6/§8/§10/§11, plus an LLM-judge **A/B** (`chunk_pair` vs `full_body`).\n\n"
    "- Spec: `docs/superpowers/specs/2026-06-17-poc-clustering-v4-paragraph-chunking-design.md`\n"
    "- Plan: `docs/superpowers/plans/2026-06-17-poc-clustering-v4-paragraph-chunking.md`\n"
    "- Pure helpers (unit-tested): `scripts/v4_chunking.py`\n\n"
    "`use_chunking=False` reproduces the v3 single-vector baseline; the **port-correctness gate** "
    "is a v3-exact 10k run (must hit F1≈0.870) before any chunk comparison is trusted."
)

CONFIG_CODE = '''# §1.4 — CONFIG (single source of truth). v4: collapses the v3 §1.4 + §1.4b overlay into one
# clean dict, bakes the v3 SHIP config (from artifacts/v3/.../P6_final-ship.json), and adds v4 keys.
CONFIG = {
    # --- Paths ---------------------------------------------------------------
    "data_dir":        Path.cwd() / "dataset" / "financial-news-multisource" / "data",
    "primary_source":  "bloomberg_reuters",
    "artifacts_dir":   Path.cwd() / "artifacts" / "v4",
    "cache_dir":       Path.cwd() / ".cache",
    "frozen_eval_path":       Path.cwd() / "artifacts" / "v2" / "labeled_eval_set.csv",       # 543-pair CI gate
    "supplemental_eval_path": Path.cwd() / "artifacts" / "v3" / "labeled_eval_supplemental.csv",

    # --- Sample size ---------------------------------------------------------
    # Port gate runs at 10k (v3-exact) to reproduce 0.870; the experiment switches to 3000.
    "target_canonical_items": 10_000,

    # --- Thresholds (v3 SHIP values) -----------------------------------------
    "tau_high":            0.94,
    "tau_low":             0.54,
    "minhash_threshold":   0.85,
    "minhash_num_perm":    128,
    "active_window_hours": 72,
    "centroid_merge_sim":  0.85,

    # --- HDBSCAN (§11) -------------------------------------------------------
    "hdbscan_min_cluster_size":         2,
    "hdbscan_min_samples":              2,
    "hdbscan_metric":                   "cosine",
    "hdbscan_cluster_selection_method": "eom",

    # --- Embedding (§6) ------------------------------------------------------
    "embed_model": "text-embedding-3-large",
    "embed_dims":  1024,
    "lede_chars":  600,

    # --- LLM models (v3 SHIP) ------------------------------------------------
    "judge_model":           "gpt-4.1-mini",
    "judge_escalation_model": "gpt-5.2",        # SHIP: escalate the whole band -> gpt-5.2
    "judge_escalation_band": (0.0, 1.01),       # [lo, hi): every gray-zone call escalates
    "metadata_model":        "gpt-4.1",
    "sonnet_model":          "claude-sonnet-4-6",
    "openai_judge_model":    "gpt-5.2",
    "gemini_judge_model":    "gemini-3.5-flash",

    # --- Rate limits ---------------------------------------------------------
    "vendor_rate_limits_rpm": {"sonnet": 25, "haiku": 40, "openai": 50, "gemini": 50},

    # --- §10 / §13 gates (v3 SHIP) -------------------------------------------
    "judge_prompt_version": "v2",
    "judge_rep":            "first",
    "use_fusion_gates":     True,
    "hdbscan_judge_gate":   True,
    "merge_include_closed": False,
    "merge_sim_v3":         None,
    "second_sweep":         False,
    "second_sweep_window_h": 336,
    "run_tag":              "v4-baseline",

    # --- Client universe / eval ----------------------------------------------
    "client_universe_size": 20,
    "eval_pairs_per_bin":   60,
    "eval_cosine_bins": [
        (0.20, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.60),
        (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80),
        (0.80, 0.85), (0.85, 1.00),
    ],
    "human_spotcheck_band": (0.55, 0.75),

    # --- v4: chunking + judge A/B --------------------------------------------
    "use_chunking":          False,        # master switch; False = v3 single-vector baseline
    "eval_seeded_sampling":  False,        # False = v3-exact sample (port gate); True = seed eval items
    "chunk_min_tokens":      25,
    "chunk_max_tokens":      400,
    "max_chunks_per_item":   12,
    "judge_text_mode":       "chunk_pair",  # "chunk_pair" | "full_body"

    # --- Determinism ---------------------------------------------------------
    "random_seed": 42,
}

CONFIG["artifacts_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["cache_dir"].mkdir(parents=True, exist_ok=True)
for _d in ("experiments", "checkpoints"):
    (CONFIG["artifacts_dir"] / _d).mkdir(parents=True, exist_ok=True)
assert CONFIG["frozen_eval_path"].exists(), "frozen v2 eval set missing — check artifacts/v2/"

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
print(f"CONFIG defined: {len(CONFIG)} keys | artifacts -> {CONFIG['artifacts_dir']}")
print(f"use_chunking={CONFIG['use_chunking']}  eval_seeded_sampling={CONFIG['eval_seeded_sampling']}  "
      f"target={CONFIG['target_canonical_items']:,}  judge_text_mode={CONFIG['judge_text_mode']!r}")
'''

SAMPLING_CODE = '''# 3.6 (v4) — Corpus sampling. Two modes, flag-gated:
#   eval_seeded_sampling=False -> v3-EXACT (sample then assign item_id) so the 10k port gate
#                                 reproduces v3's corpus and F1≈0.870.
#   eval_seeded_sampling=True  -> assign item_id on the FULL pool, then force-include every
#                                 eval-set item + random-fill to target, so the 543-pair eval
#                                 stays measurable end-to-end at small N (3000).
import uuid

def item_id_from_url(url):
    if pd.isna(url) or not url:
        return None
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(url)))

target = CONFIG["target_canonical_items"]

if CONFIG.get("eval_seeded_sampling", False):
    wdf = working_df.copy()
    wdf["item_id"] = wdf["url"].apply(item_id_from_url)
    wdf = wdf[wdf["item_id"].notna()].reset_index(drop=True)
    _eval = pd.read_csv(CONFIG["frozen_eval_path"], comment="#")
    eval_item_ids = set(_eval["item_a_id"]) | set(_eval["item_b_id"])
    items_df = v4c.build_eval_seeded_sample(wdf, eval_item_ids, target, CONFIG["random_seed"])
    _present = int(items_df["item_id"].isin(eval_item_ids).sum())
    print(f"[eval-seeded] items_df={len(items_df):,} | eval items present={_present:,}/{len(eval_item_ids):,}")
else:
    if len(working_df) > target:
        sampled = working_df.sample(n=target, random_state=CONFIG["random_seed"]).reset_index(drop=True)
    else:
        sampled = working_df.copy()
        print(f"⚠️  pool ({len(working_df):,}) < target ({target:,}); using all of it")
    sampled["item_id"] = sampled["url"].apply(item_id_from_url)
    items_df = sampled[sampled["item_id"].notna()].reset_index(drop=True)
    print(f"[v3-exact] items_df={len(items_df):,}")

print(f"items_df ready: {len(items_df):,} items × {len(items_df.columns)} columns")
'''

IMPORTS_APPEND = '''

# --- v4: unit-tested pure helpers (paragraph chunking, max-pool sim, eval seeding, judge text) ---
import sys as _sys
_sys.path.insert(0, str(Path.cwd() / "scripts"))
import v4_chunking as v4c
print("v4_chunking loaded:", [n for n in dir(v4c) if not n.startswith("_")])
'''


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def clean(cell):
    """Clear outputs/exec count so the built notebook is unexecuted."""
    cell = dict(cell)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def main():
    nb = json.loads(SRC.read_text())
    src_cells = nb["cells"]
    out = []
    for i, cell in enumerate(src_cells):
        if i in DROP:
            continue
        if i == 0:
            out.append(md_cell(TITLE_MD))
        elif i == 6:                       # imports + v4 helper loader
            base = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            out.append(code_cell(base + IMPORTS_APPEND))
        elif i == 10:                      # collapsed clean CONFIG
            out.append(code_cell(CONFIG_CODE))
        elif i == 53:                      # eval-seeded §3.6
            out.append(code_cell(SAMPLING_CODE))
        else:
            out.append(clean(cell))
    nb["cells"] = out
    DST.write_text(json.dumps(nb, indent=1))
    n_code = sum(1 for c in out if c["cell_type"] == "code")
    n_md = sum(1 for c in out if c["cell_type"] == "markdown")
    print(f"wrote {DST}: {len(out)} cells ({n_code} code, {n_md} md) | dropped {len(src_cells)-len(out)} from v3")


if __name__ == "__main__":
    main()
