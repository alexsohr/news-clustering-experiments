#!/usr/bin/env python
"""Insert the Phase-2 experiment cells into story_clustering_poc_v3.ipynb.

All locations are found by first-line markers (never fixed indices). Idempotent:
refuses to insert a cell whose marker already exists. Run from the project root.
"""
import ast, sys
import nbformat

NB_PATH = "story_clustering_poc_v3.ipynb"

MD_11_3B = """### 11.3b (v3) — Judge-gate the residual clusters (`hdbscan_judge_gate`)

The §11.3 client check passes templated same-client clusters (the Verizon-town FP mode) and R0 attributes 12 of 35 FPs to `s11_cluster`. This gate judge-verifies every surviving cluster, ≤4 calls each: n=2 → judge the pair (DIFFERENT → drop both back to singletons); n≥3 → judge medoid-vs-farthest, peeling up to 2 members on DIFFERENT (a third DIFFERENT drops the cluster); clusters of n≥4 that pass get one extra medoid-vs-seeded-random check. Peeled/dropped members return to the singleton pool via `absorbed_idxs` before 11.4 consumes it. Flag off → defines empty stats and changes nothing."""

CODE_11_3B = '''# 11.3b (v3) — Judge-gate new_residual_stories; peel/drop incoherent members.
import random as _random

hdb_gate_stats = {"clusters_in": len(new_residual_stories), "kept": 0, "dropped": 0,
                  "peeled_members": 0, "judge_calls": 0}
judge_dropped_cluster_ids = set()   # hdb cluster ids fully dropped by the gate
judge_peeled_item_ids     = set()   # item_ids peeled out of an otherwise-kept cluster

if CONFIG.get("hdbscan_judge_gate", False) and new_residual_stories:
    _sing_pos_of_item = {s["member_ids"][0]: i for i, s in enumerate(singleton_stories)}
    _hdb_id_of_item   = {s["member_ids"][0]: int(l) for s, l in zip(singleton_stories, hdb_labels)}

    async def _gate(story):
        """Return (kept_item_positions, peeled_item_positions); empty kept => drop."""
        idxs = list(story["member_idxs"])           # positions into canonical_items
        rows = [canonical_items.iloc[i] for i in idxs]
        if len(idxs) == 2:
            hdb_gate_stats["judge_calls"] += 1
            return (idxs, []) if await judge_same(rows[0], rows[1]) else ([], idxs)
        vecs = assignment_vecs[idxs]
        cent = vecs.mean(axis=0); cent /= np.linalg.norm(cent)
        med = int(np.argmax(vecs @ cent))           # medoid = nearest to centroid
        kept, peeled = list(range(len(idxs))), []
        ok = False
        for _ in range(3):                          # initial judge + up to 2 peels
            others = [k for k in kept if k != med]
            far = min(others, key=lambda k: float(vecs[k] @ vecs[med]))
            hdb_gate_stats["judge_calls"] += 1
            if await judge_same(rows[med], rows[far]):
                ok = True; break
            kept.remove(far); peeled.append(far)
            if len(kept) < 2 or len(peeled) >= 2:
                break
        if not ok:
            return [], [k for k in range(len(idxs))]
        if len(kept) >= 4:                          # one extra seeded spot-check
            rng = _random.Random(f"{CONFIG['random_seed']}|{story['story_id']}")
            extra = rng.choice([k for k in kept if k != med])
            hdb_gate_stats["judge_calls"] += 1
            if not await judge_same(rows[med], rows[extra]):
                kept.remove(extra); peeled.append(extra)
        return [idxs[k] for k in kept], [idxs[k] for k in peeled]

    _gated = []
    for _st in tqdm(new_residual_stories, desc="cluster gate"):
        _kept, _peeled = await _gate(_st)
        for _p in _peeled:                          # peeled/dropped → back to singleton pool
            _iid = canonical_items.iloc[_p]["item_id"]
            judge_peeled_item_ids.add(_iid)
            absorbed_idxs.discard(_sing_pos_of_item[_iid])
        if len(_kept) >= 2:
            _vecs = assignment_vecs[_kept]
            _cent = _vecs.mean(axis=0); _cent /= np.linalg.norm(_cent)
            _ids  = [canonical_items.iloc[i]["item_id"] for i in _kept]
            _st.update(member_idxs=_kept, member_ids=_ids, n_items=len(_kept), centroid=_cent,
                       item_clients=set().union(*[canonical_items.iloc[i]["item_clients"] for i in _kept]))
            _gated.append(_st)
            hdb_gate_stats["kept"] += 1
            hdb_gate_stats["peeled_members"] += len(_peeled)
        else:
            hdb_gate_stats["dropped"] += 1
            judge_dropped_cluster_ids.add(_hdb_id_of_item.get(_st["member_ids"][0], -2))
    new_residual_stories = _gated
    print(f"gate: {hdb_gate_stats['kept']} kept / {hdb_gate_stats['dropped']} dropped, "
          f"{hdb_gate_stats['peeled_members']} members peeled, "
          f"{hdb_gate_stats['judge_calls']} judge calls "
          f"(paid so far this run: {JUDGE_STATS['new']})")
else:
    print("hdbscan_judge_gate=False — gate inactive (stats empty)")'''

MD_11_3C = """### 11.3c-viz — Cluster-gate outcome chart

How many clusters survived the judge gate, how many members were peeled, and what the gate cost in judge calls. Compare `dropped` here against the §11.3 client-check drops — two independent coherence filters."""

CODE_11_3C = '''# 11.3c-viz — gate outcomes as bars (no-op chart when the gate is off).
if hdb_gate_stats["clusters_in"]:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _names = ["clusters in", "kept", "dropped", "members peeled", "judge calls"]
    _vals = [hdb_gate_stats["clusters_in"], hdb_gate_stats["kept"], hdb_gate_stats["dropped"],
             hdb_gate_stats["peeled_members"], hdb_gate_stats["judge_calls"]]
    bars = ax.bar(_names, _vals, color=["tab:gray", "tab:green", "tab:red", "tab:orange", "tab:blue"], alpha=0.85)
    ax.bar_label(bars)
    ax.set_title(f"§11.3b judge gate (hdbscan_judge_gate={CONFIG.get('hdbscan_judge_gate', False)})")
    plt.tight_layout(); plt.show()
else:
    print("(no clusters to chart)")'''

MD_13_0 = """### 13.0 (v3) — Second-pass singleton re-offer (`second_sweep`)

Single-pass assignment is order-sensitive: an item arriving before its story-mates seeds a story that later items may not match. This pass re-offers every surviving singleton to the multi-item stories (shared client, within `second_sweep_window_h` of the story's span, cosine ≥ τ_low), gated by the same gray-zone judge. **Caution from R0:** the supplemental slice shows long-gap high-cosine pairs are nearly all templated DIFFERENTs — this sweep is expected to need a tight window and may be rejected; the slice is its guard."""

CODE_13_0 = '''# 13.0 (v3) — Re-offer singletons to multi-stories; judge-gated absorption.
second_sweep_ids = set()
if CONFIG.get("second_sweep", False):
    import copy
    _sw = pd.Timedelta(hours=CONFIG["second_sweep_window_h"])
    sweep_stories = copy.deepcopy(final_stories)
    _multi   = [s for s in sweep_stories if s["n_items"] > 1]
    _singles = [s for s in sweep_stories if s["n_items"] == 1]
    _cents   = np.stack([s["centroid"] for s in _multi])
    _n_judge0 = JUDGE_STATS["new"] + JUDGE_STATS["cached"]
    _absorbed_count = 0
    for _s in tqdm(_singles, desc="second sweep"):
        _iid  = _s["member_ids"][0]
        _ipos = _s["member_idxs"][0]
        _row  = canonical_items.iloc[_ipos]
        _vec  = assignment_vecs[_ipos]
        _sims = _cents @ _vec
        _order = np.argsort(-_sims)
        for _k in _order[:3]:                      # top-3 candidates at most
            if _sims[_k] < CONFIG["tau_low"]: break
            _cand = _multi[_k]
            if not (_cand["item_clients"] & _row["item_clients"]): continue
            if not (_cand["first_seen_at"] - _sw <= _row["published_at"] <= _cand["last_seen_at"] + _sw):
                continue
            _rep = canonical_items.iloc[_cand["member_idxs"][0]]
            if await judge_same(_row, _rep):
                assign_to_story(_cand, _ipos, _row)
                _cents[_k] = _cand["centroid"]
                second_sweep_ids.add(_iid)
                _absorbed_count += 1
                break
    final_stories = _multi + [s for s in _singles if s["member_ids"][0] not in second_sweep_ids]
    print(f"second sweep: absorbed {_absorbed_count} of {len(_singles)} singletons "
          f"({JUDGE_STATS['new'] + JUDGE_STATS['cached'] - _n_judge0} judge lookups)")
else:
    print("second_sweep=False — pass inactive")'''

MD_13_1B = """### 13.1b (v3) — Merge-pool repair (`merge_include_closed`, `merge_sim_v3`)

§10.4 expiry closes 7,282/7,288 stories before §13 runs, and 13.1 filters to open stories — so the v2 merge pool was effectively just the §11 clusters (4 candidates all run). For a batch POC expiry is a streaming concern, not a correctness one: this override rebuilds the pool including closed multi-stories and sweeps the similarity threshold (candidate counts per τ printed and charted) before committing `merge_sim_v3` (falls back to `centroid_merge_sim`). The merge judge still gates every union."""

CODE_13_1B = '''# 13.1b (v3) — Rebuild active_multi/candidates with the repaired pool + threshold sweep.
if CONFIG.get("merge_include_closed", False) or CONFIG.get("merge_sim_v3"):
    MERGE_SIM = CONFIG.get("merge_sim_v3") or CONFIG["centroid_merge_sim"]
    _pool = [s for s in final_stories if s["n_items"] > 1
             and (CONFIG.get("merge_include_closed", False) or s["closed_at"] is None)]
    print(f"repaired pool: {len(_pool)} multi-item stories "
          f"(closed included: {CONFIG.get('merge_include_closed', False)})")
    _cents = np.stack([s["centroid"] for s in _pool])
    _sim = _cents @ _cents.T
    _iu = np.triu_indices(len(_pool), k=1)
    _shared = np.zeros(len(_iu[0]), dtype=bool)
    for _n, (_i, _j) in enumerate(zip(*_iu)):
        _shared[_n] = bool(_pool[_i]["item_clients"] & _pool[_j]["item_clients"])
    sweep_counts = {}
    for _t in (0.85, 0.825, 0.80, 0.775, 0.75, 0.70):
        sweep_counts[_t] = int(((_sim[_iu] >= _t) & _shared).sum())
    print("candidates per threshold:", sweep_counts)
    _mask = (_sim[_iu] >= MERGE_SIM) & _shared
    active_multi = _pool
    candidates = [(int(_i), int(_j), float(_sim[_i, _j]),
                   sorted(_pool[_i]["item_clients"] & _pool[_j]["item_clients"]))
                  for _i, _j, _m in zip(_iu[0], _iu[1], _mask) if _m]
    print(f"committed MERGE_SIM={MERGE_SIM}: {len(candidates)} candidates → §13.2 judge")
else:
    sweep_counts = None
    print("merge-pool repair inactive (v2 pool from 13.1 stands)")'''

MD_13_1C = """### 13.1b-viz — Candidate count vs merge threshold

The sweep that justifies the committed `merge_sim_v3`: how fast the judge-gated candidate pool grows as the threshold drops. Judge cost is ~$0.0002/candidate, so even the 0.70 point is cheap."""

CODE_13_1C = '''# 13.1b-viz — line chart of the threshold sweep.
if sweep_counts:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    _ts = sorted(sweep_counts); _cs = [sweep_counts[t] for t in _ts]
    ax.plot(_ts, _cs, marker="o", color="tab:purple")
    for _t, _c in zip(_ts, _cs):
        ax.annotate(str(_c), (_t, _c), textcoords="offset points", xytext=(0, 6), fontsize=8)
    _committed = CONFIG.get("merge_sim_v3") or CONFIG["centroid_merge_sim"]
    ax.axvline(_committed, color="green", ls="--", lw=1, label=f"committed {_committed}")
    ax.set_xlabel("centroid-sim threshold"); ax.set_ylabel("# candidates (shared client)")
    ax.set_title("Merge candidates vs threshold (repaired pool)")
    ax.legend(); plt.tight_layout(); plt.show()
else:
    print("(no sweep to chart)")'''

ENV_HOOK = '''

# v3 run-level overrides: flags/tags injected per run via the V3_CONFIG_OVERRIDES env var
# (JSON object), so measured runs never require editing this cell.
import os as _os
_env_over = json.loads(_os.environ.get("V3_CONFIG_OVERRIDES", "{}"))
if _env_over:
    CONFIG.update(_env_over)
    print(f"env overrides applied: {_env_over}")'''


def find_cell(nb, marker, cell_type=None):
    hits = [i for i, c in enumerate(nb.cells)
            if c.source.strip().startswith(marker) and (cell_type is None or c.cell_type == cell_type)]
    if len(hits) != 1:
        sys.exit(f"marker {marker!r}: expected exactly 1 hit, got {hits}")
    return hits[0]


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    if any(c.source.strip().startswith("# 11.3b (v3)") for c in nb.cells):
        sys.exit("Phase-2 cells already present — refusing to double-insert.")

    # --- bottom-up insertions -------------------------------------------------
    i_131 = find_cell(nb, "# 13.1 — Build the merge candidate pool.", "code")
    nb.cells[i_131 + 1:i_131 + 1] = [
        nbformat.v4.new_markdown_cell(MD_13_1B), nbformat.v4.new_code_cell(CODE_13_1B),
        nbformat.v4.new_markdown_cell(MD_13_1C), nbformat.v4.new_code_cell(CODE_13_1C),
    ]
    i_131md = find_cell(nb, "### 13.1 Enumerate merge candidates", "markdown")
    nb.cells[i_131md:i_131md] = [
        nbformat.v4.new_markdown_cell(MD_13_0), nbformat.v4.new_code_cell(CODE_13_0),
    ]
    i_113 = find_cell(nb, "# 11.3 — Build coherent multi-stories", "code")
    nb.cells[i_113 + 1:i_113 + 1] = [
        nbformat.v4.new_markdown_cell(MD_11_3B), nbformat.v4.new_code_cell(CODE_11_3B),
        nbformat.v4.new_markdown_cell(MD_11_3C), nbformat.v4.new_code_cell(CODE_11_3C),
    ]

    # --- in-place edits -------------------------------------------------------
    i_14b = find_cell(nb, "# 1.4b (v3) — Overlay", "code")
    if "V3_CONFIG_OVERRIDES" not in nb.cells[i_14b].source:
        nb.cells[i_14b].source += ENV_HOOK

    i_attr = find_cell(nb, "# 14.1d (v3) — Attribution", "code")
    old = ('    if None not in (ha, hb) and ha == hb != -1 and not (a in absorbed_ids and b in absorbed_ids):\n'
           '        return "cluster_dropped_client"')
    new = ('    if a in judge_peeled_item_ids or b in judge_peeled_item_ids:\n'
           '        return "cluster_dropped_judge"\n'
           '    if None not in (ha, hb) and ha == hb != -1 and not (a in absorbed_ids and b in absorbed_ids):\n'
           '        return ("cluster_dropped_judge" if ha in judge_dropped_cluster_ids\n'
           '                else "cluster_dropped_client")')
    if old not in nb.cells[i_attr].source:
        sys.exit("14.1d edit anchor not found — aborting before writing")
    nb.cells[i_attr].source = nb.cells[i_attr].source.replace(old, new)
    hook = ('absorbed_ids = {mid for s in globals().get("new_residual_stories", []) for mid in s["member_ids"]}')
    gate_defaults = (hook + '\n'
                     'judge_dropped_cluster_ids = globals().get("judge_dropped_cluster_ids", set())\n'
                     'judge_peeled_item_ids     = globals().get("judge_peeled_item_ids", set())')
    nb.cells[i_attr].source = nb.cells[i_attr].source.replace(hook, gate_defaults)

    # --- syntax check on every code cell -------------------------------------
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            try:
                compile(c.source, f"c{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as e:
                bad.append((i, str(e)))
    if bad:
        sys.exit(f"syntax failures, not writing: {bad}")

    nbformat.write(nb, NB_PATH)
    print(f"OK: notebook now {len(nb.cells)} cells; Phase-2 cells inserted, env hook + attribution edit applied.")


if __name__ == "__main__":
    main()
