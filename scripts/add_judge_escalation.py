#!/usr/bin/env python
"""Add weak-band judge escalation: judge_same(sim=...) routes the configured cosine
band to a stronger model. All call sites updated to pass the similarity they know."""
import ast, sys
import nbformat

NB = "story_clustering_poc_v3.ipynb"
nb = nbformat.read(NB, as_version=4)


def cell(marker):
    hits = [c for c in nb.cells if c.cell_type == "code" and c.source.startswith(marker)]
    assert len(hits) == 1, f"{marker!r}: {len(hits)} hits"
    return hits[0]


def sub(c, old, new):
    assert old in c.source, f"anchor missing in {c.source[:60]!r}: {old[:70]!r}"
    c.source = c.source.replace(old, new)


# --- judge cell: escalation router + sim param ------------------------------
j = cell("# 10.2 (cont., v3)")
sub(j,
    'async def judge_same(item_row, rep_row) -> bool:\n    model = CONFIG["judge_model"]',
    '''def _judge_model_for(sim):
    """Escalate the weak cosine band to a stronger model when configured."""
    esc = CONFIG.get("judge_escalation_model")
    if esc and sim is not None:
        lo, hi = CONFIG.get("judge_escalation_band", (0.60, 0.75))
        if lo <= sim < hi:
            return esc
    return CONFIG["judge_model"]

async def judge_same(item_row, rep_row, sim=None) -> bool:
    model = _judge_model_for(sim)''')
sub(j,
    "resp = await _openai_judge_client.chat.completions.create(\n            model=model, temperature=0, max_completion_tokens=5,",
    "resp = await _openai_judge_client.chat.completions.create(\n"
    "            model=model, temperature=0,\n"
    "            # escalation models may spend reasoning tokens before the answer\n"
    "            max_completion_tokens=5 if model == CONFIG[\"judge_model\"] else 4000,")

# --- loop: pass best_sim -----------------------------------------------------
loop = cell("# 10.3 (cont.)")
sub(loop, "is_same = await judge_same(row, rep_row)",
          "is_same = await judge_same(row, rep_row, sim=best_sim)")

# --- cluster gate: pass member-pair sims -------------------------------------
g = cell("# 11.3b (v3)")
sub(g,
    'if len(idxs) == 2:\n            hdb_gate_stats["judge_calls"] += 1\n            return (idxs, []) if await judge_same(rows[0], rows[1]) else ([], idxs)',
    'if len(idxs) == 2:\n            hdb_gate_stats["judge_calls"] += 1\n'
    '            _s2 = float(assignment_vecs[idxs[0]] @ assignment_vecs[idxs[1]])\n'
    '            return (idxs, []) if await judge_same(rows[0], rows[1], sim=_s2) else ([], idxs)')
sub(g,
    'hdb_gate_stats["judge_calls"] += 1\n            if await judge_same(rows[med], rows[far]):',
    'hdb_gate_stats["judge_calls"] += 1\n'
    '            if await judge_same(rows[med], rows[far], sim=float(vecs[far] @ vecs[med])):')
sub(g,
    'if not await judge_same(rows[med], rows[extra]):',
    'if not await judge_same(rows[med], rows[extra], sim=float(vecs[extra] @ vecs[med])):')

# --- judge-eval cell: pass labeled-pair cosine --------------------------------
e = cell("# 10.2c (v3)")
sub(e,
    'same = await judge_same(_rows_by_id.loc[r["item_a_id"]], _rows_by_id.loc[r["item_b_id"]])',
    'same = await judge_same(_rows_by_id.loc[r["item_a_id"]], _rows_by_id.loc[r["item_b_id"]],\n'
    '                                sim=r["cosine_sim"])')

# --- second sweep: pass centroid sim (flag-off but keep consistent) -----------
s = cell("# 13.0 (v3)")
sub(s, "if await judge_same(_row, _rep):",
       "if await judge_same(_row, _rep, sim=float(_sims[_k])):")

for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        try:
            compile(c.source, f"c{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            sys.exit(f"syntax failure cell {i}: {exc}")
nbformat.write(nb, NB)
print("escalation seam added; all judge_same call sites pass sim")
