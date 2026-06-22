#!/usr/bin/env python
"""Insert §17 (experiment ledger + progression chart + follow-ups) at the end of
story_clustering_poc_v3.ipynb, and point §16.1's decision logic at the v3 headline
metric. Run only when no driver is writing the notebook."""
import ast, sys
import nbformat

NB = "story_clustering_poc_v3.ipynb"

MD_17 = """---

## Section 17 — v3 experiment ledger & progression

Every measured run wrote one JSON record to `artifacts/v3/experiments/` (§14.7). This section renders the cross-run comparison — the without-git audit trail of what was tried, what it did, and whether it was accepted under the rule *ΔF1 ≥ +0.02, or structural ledger evidence, with no guard regression*."""

CODE_17_1 = '''# 17.1 — Load all ledger records into a comparison table.
import glob as _glob
_led = []
for _p in sorted(_glob.glob(str(CONFIG["artifacts_dir"] / "experiments" / "*.json"))):
    _r = json.loads(Path(_p).read_text())
    _m = (_r.get("metrics") or {}).get("frozen_full") or {}
    _supp = (_r.get("metrics") or {}).get("supplemental_gt72h") or {}
    _led.append({
        "run_id": _r["run_id"], "phase": _r.get("phase"), "tag": _r.get("tag"),
        "P": _m.get("precision"), "R": _m.get("recall"), "F1": _m.get("f1"),
        "FP": _m.get("fp"), "FN": _m.get("fn"), "supp_FP": _supp.get("fp"),
        "verdict": (_r.get("stop_rule") or {}).get("verdict"),
        "accepted": _r.get("accepted"),
        "judge_paid": (_r.get("cost") or {}).get("judge_calls_new"),
    })
ledger_df = pd.DataFrame(_led)
ledger_df[["P", "R", "F1"]] = ledger_df[["P", "R", "F1"]].round(3)
ledger_df'''

MD_17V = """### 17.1-viz — F1 progression across the iteration

The headline chart: frozen-set F1 per measured run (markers: green = accepted, red = rejected/aborted, gray = baseline), with precision/recall context and the SHIP/floor lines. Rejected runs show what the guards caught; the accepted path is the story of the iteration."""

CODE_17V = '''# 17.1-viz — progression chart with accept/reject markers.
_plot = ledger_df.dropna(subset=["F1"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(_plot.index, _plot["F1"], color="tab:blue", lw=1.2, alpha=0.5, zorder=1)
for _, r in _plot.iterrows():
    c = "tab:gray" if r["accepted"] is None else ("tab:green" if r["accepted"] else "tab:red")
    ax.scatter(r.name, r["F1"], color=c, s=70, zorder=3)
    ax.annotate(f'{r["tag"]}\\n{r["F1"]:.3f}', (r.name, r["F1"]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
ax.plot(_plot.index, _plot["P"], ls=":", color="tab:purple", lw=1, label="precision")
ax.plot(_plot.index, _plot["R"], ls=":", color="tab:orange", lw=1, label="recall")
ax.axhline(SHIP_LINE, color="green", ls="--", lw=1.2, label=f"SHIP {SHIP_LINE}")
ax.axhline(FLOOR_LINE, color="red", ls=":", lw=1.2, label=f"floor {FLOOR_LINE}")
ax.set_xticks(_plot.index, _plot["phase"], fontsize=9)
ax.set_ylim(0.6, 1.0); ax.set_ylabel("frozen-set metric")
ax.set_title("v3 iteration: F1 progression (green=accepted, red=rejected)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()'''

MD_17F = """### 17.2 — Follow-ups (documented, not executed)

| Item | Why deferred | Evidence |
|---|---|---|
| Merge-pool repair re-test | Rejected pre-rubric: merge judge approved template-family merges (+4 long-range FPs, 0 TPs). The v2-rubric + escalated judge may now reject those — re-sweep `merge_sim_v3` with the current judge. | P2b ledger |
| Second-sweep re-test | Aborted: cache-key analysis showed it cannot flip `judge_said_no` FNs; needs top-1 candidate + ≤7d window + current judge. | P2c ledger |
| HDBSCAN geometry sweep (5a) | §11 now contributes ~0 frozen FPs and few merges after the gate; geometry changes feed the same gate. Revisit only if §11 recall matters in production. | P4 ledgers |
| Cross-client eval slice | The frozen set is same-client by construction — cross-client merges are unmeasurable until a slice without that constraint exists. | plan §"out of scope" |
| Re-embedding (richer input) | Invalidates τ calibration + every cache; fusion attacks the same weakness for ~$0. | plan |
| GBM fusion upgrade | LogReg already at OOF AUC 0.912 (doc's GBM: 0.919) — marginal. | §8.7b |
| Judge-escalation circularity | gpt-5.2 was ensemble labeler #2; agreement with labels is partially circular. Production should validate the escalated judge against fresh human labels. | P4c/P4d ledgers |"""


def main():
    nb = nbformat.read(NB, as_version=4)
    if any(c.source.startswith("# 17.1 —") for c in nb.cells if c.cell_type == "code"):
        sys.exit("§17 already present.")
    nb.cells.extend([
        nbformat.v4.new_markdown_cell(MD_17),  nbformat.v4.new_code_cell(CODE_17_1),
        nbformat.v4.new_markdown_cell(MD_17V), nbformat.v4.new_code_cell(CODE_17V),
        nbformat.v4.new_markdown_cell(MD_17F),
    ])
    # §16.1 decision should read the v3 headline (hygiene metric), not the v2-style one.
    dec = [c for c in nb.cells if c.cell_type == "code" and "poc_f1" in c.source
           and ("decision" in c.source.lower() or "SHIP" in c.source)]
    for c in dec:
        if "metrics_hygiene" not in c.source:
            c.source = ('# v3: decision reads the hygiene headline (falls back to v2-style poc_f1)\n'
                        'poc_f1 = (metrics_hygiene["f1"] if "metrics_hygiene" in dir() else poc_f1)\n'
                        + c.source)
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            try:
                compile(c.source, f"c{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as e:
                bad.append((i, str(e)))
    if bad:
        sys.exit(f"syntax failures: {bad}")
    nbformat.write(nb, NB)
    print(f"OK: §17 appended ({len(nb.cells)} cells); §16.1 decision patched ({len(dec)} cell(s)).")


if __name__ == "__main__":
    main()
