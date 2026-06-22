#!/usr/bin/env python
"""Insert the Phase-3 fusion cells (§8.7a–d) into story_clustering_poc_v3.ipynb.
Marker-located, idempotent, syntax-checked before writing."""
import ast, sys
import nbformat

NB_PATH = "story_clustering_poc_v3.ipynb"

MD_87 = """### 8.7 (v3) — Feature-fusion scorer (replaces raw-cosine gates when `use_fusion_gates`)

The diagnosis doc §4 showed a fused scorer clears the cosine ceiling (OOF AUC 0.90+ vs 0.75 cosine-only) but v2 never wired it in. Cells 8.7a–c train a logistic regression on the 543 labeled pairs over **runtime-computable features only** (cosine, token/MinHash Jaccard, day-gap, length ratio, numeric- and capitalized-token mismatch — the last two aim at the templated-PR FP mode), cross-validated with **GroupKFold over pair-graph connected components** so pairs sharing an item never straddle train/test. Gates `p_high` (OOF precision ≥ 0.99) and `p_low` (OOF recall ≥ 0.95) replace τ_high/τ_low in §10 when the flag is on.

**Leakage protocol:** the frozen eval set is the training data — post-fusion frozen-set F1 is optimistic. The honest companions are the OOF metrics below and the supplemental slice (never trained on)."""

CODE_87A = '''# 8.7a (v3) — Pair-feature builder (everything computable at §10 decision time).
_CAPS_RE = re.compile(r"\\b[A-Z][a-zA-Z]{2,}\\b")
_NUM_RE  = re.compile(r"\\d+(?:\\.\\d+)?")

def _jac(a: set, b: set) -> float:
    if not a and not b:
        return 1.0          # two empty sets are identical, not maximally different
    return len(a & b) / max(len(a | b), 1)

def _fus_text(row):
    t = row["title"] or ""
    return t, f"{t} {(row['body'] or '')[:CONFIG['lede_chars']]}"

FUSION_FEATURES = ["cosine", "title_jac", "minhash_jac", "dt_days",
                   "len_ratio", "num_mismatch", "caps_mismatch"]

def pair_features(row_a, row_b, cosine, ts_a=None, ts_b=None):
    ta, fa = _fus_text(row_a); tb, fb = _fus_text(row_b)
    tok_a, tok_b = set(tokens(ta)), set(tokens(tb))
    mh_a = minhashes.get(row_a["item_id"]); mh_b = minhashes.get(row_b["item_id"])
    mh_j = float(mh_a.jaccard(mh_b)) if (mh_a is not None and mh_b is not None) else _jac(tok_a, tok_b)
    ts_a = row_a["published_at"] if ts_a is None else ts_a
    ts_b = row_b["published_at"] if ts_b is None else ts_b
    dt_days = min(abs((ts_a - ts_b) / pd.Timedelta(days=1)), 7.0)  # day-quantized corpus (§2.8b)
    la, lb = len(fa), len(fb)
    return [float(cosine), _jac(tok_a, tok_b), mh_j, float(dt_days),
            min(la, lb) / max(la, lb, 1),
            1.0 - _jac(set(_NUM_RE.findall(fa)), set(_NUM_RE.findall(fb))),
            1.0 - _jac(set(_CAPS_RE.findall(fa)), set(_CAPS_RE.findall(fb)))]

print(f"pair_features ready: {FUSION_FEATURES}")'''

MD_87B = """### 8.7b — Train + honest cross-validation

Groups = connected components of the pair graph (union-find over item ids): the diagnosis doc's own caveat was that pairs sharing an item leak across folds. OOF probabilities drive everything downstream — gate calibration and the reported AUC."""

CODE_87B = '''# 8.7b (v3) — LogReg with GroupKFold-by-component; OOF probabilities.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

_rows_fus = canonical_items_all.set_index("item_id", drop=False)
_tsa = pd.to_datetime(eval_df["item_a_published_at"])
_tsb = pd.to_datetime(eval_df["item_b_published_at"])
X_fus = np.array([pair_features(_rows_fus.loc[r.item_a_id], _rows_fus.loc[r.item_b_id],
                                r.cosine_sim, ta, tb)
                  for r, ta, tb in zip(eval_df.itertuples(), _tsa, _tsb)])
y_fus = (eval_df["final_label"] == "SAME").astype(int).to_numpy()

_par = {}
def _find(x):
    _par.setdefault(x, x)
    while _par[x] != x:
        _par[x] = _par[_par[x]]; x = _par[x]
    return x
for _a, _b in zip(eval_df["item_a_id"], eval_df["item_b_id"]):
    _ra, _rb = _find(_a), _find(_b)
    if _ra != _rb: _par[_ra] = _rb
fus_groups = np.array([_find(a) for a in eval_df["item_a_id"]])

oof_fus = np.zeros(len(y_fus))
for _tr, _te in GroupKFold(n_splits=5).split(X_fus, y_fus, fus_groups):
    _m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    _m.fit(X_fus[_tr], y_fus[_tr])
    oof_fus[_te] = _m.predict_proba(X_fus[_te])[:, 1]

auc_fusion = roc_auc_score(y_fus, oof_fus)
auc_cosine = roc_auc_score(y_fus, eval_df["cosine_sim"])
fusion_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_fus, y_fus)
print(f"pair-graph components: {len(set(fus_groups))} | "
      f"OOF AUC fusion={auc_fusion:.3f} vs cosine-only={auc_cosine:.3f}")'''

MD_87C = """### 8.7c — Gate calibration + runtime decision function

`p_high` = lowest OOF probability that still keeps precision ≥ 0.99 (auto-assign without judge); `p_low` = highest that keeps recall ≥ 0.95 (below it, skip the judge entirely). `fusion_gate_decision(item, story, best_sim)` scores the item against the story's **cosine-nearest member** (lexical side) + centroid cosine, and is what the §10 loop calls when `use_fusion_gates=True`. The full model spec persists to `artifacts/v3/fusion_model.json` for the AWS handoff."""

CODE_87C = '''# 8.7c (v3) — Calibrate gates on OOF; persist model; define the runtime gate.
_ord = np.argsort(-oof_fus)
_sy, _sp = y_fus[_ord], oof_fus[_ord]
_cum_tp = np.cumsum(_sy); _n = np.arange(1, len(_sy) + 1)
_prec, _rec = _cum_tp / _n, _cum_tp / max(y_fus.sum(), 1)
_hi = np.flatnonzero(_prec >= 0.99)
p_high_fus = float(_sp[_hi.max()]) if len(_hi) else 0.99
_lo = np.flatnonzero(_rec >= 0.95)
p_low_fus = float(_sp[_lo.min()]) if len(_lo) else 0.05
FUSION_GATES = {"p_high": p_high_fus, "p_low": p_low_fus}

_lr, _sc = fusion_model[-1], fusion_model[0]
_fus_spec = {
    "features": FUSION_FEATURES, "coef": _lr.coef_[0].tolist(), "intercept": float(_lr.intercept_[0]),
    "scaler_mean": _sc.mean_.tolist(), "scaler_scale": _sc.scale_.tolist(),
    "gates": FUSION_GATES, "oof_auc": auc_fusion, "oof_auc_cosine_only": auc_cosine,
    "cv": "GroupKFold5-pairgraph-components", "trained_on": _sha256_file(CONFIG["frozen_eval_path"]),
}
(CONFIG["artifacts_dir"] / "fusion_model.json").write_text(json.dumps(_fus_spec, indent=2))
fusion_summary = {"enabled": bool(CONFIG.get("use_fusion_gates", False)), "oof_auc": auc_fusion,
                  "oof_auc_cosine": auc_cosine, "gates": FUSION_GATES,
                  "cv_scheme": "GroupKFold5-pairgraph", "model_path": str(CONFIG["artifacts_dir"] / "fusion_model.json")}

def fusion_gate_decision(item_row, story, best_sim):
    """(gate_hi, gate_lo) for the §10 loop. Lexical rep = cosine-nearest member."""
    iv = vec_for(item_row["item_id"])
    mvecs = assignment_vecs[story["member_idxs"]]
    rep = canonical_items.iloc[story["member_idxs"][int(np.argmax(mvecs @ iv))]]
    p = float(fusion_model.predict_proba(np.array([pair_features(item_row, rep, best_sim)]))[0, 1])
    return p >= FUSION_GATES["p_high"], p >= FUSION_GATES["p_low"]

print(f"gates: p_high={p_high_fus:.3f} (OOF precision≥0.99), p_low={p_low_fus:.3f} (OOF recall≥0.95)")
print(f"coefficients: {dict(zip(FUSION_FEATURES, np.round(_lr.coef_[0], 2)))}")'''

MD_87D1 = """### 8.7d-viz — ROC + precision-recall: fusion vs cosine-only

Both curves on OOF predictions (honest). The gap between the two ROC curves is the value the §10 gates inherit; the PR curve shows where the 0.99-precision auto-gate sits."""

CODE_87D1 = '''# 8.7d-viz (i) — ROC + PR curves, fusion OOF vs raw cosine.
from sklearn.metrics import roc_curve, precision_recall_curve
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, score, colour in [("fusion (OOF)", oof_fus, "tab:green"),
                            ("cosine only", eval_df["cosine_sim"].to_numpy(), "tab:gray")]:
    fpr, tpr, _ = roc_curve(y_fus, score)
    axes[0].plot(fpr, tpr, color=colour, label=f"{name} AUC={roc_auc_score(y_fus, score):.3f}")
    pr, rc, _ = precision_recall_curve(y_fus, score)
    axes[1].plot(rc, pr, color=colour, label=name)
axes[0].plot([0, 1], [0, 1], ls=":", color="k", lw=0.8)
axes[0].set_title("ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend(fontsize=8)
axes[1].axhline(0.99, color="green", ls="--", lw=1, label="p_high precision bar")
axes[1].set_title("Precision-recall"); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()'''

MD_87D2 = """### 8.7d-viz — Coefficients + score distributions with the gates

Standardized-feature coefficients (sign and size are comparable). Expect `caps_mismatch`/`num_mismatch` negative — they are the template detectors that raw cosine cannot see. The right panel shows SAME vs DIFFERENT OOF scores with the calibrated gate lines."""

CODE_87D2 = '''# 8.7d-viz (ii) — coefficient bars + OOF score distributions with gate lines.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
_co = fusion_model[-1].coef_[0]
axes[0].barh(FUSION_FEATURES, _co, color=["tab:green" if c > 0 else "tab:red" for c in _co], alpha=0.85)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_title("LogReg coefficients (standardized features)")

axes[1].hist(oof_fus[y_fus == 1], bins=30, alpha=0.6, label="SAME", color="tab:green", density=True)
axes[1].hist(oof_fus[y_fus == 0], bins=30, alpha=0.6, label="DIFFERENT", color="tab:red", density=True)
axes[1].axvline(FUSION_GATES["p_high"], color="green", ls="--", lw=1.2, label=f"p_high {FUSION_GATES['p_high']:.2f}")
axes[1].axvline(FUSION_GATES["p_low"], color="orange", ls="--", lw=1.2, label=f"p_low {FUSION_GATES['p_low']:.2f}")
axes[1].set_title("OOF score distributions"); axes[1].set_xlabel("p(SAME)"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()'''


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    if any(c.source.startswith("# 8.7a (v3)") for c in nb.cells):
        sys.exit("Phase-3 cells already present.")
    hits = [i for i, c in enumerate(nb.cells)
            if c.cell_type == "code" and c.source.startswith("# 8.6 — Comparison table")]
    assert len(hits) == 1, hits
    at = hits[0] + 1
    nb.cells[at:at] = [
        nbformat.v4.new_markdown_cell(MD_87),   nbformat.v4.new_code_cell(CODE_87A),
        nbformat.v4.new_markdown_cell(MD_87B),  nbformat.v4.new_code_cell(CODE_87B),
        nbformat.v4.new_markdown_cell(MD_87C),  nbformat.v4.new_code_cell(CODE_87C),
        nbformat.v4.new_markdown_cell(MD_87D1), nbformat.v4.new_code_cell(CODE_87D1),
        nbformat.v4.new_markdown_cell(MD_87D2), nbformat.v4.new_code_cell(CODE_87D2),
    ]
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            try:
                compile(c.source, f"c{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as e:
                bad.append((i, str(e)))
    if bad:
        sys.exit(f"syntax failures: {bad}")
    nbformat.write(nb, NB_PATH)
    print(f"OK: {len(nb.cells)} cells; §8.7a-d inserted after 8.6.")


if __name__ == "__main__":
    main()
