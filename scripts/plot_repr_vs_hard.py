"""Chart: F1 by arm (representative vs hard eval) + SAME-rate by cosine bucket."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.loads(Path("artifacts/v4/repr_vs_hard_scored.json").read_text())
arms = ["single_vec", "chunk_pair", "full_body"]
repr_f1 = [res["representative"]["arms"][a]["f1"] for a in arms]
hard_f1 = [res["hard"]["arms"][a]["f1"] for a in arms]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

# Panel 1: F1 by arm, repr vs hard
x = np.arange(len(arms)); w = 0.36
ax1.bar(x - w/2, repr_f1, w, label="representative eval (24% SAME, gap 0.188)", color="#2c7fb8")
ax1.bar(x + w/2, hard_f1, w, label="hard eval (6% SAME, gap 0.021)", color="#d95f0e")
ax1.axhline(0.834, ls="--", c="gray", lw=1)
ax1.text(2.35, 0.842, "news baseline 0.834", color="gray", fontsize=8, ha="right")
for xi, v in zip(x - w/2, repr_f1): ax1.text(xi, v + .012, f"{v:.3f}", ha="center", fontsize=8)
for xi, v in zip(x + w/2, hard_f1): ax1.text(xi, v + .012, f"{v:.3f}", ha="center", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(arms); ax1.set_ylim(0, 1.0)
ax1.set_ylabel("end-to-end F1 (pairwise merge decision)")
ax1.set_title("Same pipeline + scorer — only the eval changes")
ax1.legend(fontsize=8, loc="upper center")

# Panel 2: SAME-rate by cosine bucket (representative eval)
df = pd.read_csv("artifacts/v4/eval_repr/bloomberg_eval_representative.csv")
df = df[df.final_label.isin(["SAME", "DIFFERENT"])]
g = df.assign(is_same=df.final_label.eq("SAME")).groupby("sim_bucket").is_same.agg(["mean", "count"])
ax2.bar(range(len(g)), g["mean"].values, color="#2c7fb8")
for i, (m, c) in enumerate(zip(g["mean"], g["count"])):
    ax2.text(i, m + .02, f"{m:.0%}\n(n={c})", ha="center", fontsize=8)
ax2.set_xticks(range(len(g))); ax2.set_xticklabels(g.index, rotation=20, fontsize=8)
ax2.set_ylim(0, 1.0); ax2.set_ylabel("SAME rate")
ax2.set_xlabel("single-vec cosine bucket")
ax2.set_title("Representative eval separates by cosine (monotone)")

plt.tight_layout()
out = "artifacts/v4/eval_repr/repr_vs_hard.png"
plt.savefig(out, dpi=130)
print("saved", out)
