"""Score the pipeline's pairwise merge-decision on a labeled eval (representative + hard).

Same scorer applied to BOTH evals so the only variable is the eval composition.
Per pair: arm cosine -> 7 fusion features -> gate.decide; gray zone -> gpt-4.1-mini judge.
Judge = base only (no gpt-5.4-mini escalation), matching the confident-eval judge.

Arms:
  single_vec : cosine = item mean-pool single-vector ; judge text = full body
  chunk_pair : cosine = chunk max-pool              ; judge text = matched chunk pair
  full_body  : cosine = chunk max-pool              ; judge text = full body
"""
import os, sys, json, asyncio, pickle
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
sys.path.insert(0, "scripts")
import bloomberg_clustering as bc
import v4_chunking as v4c
from openai import AsyncOpenAI
from openai import RateLimitError

OUT = Path("artifacts/v4/eval_repr")
GATE = "artifacts/v4/fusion_model_chunk.json"
BASE_MODEL = "gpt-4.1-mini"
CONC = 8

# ---- shared artifacts ---------------------------------------------------------
items = pd.read_parquet("artifacts/v4/bloomberg_items.parquet").reset_index(drop=True)
items["published_at"] = pd.to_datetime(items["published_at"], utc=True)
rows = {r["item_id"]: r for _, r in items.iterrows()}

z = np.load(OUT / "item_vecs.npz", allow_pickle=True)
sv_ids = list(z["ids"]); SV = z["sv"]
svpos = {iid: k for k, iid in enumerate(sv_ids)}
chunks = pd.read_parquet(OUT / "chunks.parquet")
chunk_vecs = np.load(OUT / "chunk_vecs.npy")
index = bc.ChunkIndex(chunks, chunk_vecs)
gate = bc.FusionGate.load(GATE)
try:
    minhashes = pickle.load(open(".cache/minhashes_3000_exact.pkl", "rb"))
    if not (isinstance(minhashes, dict) and next(iter(minhashes)) in rows):
        minhashes = None
except Exception:
    minhashes = None
print("minhashes:", "loaded" if minhashes else "None (token-jaccard fallback)")

_client = AsyncOpenAI()

async def respond_fn(model, prompt, effort):
    kwargs = dict(model=model, input=[{"role": "user", "content": prompt}], store=False,
                  text={"format": {"type": "json_schema", "name": "comparison_verdict",
                                   "strict": True, "schema": bc.VERDICT_SCHEMA}})
    if effort is not None:
        kwargs["reasoning"] = {"effort": effort}
    for attempt in range(6):
        try:
            r = await _client.responses.create(**kwargs)
            try:
                return json.loads(r.output_text)
            except Exception:
                return {"verdict": "UNCLEAR", "reason": "unparseable"}
        except RateLimitError:
            await asyncio.sleep(2 * (attempt + 1))
    return {"verdict": "UNCLEAR", "reason": "rate-limited"}

def arm_sim_and_chunks(arm, a_id, b_id):
    """Return (cosine, matched_chunk_a_text_or_None, matched_chunk_b_text_or_None)."""
    if arm == "single_vec":
        return float(SV[svpos[a_id]] @ SV[svpos[b_id]]), None, None
    va, vb = index.chunks_for(a_id), index.chunks_for(b_id)
    sim, ia, ib = v4c.max_pool_sim(va, vb)
    if arm == "chunk_pair":
        return float(sim), index.text(a_id, ia), index.text(b_id, ib)
    return float(sim), None, None   # full_body

async def score_pair(arm, judge, a_id, b_id, sem):
    ra, rb = rows[a_id], rows[b_id]
    cos, ca, cb = arm_sim_and_chunks(arm, a_id, b_id)
    feats = bc.pair_features(ra, rb, cos, minhashes=minhashes)
    hi, lo = gate.decide(feats)
    if hi:
        return True
    if not lo:
        return False
    async with sem:                                   # gray zone -> judge
        return await judge.judge_same(ra, rb, ca, cb, cos)

def metrics(y_true, y_pred):
    tp = int(((y_pred) & (y_true)).sum()); fp = int(((y_pred) & (~y_true)).sum())
    fn = int(((~y_pred) & (y_true)).sum()); tn = int(((~y_pred) & (~y_true)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                tp=tp, fp=fp, fn=fn, tn=tn, n=len(y_true))

async def score_eval(name, df):
    df = df.copy()
    df["a"] = df["item_a_id"]; df["b"] = df["item_b_id"]
    df = df[df.a.isin(rows) & df.b.isin(rows) & df.a.isin(svpos) & df.b.isin(svpos)]
    df = df[df.final_label.isin(["SAME", "DIFFERENT"])]
    y_true = df.final_label.eq("SAME").to_numpy()
    res = {"n_pairs": int(len(df)), "n_same": int(y_true.sum()), "arms": {}}
    ARMS = {"single_vec": "rb_fullbody", "chunk_pair": "rb_chunkpair", "full_body": "rb_fullbody"}
    for arm, pv in ARMS.items():
        judge = bc.TwoTierJudge(respond_fn, base_model=BASE_MODEL, escalation_model=None,
                                base_effort=None, judge_text_mode=("chunk_pair" if arm == "chunk_pair" else "full_body"),
                                prompt_version=pv, cache_dir=".cache/judge_repr")
        sem = asyncio.Semaphore(CONC)
        preds = await asyncio.gather(*[score_pair(arm, judge, a, b, sem)
                                       for a, b in zip(df.a, df.b)])
        m = metrics(y_true, np.array(preds))
        m["judge"] = dict(judge.stats)
        res["arms"][arm] = m
        print(f"  [{name}/{arm}] F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"(tp{m['tp']} fp{m['fp']} fn{m['fn']}) judge={judge.stats}")
    return res

async def main():
    repr_df = pd.read_csv(OUT / "bloomberg_eval_representative.csv")
    hard_df = pd.read_csv("artifacts/v4/bloomberg_eval_large.csv", comment="#")
    out = {}
    print("REPRESENTATIVE eval:")
    out["representative"] = await score_eval("repr", repr_df)
    print("HARD eval (bloomberg_eval_large):")
    out["hard"] = await score_eval("hard", hard_df)
    Path("artifacts/v4/repr_vs_hard_scored.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> artifacts/v4/repr_vs_hard_scored.json")

asyncio.run(main())
