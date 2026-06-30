"""Build a REPRESENTATIVE Bloomberg eval candidate-pair set.

Mirrors how the v2 news eval (-> F1 0.870) was built: candidate pairs drawn from the
real candidate-generation rule (shared client + 72h window), then STRATIFIED across
cosine buckets so SAME and DIFFERENT pairs are separated (gap ~0.1+) and the base
rate is measurable (~15%), rather than hard-negative-mined top-cosine confusers.

Outputs (artifacts/v4/eval_repr/):
  - pairs_meta.parquet    full pair metadata (ids, cosine, bucket, shared clients)
  - batch_XX.json         per-subagent labeling payloads (title + body excerpt)
  - item_vecs.npz         item single-vec (mean-pool) + chunk index, for scoring reuse

Only OBSERVE summaries are printed; nothing heavy enters the caller context.
"""
import os, sys, json, hashlib, pickle, itertools
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
sys.path.insert(0, "scripts")
import bloomberg_clustering as bc

EMBED_MODEL, EMBED_DIMS = "text-embedding-3-large", 1024
CACHE = Path(".cache/embeddings.pkl")
OUT = Path("artifacts/v4/eval_repr"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
WINDOW_H = 72
N_TARGET = 600
BODY_CHARS = 2000          # excerpt per item handed to the labeler
PAIRS_PER_BATCH = 50

import tiktoken
_enc = tiktoken.encoding_for_model(EMBED_MODEL)
count_tokens = lambda s: len(_enc.encode(s))

def embed_key(text):
    h = hashlib.sha256()
    h.update(f"{EMBED_MODEL}|{EMBED_DIMS}|chunk_v1".encode())
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]

# ---- 1. items + chunks --------------------------------------------------------
items = pd.read_parquet("artifacts/v4/bloomberg_items.parquet").reset_index(drop=True)
items["published_at"] = pd.to_datetime(items["published_at"], utc=True)
chunks = bc.build_chunks(items, count_tokens=count_tokens)
print(f"items={len(items)}  chunks={len(chunks)}  cols={list(chunks.columns)}")

# ---- 2. embed chunks (reuse cache; fetch misses via OpenAI) --------------------
cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
texts = list(chunks["chunk_text"])
miss = [t for t in dict.fromkeys(texts) if embed_key(t) not in cache]
print(f"chunk embed cache: {len(texts)-len(miss)} hit / {len(miss)} miss")
if miss:
    from openai import OpenAI
    cli = OpenAI()
    for i in range(0, len(miss), 100):
        b = miss[i:i+100]
        r = cli.embeddings.create(model=EMBED_MODEL, input=b, dimensions=EMBED_DIMS)
        for t, d in zip(b, r.data):
            cache[embed_key(t)] = np.asarray(d.embedding, dtype=np.float32)
    CACHE.write_bytes(pickle.dumps(cache))
    print(f"fetched {len(miss)} chunk embeds")
chunk_vecs = np.stack([cache[embed_key(t)] for t in texts]).astype(np.float32)
# normalize chunk vecs
chunk_vecs /= (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-9)
index = bc.ChunkIndex(chunks, chunk_vecs)

# ---- 3. item single-vec = L2-normalized mean of its chunk vectors -------------
item_ids = list(items["item_id"])
item_vec = {}
for iid in item_ids:
    v = index.chunks_for(iid)
    m = v.mean(axis=0)
    item_vec[iid] = m / (np.linalg.norm(m) + 1e-9)
SV = np.stack([item_vec[i] for i in item_ids]).astype(np.float32)
pos = {iid: k for k, iid in enumerate(item_ids)}

# ---- 4. candidate pairs: shared client + |dt| <= 72h --------------------------
clients = items["item_clients"].apply(lambda a: set(a.tolist()) if hasattr(a, "tolist") else set(a))
pub = items["published_at"].values
by_client = {}
for k, cs in enumerate(clients):
    for c in cs:
        by_client.setdefault(c, []).append(k)

win = np.timedelta64(WINDOW_H, "h")
cand = set()
for c, idxs in by_client.items():
    idxs = sorted(idxs, key=lambda k: pub[k])
    for a_i in range(len(idxs)):
        ka = idxs[a_i]
        for b_i in range(a_i + 1, len(idxs)):
            kb = idxs[b_i]
            if pub[kb] - pub[ka] > win:
                break
            cand.add((ka, kb) if ka < kb else (kb, ka))
print(f"candidate pairs (shared-client + {WINDOW_H}h): {len(cand)}")

ca = np.fromiter((p[0] for p in cand), dtype=np.int64, count=len(cand))
cb = np.fromiter((p[1] for p in cand), dtype=np.int64, count=len(cand))
cos = np.einsum("ij,ij->i", SV[ca], SV[cb])

# ---- 5. stratified sample across cosine buckets -------------------------------
edges = [0.30, 0.50, 0.60, 0.70, 0.80, 1.01]
labels = ["0.30-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-1.00"]
alloc  = [90, 110, 140, 150, 120]   # mild up-weight of decision-relevant high cosine
rng = np.random.default_rng(SEED)
buck = np.digitize(cos, edges) - 1
rows = []
for bi, (lab, want) in enumerate(zip(labels, alloc)):
    pool = np.where(buck == bi)[0]
    take = pool if len(pool) <= want else rng.choice(pool, size=want, replace=False)
    for j in take:
        rows.append((int(ca[j]), int(cb[j]), float(cos[j]), lab))
print(f"sampled {len(rows)} pairs across buckets (target {N_TARGET})")

# ---- 6. write pair metadata + per-batch labeling payloads ---------------------
meta = []
payload = []
for n, (ka, kb, cs, lab) in enumerate(rows):
    ra, rb = items.iloc[ka], items.iloc[kb]
    pid = f"rp{n:04d}"
    shared = sorted(clients[ka] & clients[kb])
    meta.append(dict(pair_id=pid, item_a_id=ra["item_id"], item_b_id=rb["item_id"],
                     cosine_sim=round(cs, 4), sim_bucket=lab, shared_clients=";".join(shared),
                     a_published=str(ra["published_at"]), b_published=str(rb["published_at"])))
    payload.append(dict(
        pair_id=pid,
        article_a=dict(title=ra["title"], body=(ra["body"] or "")[:BODY_CHARS]),
        article_b=dict(title=rb["title"], body=(rb["body"] or "")[:BODY_CHARS]),
    ))
pd.DataFrame(meta).to_parquet(OUT / "pairs_meta.parquet")
np.savez_compressed(OUT / "item_vecs.npz", ids=np.array(item_ids), sv=SV)
chunks.to_parquet(OUT / "chunks.parquet")
np.save(OUT / "chunk_vecs.npy", chunk_vecs)

nb = 0
for i in range(0, len(payload), PAIRS_PER_BATCH):
    (OUT / f"batch_{nb:02d}.json").write_text(json.dumps(payload[i:i+PAIRS_PER_BATCH], ensure_ascii=False, indent=1))
    nb += 1
print(f"wrote {nb} batch files ({PAIRS_PER_BATCH} pairs each) -> {OUT}")

# ---- 7. summary ---------------------------------------------------------------
md = pd.DataFrame(meta)
print("\nbucket distribution:")
print(md["sim_bucket"].value_counts().sort_index().to_string())
print(f"\ncosine: mean={md.cosine_sim.mean():.3f} min={md.cosine_sim.min():.3f} max={md.cosine_sim.max():.3f}")
print("done.")
