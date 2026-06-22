"""bloomberg_clustering.py — production reference pipeline for Bloomberg story clustering.

Body-rich corpus pipeline, validated in story_clustering_poc_v4 §18/§19:
  ingest -> URL+MinHash dedup -> body-paragraph chunk embeddings ->
  single-pass assignment loop (max-pool nearest-chunk similarity + fusion gate +
  two-tier chunk_pair LLM judge) -> HDBSCAN residual clustering.

Design notes for lifting into production:
  * Pure clustering logic lives here; I/O (data load, embedding, LLM calls) is injected as
    callables so you can swap the embedder / LLM provider without touching the algorithm.
  * Story state is a plain dict (see new_story); easy to serialize.
  * Reuses v4_chunking for chunk_body, max_pool_sim, build_judge_block, union_items_from_chunk_clusters.

Stages a production service would wire up:
  1. dedup_pipeline(items_df, cfg)
  2. build_chunks(items_df, count_tokens, cfg)            -> chunks_df   (then embed externally)
  3. ChunkIndex(chunks_df, chunk_vecs)                    -> chunks_for / matched-chunk lookup
  4. FusionGate.load(path)                                -> runtime gate
  5. TwoTierJudge(complete_fn, cfg)                       -> gray-zone judge
  6. await assignment_loop(items_df, index, gate, judge, cfg)
  7. await residual_cluster(singletons, index, judge, cfg)
"""
from __future__ import annotations

import re
import json
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import numpy as np
import pandas as pd

import v4_chunking as v4c  # same scripts/ dir: chunk_body, max_pool_sim, build_judge_block, union_items_from_chunk_clusters


# ============================================================================
# 1. URL canonicalization + exact-duplicate drop
# ============================================================================
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "cmpid"}


def canonicalize_url(url) -> str:
    """Lowercase scheme+host, drop tracking params + fragment + trailing slash. '' on failure."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        p = urlparse(url.strip())
    except ValueError:
        return ""
    scheme = (p.scheme or "http").lower()
    netloc = p.netloc.lower()
    path = p.path.rstrip("/") or "/"
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
         if k.lower() not in _TRACKING_PARAMS]
    q.sort()
    return urlunparse((scheme, netloc, path, "", urlencode(q), ""))


def url_hash_hex(canon_url: str) -> str:
    return hashlib.sha256((canon_url or "").encode("utf-8")).hexdigest()


# ============================================================================
# 2. MinHash / LSH near-duplicate detection
# ============================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text) -> list[str]:
    return _TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []


def shingles(toks: list[str], n: int = 5) -> set[bytes]:
    if not toks:
        return set()
    if len(toks) < n:
        return {(" ".join(toks)).encode("utf-8")}
    return {(" ".join(toks[i:i + n])).encode("utf-8") for i in range(len(toks) - n + 1)}


class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def compute_minhashes(items_df: pd.DataFrame, num_perm: int = 128) -> dict:
    """item_id -> MinHash over (title + body) 5-gram shingles."""
    from datasketch import MinHash
    out = {}
    for r in items_df.itertuples():
        sh = shingles(tokens(((r.title or "") + " " + (r.body or ""))))
        mh = MinHash(num_perm=num_perm)
        if sh:
            mh.update_batch(list(sh))
        out[r.item_id] = mh
    return out


def minhash_near_dups(items_df: pd.DataFrame, minhashes: dict, *,
                      num_perm: int = 128, threshold: float = 0.85,
                      source_priority: dict | None = None) -> pd.DataFrame:
    """Mark near-duplicates. Adds is_duplicate / duplicate_of columns. Canonical = highest-priority
    source then earliest published_at. Single-source corpora just keep the earliest."""
    from datasketch import MinHashLSH
    source_priority = source_priority or {}
    lsh = MinHashLSH(threshold=0.70, num_perm=num_perm)
    for iid, mh in minhashes.items():
        lsh.insert(iid, mh)
    pairs = set()
    for iid, mh in minhashes.items():
        for other in lsh.query(mh):
            if other != iid:
                pairs.add(tuple(sorted([iid, other])))
    uf = _UnionFind()
    for a, b in pairs:
        if minhashes[a].jaccard(minhashes[b]) >= threshold:
            uf.union(a, b)
    groups: dict = {}
    for iid in minhashes:
        groups.setdefault(uf.find(iid), []).append(iid)
    by_id = items_df.set_index("item_id")
    dup_map = {}
    for cluster in (g for g in groups.values() if len(g) >= 2):
        sub = by_id.loc[cluster].copy()
        sub["_rank"] = sub["source"].map(source_priority).fillna(99) if "source" in sub else 0
        sub = sub.sort_values(["_rank", "published_at"])
        canon = sub.index[0]
        for iid in cluster:
            if iid != canon:
                dup_map[iid] = canon
    out = items_df.copy()
    out["is_duplicate"] = out["item_id"].isin(dup_map)
    out["duplicate_of"] = out["item_id"].map(dup_map)
    return out


def dedup_pipeline(items_df: pd.DataFrame, *, num_perm: int = 128, minhash_threshold: float = 0.85,
                   source_priority: dict | None = None) -> pd.DataFrame:
    """URL canonicalize -> exact drop -> MinHash near-dup mark. Returns canonical-flagged df."""
    df = items_df.copy()
    df["canonical_url"] = df["url"].apply(canonicalize_url)
    df["url_hash"] = df["canonical_url"].apply(url_hash_hex)
    df = df[df["canonical_url"].str.len() > 0].reset_index(drop=True)
    df = (df.sort_values("published_at", kind="stable")
            .drop_duplicates(subset="url_hash", keep="first").reset_index(drop=True))
    mh = compute_minhashes(df, num_perm=num_perm)
    return minhash_near_dups(df, mh, num_perm=num_perm, threshold=minhash_threshold,
                             source_priority=source_priority)


# ============================================================================
# 3. Body-paragraph chunking + chunk index
# ============================================================================
def build_chunks(items_df: pd.DataFrame, *, count_tokens, min_tokens: int = 25,
                 max_tokens: int = 400, max_chunks: int = 12) -> pd.DataFrame:
    """One row per (item_id, chunk_idx, chunk_text). Body-only; title-only items fall back to title."""
    rows = []
    for r in items_df.itertuples():
        for j, c in enumerate(v4c.chunk_body(r.title, r.body, count_tokens=count_tokens,
                                             min_tokens=min_tokens, max_tokens=max_tokens,
                                             max_chunks=max_chunks)):
            rows.append({"item_id": r.item_id, "chunk_idx": j, "chunk_text": c})
    return pd.DataFrame(rows)


class ChunkIndex:
    """Maps item_id -> its chunk vectors + texts. chunk_vecs must be row-aligned to chunks_df."""

    def __init__(self, chunks_df: pd.DataFrame, chunk_vecs: np.ndarray):
        self.vecs = chunk_vecs
        self.ranges, self.texts, s = {}, {}, 0
        for iid, grp in chunks_df.groupby("item_id", sort=False):
            self.ranges[iid] = (s, s + len(grp))
            self.texts[iid] = grp["chunk_text"].tolist()
            s += len(grp)

    def chunks_for(self, item_id) -> np.ndarray:
        a, b = self.ranges[item_id]
        return self.vecs[a:b]

    def text(self, item_id, local_idx) -> str:
        return self.texts[item_id][local_idx]


# ============================================================================
# 4. Fusion gate (load pre-calibrated artifact) + pair features
# ============================================================================
_CAPS_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _jac(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def pair_features(row_a, row_b, cosine: float, *, minhashes: dict | None = None,
                  lede_chars: int = 600) -> list[float]:
    """The 7 fusion features. `cosine` is the chunk max-pool similarity at decision time.
    row_a/row_b are mappings with title, body, item_id, published_at."""
    fa = f"{row_a['title'] or ''} {(row_a['body'] or '')[:lede_chars]}"
    fb = f"{row_b['title'] or ''} {(row_b['body'] or '')[:lede_chars]}"
    tok_a, tok_b = set(tokens(row_a['title'] or '')), set(tokens(row_b['title'] or ''))
    if minhashes is not None and row_a["item_id"] in minhashes and row_b["item_id"] in minhashes:
        mh_j = float(minhashes[row_a["item_id"]].jaccard(minhashes[row_b["item_id"]]))
    else:
        mh_j = _jac(set(tokens(fa)), set(tokens(fb)))
    dt_days = min(abs((row_a["published_at"] - row_b["published_at"]) / pd.Timedelta(days=1)), 7.0)
    la, lb = len(fa), len(fb)
    return [float(cosine), _jac(tok_a, tok_b), mh_j, float(dt_days),
            min(la, lb) / max(la, lb, 1),
            1.0 - _jac(set(_NUM_RE.findall(fa)), set(_NUM_RE.findall(fb))),
            1.0 - _jac(set(_CAPS_RE.findall(fa)), set(_CAPS_RE.findall(fb)))]


class FusionGate:
    """Logistic-regression gate (StandardScaler + LogReg) loaded from fusion_model_chunk.json.
    decide(features) -> (gate_high, gate_low): high => auto-merge, low => send to LLM judge."""

    def __init__(self, spec: dict):
        self.features = spec["features"]
        self.coef = np.array(spec["coef"]); self.intercept = float(spec["intercept"])
        self.mean = np.array(spec["scaler_mean"]); self.scale = np.array(spec["scaler_scale"])
        self.p_high = float(spec["gates"]["p_high"]); self.p_low = float(spec["gates"]["p_low"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(open(path).read()))

    def prob(self, features: list[float]) -> float:
        z = (np.array(features) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))

    def decide(self, features: list[float]) -> tuple[bool, bool]:
        p = self.prob(features)
        return p >= self.p_high, p >= self.p_low


# ============================================================================
# 5. Two-tier LLM judge (base model; escalate the uncertain band to a stronger model)
# ============================================================================
# Structured-output schema for the verdict (OpenAI Responses API json_schema, strict).
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SAME", "DIFFERENT", "UNCLEAR"],
                    "description": "Whether the two items describe the SAME news event, DIFFERENT events, or UNCLEAR"},
        "reason": {"type": "string", "description": "Brief explanation of the verdict"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

_JUDGE_RUBRIC = (
    "You are determining whether two financial news items describe the SAME news event.\n\n"
    "ITEM A:\n{a}\n\nITEM B:\n{b}\n\n"
    "SAME = the same underlying news event involving the same primary entities, even if framing, source, "
    "or details differ. DIFFERENT = the primary event differs (e.g. preview vs results; distinct "
    "follow-ups in an ongoing saga are different events). UNCLEAR only if genuinely undecidable.\n"
    "Return your verdict (SAME / DIFFERENT / UNCLEAR) with a brief reason.")


class TwoTierJudge:
    """Gray-zone judge over the OpenAI Responses API with structured output + reasoning effort.

    `respond_fn(model, prompt, effort) -> {"verdict": str, "reason": str}` is an async callable you
    provide (wrap responses.create with VERDICT_SCHEMA). Base model handles every gray-zone call; if the
    decision-time similarity falls in `escalation_band` [lo, hi), the stronger `escalation_model` is used
    with `escalation_effort` reasoning. `effort` is passed through to respond_fn (None for non-reasoning
    base models). Verdicts cached to disk. UNCLEAR is treated as 'not SAME' (conservative: don't merge).
    """

    def __init__(self, respond_fn, *, base_model: str, escalation_model: str | None = None,
                 escalation_band: tuple[float, float] = (0.0, 0.0), base_effort: str | None = None,
                 escalation_effort: str | None = "none", cache_dir=".cache/judge",
                 judge_text_mode: str = "chunk_pair", prompt_version: str = "v4_chunkpair"):
        from pathlib import Path
        self.respond = respond_fn
        self.base_model, self.base_effort = base_model, base_effort
        self.escalation_model, self.escalation_effort = escalation_model, escalation_effort
        self.band = escalation_band
        self.mode = judge_text_mode
        self.pv = prompt_version
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"base": 0, "escalated": 0, "cached": 0}

    def _tier_for(self, sim):
        """Return (model, effort, tier_name) for a decision-time similarity."""
        lo, hi = self.band
        if self.escalation_model and sim is not None and lo <= sim < hi:
            return self.escalation_model, self.escalation_effort, "escalated"
        return self.base_model, self.base_effort, "base"

    def _key(self, model, effort, a_id, b_id) -> str:
        return hashlib.sha256(f"{model}|{effort}|{self.pv}|{a_id}|{b_id}".encode()).hexdigest()[:16]

    async def judge_same(self, item_row, rep_row, item_chunk, rep_chunk, sim) -> bool:
        model, effort, tier = self._tier_for(sim)
        a_id, b_id = sorted([item_row["item_id"], rep_row["item_id"]])
        fp = self.cache_dir / f"{self._key(model, effort, a_id, b_id)}.json"
        if fp.exists():
            self.stats["cached"] += 1
            return json.loads(fp.read_text())["verdict"] == "SAME"
        prompt = _JUDGE_RUBRIC.format(
            a=v4c.build_judge_block(item_row, self.mode, matched_chunk=item_chunk),
            b=v4c.build_judge_block(rep_row, self.mode, matched_chunk=rep_chunk))
        out = await self.respond(model, prompt, effort)              # -> {"verdict","reason"}
        verdict = (out.get("verdict") or "UNCLEAR").upper()
        fp.write_text(json.dumps({"verdict": verdict, "reason": out.get("reason", ""),
                                  "model": model, "effort": effort}))
        self.stats[tier] += 1
        return verdict == "SAME"                                      # UNCLEAR/DIFFERENT -> don't merge


# ============================================================================
# 6. Single-pass assignment loop (max-pool nearest-chunk + gate + judge)
# ============================================================================
def new_story(idx: int, row, index: ChunkIndex) -> dict:
    cv = index.chunks_for(row["item_id"])
    return {"story_id": None, "chunk_vecs": cv.copy(), "chunk_owner": [row["item_id"]] * cv.shape[0],
            "member_ids": [row["item_id"]], "member_idxs": [int(idx)],
            "item_clients": set(row["item_clients"]),
            "first_seen_at": row["published_at"], "last_seen_at": row["published_at"], "closed_at": None}


def _assign(story: dict, idx: int, row, index: ChunkIndex) -> None:
    cv = index.chunks_for(row["item_id"])
    story["chunk_vecs"] = np.vstack([story["chunk_vecs"], cv])
    story["chunk_owner"].extend([row["item_id"]] * cv.shape[0])
    story["member_ids"].append(row["item_id"]); story["member_idxs"].append(int(idx))
    story["item_clients"].update(row["item_clients"])
    if row["published_at"] > story["last_seen_at"]:
        story["last_seen_at"] = row["published_at"]


async def assignment_loop(items_df: pd.DataFrame, index: ChunkIndex, gate: FusionGate,
                          judge: TwoTierJudge, *, window_hours: int = 72, minhashes: dict | None = None,
                          lede_chars: int = 600, progress=lambda x: x) -> list[dict]:
    """Time-ordered single pass. Returns list of story dicts (story_id assigned s00000..)."""
    window = pd.Timedelta(hours=window_hours)
    df = items_df.sort_values("published_at", kind="stable")
    stories: list[dict] = []
    for _, row in progress(list(df.iterrows())):
        idx = int(row["__pos__"]); iid = row["item_id"]
        ichunks = index.chunks_for(iid); it = row["published_at"]; ic = set(row["item_clients"])
        cands = [s for s in stories if s["closed_at"] is None
                 and (it - s["last_seen_at"]) <= window and (s["item_clients"] & ic)]
        if not cands:
            stories.append(new_story(idx, row, index)); continue
        triples = [v4c.max_pool_sim(ichunks, s["chunk_vecs"]) for s in cands]
        bi = int(np.argmax([t[0] for t in triples]))
        best = cands[bi]; bsim, bia, bib = triples[bi]
        rep_id = best["chunk_owner"][bib]
        rep_row = items_df.loc[items_df["item_id"] == rep_id].iloc[0]
        feats = pair_features(row, rep_row, bsim, minhashes=minhashes, lede_chars=lede_chars)
        gate_hi, gate_lo = gate.decide(feats)
        if gate_hi:
            _assign(best, idx, row, index)
        elif gate_lo:
            item_mc = index.text(iid, bia)
            rep_mc = index.text(rep_id, bib - best["chunk_owner"].index(rep_id))
            if await judge.judge_same(row, rep_row, item_mc, rep_mc, bsim):
                _assign(best, idx, row, index)
            else:
                stories.append(new_story(idx, row, index))
        else:
            stories.append(new_story(idx, row, index))
    for i, s in enumerate(stories):
        s["story_id"] = f"s{i:05d}"
    return stories


def expire_stories(stories: list[dict], window_hours: int = 72) -> list[dict]:
    """Close stories whose newest member is older than `window_hours` before the global max."""
    if not stories:
        return stories
    max_seen = max(s["last_seen_at"] for s in stories)
    window = pd.Timedelta(hours=window_hours)
    for s in stories:
        if s["closed_at"] is None and (max_seen - s["last_seen_at"]) > window:
            s["closed_at"] = s["last_seen_at"]
    return stories


# ============================================================================
# 7. Residual clustering (HDBSCAN over chunk vectors -> union items)
# ============================================================================
async def residual_cluster(singletons: list[dict], index: ChunkIndex, judge: TwoTierJudge, *,
                           min_cluster_size: int = 2, min_samples: int = 2,
                           cluster_selection_method: str = "eom", minhashes: dict | None = None,
                           judge_gate: bool = True, lede_chars: int = 600,
                           items_df: pd.DataFrame | None = None) -> tuple[list[dict], set]:
    """Cluster leftover single-item stories by their CHUNK vectors; union items that share any
    chunk-cluster; keep clusters with a shared client; optionally judge-gate. Returns
    (new_multi_stories, absorbed_singleton_item_ids)."""
    import hdbscan
    from sklearn.metrics.pairwise import cosine_distances

    res_ids = [s["member_ids"][0] for s in singletons]
    if len(res_ids) < min_cluster_size:
        return [], set()
    chunk_vecs, owner = [], []
    for iid in res_ids:
        cv = index.chunks_for(iid); chunk_vecs.append(cv); owner.extend([iid] * cv.shape[0])
    chunk_vecs = np.vstack(chunk_vecs)
    dist = cosine_distances(chunk_vecs).astype(np.float64)
    labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size,
                             min_samples=min_samples,
                             cluster_selection_method=cluster_selection_method).fit_predict(dist)
    item_to_root = v4c.union_items_from_chunk_clusters(owner, labels)
    comps: dict = {}
    for iid, root in item_to_root.items():
        comps.setdefault(root, []).append(iid)

    sing_by_id = {s["member_ids"][0]: s for s in singletons}
    by_id = items_df.set_index("item_id", drop=False) if items_df is not None else None
    new_stories, absorbed = [], set()
    for members in comps.values():
        if len(members) < 2:
            continue
        clients = set.intersection(*[sing_by_id[m]["item_clients"] for m in members])
        if not clients:
            continue  # incoherent — leave as singletons
        if judge_gate and by_id is not None:
            # medoid vs farthest chunk pair; drop the cluster if the judge says DIFFERENT
            keep = members
            r0 = by_id.loc[members[0]]
            r1 = by_id.loc[members[1]]
            sim, ia, ib = v4c.max_pool_sim(index.chunks_for(members[0]), index.chunks_for(members[1]))
            same = await judge.judge_same(r0, r1, index.text(members[0], ia), index.text(members[1], ib), sim)
            if not same:
                continue
        new_stories.append({
            "story_id": f"r{len(new_stories):05d}",
            "member_ids": list(members), "n_items": len(members),
            "item_clients": set().union(*[sing_by_id[m]["item_clients"] for m in members]),
            "first_seen_at": min(sing_by_id[m]["first_seen_at"] for m in members),
            "last_seen_at": max(sing_by_id[m]["last_seen_at"] for m in members), "closed_at": None})
        absorbed.update(members)
    return new_stories, absorbed
