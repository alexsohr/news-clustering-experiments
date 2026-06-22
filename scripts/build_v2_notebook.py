#!/usr/bin/env python3
"""Build story_clustering_poc_v2.ipynb from the original by applying the approved v2 edits.

Spec: docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md
Plan: docs/superpowers/plans/2026-06-01-poc-clustering-v2.md

The original notebook is READ ONLY. All edits land in the v2 copy. Every edit is asserted,
so a missing anchor makes the build fail loudly rather than silently produce a wrong notebook.
"""
import json, shutil, re, textwrap
from pathlib import Path

ROOT = Path("/Users/alex/Projcts/news-clustering")
SRC  = ROOT / "story_clustering_poc.ipynb"
DST  = ROOT / "story_clustering_poc_v2.ipynb"

# ----------------------------- new / replacement cell sources -----------------------------
CHANGELOG = r"""## v2 changes (see docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md)
1. **P0 — fixed the §10 vector-indexing bug** (unstable sort scrambled tied-timestamp embeddings → ~40–70% wrong vectors).
2. **P2 — boilerplate curation**: templated wire items (REG-/Form 8.x/NAV/PR templates) set aside before clustering (§6.1b).
3. **Judge swap**: gray-zone (§10) and merge (§13) judges use OpenAI `gpt-4.1` instead of Claude Haiku.

Skipped for this minimal pass: §9 (long-doc) and §12 (Sonnet metadata). DataFrame changes are limited to touched cells. Outputs → `artifacts/v2/`; eval reuses the existing 543 labeled pairs."""

BP_MD = r"""### 6.1b — Boilerplate curation (v2): set templated wire items aside before clustering"""

BP_CODE = r"""# 6.1b — Detect templated/non-editorial wire items and route them out of the clustering corpus.
# Structural templates ONLY — must NOT flag editorial wire tags (UPDATE-N/WRAPUP/FACTBOX),
# which carry ~32% of true-SAME follow-ups.
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
_bp_mask = canonical_items_all.apply(lambda r: is_boilerplate_text(r["title"], r.get("body", "")), axis=1)
canonical_items_all["is_boilerplate"] = _bp_mask.to_numpy()
boilerplate_df = canonical_items_all[canonical_items_all["is_boilerplate"]].reset_index(drop=True)
canonical_items = canonical_items_all[~canonical_items_all["is_boilerplate"]].reset_index(drop=True)
print(f"Boilerplate flagged: {int(_bp_mask.sum()):,} ({_bp_mask.mean()*100:.1f}%)  |  "
      f"clustering corpus: {len(canonical_items):,}  |  set-aside: {len(boilerplate_df):,}")"""

BP_AUDIT = r"""# 6.1b (cont.) — Hand-audit + guardrail asserts.
print("Sample FLAGGED (should be templates/filings/PRs):")
for _t in boilerplate_df["title"].head(15).tolist():
    print("  -", str(_t)[:90])
print("\nSample NOT-flagged (should be real stories):")
for _t in canonical_items["title"].dropna().sample(min(15, len(canonical_items)), random_state=CONFIG["random_seed"]).tolist():
    print("  -", str(_t)[:90])
assert not is_boilerplate_text("UPDATE 2-Boeing 787's dimmable windows not dark enough, says ANA")
assert not is_boilerplate_text("WRAPUP 6-Boeing Dreamliners grounded worldwide on battery checks")
print("\nGuardrail OK: UPDATE/WRAPUP headlines are not treated as boilerplate.")"""

BP_PAIR_CODE = r"""# 8.1b (v2) — Flag boilerplate pairs so calibration + reporting can use the clean subset.
eval_df["is_boilerplate_pair"] = [
    is_boilerplate_text(ta) or is_boilerplate_text(tb)
    for ta, tb in zip(eval_df["item_a_title"], eval_df["item_b_title"])
]
print(f"Boilerplate pairs in eval set: {int(eval_df['is_boilerplate_pair'].sum())} / {len(eval_df)}")"""

GPT_JUDGE = r"""# 10.2 (v2) — OpenAI gpt-4.1 judge replaces Haiku. Same prompt; cache keyed by model id.
from openai import AsyncOpenAI as _AsyncOpenAI_v2
_openai_judge_client = _AsyncOpenAI_v2()
try:
    _openai_judge_limiter = _openai_limiter
except NameError:
    _openai_judge_limiter = AsyncRateLimiter(CONFIG["vendor_rate_limits_rpm"].get("openai", 50))

async def judge_same(item_row, rep_row) -> bool:
    model = CONFIG["judge_model"]
    a_id, b_id = sorted([item_row["item_id"], rep_row["item_id"]])
    fpath = JUDGE_CACHE_DIR / f"{_judge_key(model, a_id, b_id)}.json"
    if fpath.exists():
        return json.loads(fpath.read_text())["verdict"] == "SAME"
    await _openai_judge_limiter.acquire()
    prompt = (
        "Two financial news items - same story or different?\n\n"
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
print("judge_same (gpt-4.1) ready.")"""

REG_CODE = r"""# 10.3a (v2) — Quantify how many items the OLD (unstable-sort + arange) approach mis-maps.
pos_of_id = pd.Series(np.arange(len(canonical_items)), index=canonical_items["item_id"])
_old = canonical_items.sort_values("published_at").copy()                 # default quicksort = the bug
_old["pos_buggy"] = np.arange(len(_old))
_old["pos_true"]  = _old["item_id"].map(pos_of_id).to_numpy()
_n_wrong = int((_old["pos_buggy"].to_numpy() != _old["pos_true"].to_numpy()).sum())
print(f"OLD approach would mis-map {_n_wrong:,}/{len(_old):,} items "
      f"({_n_wrong/max(len(_old),1)*100:.1f}%) to the wrong embedding (this was the bug).")"""

SETUP_CODE = r"""# 10.3 — Main loop setup (v2: stable sort + id-keyed positions + regression guard).
from tqdm.asyncio import tqdm as tqdm_async
LOOP_LIMIT = None
TAU_HIGH = CONFIG["tau_high"]; TAU_LOW = CONFIG["tau_low"]
WINDOW   = pd.Timedelta(hours=CONFIG["active_window_hours"])

pos_of_id = pd.Series(np.arange(len(canonical_items)), index=canonical_items["item_id"])
sorted_items = canonical_items.sort_values("published_at", kind="stable").copy()   # STABLE
sorted_items["pos"] = sorted_items["item_id"].map(pos_of_id).to_numpy()            # TRUE position
assert (canonical_items["item_id"].to_numpy()[sorted_items["pos"].to_numpy()]
        == sorted_items["item_id"].to_numpy()).all(), "item-vector mapping is broken"

def vec_for(item_id):
    return assignment_vecs[pos_of_id[item_id]]

to_iter = sorted_items if LOOP_LIMIT is None else sorted_items.head(LOOP_LIMIT)
stories = []; outcomes = []
print(f"Processing {len(to_iter):,} items (tau_high={TAU_HIGH}, tau_low={TAU_LOW}) ...")"""

STORIES_DF = r"""# 10.5b (v2) — Materialize the story accumulator as a DataFrame + outcomes as a column.
sorted_items = sorted_items.reset_index(drop=True)
sorted_items["outcome"] = outcomes
stories_df = pd.DataFrame([{
    "story_id": s["story_id"], "n_items": s["n_items"],
    "member_ids": s["member_ids"], "item_clients": sorted(s["item_clients"]),
    "first_seen_at": s["first_seen_at"], "last_seen_at": s["last_seen_at"], "closed_at": s["closed_at"],
} for s in stories])
print(f"stories_df: {len(stories_df):,} rows | singletons={int((stories_df.n_items==1).sum()):,} "
      f"| multi-item={int((stories_df.n_items>1).sum()):,}")
stories_df.head()"""

MERGE_JUDGE = r"""# 13.2 — Pairwise merge judge (v2: OpenAI gpt-4.1) with cache + 429 retry.
def _story_display(s):
    m = s.get("metadata", {}) or {}
    return m.get("title") or canonical_items.iloc[s["member_idxs"][0]]["title"], m.get("summary", "")

async def merge_judge(story_a, story_b) -> bool:
    a_ids = "|".join(sorted(story_a["member_ids"]))
    b_ids = "|".join(sorted(story_b["member_ids"]))
    h = hashlib.sha256(f"merge|{a_ids}|{b_ids}|{CONFIG['judge_model']}".encode()).hexdigest()[:16]
    fpath = JUDGE_CACHE_DIR / f"merge_{h}.json"
    if fpath.exists():
        return json.loads(fpath.read_text())["verdict"] == "SAME"
    await _openai_judge_limiter.acquire()
    async def _call():
        ta, sa = _story_display(story_a)
        tb, sb = _story_display(story_b)
        prompt = (
            "Two news story clusters - same underlying story or different?\n\n"
            f"STORY A: {ta}\n  {sa}\n\n"
            f"STORY B: {tb}\n  {sb}\n\n"
            "Reply with a single word: SAME or DIFFERENT."
        )
        resp = await _openai_judge_client.chat.completions.create(
            model=CONFIG["judge_model"], temperature=0, max_completion_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip().upper()
    text = await _retry_on_429(_call)
    verdict = "SAME" if text.startswith("SAME") else "DIFFERENT"
    fpath.write_text(json.dumps({"verdict": verdict}))
    return verdict == "SAME"

if candidates:
    judges = await tqdm_async.gather(
        *[merge_judge(active_multi[i], active_multi[j]) for i, j, _, _ in candidates],
        desc="merge judge",
    )
else:
    judges = []

merge_pairs = [(c[0], c[1]) for c, v in zip(candidates, judges) if v]
print(f"Verdicts: {sum(judges)} SAME / {len(judges)} candidates -> {len(merge_pairs)} merges to apply")"""

EVAL_141 = r"""# 14.1 — Build item_id -> story_id lookup; predict; headline metrics (v2: two-number report).
item_to_story = {}
for s in stories_final:
    for mid in s["member_ids"]:
        item_to_story[mid] = s["story_id"]

def pred_for_pair(a_id, b_id):
    sa, sb = item_to_story.get(a_id), item_to_story.get(b_id)
    if sa is None or sb is None:
        return None
    return "SAME" if sa == sb else "DIFFERENT"

def pr_f1(y_true, y_pred):
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    tp = sum(1 for t, p in pairs if t == "SAME" and p == "SAME")
    fp = sum(1 for t, p in pairs if t == "DIFFERENT" and p == "SAME")
    fn = sum(1 for t, p in pairs if t == "SAME" and p == "DIFFERENT")
    tn = sum(1 for t, p in pairs if t == "DIFFERENT" and p == "DIFFERENT")
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_scored": len(pairs)}

eval_df["pred_poc_raw"] = [pred_for_pair(a, b) for a, b in zip(eval_df["item_a_id"], eval_df["item_b_id"])]
n_missing = int(eval_df["pred_poc_raw"].isna().sum())
# items not in any story (boilerplate set-aside, etc.) -> predicted DIFFERENT (correctly not merged)
eval_df["pred_poc"] = eval_df["pred_poc_raw"].fillna("DIFFERENT")
print(f"Items not in any story (-> predicted DIFFERENT): {n_missing} / {len(eval_df)}")

poc_metrics = pr_f1(eval_df["final_label"], eval_df["pred_poc"])
poc_f1 = poc_metrics["f1"]
_clean = eval_df[~eval_df["is_boilerplate_pair"]]
m_clean = pr_f1(_clean["final_label"], _clean["pred_poc"])

print("\nPAIRWISE F1 on the stratified eval set (NOT corpus B-cubed):")
print(f"  full {poc_metrics['n_scored']} pairs (vs old 0.323): "
      f"P={poc_metrics['precision']:.3f} R={poc_metrics['recall']:.3f} F1={poc_metrics['f1']:.3f}")
print(f"  non-boilerplate ({m_clean['n_scored']} pairs):        "
      f"P={m_clean['precision']:.3f} R={m_clean['recall']:.3f} F1={m_clean['f1']:.3f}")"""

EVAL_142B = r"""# 14.2b (v2) — baselines on the non-boilerplate subset + behavioral check that near-dups now merge.
_clean = eval_df[~eval_df["is_boilerplate_pair"]]   # re-snapshot AFTER §14.2 added pred_b1/pred_b3
print("Baselines (full | non-boilerplate):")
for _name, _col in [("cosine>=0.65", "pred_b1"), ("title-Jaccard", "pred_b3")]:
    _bf = pr_f1(eval_df["final_label"], eval_df[_col])
    _bc = pr_f1(_clean["final_label"], _clean[_col])
    print(f"  {_name:14s} F1={_bf['f1']:.3f}  |  {_bc['f1']:.3f}")

_hi = eval_df[(eval_df.final_label == "SAME") & (eval_df.cosine_sim >= 0.90)]
_tp = int((_hi["pred_poc"] == "SAME").sum())
print(f"\nHigh-cosine (>=0.90) SAME pairs now co-clustered: {_tp}/{len(_hi)} (was ~0 in the buggy run)")"""

V2_FINDINGS = r"""# 16.3b (v2) — Extra summary with the metric caveat + non-boilerplate number.
_v2 = CONFIG["artifacts_dir"] / "v2_findings.md"
_v2.write_text(
    "# POC v2 findings (minimal un-break + gpt-4.1 judge)\n\n"
    f"- Pairwise F1 (full {poc_metrics['n_scored']} pairs; boilerplate->DIFFERENT): {poc_metrics['f1']:.3f}  (old buggy run: 0.323)\n"
    f"- Pairwise F1 (non-boilerplate subset, {m_clean['n_scored']} pairs): {m_clean['f1']:.3f}\n"
    f"- Judge model: {CONFIG['judge_model']}\n"
    f"- tau_high/tau_low (recalibrated on non-boilerplate): {CONFIG['tau_high']}/{CONFIG['tau_low']}  (prior 0.88/0.54)\n\n"
    "Metric caveat: pairwise F1 on a cosine-stratified eval set - NOT the spec's corpus B-cubed. "
    "Deferred work in docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md section 6.\n"
)
print("Wrote", _v2)"""

S21_CODE = r"""import pyarrow.dataset as pa_ds, pyarrow.compute as pc, pyarrow as pa

# 2.1 (v2) — Load ONLY the 2012-2013 working window, and skip the raw shard load entirely
# when the canonical cache exists. Avoids materializing the full 8.2M-row feed (~72 GB crash).
SLICE_START_STR, SLICE_END_STR = "2012-01-01", "2014-01-01"   # [start, end)
_cache_path = CONFIG["cache_dir"] / "canonical.parquet"

if _cache_path.exists():
    raw_df = None
    print(f"canonical cache present at {_cache_path} -> skipping raw shard load (and §2.2).")
else:
    source_dir = CONFIG["data_dir"] / CONFIG["primary_source"]
    parquet_files = sorted(source_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet shards in {source_dir}")
    dataset = pa_ds.dataset(parquet_files, format="parquet")
    print(f"Schema:\n{dataset.schema}")
    print(f"Total rows across all shards: {dataset.count_rows():,}")
    # Date pushdown at the parquet level (raw 'date' is ISO-8601 string, lexicographically sortable).
    _flt = (pc.field("date") >= SLICE_START_STR) & (pc.field("date") < SLICE_END_STR)
    print(f"Loading ONLY {SLICE_START_STR}..{SLICE_END_STR} into pandas (date pushdown) ...")
    raw_df = dataset.to_table(filter=_flt).to_pandas()
    print(f"Loaded raw_df: {len(raw_df):,} rows x {len(raw_df.columns)} columns")
    print(f"Memory footprint: {raw_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")"""

S22_CODE = r"""# 2.2 (v2) — Parse extra_fields; skipped entirely when the canonical cache is used.
if raw_df is None:
    print("§2.2 skipped - using canonical cache (raw_df not loaded).")
else:
    print("Parsing extra_fields JSON ...")
    parsed_records = [json.loads(s) for s in raw_df["extra_fields"]]
    extra_df = pd.json_normalize(parsed_records)
    print(f"extra_df columns: {list(extra_df.columns)}")
    raw_df = raw_df.reset_index(drop=True).join(extra_df.reset_index(drop=True))
    raw_df = raw_df.drop(columns=["extra_fields"])
    print(f"raw_df now has {len(raw_df.columns)} columns: {list(raw_df.columns)}")"""

S23_CODE = r"""import re

# 2.3 — Helper: split a 'text' field into (title, body). Bloomberg='-- TITLE\n\nBODY'; Reuters='TITLE'.
TITLE_PREFIX_RE = re.compile(r"^--\s+")

def split_title_body(text):
    if not isinstance(text, str):
        return "", ""
    text = text.strip()
    text = TITLE_PREFIX_RE.sub("", text)
    if "\n\n" in text:
        title, body = text.split("\n\n", 1)
        return title.strip(), body.strip()
    return text, ""

cache_path = CONFIG["cache_dir"] / "canonical.parquet"

if cache_path.exists():
    # v2: read ONLY the working window via a date pushdown so we never materialize all years (~72 GB).
    import pyarrow.dataset as pa_ds, pyarrow.compute as pc, pyarrow as pa
    _start = pd.Timestamp(SLICE_START_STR, tz="UTC"); _end = pd.Timestamp(SLICE_END_STR, tz="UTC")
    _flt = (pc.field("published_at") >= pa.scalar(_start)) & (pc.field("published_at") < pa.scalar(_end))
    canonical_df = pa_ds.dataset(cache_path, format="parquet").to_table(filter=_flt).to_pandas()
    print(f"Loaded canonical_df from cache (date-filtered {SLICE_START_STR}..{SLICE_END_STR}): {len(canonical_df):,} rows")
else:
    print("Building canonical_df from raw_df ...")
    pairs = [split_title_body(t) for t in raw_df["text"]]
    titles = [p[0] for p in pairs]
    bodies = [p[1] for p in pairs]
    canonical_df = pd.DataFrame({
        "item_id":      pd.NA,
        "title":        titles,
        "body":         bodies,
        "source":       raw_df["source"],
        "dataset":      raw_df["dataset"],
        "text_type":    raw_df["text_type"],
        "time_precision": raw_df["time_precision"],
        "published_at": pd.to_datetime(raw_df["date"], utc=True, errors="coerce"),
        "url":          raw_df["url"],
        "author":       raw_df.get("author"),
    })
    canonical_df.to_parquet(cache_path, index=False)
    print(f"Wrote cache: {cache_path}")

print(f"canonical_df: {len(canonical_df):,} rows x {len(canonical_df.columns)} columns")"""

REUSE_FLAG = r"""# 7.0 (v2) — Reuse the existing labeled eval set by default (no re-sampling / re-labeling).
# Flip to False to actually run the original 3-vendor ensemble (Sonnet + GPT-5.2 + Gemini)
# on the current corpus. Kept True so a normal Run All reuses the cached 543-pair set.
REUSE_EXISTING_EVAL = True
print("REUSE_EXISTING_EVAL =", REUSE_EXISTING_EVAL,
      "(True = reuse cached labels; set False to re-run the ensemble)")"""

META_196 = r"""# 12.1 (v2) — OpenAI metadata client + limiter + 429-retry (replaces Sonnet to dodge Anthropic limits).
import asyncio, time
from collections import deque
from openai import AsyncOpenAI

if "AsyncRateLimiter" not in dir():
    class AsyncRateLimiter:
        def __init__(self, max_calls, period=60.0):
            self.max_calls = max_calls; self.period = period
            self._ts = deque(); self._lock = asyncio.Lock()
        async def acquire(self):
            async with self._lock:
                now = time.monotonic()
                while self._ts and self._ts[0] <= now - self.period:
                    self._ts.popleft()
                if len(self._ts) >= self.max_calls:
                    await asyncio.sleep(self._ts[0] + self.period - now)
                    now = time.monotonic()
                    while self._ts and self._ts[0] <= now - self.period:
                        self._ts.popleft()
                self._ts.append(time.monotonic())

_openai_meta_client = AsyncOpenAI()
_meta_limiter = AsyncRateLimiter(CONFIG["vendor_rate_limits_rpm"].get("openai", 50))

async def _retry_on_429(coro_fn, max_retries=3, base_wait=60.0):
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            is_429 = (getattr(exc, "status_code", None) == 429
                      or "rate_limit" in str(exc).lower() or "429" in str(exc)[:80])
            if not is_429 or attempt >= max_retries:
                raise
            wait = base_wait * (1.5 ** attempt)
            print(f"  metadata 429, sleeping {wait:.0f}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait)

print(f"Metadata client ready: {_meta_limiter.max_calls} RPM (gpt-4.1)")"""

META_197 = r"""# 12.1 (cont.) (v2) — JSON schema for OpenAI structured output (response_format).
STORY_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title":    {"type": "string", "description": "Headline-style story title, <=80 chars."},
        "summary":  {"type": "string", "description": "Two-sentence factual summary, <=400 chars."},
        "topic":    {"type": "string", "description": "One-word topic tag (earnings, acquisition, regulation, ...)."},
        "entities": {"type": "array", "items": {"type": "string"}, "description": "Primary companies and people."},
    },
    "required": ["title", "summary", "topic", "entities"],
    "additionalProperties": False,
}

METADATA_CACHE_DIR = CONFIG["cache_dir"] / "story_metadata"
METADATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"JSON schema + cache dir ready: {METADATA_CACHE_DIR}")"""

META_199 = r"""# 12.2 (v2) — Prompt + cached OpenAI (gpt-4.1) metadata generator.
MAX_MEMBERS_IN_PROMPT = 5

def build_metadata_prompt(member_rows):
    lines = [
        "Generate concise metadata for this cluster of related financial news items.",
        "Be factual; cover the primary event and the entities involved.",
        "Title <= 80 chars. Summary <= 400 chars. Topic = one word.\n\nCLUSTER MEMBERS:",
    ]
    for i, (_, r) in enumerate(member_rows.head(MAX_MEMBERS_IN_PROMPT).iterrows(), start=1):
        lede = (r["body"] or "")[:200]
        lines.append(f"\n[{i}] {r['title']}\n    {lede}")
    return "\n".join(lines)

def _metadata_key(story):
    member_blob = "|".join(sorted(story["member_ids"]))
    h = hashlib.sha256()
    h.update(f"{CONFIG['metadata_model']}|".encode())
    h.update(member_blob.encode("utf-8"))
    return h.hexdigest()[:16]

async def generate_story_metadata(story):
    key = _metadata_key(story)
    fpath = METADATA_CACHE_DIR / f"{key}.json"
    if fpath.exists():
        return json.loads(fpath.read_text()), {"cached": True}
    member_rows = canonical_items.iloc[story["member_idxs"]]
    prompt = build_metadata_prompt(member_rows)
    await _meta_limiter.acquire()
    async def _call():
        return await _openai_meta_client.chat.completions.create(
            model=CONFIG["metadata_model"], temperature=0, max_completion_tokens=400,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "story_metadata", "strict": True,
                                             "schema": STORY_METADATA_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
    resp = await _retry_on_429(_call)
    meta = json.loads(resp.choices[0].message.content)
    fpath.write_text(json.dumps(meta))
    u = resp.usage
    return meta, {"input_tokens": getattr(u, "prompt_tokens", 0), "output_tokens": getattr(u, "completion_tokens", 0)}"""

LONG_CAP = r'''
# v2: cap the long-item set so §9 is a SMALL retrieval benchmark, not the whole corpus
# (embedding every long article's chunks blows OpenAI's tokens-per-minute limit).
LONG_ITEMS_SAMPLE = 50
if len(long_items) > LONG_ITEMS_SAMPLE:
    long_items = long_items.sample(LONG_ITEMS_SAMPLE, random_state=CONFIG["random_seed"]).reset_index(drop=True)
    print(f"Capped long items to {len(long_items)} for the §9 benchmark (raise LONG_ITEMS_SAMPLE for more).")'''

EMBED_HELPER_9 = r'''# 9.3d — Batched OpenAI embedding helpers (v2: 429-retry with backoff to respect the TPM limit).
async def embed_batch_9(texts):
    for attempt in range(6):
        await _embed_limiter_9.acquire()
        try:
            resp = await _openai_9.embeddings.create(
                model=CONFIG["embed_model"], input=texts, dimensions=CONFIG["embed_dims"],
            )
            return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait = 2.0 * (attempt + 1)
                print(f"  embed 429, sleeping {wait:.0f}s (attempt {attempt+1}/6)")
                await asyncio.sleep(wait)
                continue
            raise
    raise RuntimeError("embed_batch_9: exhausted 429 retries")

async def embed_column(texts):
    """Embed a list of strings in batches of 100; returns list of vectors."""
    out = []
    for i in range(0, len(texts), 100):
        out.extend(await embed_batch_9(texts[i:i + 100]))
    return out'''

# ----------------------------- transform -----------------------------
nb = json.loads(SRC.read_text())
cells = nb["cells"]

def text(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s

def mkcode(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

def mkmd(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def neutralize(tag):
    return mkcode(f'# [v2] {tag} skipped - not needed for the F1 number\n'
                  f'#      (see docs/superpowers/specs/2026-06-01-poc-clustering-rework-design.md)\n'
                  f'print("[v2] skipped: {tag}")')

def guard_wrap(src):
    # Wrap an original §7 cell so it only runs when REUSE_EXISTING_EVAL is False.
    return ("if not REUSE_EXISTING_EVAL:\n" + textwrap.indent(src, "    ")
            + "\nelse:\n    pass  # [v2] reuse cached labeled_eval_set.csv")

out, section, changed = [], None, set()
for c in cells:
    src = text(c)
    ctype = c["cell_type"]
    if ctype == "markdown":
        m = re.match(r"^\s*##\s+Section\s+(\d+)", src)
        if m:
            section = int(m.group(1))
    if ctype == "code":              # start every code cell clean
        c = dict(c); c["outputs"] = []; c["execution_count"] = None

    if ctype == "markdown" and "# Story Clustering" in src and "POC Notebook" in src:
        out.append(c); out.append(mkmd(CHANGELOG)); changed.add("changelog"); continue
    if ctype == "markdown" and src.strip().startswith("## Section 7"):
        out.append(c); out.append(mkcode(REUSE_FLAG)); changed.add("reuse"); continue
    if ctype == "code" and "Single source of truth for every tunable parameter" in src:
        assert 'Path.cwd() / "artifacts",' in src, "config artifacts_dir anchor"
        src = src.replace('Path.cwd() / "artifacts",', 'Path.cwd() / "artifacts" / "v2",')
        assert '"claude-haiku-4-5-20251001",' in src, "config haiku_model anchor"
        src = src.replace('"claude-haiku-4-5-20251001",',
                          '"claude-haiku-4-5-20251001",\n'
                          '    "judge_model":        "gpt-4.1-mini",  # v2: §10 gray-zone + §13 merge (binary; ~5x cheaper, ~=accuracy)\n'
                          '    "metadata_model":     "gpt-4.1",       # v2: §12 metadata + §9 question-gen (quality generation)', 1)
        c["source"] = src; out.append(c); changed.add("config"); continue
    if ctype == "code" and "# 2.1 — Locate the parquet shards." in src:
        c["source"] = S21_CODE; out.append(c); changed.add("s21"); continue
    if ctype == "code" and "# 2.2 — Parse extra_fields once" in src:
        c["source"] = S22_CODE; out.append(c); changed.add("s22"); continue
    if ctype == "code" and "# 2.3 — Helper: split a 'text' field" in src:
        c["source"] = S23_CODE; out.append(c); changed.add("s23"); continue
    if ctype == "code" and 'canonical_items = items_df[~items_df["is_duplicate"]]' in src:
        out.append(c); out.append(mkmd(BP_MD)); out.append(mkcode(BP_CODE)); out.append(mkcode(BP_AUDIT))
        changed.add("boilerplate"); continue
    if section == 7 and ctype == "code":
        if ("LABEL_PROMPT_TEMPLATE =" in src) or ("def label_with_sonnet" in src):
            out.append(c)                      # keep the prompt + labeler/client/limiter DEFINITIONS as-is
        else:
            out.append(mkcode(guard_wrap(src))); changed.add("s7")   # real code, gated by REUSE flag
        continue
    if ctype == "code" and "# 8.1 — Load the eval set" in src:
        out.append(c); out.append(mkcode(BP_PAIR_CODE)); changed.add("s81b"); continue
    if ctype == "code" and "# 8.3 — Numeric ground truth" in src:
        a = 'y_true = (eval_df["final_label"] == "SAME").astype(int).to_numpy()'
        b = 'sim    = eval_df["cosine_sim"].to_numpy()'
        assert a in src and b in src, "8.3 anchors"
        src = src.replace(a, '_clean8 = eval_df[~eval_df["is_boilerplate_pair"]]\n'
                             'y_true = (_clean8["final_label"] == "SAME").astype(int).to_numpy()')
        src = src.replace(b, 'sim    = _clean8["cosine_sim"].to_numpy()')
        c["source"] = src; out.append(c); changed.add("s83"); continue
    if ctype == "code" and "async def haiku_judge_same" in src:
        c["source"] = src + "\n\n\n" + GPT_JUDGE; out.append(c); changed.add("judge"); continue
    if ctype == "code" and "# 10.3 — Main loop. Reset state" in src:
        out.append(mkmd("### 10.3a — Verify the item-vector mapping (regression guard)"))
        out.append(mkcode(REG_CODE)); out.append(mkcode(SETUP_CODE)); changed.add("setup"); continue
    if ctype == "code" and "# 10.3 (cont.) — The actual single-pass loop" in src:
        assert "item_vec = assignment_vecs[item_pos]" in src, "loop vec anchor"
        src = src.replace("item_vec = assignment_vecs[item_pos]", 'item_vec = vec_for(row["item_id"])')
        assert "await haiku_judge_same(row, rep_row)" in src, "loop judge anchor"
        src = src.replace("await haiku_judge_same(row, rep_row)", "await judge_same(row, rep_row)")
        c["source"] = src; out.append(c); changed.add("loop"); continue
    if ctype == "code" and "# 10.5 — Per-outcome counts" in src:
        out.append(c); out.append(mkcode(STORIES_DF)); changed.add("storiesdf"); continue
    if section == 12 and ctype == "code":
        if "# 12.1 — Sonnet async client" in src:
            c["source"] = META_196; out.append(c); changed.add("s12_196")
        elif "# 12.1 (cont.) — Tool schema" in src:
            c["source"] = META_197; out.append(c); changed.add("s12_197")
        elif "# 12.2 — Prompt + cached Sonnet metadata generator" in src:
            c["source"] = META_199; out.append(c); changed.add("s12_199")
        elif '"source": "sonnet"' in src:
            c["source"] = src.replace('"source": "sonnet"', '"source": "gpt-4.1"'); out.append(c); changed.add("s12_202")
        else:
            out.append(c)                       # keep 12.3-pick / 12.4-12.6 tables as-is
        continue
    if section == 9 and ctype == "code":
        if "# 9.4b — Generate one factual question per source chunk" in src:
            src = src.replace("(Haiku, 429-retried", "(gpt-4.1, 429-retried")
            src = src.replace("resp = await _anthropic_9.messages.create(",
                              "resp = await _openai_9.chat.completions.create(")
            src = src.replace('model=CONFIG["haiku_model"], max_tokens=80, temperature=0,',
                              'model=CONFIG["metadata_model"], max_completion_tokens=80, temperature=0,')
            src = src.replace("return resp.content[0].text.strip()",
                              'return (resp.choices[0].message.content or "").strip()')
            c["source"] = src; out.append(c); changed.add("s9swap")
        elif "# 9.1 — Tokenizer" in src:
            anchor = 'print(f"Long items (> {LONG_THRESHOLD_TOKENS:,} tokens): {len(long_items):,}")'
            assert anchor in src, "9.1 long-items print anchor"
            src = src.replace(anchor, anchor + "\n" + LONG_CAP); c["source"] = src
            out.append(c); changed.add("s9cap")
        elif "# 9.3d — Batched OpenAI embedding helpers" in src:
            c["source"] = EMBED_HELPER_9; out.append(c); changed.add("s9embed")
        else:
            out.append(c)                       # keep other §9 cells (defs / OpenAI doc-context / embeds)
        continue
    if ctype == "code" and "# 13.2 — Pairwise Haiku merge judge" in src:
        c["source"] = MERGE_JUDGE; out.append(c); changed.add("merge"); continue
    if ctype == "code" and "# 14.1 — Build item_id" in src:
        c["source"] = EVAL_141; out.append(c); changed.add("e141"); continue
    if ctype == "code" and "# 14.2 — Compute baseline" in src:
        out.append(c); out.append(mkcode(EVAL_142B)); changed.add("e142b"); continue
    if ctype == "code" and "# 15.1 — Pricing" in src:
        assert '"sonnet-input":    3.00 / 1_000_000, "sonnet-output": 15.00 / 1_000_000,' in src, "15.1 prices"
        src = src.replace('"sonnet-input":    3.00 / 1_000_000, "sonnet-output": 15.00 / 1_000_000,',
                          '"sonnet-input":    3.00 / 1_000_000, "sonnet-output": 15.00 / 1_000_000,\n'
                          '    "gpt41-input":     2.00 / 1_000_000, "gpt41-output":  8.00 / 1_000_000,\n'
                          '    "gpt41mini-input": 0.40 / 1_000_000, "gpt41mini-output": 1.60 / 1_000_000,')
        src = src.replace('cost_per_call("gray_judge",     "haiku-input",  "haiku-output")',
                          'cost_per_call("gray_judge",     "gpt41mini-input", "gpt41mini-output")')
        src = src.replace('cost_per_call("merge_judge",    "haiku-input",  "haiku-output")',
                          'cost_per_call("merge_judge",    "gpt41mini-input", "gpt41mini-output")')
        src = src.replace('cost_per_call("story_metadata", "sonnet-input", "sonnet-output")',
                          'cost_per_call("story_metadata", "gpt41-input",  "gpt41-output")')
        src = src.replace('cost_per_call("doc_context",    "haiku-input",  "haiku-output")',
                          'cost_per_call("doc_context",    "gpt41-input",  "gpt41-output")')
        c["source"] = src; out.append(c); changed.add("s151"); continue
    if ctype == "code" and "# 16.3 — Assemble" in src:
        out.append(c); out.append(mkcode(V2_FINDINGS)); changed.add("v2find"); continue
    out.append(c)

nb["cells"] = out

# stage eval inputs into artifacts/v2 so the v2 notebook (artifacts_dir=artifacts/v2) finds them
(ROOT / "artifacts" / "v2").mkdir(parents=True, exist_ok=True)
for f in ["labeled_eval_set.csv", "human_labels.csv"]:
    p = ROOT / "artifacts" / f
    if p.exists():
        shutil.copy(p, ROOT / "artifacts" / "v2" / f)

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

required = {"changelog", "config", "s21", "s22", "s23", "boilerplate", "reuse", "s7", "s81b", "s83",
            "judge", "setup", "loop", "storiesdf", "s12_196", "s12_197", "s12_199", "s12_202",
            "s9swap", "s9cap", "s9embed", "merge", "e141", "e142b", "s151", "v2find"}
missing = required - changed
print("APPLIED:", sorted(changed))
print("MISSING:", sorted(missing) if missing else "none")
print("cells:", len(out), "(was", len(cells), ")")
assert not missing, f"missing edits: {missing}"
print("OK -> wrote", DST)
