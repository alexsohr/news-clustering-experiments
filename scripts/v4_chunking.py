"""Pure helpers for v4 paragraph-chunking clustering. No I/O, no notebook globals.

Used by story_clustering_poc_v4.ipynb and tested in test_v4_chunking.py.
"""
from __future__ import annotations
import re
from collections import defaultdict, Counter

import numpy as np


# ----------------------------------------------------------------------------
# Paragraph chunking
# ----------------------------------------------------------------------------
_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(body):
    """Split a body into paragraphs on blank lines, falling back to single \\n.

    Non-str / None -> []. Strips each paragraph and drops empties.
    """
    if not isinstance(body, str):
        return []
    parts = _PARA_SPLIT.split(body)
    if len(parts) == 1:                      # no blank-line breaks -> fall back to single \n
        parts = body.split("\n")
    return [p.strip() for p in parts if p.strip()]


def _split_oversize(text, count_tokens, max_tokens):
    """Break a too-long chunk into <= max_tokens pieces.

    Sentence boundaries first; any single sentence still over the cap is packed
    word-by-word (handles bodies with no sentence punctuation).
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces, cur = [], []

    def flush():
        if cur:
            pieces.append(" ".join(cur))
            cur.clear()

    for unit in _SENT_SPLIT.split(text):
        if count_tokens(unit) > max_tokens:          # a single sentence is itself too big
            flush()
            words = unit.split()
            w = []
            for word in words:
                w.append(word)
                if count_tokens(" ".join(w)) >= max_tokens:
                    pieces.append(" ".join(w))
                    w = []
            if w:
                pieces.append(" ".join(w))
        else:
            cur.append(unit)
            if count_tokens(" ".join(cur)) >= max_tokens:
                flush()
    flush()
    return pieces or [text]


def chunk_body(title, body, *, count_tokens, min_tokens=25, max_tokens=400, max_chunks=12):
    """Return body-only chunk texts for one item.

    - tiny paragraphs (< min_tokens) merge into the previous chunk
    - oversize chunks (> max_tokens) are sentence/word split
    - result capped to the first max_chunks
    - empty/unusable body -> [title] (title-only fallback), or [] if title also empty

    `count_tokens` is a required callable str -> int.
    """
    title = (title or "").strip()
    paras = split_paragraphs(body)
    if not paras:                                  # title-only / empty-body fallback
        return [title] if title else []

    # merge tiny paragraphs into the previous chunk (or the next if there is no previous)
    merged = []
    for p in paras:
        if merged and count_tokens(p) < min_tokens:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    if len(merged) > 1 and count_tokens(merged[0]) < min_tokens:
        merged[1] = merged[0] + " " + merged[1]
        merged = merged[1:]

    # sentence/word-split oversize chunks
    chunks = []
    for c in merged:
        chunks.extend(_split_oversize(c, count_tokens, max_tokens))
    return chunks[:max_chunks]


# ----------------------------------------------------------------------------
# Chunk-level similarity
# ----------------------------------------------------------------------------
def max_pool_sim(vecs_a, vecs_b):
    """Nearest-chunk similarity for two unit-normalized row matrices.

    Returns (max_cosine, argmax_a_row, argmax_b_row). Empty either side -> (-1.0, -1, -1).
    """
    a = np.asarray(vecs_a)
    b = np.asarray(vecs_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return (-1.0, -1, -1)
    sims = a @ b.T
    ia, ib = np.unravel_index(int(sims.argmax()), sims.shape)
    return (float(sims[ia, ib]), int(ia), int(ib))


# ----------------------------------------------------------------------------
# Eval-seeded corpus sampling
# ----------------------------------------------------------------------------
def build_eval_seeded_sample(working_df, eval_item_ids, target, seed, id_col="item_id"):
    """All rows whose id_col is in eval_item_ids, plus a seeded random fill up to target.

    Repeatable. If the eval rows already meet/exceed target, returns just the eval rows
    (eval measurability takes priority over the exact target size).
    """
    import pandas as pd

    eval_set = set(eval_item_ids)
    present = working_df[working_df[id_col].isin(eval_set)].drop_duplicates(id_col)
    n_fill = max(0, target - len(present))
    remaining = working_df[~working_df[id_col].isin(eval_set)]
    if n_fill:
        fill = remaining.sample(n=min(n_fill, len(remaining)), random_state=seed)
    else:
        fill = remaining.iloc[:0]
    return pd.concat([present, fill]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Judge text builders
# ----------------------------------------------------------------------------
def build_judge_block(row, mode, matched_chunk=None):
    """One per-item block for the judge prompt.

    mode="chunk_pair" -> uses matched_chunk as the text (no lede, no full body).
    mode="full_body"  -> uses row["body"].
    """
    title = row["title"] or ""
    pub = row["published_at"]
    if mode == "chunk_pair":
        text = matched_chunk or ""
    elif mode == "full_body":
        text = row["body"] or ""
    else:
        raise ValueError(f"unknown judge_text_mode: {mode!r}")
    return f"  title: {title}\n  published_at: {pub}\n  text: {text}"


# ----------------------------------------------------------------------------
# Chunk-cluster -> item union-find (§11 mapping + over-merge diagnostic)
# ----------------------------------------------------------------------------
def union_items_from_chunk_clusters(chunk_item_ids, chunk_labels):
    """Union items that own chunks in the same non-noise (!= -1) HDBSCAN cluster.

    Returns {item_id: root}; every distinct item_id is a key (noise-only -> itself).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for iid in chunk_item_ids:               # ensure every item is a node
        find(iid)
    by_cluster = defaultdict(list)
    for iid, lab in zip(chunk_item_ids, chunk_labels):
        if lab != -1:
            by_cluster[lab].append(iid)
    for members in by_cluster.values():
        for m in members[1:]:
            union(members[0], m)
    return {iid: find(iid) for iid in parent}


def component_sizes(item_to_root):
    """Size of each transitive component (the §11 over-merge diagnostic)."""
    return dict(Counter(item_to_root.values()))
