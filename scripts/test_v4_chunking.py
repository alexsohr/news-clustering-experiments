"""Unit tests for v4_chunking pure helpers. Run: python -m pytest scripts/test_v4_chunking.py -v"""
import importlib
import numpy as np
import pandas as pd
import pytest


# ----- Task 0.1: module smoke -------------------------------------------------
def test_module_imports():
    mod = importlib.import_module("v4_chunking")
    assert mod is not None


# ----- Task 1.1: split_paragraphs --------------------------------------------
from v4_chunking import split_paragraphs


def test_split_on_blank_lines():
    assert split_paragraphs("A para.\n\nB para.\n\nC.") == ["A para.", "B para.", "C."]


def test_split_fallback_single_newline():
    assert split_paragraphs("line one\nline two") == ["line one", "line two"]


def test_split_strips_and_drops_empty():
    assert split_paragraphs("  x  \n\n\n\n  y ") == ["x", "y"]


def test_split_none_and_nonstr():
    assert split_paragraphs(None) == []
    assert split_paragraphs(123) == []
    assert split_paragraphs("") == []


# ----- Task 1.2: chunk_body ---------------------------------------------------
from v4_chunking import chunk_body

WC = lambda s: len(s.split())   # word-count token proxy for tests


def test_chunk_empty_body_falls_back_to_title():
    assert chunk_body("Apple beats Q3", "", count_tokens=WC) == ["Apple beats Q3"]
    assert chunk_body("Apple beats Q3", None, count_tokens=WC) == ["Apple beats Q3"]


def test_chunk_empty_title_and_body_is_empty():
    assert chunk_body("", "", count_tokens=WC) == []


def test_chunk_body_only_excludes_title():
    out = chunk_body("HEADLINE", "Para one has several words here.\n\nPara two also has words.",
                     count_tokens=WC, min_tokens=2)
    assert out == ["Para one has several words here.", "Para two also has words."]
    assert all("HEADLINE" not in c for c in out)


def test_tiny_paragraph_merges_into_previous():
    body = "This first paragraph has more than three words total.\n\ntiny\n\nAnother real paragraph here now."
    out = chunk_body("T", body, count_tokens=WC, min_tokens=3)
    assert out[0].endswith("tiny")
    assert len(out) == 2


def test_oversize_paragraph_is_split():
    body = " ".join(["word"] * 50)            # one 50-word paragraph
    out = chunk_body("T", body, count_tokens=WC, min_tokens=1, max_tokens=20)
    assert len(out) >= 3
    assert all(WC(c) <= 20 for c in out)


def test_cap_keeps_first_n():
    body = "\n\n".join([f"Paragraph number {i} with enough words." for i in range(20)])
    out = chunk_body("T", body, count_tokens=WC, min_tokens=1, max_chunks=12)
    assert len(out) == 12
    assert "number 0" in out[0]


# ----- Task 1.3: max_pool_sim -------------------------------------------------
from v4_chunking import max_pool_sim


def test_max_pool_identity_pair():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0]])               # matches row 1 of a
    sim, ia, ib = max_pool_sim(a, b)
    assert abs(sim - 1.0) < 1e-9 and ia == 1 and ib == 0


def test_max_pool_picks_best_pair():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.7071, 0.7071], [1.0, 0.0]])
    sim, ia, ib = max_pool_sim(a, b)
    assert ib == 1 and abs(sim - 1.0) < 1e-9


def test_max_pool_empty_side():
    assert max_pool_sim(np.zeros((0, 3)), np.ones((2, 3))) == (-1.0, -1, -1)


# ----- Task 1.4: build_eval_seeded_sample ------------------------------------
from v4_chunking import build_eval_seeded_sample


def _df(n):
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(n)], "v": range(n)})


def test_seed_includes_all_eval_items():
    df = _df(100); evals = {"i3", "i7", "i42"}
    out = build_eval_seeded_sample(df, evals, target=10, seed=1)
    assert evals.issubset(set(out["item_id"]))
    assert len(out) == 10


def test_seed_is_repeatable():
    df = _df(100); evals = {"i1"}
    a = build_eval_seeded_sample(df, evals, target=20, seed=1)
    b = build_eval_seeded_sample(df, evals, target=20, seed=1)
    assert list(a["item_id"]) == list(b["item_id"])


def test_eval_larger_than_target_keeps_all_eval():
    df = _df(100); evals = {f"i{k}" for k in range(30)}
    out = build_eval_seeded_sample(df, evals, target=10, seed=1)
    assert set(out["item_id"]) == evals and len(out) == 30


# ----- Task 1.5: build_judge_block -------------------------------------------
from v4_chunking import build_judge_block

ROW = {"title": "Apple beats Q3", "published_at": "2012-07-24", "body": "FULL BODY TEXT here."}


def test_judge_block_chunk_pair_uses_matched_chunk_not_body():
    blk = build_judge_block(ROW, "chunk_pair", matched_chunk="the matched paragraph")
    assert "the matched paragraph" in blk
    assert "FULL BODY TEXT" not in blk
    assert "Apple beats Q3" in blk and "2012-07-24" in blk


def test_judge_block_full_body_uses_body():
    blk = build_judge_block(ROW, "full_body")
    assert "FULL BODY TEXT here." in blk


def test_judge_block_bad_mode_raises():
    with pytest.raises(ValueError):
        build_judge_block(ROW, "title_lede")


# ----- Task 1.6: union_items_from_chunk_clusters + component_sizes ------------
from v4_chunking import union_items_from_chunk_clusters, component_sizes


def test_union_items_sharing_cluster_are_unioned():
    ids    = ["A", "B", "C"]
    labels = [ 0,   0,  -1 ]          # A,B share cluster 0; C is noise
    m = union_items_from_chunk_clusters(ids, labels)
    assert m["A"] == m["B"] and m["C"] != m["A"]


def test_union_transitive_merge_across_clusters():
    ids    = ["A", "B", "B", "C"]
    labels = [ 0,   0,   1,   1 ]
    m = union_items_from_chunk_clusters(ids, labels)
    assert m["A"] == m["B"] == m["C"]
    assert max(component_sizes(m).values()) == 3


def test_union_noise_only_items_are_singletons():
    ids = ["A", "B"]; labels = [-1, -1]
    m = union_items_from_chunk_clusters(ids, labels)
    assert m["A"] != m["B"]
    assert set(component_sizes(m).values()) == {1}
