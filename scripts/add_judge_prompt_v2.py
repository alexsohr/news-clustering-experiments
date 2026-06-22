#!/usr/bin/env python
"""Add the rubric-aligned v2 judge prompt to build_judge_prompt (10.2 cont. cell)."""
import nbformat

NB = "story_clustering_poc_v3.ipynb"
nb = nbformat.read(NB, as_version=4)
hits = [c for c in nb.cells if c.cell_type == "code" and c.source.startswith("# 10.2 (cont., v3)")]
assert len(hits) == 1

OLD = '''def build_judge_prompt(item_row, rep_row, variant="plain"):
    return (
        "Two financial news items - same story or different?\\n\\n"
        f"ITEM A: {item_row['title']}\\n  {(item_row['body'] or '')[:200]}\\n\\n"
        f"ITEM B: {rep_row['title']}\\n  {(rep_row['body'] or '')[:200]}\\n\\n"
        "Reply with a single word: SAME or DIFFERENT."
    )'''

NEW = '''def build_judge_prompt(item_row, rep_row, variant="plain"):
    pv = CONFIG.get("judge_prompt_version", "v1")
    if pv == "v1":   # legacy prompt — keeps the entire v2-era verdict cache valid
        return (
            "Two financial news items - same story or different?\\n\\n"
            f"ITEM A: {item_row['title']}\\n  {(item_row['body'] or '')[:200]}\\n\\n"
            f"ITEM B: {rep_row['title']}\\n  {(rep_row['body'] or '')[:200]}\\n\\n"
            "Reply with a single word: SAME or DIFFERENT."
        )
    # v2 — rubric-aligned with the §7.3 ensemble labelers (event-level definition, dates,
    # full ledes) + an explicit saga-vs-event clause. R2 attribution showed both error
    # modes were rubric mismatches: saga follow-ups judged SAME (FPs), cross-source
    # restatements of one event judged DIFFERENT (FNs).
    def _blk(r):
        return (f"  title: {r['title']}\\n"
                f"  lede:  {(r['body'] or '')[:CONFIG['lede_chars']]}\\n"
                f"  published_at: {r['published_at']}")
    return (
        "You are determining whether two financial news items describe the SAME news event.\\n\\n"
        f"ITEM A:\\n{_blk(item_row)}\\n\\n"
        f"ITEM B:\\n{_blk(rep_row)}\\n\\n"
        "Two items are the SAME story if they describe the same underlying news event "
        "involving the same primary entities — even if framing, source, or details differ. "
        'They are DIFFERENT if the primary event differs (e.g., "Apple Q3 earnings beat" vs '
        '"Apple Q3 earnings preview" are different events despite same topic; follow-ups in an '
        "ongoing saga that report distinct developments are DIFFERENT events).\\n\\n"
        "Reply with a single word: SAME or DIFFERENT."
    )'''

assert OLD in hits[0].source, "anchor missing"
hits[0].source = hits[0].source.replace(OLD, NEW)
compile(hits[0].source, "judgecell", "exec")
nbformat.write(nb, NB)
print("build_judge_prompt v2 added (v1 path byte-identical)")
