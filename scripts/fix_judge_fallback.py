#!/usr/bin/env python
"""Split judge_same into _judge_once(model,...) + a wrapper that falls back to the
base judge model when an escalated call fails (timeout etc.). Slice-based rewrite."""
import ast, sys
import nbformat

NB = "story_clustering_poc_v3.ipynb"
nb = nbformat.read(NB, as_version=4)
hits = [c for c in nb.cells if c.cell_type == "code" and c.source.startswith("# 10.2 (cont., v3)")]
assert len(hits) == 1
src = hits[0].source

start = src.index("async def judge_same(")
end = src.index('print(f"judge_same ready')
NEW = '''async def _judge_once(model, item_row, rep_row) -> bool:
    a_id, b_id = sorted([item_row["item_id"], rep_row["item_id"]])
    fpath = JUDGE_CACHE_DIR / f"{_judge_key(model, a_id, b_id)}.json"
    if fpath.exists():
        JUDGE_STATS["cached"] += 1
        return json.loads(fpath.read_text())["verdict"] == "SAME"
    await _openai_judge_limiter.acquire()
    prompt = build_judge_prompt(item_row, rep_row)
    async def _call():
        resp = await _openai_judge_client.chat.completions.create(
            model=model, temperature=0,
            # escalation models may spend reasoning tokens before the answer
            max_completion_tokens=5 if model == CONFIG["judge_model"] else 4000,
            timeout=60 if model == CONFIG["judge_model"] else 180,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip().upper()
    text = await _retry_on_429(_call)
    verdict = "SAME" if text.startswith("SAME") else "DIFFERENT"
    fpath.write_text(json.dumps({"verdict": verdict}))
    JUDGE_STATS["new"] += 1
    return verdict == "SAME"

async def judge_same(item_row, rep_row, sim=None) -> bool:
    model = _judge_model_for(sim)
    try:
        return await _judge_once(model, item_row, rep_row)
    except Exception as exc:
        if model != CONFIG["judge_model"]:   # graceful degradation: escalated -> base judge
            print(f"  ⚠ {type(exc).__name__} from {model}; falling back to {CONFIG['judge_model']}")
            return await _judge_once(CONFIG["judge_model"], item_row, rep_row)
        raise

'''
hits[0].source = src[:start] + NEW + src[end:]
compile(hits[0].source, "judgecell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
nbformat.write(nb, NB)
print("judge_same now falls back to the base judge on escalated-call failure")
