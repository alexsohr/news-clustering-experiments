#!/usr/bin/env python
"""Headless cell-by-cell executor for story_clustering_poc_v3.ipynb.

Executes code cells in notebook order on a fresh ipykernel, saving each cell's
outputs + execution_count back into the notebook after it finishes, and printing
one status line per cell. Stops on first error (notebook stays saved up to there).

Usage: python -u scripts/run_v3_cells.py [--start N] [--stop N]
  --start N  first cell index to execute (default 0)
  --stop  N  last cell index to execute, inclusive (default: the §14 ledger cell)
Skips: §12 code cells (iteration plan: Phase 6 only).
"""
import argparse, json, os, re, sys, time
from queue import Empty

import nbformat
from jupyter_client.manager import start_new_kernel

NB_PATH = "story_clustering_poc_v3.ipynb"
CELL_TIMEOUT = 2400  # max quiet seconds per cell (labeling/judge cells run ~10 min)

PREAMBLE = """
%matplotlib inline
try:
    import plotly.io as _pio
    _pio.renderers.default = "notebook"
except Exception:
    pass
"""


def section_bounds(nb):
    bounds = {}
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown":
            m = re.match(r"\s*## Section (\d+)", c.source)
            if m:
                bounds[int(m.group(1))] = i
    return bounds


def run_source(kc, source, timeout=CELL_TIMEOUT):
    msg_id = kc.execute(source)
    outputs, status, exec_count = [], "ok", None
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except Empty:
            return outputs, "timeout", exec_count
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t, c = msg["msg_type"], msg["content"]
        if t == "execute_input":
            exec_count = c.get("execution_count")
        elif t == "stream":
            if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                outputs[-1]["text"] += c["text"]
            else:
                outputs.append(nbformat.v4.new_output("stream", name=c["name"], text=c["text"]))
        elif t in ("display_data", "execute_result"):
            kw = {"data": c["data"], "metadata": c.get("metadata", {})}
            if t == "execute_result":
                kw["execution_count"] = c.get("execution_count")
            outputs.append(nbformat.v4.new_output(t, **kw))
        elif t == "error":
            outputs.append(nbformat.v4.new_output(
                "error", ename=c["ename"], evalue=c["evalue"], traceback=c["traceback"]))
            status = "error"
        elif t == "status" and c["execution_state"] == "idle":
            break
    return outputs, status, exec_count


def trim_streams(outputs, keep=8000):
    for o in outputs:
        if o["output_type"] == "stream" and len(o["text"]) > keep:
            # keep the tail; tqdm \r spam compresses to the final state
            o["text"] = "...[trimmed]...\n" + o["text"][-keep:]
    return outputs


def last_line(outputs):
    for o in reversed(outputs):
        if o["output_type"] == "stream":
            lines = [l for l in o["text"].replace("\r", "\n").splitlines() if l.strip()]
            if lines:
                return lines[-1][:110]
        if o["output_type"] == "error":
            return f'{o["ename"]}: {o["evalue"][:90]}'
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()

    nb = nbformat.read(NB_PATH, as_version=4)
    bounds = section_bounds(nb)
    if os.environ.get("V3_INCLUDE_S12"):   # Phase-6 final run executes §12 too
        s12_range = range(0, 0)
    else:
        s12_range = range(bounds.get(12, 10**9), bounds.get(13, -1))
    ledger_stop = args.stop
    if ledger_stop is None:  # default: stop after the §14 ledger cell (last code cell before §15)
        ledger_stop = max(i for i in range(len(nb.cells))
                          if nb.cells[i].cell_type == "code" and i < bounds[15])

    todo = [i for i, c in enumerate(nb.cells)
            if c.cell_type == "code" and args.start <= i <= ledger_stop]
    print(f"notebook: {len(nb.cells)} cells | executing {len(todo)} code cells "
          f"({args.start}..{ledger_stop}), skipping §12 ({s12_range.start}..{s12_range.stop - 1})",
          flush=True)

    km, kc = start_new_kernel(kernel_name="python3", cwd=".")
    print("kernel started", flush=True)
    try:
        run_source(kc, PREAMBLE, timeout=120)
        t_all = time.time()
        for n, i in enumerate(todo, 1):
            if i in s12_range:
                print(f"[{n}/{len(todo)}] cell {i:3d}  SKIP (§12 — Phase 6 only)", flush=True)
                continue
            head = nb.cells[i].source.strip().splitlines()[0][:70] if nb.cells[i].source.strip() else "(empty)"
            t0 = time.time()
            outputs, status, exec_count = run_source(kc, nb.cells[i].source)
            nb.cells[i].outputs = trim_streams(outputs)
            nb.cells[i].execution_count = exec_count
            nbformat.write(nb, NB_PATH)
            print(f"[{n}/{len(todo)}] cell {i:3d}  {status.upper():7s} {time.time()-t0:7.1f}s  {head}",
                  flush=True)
            tail = last_line(outputs)
            if tail:
                print(f"          └─ {tail}", flush=True)
            if status != "ok":
                print(f"STOPPED at cell {i} ({status}); notebook saved with outputs so far.", flush=True)
                sys.exit(1)
        print(f"DONE: {len(todo)} cells in {(time.time()-t_all)/60:.1f} min", flush=True)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
