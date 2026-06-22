#!/usr/bin/env python
"""Export per-item cluster-map data (UMAP coords + story membership + titles) to
presentation/cluster_map_data.json. Runs §1–§6 of the v3 notebook headlessly
(everything cached → ~1–2 min), then joins the post-§11 story checkpoint."""
import json, re, sys
sys.path.insert(0, "scripts")
from run_v3_cells import run_source, PREAMBLE  # reuse the proven executor

import nbformat
from jupyter_client.manager import start_new_kernel

nb = nbformat.read("story_clustering_poc_v3.ipynb", as_version=4)
bounds = {}
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown":
        m = re.match(r"\s*## Section (\d+)", c.source)
        if m: bounds[int(m.group(1))] = i

todo = [i for i in range(bounds[7]) if nb.cells[i].cell_type == "code"]
km, kc = start_new_kernel(kernel_name="python3", cwd=".")
print(f"kernel up; running {len(todo)} cells (§1–§6, cached)")
try:
    run_source(kc, PREAMBLE, timeout=120)
    for i in todo:
        outputs, status, _ = run_source(kc, nb.cells[i].source, timeout=600)
        if status != "ok":
            for o in outputs:
                if o["output_type"] == "error":
                    print("\n".join(o["traceback"])[-1500:])
            sys.exit(f"cell {i} failed ({status})")
    print("§1–§6 state ready; exporting map data")

    PAYLOAD = '''
import pickle, json
ck = pickle.load(open("artifacts/v3/checkpoints/post_s11.pkl", "rb"))
stories = ck["final_stories"]
sid_of_pos, n_of_pos, story_color = {}, {}, {}
rank = 0
for s in sorted(stories, key=lambda s: -s["n_items"]):
    if s["n_items"] > 1:
        story_color[s["story_id"]] = rank; rank += 1
    for p in s["member_idxs"]:
        sid_of_pos[p] = s["story_id"]; n_of_pos[p] = s["n_items"]
data = {"x": [], "y": [], "z": [], "title": [], "n": [], "ci": []}
for p in range(len(canonical_items)):
    cx, cy, cz = umap_coords[p]
    data["x"].append(round(float(cx), 3)); data["y"].append(round(float(cy), 3)); data["z"].append(round(float(cz), 3))
    data["title"].append(str(canonical_items.iloc[p]["title"])[:90])
    data["n"].append(int(n_of_pos.get(p, 1)))
    data["ci"].append(int(story_color.get(sid_of_pos.get(p), -1)))
json.dump(data, open("presentation/cluster_map_data.json", "w"))
print(f"exported {len(data['x'])} items | {rank} multi-item stories | "
      f"{sum(1 for c in data['ci'] if c < 0)} singletons")
'''
    outputs, status, _ = run_source(kc, PAYLOAD, timeout=300)
    for o in outputs:
        if o["output_type"] == "stream": print(o["text"], end="")
        if o["output_type"] == "error": print("\n".join(o["traceback"])[-1500:]); sys.exit(1)
finally:
    kc.stop_channels(); km.shutdown_kernel(now=True)
