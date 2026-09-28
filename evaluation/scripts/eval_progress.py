"""Live progress of a chunked VLN-CE eval: SR / SPL so far and ETA, from the .partial JSONs.

    python scripts/eval_progress.py eval_out/<run>/VLN-CE-v1/val_unseen
    watch -n 60 python scripts/eval_progress.py eval_out/<run>/VLN-CE-v1/val_unseen

Each chunk writes val_unseen_<N>-<i>.json.partial after every finished episode, and the final
val_unseen_<N>-<i>.json when done. Finished chunks are read from the final file.
"""
import glob
import json
import math
import os
import re
import sys
import time

d = sys.argv[1] if len(sys.argv) > 1 else "."
chunks = {}
for f in glob.glob(os.path.join(d, "*_*-*.json")) + glob.glob(os.path.join(d, "*_*-*.json.partial")):
    m = re.search(r"_(\d+)-(\d+)\.json(\.partial)?$", f)
    key = int(m.group(2))
    if key in chunks and chunks[key]["final"]:
        continue
    try:
        blob = json.load(open(f))
    except (json.JSONDecodeError, OSError):
        continue  # being replaced right now; the next call will read it
    if m.group(3):
        p = blob["_progress"]
        chunks[key] = {"final": False, "eps": blob["episodes"], "done": p["done"], "total": p["total"],
                       "elapsed": p["elapsed_s"], "age": time.time() - os.path.getmtime(f)}
    else:
        chunks[key] = {"final": True, "eps": blob, "done": len(blob), "total": len(blob), "elapsed": None, "age": 0}

if not chunks:
    sys.exit(f"no results yet in {d}")

eps = {k: v for c in chunks.values() for k, v in c["eps"].items()}
n = len(eps)
mean = lambda k: sum(float(v[k]) for v in eps.values()) / n
sr = mean("success")
se = math.sqrt(sr * (1 - sr) / n)
print(f"{d}\n")
print(f"{'chunk':>5} {'done':>11} {'rate s/ep':>9} {'ETA':>8}  status")
etas = []
for k in sorted(chunks):
    c = chunks[k]
    if c["final"]:
        print(f"{k:>5} {c['done']:>5}/{c['total']:<5} {'':>9} {'':>8}  finished")
        continue
    rate = c["elapsed"] / max(c["done"], 1)
    eta = rate * (c["total"] - c["done"]) - c["age"]
    etas.append(eta)
    print(f"{k:>5} {c['done']:>5}/{c['total']:<5} {rate:>9.0f} {max(eta, 0) / 3600:>7.1f}h  running (last update {c['age'] / 60:.0f} min ago)")
total = sum(c["total"] for c in chunks.values())
print(f"\nepisodes {n}/{total}   SR {sr:.3f} ± {1.96 * se:.3f} (95%)   SPL {mean('spl'):.3f}   "
      f"OSR {mean('oracle_success'):.3f}   NE {mean('distance_to_goal'):.2f} m   "
      f"500-step timeouts {sum(float(v['steps_taken']) >= 500 for v in eps.values())}")
if etas:
    print(f"ETA (slowest chunk): {max(etas) / 3600:.1f} h  -> ~{time.strftime('%a %H:%M', time.localtime(time.time() + max(etas)))}")
print("note: chunks run episodes scene by scene, so the early SR is a biased estimate until every chunk is well along.")
