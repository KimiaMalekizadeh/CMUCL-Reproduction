"""Execute notebooks 00-05 in order (each one depends on the artifacts written by the previous one).

    python run_all.py                              # official Citeseer, as many nodes as fit 45 min on this CPU
    python run_all.py --budget-min 60
    python run_all.py --n-nodes full               # GPU / patience
    python run_all.py --only 04 05                 # re-run a subset
Executed copies (with outputs) are written to artifacts/executed/.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT = ROOT / "artifacts" / "executed"


def run(n_nodes=None, budget_min=None, max_batches=None, only=None):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if n_nodes: os.environ["CMUCL_N_NODES"] = str(n_nodes)
    if budget_min: os.environ["CMUCL_BUDGET_MIN"] = str(budget_min)
    if max_batches: os.environ["CMUCL_MAX_BATCHES"] = str(max_batches)
    OUT.mkdir(parents=True, exist_ok=True)
    nbs = sorted(NB_DIR.glob("0[0-5]_*.ipynb"))
    if only: nbs = [p for p in nbs if p.name[:2] in only]
    for p in nbs:
        t0 = time.time(); print(f"=== {p.name} ...", flush=True)
        nb = nbformat.read(p, as_version=4)
        try:
            NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        finally:
            nbformat.write(nb, OUT / p.name)       # keep partial output for debugging even on failure
        print(f"    done in {time.time()-t0:.0f}s", flush=True)
    print("All finished. Executed notebooks:", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", help='"auto" (default) | "full" | integer'); ap.add_argument("--budget-min", type=float)
    ap.add_argument("--max-batches", type=int); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(a.n_nodes, a.budget_min, a.max_batches, a.only)
