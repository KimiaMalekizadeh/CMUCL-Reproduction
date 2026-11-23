"""Run this FIRST (in a terminal, not in Jupyter):   python diagnose.py
Prints every step before running it, so if something hangs or the process dies you can see exactly where.
Also measures how long one official-size training step takes on your CPU."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)   # if any step is stuck, print WHERE (file + line) every 60 s
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step(name, fn):
    print(f"[..] {name}", flush=True)
    t0 = time.time()
    try:
        r = fn()
        print(f"[OK] {name}  ({time.time()-t0:.1f}s)  {r if r is not None else ''}", flush=True)
        return r
    except Exception:
        print(f"[FAIL] {name}"); traceback.print_exc(); sys.exit(1)


print("python", sys.version.split()[0], "|", platform.platform(), "| cpu count", os.cpu_count())
print("project path:", Path(__file__).resolve().parent, "(non-ASCII characters in this path can cause trouble on Windows)")
for _m in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "networkx"):
    step(f"import {_m}", lambda m=_m: __import__(m).__version__)
torch = step("import torch", lambda: __import__("torch"))
print("     torch", torch.__version__, "| threads", torch.get_num_threads(), "| cuda", torch.cuda.is_available())
step("import torch_geometric + tiny GCN", lambda: __import__("torch_geometric.nn", fromlist=["GCNConv"]).GCNConv(4, 4)(torch.randn(5, 4), torch.tensor([[0, 1], [1, 2]])).shape)
from cmucl_nb import common as C
step("fetch official code", lambda: str(C.fetch_official_code()))
import random
import model as official_model  # noqa
from simple_tokenizer import SimpleTokenizer
tok = step("build tokenizer", lambda: SimpleTokenizer())
words = "graph node edge network learning model data anomaly detection neural text semantic citation paper method".split()
texts = [" ".join(random.choice(words) for _ in range(150)) for _ in range(200)]
step("tokenise 200 texts (should take ~0.1-1 s)", lambda: sum(len(tok.encode(t)) for t in texts))
cfg = C.make_config("Citeseer")
print("Benchmarking the OFFICIAL-size text encoder on a small micro-batch (safe: low memory) and extrapolating to a full batch.")
t = step("benchmark training step", lambda: C.benchmark_batch_times(cfg))
nb_full = -(-3186 // cfg["batch_size"])
print(f"\n=> one training step ~ {t['t_train']:.0f}s, one embedding batch ~ {t['t_embed']:.0f}s")
print(f"=> full Citeseer would need ~ {C.estimate_minutes(3186, cfg, t):.0f} min on this machine")
for b in (30, 45, 60):
    n, est = C.plan_nodes(cfg, 3186, t, b)
    print(f"   budget {b} min -> {n} nodes (est. {est:.0f} min)")
try:
    import psutil
    print(f"RAM total {psutil.virtual_memory().total/2**30:.1f} GB | this process now {psutil.Process().memory_info().rss/2**30:.1f} GB")
except ImportError:
    print("(pip install psutil to also print memory usage)")
print("\nAll good. Next:  python run_all.py")
