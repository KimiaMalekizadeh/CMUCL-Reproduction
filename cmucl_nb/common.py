"""Shared utilities: paths, config, official-code fetching, official dataset loading, node-subset selection, CPU timing.

Everything here is *infrastructure*. The model, tokenizer, neighbour sampler and the training / inference calls come
from the official repository (https://github.com/yimingxu24/CMUCL), cloned at runtime into ./third_party/CMUCL.

Only deviation from the official pipeline (needed because this is meant to run on a CPU): the number of NODES can be
reduced (see subset_snowball). Architecture and hyper-parameters are the official ones.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: duplicate OpenMP runtimes can kill the Jupyter kernel

import argparse
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"
DATA = ROOT / "data"
OFFICIAL = ROOT / "third_party" / "CMUCL"
OFFICIAL_URL = "https://github.com/yimingxu24/CMUCL.git"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1Suws6A-v0jBQpKeMphD0CEgxBjK-72AC"
for _p in (ART, FIG, DATA):
    _p.mkdir(parents=True, exist_ok=True)

# Official hyper-parameters for Citeseer (official README command + paper Appendix B + argparse defaults of main_train.py).
PAPER = dict(
    lr=2e-4, epoch_num=2, gamma=5e-3, batch_size=64, neigh_num=3, scale=0.07,
    gnn_layers=2, gnn_input=768, gnn_hid=128, gnn_output=128,
    context_length=128, embed_dim=128,
    transformer_heads=8, transformer_layers=12, transformer_width=512, vocab_size=49408,
    rounds=256, seed=1,
    edge_coef=10, optim=0, norm=0, reg=0, drop_prob=0.2, aggregation_times=2, gpu=0,
)
EXPECTED_CITESEER = dict(nodes=3186, edges=3432, anomalies=128)   # official README, Table 2 of the paper


def make_config(dataset="Citeseer"):
    cfg = dict(PAPER)
    if os.environ.get("CMUCL_BATCH_SIZE"):      # optional RAM fallback; NOT the official setting
        cfg["batch_size"] = int(os.environ["CMUCL_BATCH_SIZE"])
    cfg.update(dataset=dataset)
    return cfg


def save_config(cfg):
    (ART / "config.json").write_text(json.dumps(cfg, indent=2))


def load_config():
    return json.loads((ART / "config.json").read_text())


def as_args(cfg):
    return argparse.Namespace(**cfg)


def get_device(cfg=None):
    import torch
    return torch.device(f"cuda:{(cfg or {}).get('gpu', 0)}" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    """Same seeding as the official main_train.py."""
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


OFFICIAL_ZIP = "https://codeload.github.com/yimingxu24/CMUCL/zip/refs/heads/main"


def _download_official_zip(dest):
    """Fallback when git is unavailable: download the repo ZIP and put its FILES directly into dest."""
    import io
    import urllib.request
    import zipfile
    with urllib.request.urlopen(OFFICIAL_ZIP, timeout=120) as r:
        data = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            parts = info.filename.split("/", 1)            # strip the top folder "CMUCL-main/"
            if len(parts) < 2 or not parts[1] or info.is_dir():
                continue
            target = dest / parts[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))


def fetch_official_code():
    """Get the official repo into ./third_party/CMUCL (git clone -> ZIP download -> clear manual instructions)."""
    if not (OFFICIAL / "model.py").exists():
        OFFICIAL.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        try:
            r = subprocess.run(["git", "clone", "--depth", "1", OFFICIAL_URL, str(OFFICIAL)], capture_output=True, text=True)
            if r.returncode != 0:
                errors.append(f"git clone failed: {r.stderr.strip()[:400]}")
        except Exception as e:
            errors.append(f"git not usable: {e}")
        if not (OFFICIAL / "model.py").exists():
            try:
                _download_official_zip(OFFICIAL)
            except Exception as e:
                errors.append(f"ZIP download failed: {e}")
        if not (OFFICIAL / "model.py").exists():
            raise RuntimeError(
                "Could not obtain the official code automatically:\n  " + "\n  ".join(errors) +
                f"\nManual fix: download {OFFICIAL_URL[:-4]}/archive/refs/heads/main.zip in a browser, unzip it and copy the FILES "
                f"(model.py, data.py, simple_tokenizer.py, bpe_simple_vocab_16e6.txt.gz, ...) directly into:\n  {OFFICIAL}")
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    return OFFICIAL


# ----------------------------------------------------------------------------------------------------------------
# Official dataset (NOT generated here)
# ----------------------------------------------------------------------------------------------------------------
def find_dataset_files(name):
    """Look anywhere under ./data for <name>.mat and the csv next to it (case-insensitive)."""
    for mat in DATA.rglob("*.mat"):
        if mat.stem.lower() == name.lower():
            csvs = [c for c in mat.parent.glob("*.csv")]
            same = [c for c in csvs if c.stem.lower() == name.lower()]
            if same or len(csvs) == 1:
                return mat, (same or csvs)[0]
    return None, None


def require_dataset(name):
    mat, csv = find_dataset_files(name)
    if mat is None:
        raise FileNotFoundError(
            f"\nOfficial dataset '{name}' not found under {DATA}.\n"
            f"1) Open {DRIVE_FOLDER}\n"
            f"2) Download ONLY the '{name}' files (the folder also holds multi-GB datasets).\n"
            f"3) Put {name}.mat and {name}.csv anywhere under {DATA} (e.g. {DATA / 'processed' / name}).\n"
            f"No synthetic data is generated on purpose.")
    return mat, csv


def adj_to_edge_index(adj):
    """Symmetric (2, E) list = both directions of every edge (what DGL produced in the official script)."""
    import scipy.sparse as sp
    A = sp.coo_matrix(adj)
    A = ((A + A.T) > 0).tocoo()
    return np.vstack((A.row, A.col)).astype(np.int64)


def load_official_dataset(name):
    """Mirror of main_train.load_mat + the StandardScaler step of the official script. Returns the FULL dataset."""
    import pandas as pd
    import scipy.io as sio
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler
    mat_path, csv_path = require_dataset(name)
    mat = sio.loadmat(str(mat_path))
    text = pd.read_csv(csv_path)
    label = mat["Label"] if "Label" in mat else mat["gnd"]
    attr = mat["Attributes"] if "Attributes" in mat else mat["X"]
    net = mat["Network"] if "Network" in mat else mat["A"]
    feats = StandardScaler().fit_transform(sp.lil_matrix(attr).toarray()).astype(np.float32)
    A = sp.csr_matrix(net)
    n_self = int((A.diagonal() != 0).sum())
    off = A.copy(); off.setdiag(0); off.eliminate_zeros()
    return dict(name=name, n=feats.shape[0], nnz=int(A.nnz), n_selfloops=n_self,
                n_offdiag_edges=int(((off + off.T) > 0).nnz // 2), edge_index=adj_to_edge_index(sp.csr_matrix(net)), feats=feats,
                labels=np.squeeze(np.array(label)).astype(int), texts=text["text"].astype(str).to_numpy(),
                orig_index=np.arange(feats.shape[0]), types=None)


def subset_snowball(ds, n_keep, seed=0):
    """Pick ~n_keep nodes of the OFFICIAL graph, keeping real edges/texts/labels/features (nothing is generated).

    Multi-source BFS ("snowball") from random seeds so that the induced subgraph stays connected; the seed set contains
    anomalies at the dataset's anomaly rate so the subset is not left without anomalies. Nodes that end up without any
    edge are dropped (the official sampler ignores them anyway). Features were scaled on the full data first.
    """
    n = ds["n"]
    if n_keep >= n:
        return ds
    rng = np.random.RandomState(seed)
    ei, labels = ds["edge_index"], ds["labels"]
    nbrs = [[] for _ in range(n)]
    for u, v in ei.T:
        nbrs[int(u)].append(int(v))
    has_edge = np.array([len(a) > 0 for a in nbrs])
    n_seed = max(8, n_keep // 10)
    n_anom_seed = int(round(labels.mean() * n_seed))
    anom = rng.permutation(np.where((labels == 1) & has_edge)[0])
    norm = rng.permutation(np.where((labels == 0) & has_edge)[0])
    seeds = list(anom[:n_anom_seed]) + list(norm[:n_seed - n_anom_seed])
    visited, order, queue = set(), [], []
    for s in rng.permutation(seeds):
        visited.add(int(s)); order.append(int(s)); queue.append(int(s))
    spare = iter(norm[n_seed - n_anom_seed:])
    qi = 0
    while len(order) < n_keep:
        if qi >= len(queue):                                # component exhausted -> new random seed
            s = next(spare)
            if int(s) in visited: continue
            visited.add(int(s)); order.append(int(s)); queue.append(int(s)); continue
        u = queue[qi]; qi += 1
        for v in nbrs[u]:
            if v not in visited and len(order) < n_keep:
                visited.add(v); order.append(v); queue.append(v)
    keep = np.array(sorted(order))
    new_id = -np.ones(n, dtype=np.int64); new_id[keep] = np.arange(len(keep))
    m = (new_id[ei[0]] >= 0) & (new_id[ei[1]] >= 0)
    ei2 = np.vstack((new_id[ei[0][m]], new_id[ei[1][m]]))
    connected = np.bincount(ei2[0], minlength=len(keep)) > 0   # drop nodes left without any edge
    keep2 = keep[connected]
    new_id2 = -np.ones(n, dtype=np.int64); new_id2[keep2] = np.arange(len(keep2))
    m2 = (new_id2[ei[0]] >= 0) & (new_id2[ei[1]] >= 0)
    ei3 = np.vstack((new_id2[ei[0][m2]], new_id2[ei[1][m2]]))
    out = dict(ds)
    out.update(n=len(keep2), edge_index=ei3, feats=ds["feats"][keep2], labels=labels[keep2],
               texts=ds["texts"][keep2], orig_index=ds["orig_index"][keep2])
    return out


def save_dataset(ds):
    with open(ART / "dataset.pkl", "wb") as f:
        pickle.dump(ds, f)


def load_saved_dataset():
    with open(ART / "dataset.pkl", "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------------------------------------------
# CPU timing -> choose how many nodes fit into a time budget (architecture untouched)
# ----------------------------------------------------------------------------------------------------------------
def ram_available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def set_cpu_threads(frac=0.5):
    """Use only a fraction of the logical CPUs so the laptop stays responsive (no effect on results)."""
    import torch
    if torch.cuda.is_available():
        return
    n = max(1, int((os.cpu_count() or 2) * float(os.environ.get("CMUCL_THREAD_FRAC", frac))))
    torch.set_num_threads(n)


def enable_checkpointing(model):
    """Gradient checkpointing of the transformer blocks, applied from OUTSIDE (official model.py is untouched).
    The transformer has no dropout, so forward values and gradients are identical; activation memory drops roughly
    10x at the price of ~30% more compute. Without this, one official-size batch (256 sequences x 128 tokens through
    12 layers x width 512) can need ~10 GB of RAM and freeze a laptop."""
    from torch.utils.checkpoint import checkpoint
    for blk in model.transformer.resblocks:
        if getattr(blk, "_ckpt", False):
            continue
        orig = blk.forward
        blk.forward = (lambda x, _o=orig: checkpoint(_o, x, use_reentrant=False))
        blk._ckpt = True


def prepare_model(model):
    """CPU-friendliness only: thread limit + checkpointing (set CMUCL_CHECKPOINT=0 to disable)."""
    import torch
    if not torch.cuda.is_available():
        set_cpu_threads()
        if os.environ.get("CMUCL_CHECKPOINT", "1") != "0":
            enable_checkpointing(model)
    return model


def benchmark_batch_times(cfg, micro=16, verbose=True):
    """SAFE timing: runs a small micro-batch (default 16 sequences) of the official-size text encoder and extrapolates
    linearly to the real batch (batch_size x (1+neigh_num) sequences). It never builds the full-size batch in memory."""
    import torch
    fetch_official_code()
    from model import CMUCL
    args, dev = as_args(cfg), get_device(cfg)
    avail = ram_available_gb()
    if verbose and avail is not None:
        print(f"  available RAM: {avail:.1f} GB | torch threads: {torch.get_num_threads()}", flush=True)
    torch.manual_seed(0)
    m = CMUCL(args).to(dev)
    prepare_model(m)
    if verbose:
        print(f"  threads now: {torch.get_num_threads()} | checkpointing: {os.environ.get('CMUCL_CHECKPOINT', '1') != '0'}", flush=True)
    ids = torch.randint(1, 49000, (micro, cfg["context_length"]), device=dev)
    ids[:, -1] = 49407                                      # EOS at the last position = worst case (full-length texts)
    m.train(); ts = []
    for r in range(2):
        t0 = time.perf_counter()
        loss = m.encode_text(ids).pow(2).mean()
        m.optim.zero_grad(); loss.backward()
        dt = time.perf_counter() - t0
        if verbose: print(f"  micro-batch ({micro} seqs) fwd+bwd {r+1}/2: {dt:.1f}s" + ("  (warm-up, discarded)" if r == 0 else ""), flush=True)
        if r > 0: ts.append(dt)
    t0 = time.perf_counter(); m.optim.step(); t_opt = time.perf_counter() - t0
    seqs = cfg["batch_size"] * (1 + cfg["neigh_num"])
    m.eval()
    with torch.no_grad():
        t0 = time.perf_counter(); m.encode_text(ids); t_e = time.perf_counter() - t0
    return dict(t_train=float(np.mean(ts)) * seqs / micro + t_opt, t_embed=t_e * cfg["batch_size"] / micro)


def estimate_minutes(n_nodes, cfg, t):
    """Whole pipeline (notebooks 00-05) estimate in minutes: training + embedding + 4 probe/PCA forwards + inference rounds."""
    nb = int(np.ceil(n_nodes / cfg["batch_size"]))
    sec = nb * (cfg["epoch_num"] * t["t_train"] + t["t_embed"]) + (cfg["epoch_num"] + 4) * t["t_train"] / 3 + 60 + 0.05 * nb * cfg["rounds"] / 10
    return 1.15 * sec / 60                                  # 15% safety margin


def plan_nodes(cfg, n_full, t, budget_min):
    best = 0
    for nb in range(2, int(np.ceil(n_full / cfg["batch_size"])) + 1):
        if estimate_minutes(nb * cfg["batch_size"], cfg, t) <= budget_min:
            best = nb
    n = min(n_full, max(best, 2) * cfg["batch_size"])
    return n, estimate_minutes(n, cfg, t)


def savefig(name, fig=None):
    import matplotlib.pyplot as plt
    (fig or plt.gcf()).savefig(FIG / f"{name}.png", dpi=140, bbox_inches="tight")
