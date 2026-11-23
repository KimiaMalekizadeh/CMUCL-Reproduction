"""Transparent mirror of the official CMUCL.forward() / CMUCL.inference() maths (model.py).

The official methods return only a scalar loss / the final per-node score, so the intermediate objects the paper
talks about (the four cross-modal and two uni-modal similarity matrices, per-view losses and per-view anomaly
scores) are not observable. This module recomputes them with the SAME operations, step by step, and the notebooks
assert numerically that the totals equal the official outputs. Nothing here is used for training.

Paper view  <->  official variable
  nn (Eq.4, cross inner, node-node)      logits1 / sg_st : graph node  x text node
  cc (Eq.5, cross inner, ctx-ctx)        logits4 / tg_tt : graph ctx   x text ctx
  nc (Eq.8, cross inter, node-ctx)       logits2 / sg_tt : graph node  x text ctx
  cn (Eq.9, cross inter, ctx-node)       logits3 / tg_st : graph ctx   x text node
  gg (Eq.11, uni, graph node-ctx)        logits5 / sg_tg
  tt (Eq.11, uni, text  node-ctx)        logits6 / st_tt
"""
import torch
import torch.nn.functional as F

VIEW_INFO = {
    "nn": dict(code="sg_st", a="hG", b="hT", eq="Eq.4", kind="cross / inner-scale"),
    "cc": dict(code="tg_tt", a="hG_ctx", b="hT_ctx", eq="Eq.5", kind="cross / inner-scale"),
    "nc": dict(code="sg_tt", a="hG", b="hT_ctx", eq="Eq.8", kind="cross / inter-scale"),
    "cn": dict(code="tg_st", a="hG_ctx", b="hT", eq="Eq.9", kind="cross / inter-scale"),
    "gg": dict(code="sg_tg", a="hG", b="hG_ctx", eq="Eq.11 (graph)", kind="uni-modal"),
    "tt": dict(code="st_tt", a="hT", b="hT_ctx", eq="Eq.11 (text)", kind="uni-modal"),
}
CROSS = ("nn", "cc", "nc", "cn")
UNI = ("gg", "tt")


def l2n(z):
    return z / z.norm(dim=-1, keepdim=True)


def batch_features(model, x, edge_index, s_n, t_n, s_text, t_text, neigh_num):
    """First half of CMUCL.forward: node-level (h) and context-level (h~) embeddings, un-normalised."""
    s_graph, embs = model.encode_graph(s_n, x, edge_index)
    t_graph = embs[t_n.numpy().reshape(-1)]
    s_txt = model.encode_text(s_text)
    t_txt = model.encode_text(t_text)
    B = s_graph.shape[0]
    return dict(
        hG=s_graph, hT=s_txt,
        hG_ctx=t_graph.reshape(B, neigh_num, -1).mean(dim=1),   # Readout = mean of the sampled neighbours (Eq.3)
        hT_ctx=t_txt.reshape(B, neigh_num, -1).mean(dim=1),
    )


def stored_features(graph_feats, text_feats, s_n, t_n, neigh_num):
    """Same as above but from pre-computed embeddings (what CMUCL.inference does)."""
    s, t = s_n.numpy(), t_n.numpy().reshape(-1)
    B = len(s)
    return dict(
        hG=graph_feats[s], hT=text_feats[s],
        hG_ctx=graph_feats[t].reshape(B, neigh_num, -1).mean(dim=1),
        hT_ctx=text_feats[t].reshape(B, neigh_num, -1).mean(dim=1),
    )


def similarity_logits(feats, logit_scale):
    """The six B x B matrices: logit_scale * <a_i, b_j> on L2-normalised vectors (cosine / tau)."""
    z = {k: l2n(v) for k, v in feats.items()}
    return {k: logit_scale * z[v["a"]] @ z[v["b"]].t() for k, v in VIEW_INFO.items()}


def infonce_code(logits):
    """Symmetric InfoNCE exactly as implemented: (CE over rows + CE over columns) / 2."""
    y = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def infonce_paper(logits):
    """Single-direction InfoNCE, literally Eq.4 (rows only)."""
    y = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, y)


def view_losses(logits):
    return {k: infonce_code(l) for k, l in logits.items()}


def total_loss(losses, gamma):
    cross = sum(losses[k] for k in CROSS)          # Eq.10 = inner (Eq.6) + inter (Eq.7)
    uni = sum(losses[k] for k in UNI)              # Eq.11
    return cross + gamma * uni, cross, uni         # Eq.12


def view_score(logits, use_ce=True):
    """Per-node score of ONE view, mirroring CMUCL.inference: neg_row + neg_col - 4*pos + (CE_row + CE_col)/4."""
    B = logits.shape[0]
    y = torch.arange(B, device=logits.device)
    pos = torch.diag(logits)
    m = logits.clone()
    m.fill_diagonal_(0.0)
    neg_row, neg_col = m.mean(dim=1), m.T.mean(dim=1)   # note: divides by B (diagonal zeroed), not B-1 as in Eq.13
    ce = (F.cross_entropy(logits, y, reduction="none") + F.cross_entropy(logits.T, y, reduction="none")) / 4
    score = neg_row + neg_col - pos * 4 + (ce if use_ce else 0.0)
    return dict(score=score, pos=pos, neg_row=neg_row, neg_col=neg_col, ce=ce)


def anomaly_score(logits, gamma, use_ce=True, return_views=False):
    parts = {k: view_score(l, use_ce) for k, l in logits.items()}
    s = sum(parts[k]["score"] for k in CROSS) + gamma * sum(parts[k]["score"] for k in UNI)
    return (s, parts) if return_views else s


# ---------------------------------------------------------------------------------------------------------------
# Convenience helpers shared by notebooks 02-04 (probe batch = one fixed batch used to *look inside* the model)
# ---------------------------------------------------------------------------------------------------------------
def load_probe(art_dir, texts, tokenize, context_length, device):
    p = torch.load(art_dir / "probe.pt")
    s_n, t_n = p["s_n"], p["t_n"]
    s_text = tokenize(texts[s_n.numpy()].tolist(), context_length=context_length).to(device)
    t_text = tokenize(texts[t_n.numpy().reshape(-1)].tolist(), context_length=context_length).to(device)
    return s_n, t_n, s_text, t_text


def probe_logits(model, x, edge_index, probe, neigh_num):
    s_n, t_n, s_text, t_text = probe
    with torch.no_grad():
        feats = batch_features(model, x, edge_index, s_n, t_n, s_text, t_text, neigh_num)
        return feats, similarity_logits(feats, model.logit_scale.exp())


def plot_cos(ax, logits, scale, title, anomalous=None):
    """Heat-map of cosine similarities (= logits / scale); diagonal = positive pairs; red ticks = anomalous nodes."""
    import numpy as np
    M = (logits / scale).detach().cpu().numpy()
    im = ax.imshow(M, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if anomalous is not None and len(anomalous):
        ax.scatter(anomalous, np.full(len(anomalous), -1.5), c="red", s=8, marker="v", clip_on=False)
        ax.scatter(np.full(len(anomalous), -1.5), anomalous, c="red", s=8, marker=">", clip_on=False)
    return im
