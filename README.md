# CMUCL — a step-by-step notebook reproduction (CPU-friendly)

> ## ⚠️ This is NOT my paper and NOT my method
> This repository is an **independent study / reproduction aid** of the paper
> **"Text-Attributed Graph Anomaly Detection via Multi-Scale Cross- and Uni-Modal Contrastive Learning" (CMUCL)**
> by Yiming Xu, Xu Hua, Zhen Peng, Bin Shi, Jiarun Chen, Xingbo Fu, Song Wang and Bo Dong — [arXiv:2508.00513](https://arxiv.org/abs/2508.00513).
> All credit for the method, the official code and the datasets belongs to the original authors.
> Official code: https://github.com/yimingxu24/CMUCL
>
> I wrote these notebooks to **understand the paper**, by running each part of the pipeline and looking at its output.

## What is in here
* The **official code is not copied** into this repo. It is cloned at runtime into `third_party/CMUCL` (git-ignored) and used unchanged
  (model, tokenizer, neighbour sampler, `forward()` for training, `inference()` for scoring).
* `cmucl_nb/views.py` is **my own code**: a transparent re-computation of the official loss / anomaly score that exposes the
  intermediate objects (six similarity matrices, per-view losses and scores). The notebooks `assert` that it matches the official
  output numerically (e.g. total loss 18.2006 vs official 18.2006).
* The **official Citeseer data is not included**. Download it from the Google-Drive link in the official README and put
  `Citeseer.mat` and `Citeseer.csv` under `data/processed/Citeseer/`.

| notebook | paper section | what you see |
|---|---|---|
| 00_setup_and_data | §3, App. A | TAG statistics, degree histogram, texts, ego-graph, CPU time budget |
| 01_text_and_graph_encoders | §4.1, Eq.1–3 | tokenisation, transformer & GCN step by step, context readout |
| 02_cross_modal_contrast | §4.2, Eq.4–10 | four cross-modal similarity matrices, losses |
| 03_unimodal_and_objective | §4.3–4.4, Eq.11–12 | two uni-modal matrices, total loss, check against official `forward` |
| 04_joint_training | §4.4 | official training loop, loss curves, before/after matrices |
| 05_anomaly_score_and_evaluation | §4.5, Eq.13–15 | per-view scores, mean+std estimator, AUC/AP, ablations |
| 06_run_all | — | runs 00→05 |

## What differs from the paper's setting (read before comparing numbers)
* **Only the number of nodes is reduced** (CPU-only laptop): a real subgraph of official Citeseer, 576 of 3186 nodes
  (all rows are exact slices of the official files). Architecture and hyper-parameters are the official Citeseer ones
  (12-layer/512 transformer, batch 64, R=256, γ=0.005, 2 epochs, lr 2e-4).
* **Memory-only changes for CPU** (outside the official code): half of the CPU threads, and gradient checkpointing of the transformer blocks.
* **Single seed** (the paper averages 5) and **only 23 anomalies** in the subset → AUC/AP are very noisy and are **not comparable to the paper's table**.

## Results (576-node subset, seed 1)
| | value |
|---|---|
| Training loss, first → last batch | 18.11 → 13.83 (18 steps) |
| Cross-modal view losses on a fixed probe batch, init → epoch 2 | ≈4.5 → ≈2.4 (chance level ln 64 ≈ 4.16) |


### Observations about the official Citeseer files (my own analysis, verified on the subset)
* In `Citeseer.mat` every node has a **self-loop** and real edges are rare: in the 576-node subset 108 nodes have a real neighbour
  (22 of the 23 anomalies, 86 of 553 normal nodes). In a probe batch ~91 % of the sampled "neighbours" were the node itself.
* Therefore the neighbourhood context is, for most nodes, the node itself, and a trivial rule "node has a real neighbour" already reaches
  AUC ≈ 90 on this subset. Keep this in mind when reading any AUC on this data.

## Where the official code differs from the paper text (verified by reading the code)
| topic | paper | official code |
|---|---|---|
| context embedding (Eq.3) | mean over all neighbours | mean over `neigh_num=3` neighbours sampled with replacement |
| InfoNCE (Eq.4) | one direction, fixed τ=0.07 | symmetric (rows+cols)/2, τ **learnable** (init 1/0.07) |
| std in Eq.15 | divide by R | `torch.std` (divide by R−1) |
| negative term in Eq.13 | mean over N−1 negatives | mean over N (diagonal zeroed); score = neg_row + neg_col − 4·pos + CE/4 |
| scalability claim | "does not require reading the entire graph" | GCN runs over the full graph in every batch |
| nodes without edges | — | never enter `DataHelper`, keep score 0, still counted in AUC |

## Run
```
python -m pip install -r requirements.txt
python diagnose.py                     # checks imports and times the official-size text encoder on your CPU
python run_all.py                      # notebooks 00 -> 05; picks the number of nodes that fits ~45 min
python run_all.py --only 05            # re-run a single notebook (00 deletes earlier outputs, so don't re-run it by accident)
```
Environment overrides: `CMUCL_N_NODES`, `CMUCL_BUDGET_MIN`, `CMUCL_MAX_BATCHES`, `CMUCL_BATCH_SIZE` (RAM fallback, **not** the official setting).

## Licence note
The official repository snapshot I inspected had **no LICENSE file**, which is why its code and data are linked, not copied.
Please check the upstream licence before redistributing any of it.

## Citation
```
@article{xu2025cmucl,
  title   = {Text-Attributed Graph Anomaly Detection via Multi-Scale Cross- and Uni-Modal Contrastive Learning},
  author  = {Xu, Yiming and Hua, Xu and Peng, Zhen and Shi, Bin and Chen, Jiarun and Fu, Xingbo and Wang, Song and Dong, Bo},
  journal = {arXiv preprint arXiv:2508.00513},
  year    = {2025}
}
```
