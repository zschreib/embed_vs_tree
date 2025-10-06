#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from scipy.spatial import procrustes

from utils import (
    cosine_distance_matrix,
    get_logger,
    intersect_ids,
    load_embedding,
    patristic_distance_matrix,
    read_tree_tip_labels,
    set_seed,
)

LOG = get_logger("procrustes")


def _load_dir_embeddings(emb_dir: str, layer: int) -> Tuple[List[str], np.ndarray]:
    files = sorted(f for f in os.listdir(emb_dir) if f.lower().endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {emb_dir}")
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    seen: Dict[str, int] = {}
    for i, fname in enumerate(files):
        emb, eid = load_embedding(os.path.join(emb_dir, fname), layer=layer)
        if eid in seen:
            raise ValueError(f"Duplicate embedding ID: {eid}")
        seen[eid] = i
        ids.append(eid)
        vecs.append(emb)
    X = np.vstack(vecs)
    LOG.info("Loaded %d embeddings with dim=%d", X.shape[0], X.shape[1])
    return ids, X


def _save_coords_tsv(path: str, ids: List[str], coords: np.ndarray) -> None:
    k = coords.shape[1]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(["id"] + [f"PC{i+1}" for i in range(k)]) + "\n")
        for name, row in zip(ids, coords):
            fh.write("\t".join([name] + [f"{v:.8g}" for v in row]) + "\n")


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "procrustes",
        help="PCoA ordinations for tree and embeddings, then Procrustes alignment with permutation p-value.",
        description="Compare geometry of tree patristic vs embedding cosine distances via PCoA + Procrustes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--emb-dir", required=True, help="Directory with ESM2 .pt files.")
    p.add_argument("--tree", required=True, help="Newick tree file (AA tree).")
    p.add_argument("--layer", type=int, default=36, help="ESM2 layer (mean_representations[layer]).")
    p.add_argument("--pcoa-dims", type=int, default=10, help="Number of PCoA axes to keep.")
    p.add_argument("--permutations", type=int, default=9999, help="Permutation count for Procrustes p-value.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--log1p-patristic", action="store_true", help="Apply log1p to patristic distances.")
    p.add_argument("--out-prefix", default=None, help="If set, save aligned coordinate TSVs.")
    p.set_defaults(_handler=run_from_namespace)
    return p


def run_from_namespace(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    LOG.info("Loading embeddings from %s (layer=%d)", args.emb_dir, args.layer)
    emb_ids_all, X_all = _load_dir_embeddings(args.emb_dir, layer=args.layer)

    LOG.info("Reading tree tips from %s", args.tree)
    tree_tips_all = read_tree_tip_labels(args.tree)

    ids = intersect_ids(emb_ids_all, tree_tips_all)
    LOG.info("Intersection size: %d tips", len(ids))
    if len(ids) < 10:
        raise ValueError(f"Too few overlapping IDs for Procrustes/PCoA: n={len(ids)}")

    idx_map = {eid: i for i, eid in enumerate(emb_ids_all)}
    X = np.vstack([X_all[idx_map[k]] for k in ids])

    LOG.info("Computing cosine embedding distances")
    d_embed = cosine_distance_matrix(X)

    LOG.info("Computing patristic distances")
    d_pat = patristic_distance_matrix(args.tree, ids)
    if args.log1p_patristic:
        LOG.info("Applying log1p to patristic distances")
        d_pat = np.log1p(d_pat)

    LOG.info("Running PCoA (dims=%d)", args.pcoa_dims)
    pcoa_tree = pcoa(DistanceMatrix(d_pat, ids=ids))
    pcoa_embd = pcoa(DistanceMatrix(d_embed, ids=ids))

    k = args.pcoa_dims
    A = pcoa_tree.samples.iloc[:, :k].to_numpy()
    B = pcoa_embd.samples.iloc[:, :k].to_numpy()

    LOG.info("Running Procrustes alignment (SciPy)")
    Ahat, Bhat, m2_obs = procrustes(A, B)  
    R = float(np.sqrt(max(0.0, 1.0 - m2_obs)))

    LOG.info("Permutation test (n=%d)", args.permutations)
    rng = np.random.default_rng(args.seed)
    count = 0
    for _ in range(args.permutations):
        perm = rng.permutation(B.shape[0])
        _, _, m2 = procrustes(A, B[perm, :])
        if m2 <= m2_obs:
            count += 1
    pval = (count + 1) / (args.permutations + 1)

    print("=== PCoA + Procrustes ===")
    print(f"Tips (intersection): {len(ids)}")
    print(f"PCoA dims (k):      {k}")
    print(f"Procrustes m^2:     {m2_obs:.4f}   (lower is better)")
    print(f"Procrustes R:       {R:.4f}        (R = sqrt(1 - m^2))")
    print(f"Permutations:       {args.permutations}")
    print(f"p-value:            {pval:.4e}")
    var_tree = float(pcoa_tree.proportion_explained.iloc[:k].sum())
    var_embd = float(pcoa_embd.proportion_explained.iloc[:k].sum())
    print(f"Tree PCoA variance (k): {var_tree:.3f}")
    print(f"Emb PCoA variance (k):  {var_embd:.3f}")

    if args.out_prefix:
        out_dir = os.path.dirname(args.out_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _save_coords_tsv(f"{args.out_prefix}.tree_pcoa_aligned.tsv", ids, Ahat)
        _save_coords_tsv(f"{args.out_prefix}.embed_pcoa_aligned.tsv", ids, Bhat)
        LOG.info("Saved aligned coordinates: %s, %s",
                 f"{args.out_prefix}.tree_pcoa_aligned.tsv",
                 f"{args.out_prefix}.embed_pcoa_aligned.tsv")
