#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from skbio import DistanceMatrix
from skbio.stats.distance import mantel

from utils import (
    cosine_distance_matrix,
    get_logger,
    intersect_ids,
    load_embedding,
    patristic_distance_matrix,
    read_tree_tip_labels,
    set_seed,
    save_matrix_tsv
)

LOG = get_logger("mantel")


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

def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "mantel",
        help="Global concordance via Mantel (patristic vs cosine embedding distances).",
        description="Compute Mantel (Spearman) correlation between patristic and embedding cosine distances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--emb-dir",
        required=True,
        help="Directory with ESM2 .pt files (needs mean_representations[36] and label).",
    )
    p.add_argument("--layer", type=int, default=36,
        help="ESM2 layer to use from mean_representations.",
    )
    p.add_argument(
        "--tree",
        required=True,
        help="Newick tree file (AA tree; tip labels must match embedding labels).",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=9999,
        help="Number of permutations for Mantel p-value.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--log1p-patristic",
        action="store_true",
        help="Apply log1p to patristic distances (sensitivity).",
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "If set, write tsv: <prefix>.ids.txt, <prefix>.embed_dist.tsv, "
            "<prefix>.patristic_dist.tsv"
        ),
    )
    p.set_defaults(_handler=run_from_namespace)
    return p


def run_from_namespace(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    LOG.info("Loading embeddings from %s", args.emb_dir)
    emb_ids_all, X_all = _load_dir_embeddings(args.emb_dir, layer=args.layer)

    LOG.info("Reading tree tips from %s", args.tree)
    tree_tips_all = read_tree_tip_labels(args.tree)

    # Debug: missing counts
    missing_in_tree = sorted(set(emb_ids_all) - set(tree_tips_all))
    missing_in_emb = sorted(set(tree_tips_all) - set(emb_ids_all))
    LOG.info("IDs present only in embeddings: %d", len(missing_in_tree))
    LOG.info("IDs present only in tree: %d", len(missing_in_emb))

    ids = intersect_ids(emb_ids_all, tree_tips_all)
    LOG.info("Intersection size: %d tips", len(ids))
    if len(ids) < 10:
        raise ValueError(f"Too few overlapping IDs for Mantel: n={len(ids)} (need = 10)")

    idx_map = {eid: i for i, eid in enumerate(emb_ids_all)}
    X = np.vstack([X_all[idx_map[k]] for k in ids])

    LOG.info("Computing cosine embedding distances")
    d_embed = cosine_distance_matrix(X)

    LOG.info("Computing patristic distances")
    d_pat = patristic_distance_matrix(args.tree, ids)
    if args.log1p_patristic:
        LOG.info("Applying log1p to patristic distances")
        d_pat = np.log1p(d_pat)

    # Debug: checks on matrices
    if not (np.isfinite(d_embed).all() and np.isfinite(d_pat).all()):
        raise ValueError("Non-finite values detected in distance matrices.")
    if not (np.allclose(d_embed, d_embed.T) and np.allclose(d_pat, d_pat.T)):
        raise ValueError("Distance matrices must be symmetric.")

    LOG.info("Running Mantel (Spearman, permutations=%d)", args.permutations)
    r, p, _ = mantel(
        DistanceMatrix(d_pat, ids=ids),
        DistanceMatrix(d_embed, ids=ids),
        method="spearman",
        permutations=args.permutations,
        strict=False,
        alternative="two-sided",
    )

    print("=== Global Concordance (Mantel, Spearman) ===")
    print(f"Tips (intersection): {len(ids)}")
    print(f"Permutations:        {args.permutations}")
    print(f"log1p(patristic):    {args.log1p_patristic}")
    print(f"r: {r:.4f}")
    print(f"p: {p:.2e}")

    if args.out_prefix:
        out_dir = os.path.dirname(args.out_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ids_path = f"{args.out_prefix}.ids.txt"
        with open(ids_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(ids) + "\n")
        save_matrix_tsv(f"{args.out_prefix}.embed_dist.tsv", d_embed, ids)
        save_matrix_tsv(f"{args.out_prefix}.patristic_dist.tsv", d_pat, ids)
        LOG.info(
            "Saved: %s, %s, %s",
            ids_path,
            f"{args.out_prefix}.embed_dist.tsv",
            f"{args.out_prefix}.patristic_dist.tsv",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global concordance between phylogeny and embeddings (Mantel).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log1p-patristic", action="store_true")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()
    run_from_namespace(args)


if __name__ == "__main__":
    main()
