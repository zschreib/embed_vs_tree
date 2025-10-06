#!/usr/bin/env python3
"""
embed_vs_tree_run.py

Subcommands:
  - mantel     : Global concordance (Mantel Spearman) between tree patristic and embedding cosine distances
  - procrustes : PCoA of both distance matrices + Procrustes alignment with permutation test

Quick help:
  python embed_vs_tree_run.py mantel --help
  python embed_vs_tree_run.py procrustes --help


Examples:
  # Mantel (pretrained layer 36)
  python embed_vs_tree_run.py mantel \
    --emb-dir data/embeddings_pretrained \
    --tree data/polA.tree.nwk \
    --layer 36 \
    --permutations 9999 \
    --out-prefix results/polA_mantel_r36

  # Procrustes (PCoA dims = 10)
  python embed_vs_tree_run.py procrustes \
    --emb-dir data/embeddings_pretrained \
    --tree data/polA.tree.nwk \
    --layer 36 \
    --pcoa-dims 10 \
    --permutations 9999 \
    --out-prefix results/polA_proc_r36_k10

"""

from __future__ import annotations
import argparse
import os
import sys

def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import mantel_concordance as mantel_cmd
    import procrustes_alignment as procrustes_cmd

    parser = argparse.ArgumentParser(prog="embed_vs_tree")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    mantel_cmd.add_subparser(subparsers)
    procrustes_cmd.add_subparser(subparsers)

    args = parser.parse_args()
    if hasattr(args, "_handler"):
        args._handler(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
