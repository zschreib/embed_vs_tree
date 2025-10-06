from __future__ import annotations

import gzip
import logging
import os
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from skbio import DistanceMatrix
from skbio.tree import TreeNode

# ---------------------------
# Logging and set seed
# ---------------------------

# Logging information
def get_logger(name: str = "embed_vs_tree") -> logging.Logger:
    """Return a configured logger (stdout, INFO)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger

# Global seed
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------
# Embedding utilities
# ---------------------------

# ESM2-3B pre-trained (esm2_t36_3B_UR50D) layer 36 = pooled embedding final layer (default)
def load_embedding(file_path: str, layer: int = 36):

    data = torch.load(file_path, map_location="cpu")
    if "mean_representations" not in data:
        raise KeyError(f"'mean_representations' missing in {os.path.basename(file_path)}")

    if layer not in data["mean_representations"]:
        raise KeyError(f"Layer {layer} missing in {os.path.basename(file_path)}")

    if "label" not in data:
        raise KeyError(f"'label' missing in {os.path.basename(file_path)}")

    emb = data["mean_representations"][layer]
    if hasattr(emb, "numpy"):
        emb = emb.numpy()
    emb = np.asarray(emb, dtype=np.float32).ravel()
    if emb.ndim != 1:
        raise ValueError(f"Embedding must be 1D after ravel; got {emb.shape} in {os.path.basename(file_path)}")

    return emb, data["label"]

def cosine_distance_matrix(X: np.ndarray, *, dtype: Optional[np.dtype] = np.float32) -> np.ndarray:
    return pairwise_distances(X, metric="cosine").astype(dtype, copy=False)

# ---------------------------
# Tree utilities
# ---------------------------

def read_tree_tip_labels(tree_path: str) -> List[str]:
    tree = TreeNode.read(tree_path, format="newick", convert_underscores=False)
    return [t.name for t in tree.tips()]

def patristic_distance_matrix(tree_path: str, tip_order: List[str]) -> np.ndarray:

    tree = TreeNode.read(tree_path, format="newick", convert_underscores=False)
    tree_tips = {t.name for t in tree.tips()}
    missing = [t for t in tip_order if t not in tree_tips]
    if missing:
        raise ValueError(
            f"{len(missing)} IDs not found in tree tips (showing up to 10): "
            + ", ".join(missing[:10])
        )
    dm = tree.tip_tip_distances(tip_order)
    return dm.to_data_frame().to_numpy(dtype=np.float64)

# ---------------------------
# ID alignment and I/O
# ---------------------------

def intersect_ids(emb_ids: Iterable[str], tree_tip_ids: Iterable[str]) -> List[str]:
    inter = sorted(set(emb_ids) & set(tree_tip_ids))
    if not inter:
        raise ValueError("No overlapping IDs between embeddings and tree tips.")
    return inter
    
def save_matrix_tsv(path: str, matrix: np.ndarray, ids: List[str]) -> None:
    n = len(ids)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(["id"] + ids) + "\n")
        for i in range(n):
            row = "\t".join([ids[i]] + [f"{matrix[i, j]:.8g}" for j in range(n)])
            fh.write(row + "\n")
