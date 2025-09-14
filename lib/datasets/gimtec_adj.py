"""
GIMtec 邻接矩阵统一发现/构建/加载（与模型解耦）。

规则：
- 统一目录：data/adj/<DATASET>/
- 文件命名：<DATASET>_adj_<graph_tag>[ _<adj_model> ]_<YYYYMMDD_HHMMSS>.npy
- 优先级：按修改时间选最新的 *_adj_*.npy；若无，则尝试 data/<DATASET>/<DATASET>.npy；
  若仍无且 DATASET in {GIMtec, TEC}，构建 71×73 网格 8 邻接并保存。
"""
from __future__ import annotations

import os
import glob
import time
import numpy as np
from typing import Tuple

from lib.predifineGraph import constructGraph, get_adjacency_matrix


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _adj_store_dir(dataset: str) -> str:
    return os.path.join('..', 'data', 'adj', dataset)


def latest_adj_path(dataset: str) -> str | None:
    """返回 data/adj/<DATASET>/ 中最新的 *_adj_*.npy 路径。"""
    store = _adj_store_dir(dataset)
    if not os.path.isdir(store):
        return None
    cands = sorted(glob.glob(os.path.join(store, f"{dataset}_adj_*.npy")), key=os.path.getmtime)
    return cands[-1] if cands else None


def build_grid8_adj(H: int = 71, W: int = 73) -> np.ndarray:
    """71×73 网格 8 邻接（0/1）。"""
    return constructGraph(H, W).astype(np.float32)


def save_adj(dataset: str, A: np.ndarray, graph_tag: str = 'grid8', adj_model: str | None = None) -> str:
    store = _adj_store_dir(dataset)
    _ensure_dir(store)
    ts = time.strftime('%Y%m%d_%H%M%S')
    if adj_model:
        fname = f"{dataset}_adj_{graph_tag}_{adj_model}_{ts}.npy"
    else:
        fname = f"{dataset}_adj_{graph_tag}_{ts}.npy"
    out = os.path.join(store, fname)
    np.save(out, A)
    return out


def load_or_build_adj(dataset: str, num_nodes: int, graph_tag: str = 'grid8', adj_model: str | None = None) -> Tuple[np.ndarray, str]:
    """统一入口：返回 (A, path)。优先 data/adj/<DATASET>/ 最新；否则 data/<DATASET>/<DATASET>.npy；
    若都无且是 GIMtec/TEC，则构建 8 邻接并保存到统一目录。
    """
    # 最新的统一目录文件
    path_latest = latest_adj_path(dataset)
    if path_latest and os.path.exists(path_latest):
        return np.load(path_latest).astype(np.float32), path_latest

    # 兼容旧路径 data/<DATASET>/<DATASET>.npy
    legacy = os.path.join('..', 'data', dataset, f'{dataset}.npy')
    if os.path.exists(legacy):
        A, _ = get_adjacency_matrix(legacy, num_nodes)
        return A, legacy

    # 构建网格 8 邻接（适用于 GIMtec/TEC）
    if dataset.lower() in ['gimtec', 'tec']:
        A = build_grid8_adj(71, 73)
        out = save_adj(dataset, A, graph_tag=graph_tag or 'grid8', adj_model=adj_model)
        return A, out

    # 其他数据集：返回零矩阵占位（或抛错）
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    return A, ''

