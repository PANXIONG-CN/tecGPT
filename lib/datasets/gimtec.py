"""
GIMtec 数据集插件加载器

用途：为非 CSA/OptimizedCSA 模型（例如 GWN）提供通用的 [T, N] 序列，
遵循现有 load_st_dataset 插件约定：expose load(args) -> np.ndarray。
最小侵入：不改变通用管线，只新增该插件文件。
"""
from __future__ import annotations

import os
import numpy as np


def _load_year(base_dir: str, year: int) -> np.ndarray:
    path = os.path.join(base_dir, f"TEC_{year}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    arr = arr.astype(np.float32)
    # 原始数据单位通常放大 10 倍，保持与 vendor 一致做一个/10 的缩放
    arr = arr / 10.0
    # 兼容可能的 [D, M, H, W] 形态 -> [T, H, W]
    if arr.ndim == 4:
        d0, d1, h, w = arr.shape
        arr = arr.reshape(d0 * d1, h, w)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")
    return arr


def load(args) -> np.ndarray:
    """
    返回 [T, N] 时间序列（单通道）。
    - 拼接 2009-2022 年，按时间顺序串联。
    - 下游通用管线会自动拼接 day/week 特征，并进行窗口化与归一化。
    """
    base_dir = os.path.join('..', 'data', 'GIMtec')
    years = list(range(2009, 2023))
    parts = [_load_year(base_dir, y) for y in years if os.path.exists(os.path.join(base_dir, f"TEC_{y}.npy"))]
    if not parts:
        raise RuntimeError(f"No TEC_*.npy found under {base_dir}")
    data = np.concatenate(parts, axis=0)  # [T, H, W]
    T, H, W = data.shape
    data = data.reshape(T, H * W)  # [T, N]
    # 为下游提供一些默认元信息（若未在 conf 中显式设置）
    if not hasattr(args, 'interval'):
        args.interval = 120  # minutes per step（GIMtec 每帧≈2小时）
    if not hasattr(args, 'week_day'):
        args.week_day = 7
    if not hasattr(args, 'week_start'):
        args.week_start = 4
    if not hasattr(args, 'holiday_list'):
        args.holiday_list = []
    return data
