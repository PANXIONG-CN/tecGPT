import os
import glob
import numpy as np

def _stack_years(base_dir: str):
    # Prefer yearly shards TEC_YYYY.npy
    yearly = sorted(glob.glob(os.path.join(base_dir, 'TEC_*.npy')))
    arrays = []
    for fp in yearly:
        arr = np.load(fp)
        # normalize possible shapes
        if arr.ndim == 4:
            # common forms: [T,H,W,C] or [D,M,H,W]; unify to [T,H,W]
            if arr.shape[-1] in (1, 2, 3, 4):
                # treat last dim as channel
                arr = arr[..., 0]
            else:
                # fold first two dims as time
                d0, d1, h, w = arr.shape
                arr = arr.reshape(d0*d1, h, w)
        arrays.append(arr)
    if arrays:
        tec = np.concatenate(arrays, axis=0)
        return tec
    # fallback single files
    for name in ['TEC.npy', 'TEC.npz']:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            if name.endswith('.npy'):
                return np.load(p)
            else:
                tmp = np.load(p)
                return tmp['data'] if 'data' in tmp.files else tmp[list(tmp.files)[0]]
    raise FileNotFoundError('No TEC files found in {}'.format(base_dir))

def load(args):
    """
    Returns base data [T, N] or [T, N, 1] for GIMtec (TEC) dataset.
    Data dir: ../data/GIMtec/
    """
    base_dir = os.path.join('..', 'data', 'GIMtec')
    # If debug, only load the first available yearly file to speed up
    yearly = sorted(glob.glob(os.path.join(base_dir, 'TEC_*.npy')))
    if getattr(args, 'debug', False) and yearly:
        arr = np.load(yearly[0])
        if arr.ndim == 4:
            if arr.shape[-1] in (1, 2, 3, 4):
                arr = arr[..., 0]
            else:
                d0, d1, h, w = arr.shape
                arr = arr.reshape(d0*d1, h, w)
        tec = arr
    else:
        tec = _stack_years(base_dir)
    # Optional scale (common TECU -> /10)
    tec = tec.astype(np.float32) / 10.0
    # debug slice to speed smoke runs
    # Keep small time-span for smoke runs to avoid OOM since this repo preloads
    # all X/Y tensors to GPU inside the dataloader. If debug is True, or if the
    # run looks like a short ori evaluation (epochs <= 10), cap to 200 steps.
    if getattr(args, 'debug', False) or (getattr(args, 'mode', 'ori') == 'ori' and getattr(args, 'epochs', 9999) <= 10):
        max_steps = min(200, tec.shape[0])
        tec = tec[:max_steps]
    # Ensure val/test have enough span to build windows
    T = tec.shape[0]
    lag = getattr(args, 'lag', 12)
    horizon = getattr(args, 'horizon', 12)
    need = max(0, lag + horizon)
    if T > 0 and need > 0:
        min_ratio = min(0.4, max(0.05, (need + 1) / float(T)))
        if getattr(args, 'test_ratio', None) is not None and args.test_ratio < min_ratio:
            args.test_ratio = min_ratio
        if getattr(args, 'val_ratio', None) is not None and args.val_ratio < min_ratio:
            args.val_ratio = min_ratio
    # unify to [T, N, 1]
    if tec.ndim == 3:
        T, H, W = tec.shape
        args.height = getattr(args, 'height', H)
        args.width = getattr(args, 'width', W)
        data = tec.reshape(T, H*W, 1)
    elif tec.ndim == 2:
        T, N = tec.shape
        data = tec.reshape(T, N, 1)
    elif tec.ndim == 4:
        T, H, W, C = tec.shape
        args.height = getattr(args, 'height', H)
        args.width = getattr(args, 'width', W)
        data = tec[..., 0].reshape(T, H*W, 1)
    else:
        raise ValueError('Unexpected TEC array dims: {}'.format(tec.shape))
    # default temporal meta for downstream time_add
    args.interval = getattr(args, 'interval', 60)
    args.week_day = getattr(args, 'week_day', 7)
    args.week_start = getattr(args, 'week_start', 5)
    return data
