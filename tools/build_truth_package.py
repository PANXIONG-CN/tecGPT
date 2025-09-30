import os
import json
import time
from pathlib import Path

import numpy as np


def build_gimtec_truth(dataset: str, lag: int, horizon: int, stride: int = 1, prefix_boundary: bool = True,
                        normalizer: str = 'tec01', out_root: str = 'coordinates/truth') -> str:
    """Build centralized truth package for GIMtec/TEC protocol.

    Returns path to the generated .npz.
    """
    from lib.Params_pretrain import parse_args as _parse
    from lib.dataloader import get_dataloader

    # Fabricate minimal args
    class _Args:
        pass
    args = _parse(device='cpu')
    args.dataset = dataset
    args.lag = lag
    args.horizon = horizon
    setattr(args, 'stride_horizon', False)
    setattr(args, 'prefix_boundary', prefix_boundary)
    # loaders
    _, _, test_loader, scaler_data, *_ = get_dataloader(args, normalizer=normalizer, single=True)
    ys = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            _, target, _ = batch if len(batch) == 3 else (batch[0], batch[1], None)
        else:
            _, target = batch
        ys.append(target[..., :1].cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    # inverse
    try:
        y_true = scaler_data.inverse_transform(y_true)
    except Exception:
        pass
    # digest
    import hashlib
    h = hashlib.md5(); h.update(y_true.tobytes()); digest = 'md5:' + h.hexdigest()

    # save
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_from = 'lastyear' if prefix_boundary else 'none'
    fname = f"gimtec__lag{lag}_hor{horizon}_stride{stride}_prefix-{prefix_from}.npz"
    fpath = out_dir / fname
    np.savez_compressed(str(fpath), y_true=y_true, digest=digest)
    print('Truth package saved to', fpath)
    return str(fpath)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='GIMtec')
    p.add_argument('--lag', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--prefix_boundary', type=lambda x: str(x).lower()=='true', default=True)
    args = p.parse_args()
    build_gimtec_truth(args.dataset, args.lag, args.horizon, args.stride, args.prefix_boundary)

