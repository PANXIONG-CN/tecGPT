import os
import io
import time
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


# --------------------------- helpers ---------------------------

DATASET_SLUGS = {
    'gimtec': 'gim',
    'tec': 'gim',
    'china_hires': 'china_hires',
    'metr_la': 'metrla',
    'nyc_taxi': 'nyc_taxi',
    'nyc_bike': 'nyc_bike',
    'pems08': 'pems08',
}


def _now_utc_ts() -> str:
    # UTC ISO-like without colons: %Y%m%dT%H%M%SZ
    import datetime as _dt
    return _dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def dataset_slug(name: str) -> str:
    if not name:
        return 'unknown'
    key = name.strip().lower()
    return DATASET_SLUGS.get(key, key)


def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters()) if model is not None else 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0
    return int(total), int(trainable)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # shapes: [B,T,N,(D)] or broadcastable; compute over all elements
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    # filter NaNs
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float('nan')
    yt = yt[m]
    yp = yp[m]
    ss_res = np.sum((yt - yp) ** 2)
    yt_mean = np.mean(yt)
    ss_tot = np.sum((yt - yt_mean) ** 2)
    if ss_tot <= 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float('nan')
    yt = yt[m]
    yp = yp[m]
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = (np.linalg.norm(yt) * np.linalg.norm(yp))
    if denom == 0:
        return float('nan')
    return float(np.dot(yt, yp) / denom)


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float('nan'), float('nan')
    diff = (yp - yt)[m]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _compute_digest(arr: np.ndarray) -> str:
    # simple md5 of raw bytes for truth package
    h = hashlib.md5()
    h.update(arr.tobytes())
    return 'md5:' + h.hexdigest()


# ------------------------ main entrypoints ---------------------

def prepare_outdir(args, run_tag: Optional[str] = None) -> Path:
    dataset_name = getattr(args, 'dataset', 'DATASET')
    ds_slug = dataset_slug(dataset_name)
    model_slug_val = getattr(args, 'model', 'model').lower()
    seed = int(getattr(args, 'seed', 0))
    tag = run_tag or _now_utc_ts()
    base = Path('outputs') / ds_slug / model_slug_val / f'seed_{seed}' / tag
    ensure_dir(base)
    # propagate back for the rest of the pipeline
    args.log_dir = str(base)
    # record ts_utc + run_tag for manifest use
    try:
        if not getattr(args, 'ts_utc', None):
            args.ts_utc = tag.split('-', 1)[0]
        args.run_tag = tag
    except Exception:
        pass
    # best-effort dataset-level symlink (e.g., outputs/GIMtec -> outputs/gim)
    try:
        ds_upper = str(dataset_name)
        if ds_upper and ds_upper.lower() != ds_slug:
            legacy = Path('outputs') / ds_upper
            target = Path('outputs') / ds_slug
            if not legacy.exists():
                legacy.symlink_to(target, target_is_directory=True)
    except Exception:
        pass
    return base


def _build_truth_if_needed(args, out_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """For GIMtec/TEC only: build a centralized truth package if missing and return (ref, digest).
    Other datasets return (None, None).
    """
    try:
        ds_key = str(getattr(args, 'dataset', '')).lower()
        if ds_key not in ['gimtec', 'tec']:
            return None, None
        # construct protocol key
        lag = int(getattr(args, 'lag', 12))
        horizon = int(getattr(args, 'horizon', 12))
        stride = 1 if bool(getattr(args, 'stride_horizon', False)) else 1
        prefix_from = 'lastyear' if bool(getattr(args, 'prefix_boundary', True)) else 'none'
        truth_root = Path('coordinates') / 'truth'
        ensure_dir(truth_root)
        fname = f"gimtec__lag{lag}_hor{horizon}_stride{stride}_prefix-{prefix_from}.npz"
        fpath = truth_root / fname
        if not fpath.exists():
            # auto-build once
            from lib.dataloader import get_dataloader
            # Build loaders with the same args but ensure GIM protocol
            train_loader, val_loader, test_loader, scaler_data, *_ = get_dataloader(args, normalizer=getattr(args, 'normalizer', 'tec01'), single=True)
            ys = []
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    _, target, _ = batch if len(batch) == 3 else (batch[0], batch[1], None)
                else:
                    _, target = batch
                ys.append(target[..., :1].cpu().numpy())
            if len(ys) == 0:
                return None, None
            y_true = np.concatenate(ys, axis=0)
            # inverse transform to TECU domain
            try:
                y_true = scaler_data.inverse_transform(y_true)
            except Exception:
                pass
            digest = _compute_digest(y_true)
            np.savez_compressed(str(fpath), y_true=y_true, digest=digest)
        # read digest
        z = np.load(str(fpath))
        digest = str(z.get('digest')) if 'digest' in z.files else None
        return str(fpath), digest
    except Exception:
        return None, None


def _collect_env() -> Dict:
    env = {}
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda'] = getattr(torch.version, 'cuda', None)
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['gpu_name'] = torch.cuda.get_device_name(0)
            env['capability'] = torch.cuda.get_device_capability(0)
    except Exception:
        pass
    try:
        import numpy
        env['numpy'] = numpy.__version__
    except Exception:
        pass
    try:
        import sys
        env['python'] = sys.version
    except Exception:
        pass
    return env


def _save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_stepwise_csv(path: Path, rows: List[Dict[str, object]]):
    # columns: seed,dataset,model_slug,horizon,step,rmse
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['seed', 'dataset', 'model_slug', 'horizon', 'step', 'rmse']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def _wandb_log_artifact(run, file_path: Path, atype: str = 'file', aliases: Optional[List[str]] = None):
    if wandb is None or run is None:
        return
    if not file_path.exists():
        return
    try:
        art = wandb.Artifact(name=f"{file_path.stem}-{run.id}", type=atype)
        art.add_file(str(file_path))
        run.log_artifact(art, aliases=aliases or ['latest'])
    except Exception:
        pass


def post_run_collect_and_upload(run, args, model, out_dir: Path) -> None:
    """Unify final result saving and small-file uploading.

    - Create metrics.json, stepwise_rmse.csv, compute_cost.json, manifest.json if missing.
    - Upload small files to W&B (if online).
    - Prepare OSS hints in W&B summary (actual OSS upload handled by shell script).
    """
    out_dir = Path(out_dir)
    # Truth package (GIMtec only, auto-build if missing)
    truth_ref, truth_digest = _build_truth_if_needed(args, out_dir)

    # Metrics and arrays paths (may be produced already by Trainer)
    preds_files = list(out_dir.glob("*_test_preds.npy"))
    preds_path = preds_files[0] if preds_files else (out_dir / 'predictions.npy')

    # If predictions.npy missing but *_test_preds.npy exists, copy/link
    try:
        if preds_files and not preds_path.exists():
            import shutil
            shutil.copy2(str(preds_files[0]), str(preds_path))
    except Exception:
        pass

    # Prepare metrics from arrays if available
    metrics_path = out_dir / 'metrics.json'
    stepwise_path = out_dir / 'stepwise_rmse.csv'
    if preds_path.exists():
        try:
            yp = np.load(str(preds_path))
            # Try load y_true from truth package or reconstruct from saved arrays if present
            ytrue = None
            if truth_ref and Path(truth_ref).exists():
                z = np.load(str(truth_ref))
                ytrue = z['y_true'] if 'y_true' in z.files else None
            # fallback: search *_true.npy
            if ytrue is None:
                t_candidates = list(out_dir.glob("*_test_true.npy"))
                if t_candidates:
                    ytrue = np.load(str(t_candidates[0]))
            if ytrue is not None and ytrue.shape[:3] == yp.shape[:3]:
                # overall
                ss_res = float(np.nansum((yp - ytrue) ** 2))
                mu = float(np.nanmean(ytrue))
                ss_tot = float(np.nansum((ytrue - mu) ** 2))
                rrse = float('nan') if ss_tot <= 0 else float(np.sqrt(ss_res / ss_tot))
                overall_mae, overall_rmse = mae_rmse(ytrue, yp)
                overall_r2 = float('nan') if ss_tot <= 0 else float(1.0 - (ss_res / ss_tot))
                overall_corr = corr_score(ytrue, yp)
                # per-horizon rows for unified CSV (seed,dataset,model_slug,horizon,step,rmse)
                rows = []
                H = yp.shape[1]
                ds_slug = dataset_slug(getattr(args, 'dataset', ''))
                model_slug = str(getattr(args, 'model', 'model')).lower()
                seed = int(getattr(args, 'seed', 0))
                for h in range(H):
                    # stepwise RMSE per horizon
                    _, rmse_h = mae_rmse(ytrue[:, h, ...], yp[:, h, ...])
                    rows.append({'seed': seed, 'dataset': ds_slug, 'model_slug': model_slug,
                                 'horizon': h + 1, 'step': h + 1, 'rmse': round(rmse_h, 6) if rmse_h == rmse_h else None})
                _save_stepwise_csv(stepwise_path, rows)
                _save_json(metrics_path, {
                    'overall': {
                        'MAE': overall_mae,
                        'RMSE': overall_rmse,
                        'R2': overall_r2,
                        'CORR': overall_corr,
                        'RRSE': rrse,
                    }
                })
        except Exception:
            pass

    # compute_cost.json
    cc_path = out_dir / 'compute_cost.json'
    if not cc_path.exists():
        cc = {
            'start_ts': getattr(args, '_run_start_ts', None),
            'end_ts': _now_ts(),
            'total_seconds': None,
            'epochs': int(getattr(args, 'epochs', 0)),
            'steps': int(getattr(args, 'batch_seen', 0)) if hasattr(args, 'batch_seen') else None,
            'throughput_samples_per_sec': None,
            'gpu_name': None,
            'max_memory_train_bytes': None,
            'max_memory_infer_bytes': None,
            'amp_mode': 'bf16' if bool(getattr(args, 'amp', False)) else 'none',
            'total_params': None,
            'trainable_params': None,
        }
        # params
        try:
            total, trainable = count_params(model)
            cc['total_params'] = total
            cc['trainable_params'] = trainable
        except Exception:
            pass
        # env
        env = _collect_env()
        if 'gpu_name' in env:
            cc['gpu_name'] = env['gpu_name']
        # timing estimates if recorded on args
        if hasattr(args, '_train_start_time') and hasattr(args, '_train_end_time'):
            cc['total_seconds'] = float(args._train_end_time - args._train_start_time)
        # save
        _save_json(cc_path, cc)

    # manifest.json
    man_path = out_dir / 'manifest.json'
    if not man_path.exists():
        man = {
            'dataset': getattr(args, 'dataset', ''),
            'dataset_slug': dataset_slug(getattr(args, 'dataset', '')),
            'model': getattr(args, 'model', ''),
            'seed': int(getattr(args, 'seed', 0)),
            'units': 'TECU' if str(getattr(args, 'dataset', '')).lower() in ['gimtec', 'tec'] else 'raw',
            'normalized': False,
            'args': {k: getattr(args, k) for k in sorted(vars(args).keys()) if not k.startswith('_')},
            'env': _collect_env(),
            'paths': {
                'root': str(out_dir),
                'predictions': str(preds_path) if preds_path.exists() else None,
            },
            'truth_ref': truth_ref,
            'truth_digest': truth_digest,
            'ts_utc': getattr(args, 'ts_utc', None) or _now_utc_ts(),
            'run_id': getattr(args, 'wandb_run_id', None),
            'run_tag': getattr(args, 'run_tag', None),
            'oss_prefix': f"Ion-Phys-Toolkit/{dataset_slug(getattr(args,'dataset',''))}/"
                           f"{str(getattr(args,'model','')).lower()}/seed_{int(getattr(args,'seed',0))}/"
                           f"{getattr(args,'run_tag', '')}/",
        }
        _save_json(man_path, man)

    # Upload small files to W&B
    _wandb_log_artifact(run, metrics_path, atype='results')
    _wandb_log_artifact(run, stepwise_path, atype='results')
    _wandb_log_artifact(run, cc_path, atype='results')
    _wandb_log_artifact(run, man_path, atype='results')
