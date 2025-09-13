import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from lib.datasets.gimtec_vendor import _load_year, _split_data, FIXED_MIN, FIXED_MAX, H, W
from lib.normalization import MinMax01Scaler, NScaler


def _prefix_concat(base, prefix_src, time_step):
    return np.concatenate([prefix_src[-time_step:], base], axis=0)


def build_year_loader(args, year, time_step, predict_step, prefix_from=None, batch_size=None):
    base_dir = os.path.join('..', 'data', 'GIMtec')
    y = _load_year(base_dir, year)
    if prefix_from is not None:
        y_prefix = _load_year(base_dir, prefix_from)
        y = _prefix_concat(y, y_prefix, time_step)
    x, y_t = _split_data(y, time_step, predict_step)
    bx, tx = x.shape[:2]
    x = x.reshape(bx, tx, H * W, 1)
    y_t = y_t.reshape(bx, predict_step, H * W, 1)
    ds = _TensorDataset(x, y_t)
    loader = DataLoader(ds, batch_size=(batch_size or args.batch_size), shuffle=False,
                        drop_last=False, pin_memory=torch.cuda.is_available())
    scaler_data = MinMax01Scaler(FIXED_MIN, FIXED_MAX)
    return loader, scaler_data


class _TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluate_per_year(model, trainer):
    """Evaluate CSA-WTConvLSTM on GIMtec year by year and log metrics.
    Matches original dataset splits and prefix stitching.
    """
    args = trainer.args
    logger = trainer.logger
    time_step = args.lag
    predict_step = args.horizon

    # Val years
    val_years = [(2013, [2009, 2010, 2011, 2012]), (2017, [2016])]
    # Test years
    test_years = [(2015, [2014]), (2019, [2018]), (2020, [2019]), (2021, [2020]), (2022, [2021])]

    def _run_for_year(year, prefixes):
        prefix_from = prefixes[-1] if prefixes else None
        loader, scaler = build_year_loader(args, year, time_step, predict_step, prefix_from=prefix_from,
                                           batch_size=args.batch_size)
        trainer.test(model, args, loader, scaler, logger)

    logger.info('==== Year-wise Validation Metrics ====')
    for year, prefixes in val_years:
        logger.info(f'-- Year {year} --')
        _run_for_year(year, prefixes)

    logger.info('==== Year-wise Test Metrics ====')
    for year, prefixes in test_years:
        logger.info(f'-- Year {year} --')
        _run_for_year(year, prefixes)


def compute_yearwise_metrics(model, trainer):
    """Return JSON-style metrics overall and per-year for GIMtec.
    Metrics include normalized MSE/RMSE (on [0,1]) and real TECU MAE/RMSE.
    """
    import json
    args = trainer.args
    time_step = args.lag
    predict_step = args.horizon

    years = {
        '2015': ['2014'],
        '2019': ['2018'],
        '2020': ['2019'],
        '2021': ['2020'],
        '2022': ['2021'],
    }

    FIX_MAX = 147.1

    def _eval_loader(loader):
        import torch
        from lib.metrics import MSE_torch, MAE_torch, RMSE_torch
        y_true_norm_all = []
        y_pred_norm_all = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(args.device).float()
                y = y.to(args.device).float()
                out = model(x, label=None)[0]  # predictor returns tuple-compatible
                y_true_norm_all.append(y)
                y_pred_norm_all.append(out)
        y_true_norm = torch.cat(y_true_norm_all, dim=0)
        y_pred_norm = torch.cat(y_pred_norm_all, dim=0)
        mse_norm = MSE_torch(y_pred_norm, y_true_norm, None).item()
        rmse_norm = (mse_norm ** 0.5)
        # to TECU
        y_true = y_true_norm * FIX_MAX
        y_pred = y_pred_norm * FIX_MAX
        rmse = RMSE_torch(y_pred, y_true, None).item()
        mae, _ = MAE_torch(y_pred, y_true, None)
        mae = mae.item()
        count = y_true.shape[0] * y_true.shape[1]
        return mse_norm, rmse_norm, rmse, mae, int(count)

    overall = {
        'mse_norm': 0.0,
        'rmse_norm': 0.0,
        'rmse_real_TECU': 0.0,
        'mae_real_TECU': 0.0,
        'relative_error_percent': 0.0,
        'count_frames': 0,
        'shape_per_frame': [1, getattr(args, 'height', 71), getattr(args, 'width', 73)],
        'per_year': {}
    }

    total_mse_sum = 0.0
    total_rmse_sum = 0.0
    total_mae_sum = 0.0
    total_frames = 0

    # per year
    for year, prefixes in years.items():
        loader, _ = build_year_loader(args, int(year), time_step, predict_step, prefix_from=int(prefixes[-1]))
        mse_n, rmse_n, rmse_tecu, mae_tecu, frames = _eval_loader(loader)
        overall['per_year'][year] = {
            'mse_norm': mse_n,
            'rmse_norm': rmse_n,
            'rmse_real_TECU': rmse_tecu,
            'mae_real_TECU': mae_tecu,
            'relative_error_percent': rmse_n * 100.0,
            'windows': frames // predict_step,
        }
        total_mse_sum += mse_n * frames
        total_rmse_sum += rmse_n * frames
        total_mae_sum += mae_tecu * frames
        total_frames += frames

    # overall (weighted by frames)
    if total_frames > 0:
        mse_norm = total_mse_sum / total_frames
        rmse_norm = total_rmse_sum / total_frames
        mae_real = total_mae_sum / total_frames
        overall['mse_norm'] = mse_norm
        overall['rmse_norm'] = rmse_norm
        overall['rmse_real_TECU'] = rmse_norm * FIX_MAX
        overall['mae_real_TECU'] = mae_real
        overall['relative_error_percent'] = rmse_norm * 100.0
        overall['count_frames'] = total_frames
    return overall

