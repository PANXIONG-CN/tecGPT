import torch
import os
import time
import copy
import numpy as np
import math
from time import perf_counter
try:
    import wandb
except Exception:
    wandb = None
try:
    import requests, oss2
except Exception:
    requests = None
    oss2 = None
from threading import Thread
from lib import results_io
from lib.logger import get_logger
from lib.metrics import All_Metrics, MAE_torch, RMSE_torch, CORR_torch
try:
    # Optional imports for physics PINN (kept model-agnostic)
    from lib.datasets.gimtec_adj import load_or_build_adj  # unified adjacency for GIMtec/TEC
    from lib.physics.ppinn import PPinnLoss
except Exception:
    load_or_build_adj = None
    PPinnLoss = None

# --- Local W&B proxy injection helpers -------------------------------------
def _with_wandb_proxy():
    """Temporarily inject SOCKS5 proxy envs for W&B calls only.
    Reads `PROXY_SOCKS5` (e.g., socks5h://host:port) and sets HTTP(S)_PROXY/ALL_PROXY.
    Returns a dict of previous values to restore.
    """
    try:
        p = os.getenv('PROXY_SOCKS5', '')
        if not p:
            return None
        prev = {k: os.environ.get(k) for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')}
        os.environ['HTTP_PROXY'] = p
        os.environ['HTTPS_PROXY'] = p
        os.environ['ALL_PROXY'] = p
        return prev
    except Exception:
        return None


def _restore_proxy(prev):
    try:
        if prev is None:
            return
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    except Exception:
        pass

# --- AMP/BF16 helpers -------------------------------------------------------
def _has_bf16_cuda() -> bool:
    """Detect bf16 support on the current CUDA device.
    Prefer torch APIs; fall back to compute capability heuristic (>= 8.0).
    """
    if not torch.cuda.is_available():
        return False
    try:
        # PyTorch 2.x exposes this helper
        return bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    except Exception:
        pass
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False

class Trainer(object):
    def __init__(self, model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        # self.model_stu = model_stu
        self.args = args
        self.loss = loss
        self.loss_kl = loss_kl
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.batch_seen = 0
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        # Best checkpoint naming will be initialized after logger is created
        self.best_run_path = None
        self.best_link_path = None
        self.best_path = None
        self.loss_figure_path = None
        # W&B state
        self._wandb_status_logged = False
        self._wandb_first_log_done = False
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        # build informative logfile name: <timestamp>_<dataset>_<model>_<mode>.log
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            ds = getattr(args, 'dataset', 'DATASET')
            mode = getattr(args, 'mode', 'train')
            if mode == 'pretrain':
                arch = getattr(args, 'arch_tag', 'gptst')
                tgt = getattr(args, 'target_model', 'generic')
                gtag = getattr(args, 'graph_tag', 'na')
                filename = f"{ts}_{ds}_{arch}_{mode}_{tgt}_{gtag}.log"
                logger_name = arch
            else:
                md = getattr(args, 'model', 'MODEL')
                filename = f"{ts}_{ds}_{md}_{mode}.log"
                logger_name = md
        except Exception:
            filename = f"{ts}_run.log"
            logger_name = getattr(args, 'model', 'run')
        self.log_filename = filename
        self.logger = get_logger(args.log_dir, name=logger_name, debug=args.debug, filename=filename)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # log command line and basic env/device info
        try:
            if hasattr(args, 'cmdline'):
                self.logger.info('Command: {}'.format(args.cmdline))
            import torch
            self.logger.info('Torch: {}  CUDA: {}'.format(torch.__version__, torch.cuda.is_available()))
            if torch.cuda.is_available():
                self.logger.info('GPU: {}'.format(torch.cuda.get_device_name(0)))
                self.logger.info('CUDA_VISIBLE_DEVICES={}'.format(os.environ.get('CUDA_VISIBLE_DEVICES')))
            # dump key args
            for k in sorted(vars(args).keys()):
                try:
                    self.logger.info(f'{k}: {getattr(args,k)}')
                except Exception:
                    pass
        except Exception:
            pass
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

        # Initialize best checkpoint naming now that log_filename is known
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_stem = os.path.splitext(self.log_filename)[0] if getattr(self, 'log_filename', '') else ''
        if not base_stem:
            # Fallback naming
            ds = getattr(args, 'dataset', 'DATASET')
            md = getattr(args, 'model', 'MODEL')
            mode = getattr(args, 'mode', 'train')
            # Pretrain uses arch tag
            if mode == 'pretrain':
                arch = getattr(args, 'arch_tag', 'gptst')
                tgt = getattr(args, 'target_model', 'generic')
                gtag = getattr(args, 'graph_tag', 'na')
                base_stem = f"{ts}_{ds}_{arch}_{mode}_{tgt}_{gtag}"
            else:
                base_stem = f"{ts}_{ds}_{md}_{mode}"
        self.best_run_path = os.path.join(self.args.log_dir, f"{base_stem}.pth")
        self.best_link_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_path = self.best_run_path
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        # ====== Physics PINN (optional; minimal intrusion) ======
        self.use_pinn = bool(getattr(self.args, 'use_pinn', False)) and (PPinnLoss is not None)
        self.ppinn = None
        if self.use_pinn:
            try:
                # Build/load adjacency where applicable (e.g., GIMtec/TEC grid8); failure falls back to None
                A = None
                if load_or_build_adj is not None:
                    A_np, _ = load_or_build_adj(getattr(self.args, 'dataset', 'GIMtec'),
                                                getattr(self.args, 'num_nodes', 71*73),
                                                graph_tag=getattr(self.args, 'graph_tag', 'grid8'),
                                                adj_model=getattr(self.args, 'adj_model', None))
                    A = torch.from_numpy(A_np.astype('float32')).to(self.args.device)
            except Exception:
                A = None
            dtm = int(getattr(self.args, 'interval', 120))
            self.ppinn = PPinnLoss(
                scaler_data=self.scaler,
                A=A,
                dt_minutes=dtm,
                tec_ref=float(getattr(self.args, 'tec_ref', 50.0)),
                t_ref_sec=float(getattr(self.args, 't_ref_sec', 7200.0)),
                rot_cap_tecu_per_min=float(getattr(self.args, 'rot_cap', 0.5)),
                roti_cap_scale=float(getattr(self.args, 'roti_cap_scale', 0.7)),
                kappa_nd=float(getattr(self.args, 'kappa_nd', 0.05)),
                roti_w=3,
                use_diffusion=bool(getattr(self.args, 'use_diffusion', True)),
                use_drivers=bool(getattr(self.args, 'use_drivers', False)),
            ).to(self.args.device)
            # Ensure PPINN parameters (e.g., log_vars) are optimized
            try:
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    self.optimizer.add_param_group({'params': self.ppinn.parameters(),
                                                    'lr': float(getattr(self.args, 'lr_init', 1e-3))})
                    self.logger.info('PPINN parameters added to optimizer.')
            except Exception as e:
                self.logger.warning(f'Failed to add PPINN params to optimizer: {e}')
        # Prepare robust drivers path under repo root
        try:
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._drivers_path = os.path.join(_repo_root, 'data', 'GIMtec', 'drivers', 'drivers_2009_2022_2h.npz')
        except Exception:
            self._drivers_path = "./data/GIMtec/drivers/drivers_2009_2022_2h.npz"

    # ------------------------ W&B utilities ------------------------
    def _wandb_prepare(self):
        """Ensure W&B run exists (retry init once) and log status line once.
        Uses local proxy injection around wandb.init to avoid global side effects.
        """
        if wandb is None:
            return None
        run = getattr(wandb, 'run', None)
        if run is None:
            prev = _with_wandb_proxy()
            try:
                # reconstruct entity/project/tags like model/Run.py
                from lib import results_io as _rio
                ds_slug = _rio.dataset_slug(getattr(self.args, 'dataset', ''))
                md_slug = str(getattr(self.args, 'model', 'model')).lower()
                tags = [
                    f"seed_{int(getattr(self.args,'seed',0))}",
                    f"amp_{bool(getattr(self.args,'amp',False))}",
                    f"pinn_{bool(getattr(self.args,'use_pinn',False))}",
                    f"drivers_{bool(getattr(self.args,'use_drivers',False))}",
                    f"adv_{bool(getattr(self.args,'use_adv',False))}",
                    f"nodiff_{not bool(getattr(self.args,'use_diffusion',True))}",
                    f"nochem_{bool(getattr(self.args,'nochem',False))}",
                    f"acc_{int(getattr(self.args,'accumulate_steps',1))}",
                    f"ys_{bool(getattr(self.args,'year_split',False))}",
                ]
                project = os.getenv('WANDB_PROJECT', 'Ion-Phys-Toolkit')
                entity = os.getenv('WANDB_ENTITY', None)
                run = wandb.init(project=project,
                                 entity=entity,
                                 config=vars(self.args),
                                 group=f"{ds_slug}-{md_slug}",
                                 tags=tags)
                try:
                    run.name = f"{ds_slug}-{md_slug}-seed{int(getattr(self.args,'seed',0))}-{getattr(self.args,'ts_utc','')}"
                except Exception:
                    pass
                try:
                    self.args.wandb_run_id = getattr(run, 'id', None)
                except Exception:
                    pass
            except Exception:
                run = None
            finally:
                _restore_proxy(prev)
        # Log status line once
        if not self._wandb_status_logged:
            offline = str(os.getenv('WANDB_MODE', '')).lower() == 'offline'
            proxy = os.getenv('PROXY_SOCKS5', '')
            status = 'online' if (run is not None and not offline) else 'offline'
            proxy_msg = f"enabled ({proxy})" if proxy else 'disabled'
            try:
                self.logger.info(f"W&B status: {status} + 代理: {proxy_msg}")
            except Exception:
                pass
            self._wandb_status_logged = True
        return run

    def _wandb_log(self, data: dict, step: int = None):
        if wandb is None:
            return
        try:
            run = self._wandb_prepare()
            if run is None:
                return
            prev = _with_wandb_proxy()
            try:
                wandb.log(data, step=step)
            finally:
                _restore_proxy(prev)
        except Exception:
            pass

    def _wandb_log_artifact(self, artifact, aliases=None):
        if wandb is None or artifact is None:
            return
        try:
            run = self._wandb_prepare()
            if run is None:
                return
            prev = _with_wandb_proxy()
            try:
                wandb.run.log_artifact(artifact, aliases=aliases or ['latest'])
            finally:
                _restore_proxy(prev)
        except Exception:
            pass

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        count_steps = 0
        val_sse = 0.0
        val_cnt = 0
        # val_pred = []
        # val_true = []

        # AMP dtype selection: bf16 preferred on supported GPUs (e.g., H800)
        use_amp_flag = bool(getattr(self.args, 'amp', False)) and torch.cuda.is_available()
        has_bf16 = _has_bf16_cuda()
        amp_dtype = torch.bfloat16 if (use_amp_flag and has_bf16) else (torch.float16 if use_amp_flag else None)
        # Compat autocast context (torch.amp preferred; fallback to torch.cuda.amp)
        try:
            from torch.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast('cuda', enabled=enabled, dtype=dtype)
        except Exception:
            from torch.cuda.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast(enabled=enabled, dtype=dtype)

        dyn_caps_all = []  # collect dynamic rot_cap per-batch (TECU/min) if available
        dyn_valid_rot_all = []
        dyn_valid_roti_all = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # support optional (data, target, start_idx) for driver slicing; default to 2-tuple
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    data, target, start_idx = batch
                else:
                    data, target = batch
                    start_idx = None
                data = data.to(self.args.device, non_blocking=True)
                target = target.to(self.args.device, non_blocking=True)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                if self.args.mode == 'pretrain':
                    label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                else:
                    label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                # Optional drivers input for de_gwn
                drivers_input = None
                if (str(getattr(self.args, 'model', '')).lower() == 'gwn') and bool(getattr(self.args, 'use_drivers', False)):
                    try:
                        from lib.physics.drivers import DriversStore
                        if not hasattr(self, '_drivers'):
                            self._drivers = DriversStore(self._drivers_path, device=self.args.device)
                        B = data.size(0); T = data.size(1); N = data.size(2)
                        sidx = int(start_idx) if start_idx is not None else 0
                        d = self._drivers.slice_bt(sidx, T, N)
                        ae = d['AE']; dst = d['Dst']; kp = d['Kp']; f107 = d['F107']; cosz = d.get('cosSZA')
                        if cosz is None:
                            import torch as _torch
                            cosz = _torch.zeros_like(ae)
                        # shape [1,T,N] -> [B,T,N,1]; stack last dim
                        drivers_input = torch.stack([ae, dst, kp, f107, cosz], dim=-1).expand(B, -1, -1, -1)
                    except Exception:
                        drivers_input = None
                with _amp_ctx(use_amp_flag, amp_dtype):
                    if self.args.mode == 'pretrain':
                        output, _, mask, _, _ = self.model(data, label)
                    else:
                        # pass optional drivers_input for de_gwn
                        try:
                            output, _, mask, _, _ = self.model(data, label=None, batch_seen=None, drivers_input=drivers_input)
                        except TypeError:
                            output, _, mask, _, _ = self.model(data, label=None)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label[..., :self.args.output_dim])
                with _amp_ctx(use_amp_flag, amp_dtype):
                    if self.args.mode == 'pretrain':
                        loss, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                    else:
                        res = self.loss(output, target[..., :self.args.output_dim])
                        loss = res[0] if isinstance(res, tuple) else res
                        # streaming RMSE in TECU domain
                        try:
                            pred_inv = self.scaler.inverse_transform(output[..., :self.args.output_dim])
                            true_inv = self.scaler.inverse_transform(target[..., :self.args.output_dim])
                            diff = (pred_inv - true_inv)
                            val_sse += float(torch.nansum(diff * diff).item())
                            val_cnt += int(diff.numel())
                        except Exception:
                            pass
                        # Physics PINN (optional) — zero-driver works without start_idx
                        if self.ppinn is not None:
                            try:
                                y_pred01 = output[..., :1]
                                drivers = None
                                # external drivers can be enabled later with dataset returning start_idx
                                if bool(getattr(self.args, 'use_drivers', False)) and (start_idx is not None):
                                    from lib.physics.drivers import DriversStore
                                    if not hasattr(self, '_drivers'):
                                        self._drivers = DriversStore(self._drivers_path, device=self.args.device)
                                    B, T, N = y_pred01.shape[0], y_pred01.shape[1], y_pred01.shape[2]
                                    drivers = self._drivers.slice_bt(int(start_idx), T, N)
                                    drivers['use_adv'] = bool(getattr(self.args, 'use_adv', False))
                                    if bool(getattr(self.args, 'nochem', False)):
                                        drivers['alpha_nd'] = 0.0
                                        drivers['beta_nd'] = 0.0
                                lam = float(getattr(self.args, 'lambda_phys', 1.0))
                                if lam != 0.0:
                                    # Avoid double-counting data term: exclude data from PPINN aggregation
                                    loss_phys, _ = self.ppinn(y_pred01, None, drivers=drivers)
                                    if torch.isfinite(loss_phys).item():
                                        loss = loss + lam * loss_phys
                            except Exception:
                                pass
                # accumulate only finite losses
                if torch.isfinite(loss).item():
                    total_val_loss += loss.item()
                    count_steps += 1
        denom = count_steps if count_steps > 0 else max(1, len(val_dataloader))
        val_loss = total_val_loss / denom
        val_rmse = float(math.sqrt(val_sse / val_cnt)) if val_cnt > 0 else float('nan')
        self._last_val_rmse = val_rmse
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}, RMSE: {:.6f}'.format(epoch, val_loss, 0.0 if (val_rmse!=val_rmse) else val_rmse))
        try:
            self._wandb_log({'val/loss': val_loss, 'val/rmse': val_rmse}, step=self.batch_seen)
        except Exception:
            pass

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        count_steps = 0
        total_flow_loss = 0
        total_s_loss = 0
        # AMP dtype selection: bf16 preferred on supported GPUs (e.g., H800)
        use_amp_flag = bool(getattr(self.args, 'amp', False)) and torch.cuda.is_available()
        has_bf16 = _has_bf16_cuda()
        amp_dtype = torch.bfloat16 if (use_amp_flag and has_bf16) else (torch.float16 if use_amp_flag else None)
        accum = max(1, int(getattr(self.args, 'accumulate_steps', 1)))
        # AMP helpers: use new torch.amp API if available
        try:
            from torch.amp import GradScaler as _GradScaler
            def _make_scaler(enabled):
                return _GradScaler('cuda', enabled=enabled)
        except Exception:
            from torch.cuda.amp import GradScaler as _GradScaler
            def _make_scaler(enabled):
                return _GradScaler(enabled=enabled)
        try:
            from torch.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast('cuda', enabled=enabled, dtype=dtype)
        except Exception:
            from torch.cuda.amp import autocast as _autocast
            def _amp_ctx(enabled, dtype):
                return _autocast(enabled=enabled, dtype=dtype)
        # Enable GradScaler only for FP16 (bf16 path runs without scaling)
        scaler = _make_scaler(enabled=(use_amp_flag and not has_bf16))
        grad_clip = float(getattr(self.args, 'grad_clip', 0.0))
        for batch_idx, batch in enumerate(self.train_loader):
            self.batch_seen += 1
            # support optional (data, target, start_idx)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, target, start_idx = batch
            else:
                data, target = batch
                start_idx = None
            data = data.to(self.args.device, non_blocking=True)
            target = target.to(self.args.device, non_blocking=True)
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            if self.args.mode == 'pretrain':
                label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            else:
                label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
            if batch_idx % accum == 0:
                self.optimizer.zero_grad(set_to_none=True)

            if self.args.mode == 'pretrain':
                with _amp_ctx(use_amp_flag, amp_dtype):
                    output, out_time, mask, probability, eb = self.model(data, label, self.batch_seen, epoch)
                    loss_flow, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                    if epoch > self.args.change_epoch:
                        loss_s = self.loss_kl(probability.log(), eb) * 0.1
                        loss = loss_flow + loss_s
                    else:
                        loss = loss_flow
            else:
                with _amp_ctx(use_amp_flag, amp_dtype):
                    # Optional drivers input for de_gwn
                    drivers_input = None
                    if (str(getattr(self.args, 'model', '')).lower() == 'gwn') and bool(getattr(self.args, 'use_drivers', False)):
                        try:
                            from lib.physics.drivers import DriversStore
                            if not hasattr(self, '_drivers'):
                                self._drivers = DriversStore(self._drivers_path, device=self.args.device)
                            B = data.size(0); T = data.size(1); N = data.size(2)
                            sidx = int(start_idx) if start_idx is not None else 0
                            d = self._drivers.slice_bt(sidx, T, N)
                            ae = d['AE']; dst = d['Dst']; kp = d['Kp']; f107 = d['F107']; cosz = d.get('cosSZA')
                            if cosz is None:
                                import torch as _torch
                                cosz = _torch.zeros_like(ae)
                            drivers_input = torch.stack([ae, dst, kp, f107, cosz], dim=-1).expand(B, -1, -1, -1)
                        except Exception:
                            drivers_input = None
                    try:
                        output, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen, drivers_input=drivers_input)
                    except TypeError:
                        t0 = perf_counter()
                        output, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen)
                    res = self.loss(output, target[..., :self.args.output_dim])
                    # loss functions may return (mean_loss, elementwise) tuples (e.g., mask_mae)
                    loss = res[0] if isinstance(res, tuple) else res
                    # Physics PINN (optional)
                    if self.ppinn is not None:
                        try:
                            y_pred01 = output[..., :1]
                            drivers = None
                            if bool(getattr(self.args, 'use_drivers', False)) and (start_idx is not None):
                                from lib.physics.drivers import DriversStore
                                if not hasattr(self, '_drivers'):
                                    self._drivers = DriversStore(self._drivers_path, device=self.args.device)
                                B, T, N = y_pred01.shape[0], y_pred01.shape[1], y_pred01.shape[2]
                                drivers = self._drivers.slice_bt(int(start_idx), T, N)
                                drivers['use_adv'] = bool(getattr(self.args, 'use_adv', False))
                                if bool(getattr(self.args, 'nochem', False)):
                                    drivers['alpha_nd'] = 0.0
                                    drivers['beta_nd'] = 0.0
                            lam = float(getattr(self.args, 'lambda_phys', 1.0))
                            if lam != 0.0:
                                # Avoid double-counting data term in training
                                loss_phys, _ = self.ppinn(y_pred01, None, drivers=drivers)
                                if torch.isfinite(loss_phys).item():
                                    loss = loss + lam * loss_phys
                        except Exception:
                            pass
                    # step time measurement
                    try:
                        step_times_ms.append((perf_counter() - t0) * 1000.0)
                    except Exception:
                        pass

            loss_to_back = loss / accum
            if use_amp_flag:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            # add max grad clipping
            if grad_clip and grad_clip > 0:
                # unscale before clipping in scaled fp16 path
                if use_amp_flag and not has_bf16:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            elif self.args.grad_norm:
                if use_amp_flag and not has_bf16:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == self.train_per_epoch:
                if use_amp_flag:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
            if torch.isfinite(loss).item():
                total_loss += loss.item()
                count_steps += 1
            # calculate total loss
            if self.args.mode == 'pretrain':
                total_flow_loss += loss_flow.item()
                if epoch > self.args.change_epoch:
                    total_s_loss += loss_s.item()
            #log information
            # maintain EMA
            try:
                if ema is None:
                    ema = loss.item()
                else:
                    ema = 0.9 * ema + 0.1 * loss.item()
                self._ema_loss = ema
            except Exception:
                pass
            if batch_idx % max(1, int(getattr(self.args, 'log_step', 50))) == 0:
                running_avg = (total_loss / max(1, count_steps)) if 'count_steps' in locals() else loss.item()
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} (running_avg: {:.6f})'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item(), running_avg))
                # wandb real-time metrics
                try:
                    logd = {'train/loss': float(loss.item()), 'train/loss_smooth': float(ema)}
                    try:
                        lr = float(self.optimizer.param_groups[0]['lr'])
                        logd['train/lr'] = lr
                    except Exception:
                        pass
                    if bool(getattr(self.args, 'grad_norm', False)):
                        try:
                            total_norm = 0.0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    g = p.grad.detach().data
                                    total_norm += float(torch.norm(g, 2).item()) ** 2
                            logd['train/grad_norm'] = float(math.sqrt(total_norm))
                        except Exception:
                            pass
                    self._wandb_log(logd, step=self.batch_seen)
                except Exception:
                    pass
        denom = count_steps if count_steps > 0 else self.train_per_epoch
        train_epoch_loss = total_loss/denom
        if self.args.mode == 'pretrain':
            train_epoch_flow_loss = total_flow_loss/self.train_per_epoch
            train_epoch_s_loss = total_s_loss / self.train_per_epoch
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} averaged Loss_s: {:.6f}'.format(epoch, train_epoch_flow_loss, train_epoch_s_loss))
        else:
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} (per-batch mean)'.format(epoch, train_epoch_loss))

        # epoch-level average step time
        try:
            if step_times_ms:
                self._wandb_log({'train/avg_step_time_ms_per_epoch': float(sum(step_times_ms)/len(step_times_ms))}, step=self.batch_seen)
        except Exception:
            pass
        # learning rate scheduling (non-plateau handled here; plateau handled in train() after val)
        if self.lr_scheduler is not None and getattr(self.args, 'scheduler', 'none') != 'plateau':
            if self.args.lr_decay:
                self.lr_scheduler.step()
        # train_epoch_flow_loss for params selecting
        if self.args.mode == 'pretrain':
            return train_epoch_flow_loss
        else:
            return train_epoch_loss

    def train(self):
        best_model = None
        best_model_test = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        up_epoch = [int(i) for i in list(self.args.up_epoch.split(','))]
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            if epoch in up_epoch:
                best_loss = float('inf')
            if self.args.mode == 'pretrain':
                if train_epoch_loss < best_loss:
                    best_loss = train_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
            else:
                if self.val_loader == None:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                val_loss_list.append(val_epoch_loss)
                # step plateau scheduler with validation loss
                if self.lr_scheduler is not None and getattr(self.args, 'scheduler', 'none') == 'plateau':
                    self.lr_scheduler.step(val_epoch_loss)
                min_delta = float(getattr(self.args, 'early_stop_min_delta', 0.0))
                if (best_loss - val_epoch_loss) > min_delta:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                best_model_test = copy.deepcopy(self.model)
                # Best snapshot: save symlink immediately, async upload to OSS, and write snapshot JSON
                try:
                    # save checkpoint to the named path and refresh best_model.pth symlink
                    import os as _os
                    base_stem = _os.path.splitext(self.log_filename)[0] if hasattr(self, 'log_filename') else 'supervised_best'
                    named_pth = _os.path.join(self.args.log_dir, f"{base_stem}.pth")
                    torch.save(best_model, named_pth)
                    self.logger.info("Saving current model to " + named_pth)
                    best_link = _os.path.join(self.args.log_dir, 'best_model.pth')
                    try:
                        if _os.path.islink(best_link) or _os.path.exists(best_link):
                            _os.remove(best_link)
                        _os.symlink(_os.path.basename(named_pth), best_link)
                        self.logger.info("Updated symlink -> best_model.pth")
                    except Exception as e:
                        self.logger.warning(f"Failed to create best_model.pth symlink: {e}")
                    # build s3 uri and async upload if OSS creds present
                    try:
                        ds_slug = results_io.dataset_slug(getattr(self.args, 'dataset', ''))
                        md_slug = str(getattr(self.args, 'model', 'model')).lower()
                        seed = int(getattr(self.args, 'seed', 0))
                        run_tag = getattr(self.args, 'run_tag', '')
                        bucket = os.getenv('OSS_BUCKET', '')
                        prefix = f"Ion-Phys-Toolkit/{ds_slug}/{md_slug}/seed_{seed}/{run_tag}/"
                        s3_uri = f"s3://{bucket}/{prefix}best_model.pth" if bucket else None
                        def _upload_best(path: str, key: str):
                            if requests is None or oss2 is None or not bucket:
                                return
                            try:
                                ak = os.environ['OSS_ACCESS_KEY_ID']; sk = os.environ['OSS_ACCESS_KEY_SECRET']
                                endpoint = os.environ.get('OSS_ENDPOINT','oss-accelerate.aliyuncs.com')
                                sess = requests.Session(); sess.trust_env = False
                                auth = oss2.Auth(ak, sk)
                                bkt = oss2.Bucket(auth, 'https://'+endpoint, bucket, session=oss2.Session(sess))
                                key_full = prefix + key
                                sz = os.path.getsize(path)
                                if sz <= 50*1024*1024:
                                    with open(path, 'rb') as f:
                                        bkt.put_object(key_full, f)
                                else:
                                    import tarfile
                                    tmp = path + '.tar.gz'
                                    with tarfile.open(tmp, 'w:gz') as tar:
                                        tar.add(path, arcname=os.path.basename(path))
                                    oss2.resumable_upload(bkt, key_full+'.tar.gz', tmp, num_threads=6, part_size=10*1024*1024)
                                    try:
                                        os.remove(tmp)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        try:
                            Thread(target=_upload_best, args=(best_link, 'best_model.pth'), daemon=True).start()
                        except Exception:
                            pass
                        # write best_snapshot.json
                        try:
                            import json
                            snap = {
                                'epoch': int(epoch),
                                'step': int(self.batch_seen),
                                'metrics': {
                                    'val_loss': float(best_loss),
                                    'val_rmse': float(getattr(self, '_last_val_rmse', float('nan'))),
                                },
                                'timestamp': getattr(self.args, 'ts_utc', ''),
                                'run_tag': run_tag,
                                'wandb_run_id': getattr(self.args, 'wandb_run_id', None),
                                'checkpoint_reference_uri': s3_uri,
                                'oss_prefix': prefix,
                            }
                            with open(os.path.join(self.args.log_dir, 'best_snapshot.json'), 'w') as f:
                                json.dump(snap, f, indent=2)
                            # upload snapshot artifact online
                            try:
                                if wandb is not None:
                                    art = wandb.Artifact(name=f"best-snapshot-{getattr(self.args,'wandb_run_id','')}", type='results')
                                    art.add_file(os.path.join(self.args.log_dir, 'best_snapshot.json'))
                                    self._wandb_log_artifact(art, aliases=['best'])
                                    self._wandb_log({'best/val_loss': float(best_loss), 'best/val_rmse': float(getattr(self, '_last_val_rmse', float('nan')))}, step=self.batch_seen)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # Save the best model to file (always)
        try:
            target_state = best_model if 'best_model' in locals() else self.model.state_dict()
            # pretrain: save named + symlink latest_pretrain.pth; supervised: save best_model.pth
            if getattr(self.args, 'mode', '') == 'pretrain':
                # save named
                torch.save(target_state, self.best_path)
                self.logger.info("Saving pretrain model to " + self.best_path)
                # create/refresh symlink latest_pretrain.pth
                try:
                    import os as _os
                    latest_link = _os.path.join(self.args.log_dir, 'latest_pretrain.pth')
                    if _os.path.islink(latest_link) or _os.path.exists(latest_link):
                        try:
                            _os.remove(latest_link)
                        except Exception:
                            pass
                    _os.symlink(_os.path.basename(self.best_path), latest_link)
                    self.logger.info("Updated symlink -> latest_pretrain.pth")
                except Exception as e:
                    self.logger.warning(f"Failed to create latest_pretrain.pth symlink: {e}")
            else:
                # supervised: save with log basename and create/refresh symlink best_model.pth
                try:
                    import os as _os
                    base_stem = _os.path.splitext(self.log_filename)[0] if hasattr(self, 'log_filename') else 'supervised_best'
                    named_pth = _os.path.join(self.args.log_dir, f"{base_stem}.pth")
                    torch.save(target_state, named_pth)
                    self.logger.info("Saving current model to " + named_pth)
                    # symlink best_model.pth -> named file
                    best_link = _os.path.join(self.args.log_dir, 'best_model.pth')
                    try:
                        if _os.path.islink(best_link) or _os.path.exists(best_link):
                            _os.remove(best_link)
                        _os.symlink(_os.path.basename(named_pth), best_link)
                        self.logger.info("Updated symlink -> best_model.pth")
                    except Exception as e:
                        self.logger.warning(f"Failed to create best_model.pth symlink: {e}")
                except Exception as e:
                    # fallback to legacy path
                    import torch as _torch
                    _torch.save(target_state, self.best_path)
                    self.logger.info("Saving current model to " + self.best_path)
        except Exception as e:
            self.logger.warning(f"Failed to save model: {e}")

        #test
        # self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        if self.args.mode == 'pretrain':
            model_for_test = best_model_test if best_model_test is not None else self.model
            self.test(model_for_test, self.args, self.train_loader, self.scaler, self.logger)
        else:
            # save arrays: default save test only (float16); disable val arrays by default
            self.test(best_model_test, self.args, self.test_loader, self.scaler, self.logger,
                      save_arrays=True, save_tag='test', log_filename=self.log_filename)
            if self.val_loader is not None:
                self.test(best_model_test, self.args, self.val_loader, self.scaler, self.logger,
                          save_arrays=False, save_tag='val', log_filename=self.log_filename)
            # Optional: year-wise evaluation for GIMtec + CSA_WTConvLSTM to match original repo analysis
            try:
                if getattr(self.args, 'model', None) == 'CSA_WTConvLSTM' and self.args.dataset.lower() in ['gimtec', 'tec']:
                    from lib.eval_gimtec_yearwise import evaluate_per_year
                    evaluate_per_year(best_model_test, self)
                # Always save JSON metrics for GIMtec/TEC at end of training
                if self.args.dataset.lower() in ['gimtec', 'tec']:
                    from lib.eval_gimtec_yearwise import compute_yearwise_metrics
                    metrics = compute_yearwise_metrics(best_model_test, self)
                    import json, os, torch
                    # derive json filename from log filename to keep naming consistent
                    base_stem = os.path.splitext(self.log_filename)[0]
                    json_name = f"{base_stem}.json"
                    out_path = os.path.join(self.args.log_dir, json_name)
                    # Inject meta (env/oom/model/data/paths)
                    try:
                        preds_path = os.path.join(self.args.log_dir, f"{base_stem}_test_preds.npy")
                        ckpt_path = getattr(self, 'best_run_path', os.path.join(self.args.log_dir, 'best_model.pth'))
                        metrics['meta'] = {
                            'run_name': base_stem,
                            'paths': {
                                'preds': preds_path,
                                'log': os.path.join(self.args.log_dir, self.log_filename),
                                'ckpt': ckpt_path,
                            },
                            'env': {
                                'amp': 'bf16' if bool(getattr(self.args, 'amp', False)) else 'none',
                                'tf32': bool(getattr(torch.backends.cuda.matmul, 'allow_tf32', True)),
                                'ln_enabled': bool(getattr(self.args, 'use_ln', False)),
                            },
                            'oom_fallback': {
                                'initial_batch_size': int(getattr(self.args, 'batch_size', 0)),
                                'final_batch_size': int(getattr(self.args, 'batch_size', 0)),
                                'events': [],
                            },
                            'model': {
                                'gpt2_name': str(getattr(self.args, 'hf_model_name', 'gpt2-large')),
                                'llm_layers': int(getattr(self.args, 'llm_layers', 0)),
                                'frozen': True,
                            },
                            'data': {
                                'dataset': str(self.args.dataset),
                                'channels': int(getattr(self.args, 'input_base_dim', 1)),
                                'stride': 1,
                                'interval_minutes': int(getattr(self.args, 'interval', 120)),
                                'lag': int(getattr(self.args, 'lag', 12)),
                                'horizon': int(getattr(self.args, 'horizon', 12)),
                            }
                        }
                    except Exception:
                        pass
                    with open(out_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    self.logger.info('Saved JSON metrics to {}'.format(out_path))
            except Exception as e:
                self.logger.warning(f'Year-wise evaluation skipped due to error: {e}')


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        # Save to run_name.pth
        torch.save(state, self.best_run_path)
        # Update best_model.pth symlink (or copy if symlink unsupported)
        try:
            if os.path.islink(self.best_link_path) or os.path.exists(self.best_link_path):
                try:
                    os.remove(self.best_link_path)
                except Exception:
                    pass
            os.symlink(os.path.basename(self.best_run_path), self.best_link_path)
        except Exception:
            # fallback: copy file
            import shutil
            try:
                shutil.copyfile(self.best_run_path, self.best_link_path)
            except Exception:
                pass
        self.logger.info("Saving current best model to " + self.best_run_path + " (link: best_model.pth)")

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None, save_arrays=False, save_tag='test', log_filename=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        _batch_start_idx = []
        _batch_T = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    data, target, sidx = batch
                else:
                    data, target = batch
                    sidx = None
                data = data.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                data = data[..., :args.input_base_dim + args.input_extra_dim]
                if args.mode == 'pretrain':
                    output, _, mask, _, _ = model(data, None, None, args.epochs)
                    label = data[..., :args.output_dim]
                    y_true.append(label*mask)
                    y_pred.append(output*mask)
                else:
                    output, _, mask, _, _ = model(data, label=None)
                    label = target[..., :args.output_dim]
                    y_true.append(label)
                    y_pred.append(output)
                try:
                    _batch_T.append(int(output.shape[1]))
                except Exception:
                    _batch_T.append(None)
                _batch_start_idx.append(int(sidx) if sidx is not None else None)
                # dynamic rot_cap series (TECU/min) if drivers available
                try:
                    if bool(getattr(self.args, 'use_pinn', False)) and bool(getattr(self.args, 'use_drivers', False)):
                        from lib.physics.drivers import DriversStore
                        if not hasattr(self, '_drivers'):
                            self._drivers = DriversStore(self._drivers_path, device='cpu')
                        T = int(output.shape[1]); N = int(output.shape[2])
                        if sidx is not None and T >= 2:
                            d = self._drivers.slice_bt(int(sidx), T, N)
                            if 'Dst' in d:
                                import numpy as _np
                                dst = d['Dst'].cpu().numpy()  # [1,T,N]
                                dstn = _np.clip(-dst, 0.0, None)
                                k = float(getattr(self.args, 'rot_cap_k', 0.01))
                                base = float(getattr(self.args, 'rot_cap', 0.5))
                                cap = base * (1.0 + k * (dstn / 100.0))  # [1,T,N], TECU/min
                                cap = cap[:, 1:, :]  # align to ROT steps
                                # average over nodes for a compact per-step series
                                cap_series = cap.mean(axis=2).reshape(-1)  # [T-1]
                                dyn_caps_all.append(cap_series)
                                # dynamic validity (optional)
                                # compute ROT and ROTI in TECU/min domain
                                yp_bt = output.detach().cpu().numpy()  # [B,T,N,1]
                                yt_bt = y_true[-1].detach().cpu().numpy() if len(y_true)>0 else None
                                rot = (yp_bt[:, 1:, :, 0] - yp_bt[:, :-1, :, 0]) / float(getattr(self.args, 'interval', 120))
                                rot_valid = (abs(rot) <= cap)
                                dyn_valid_rot_all.append(rot_valid.mean())
                                # roti rolling std
                                if rot.shape[1] >= 3:
                                    w = 3
                                    roti = _np.stack([rot[:, i:i+w, :].std(axis=1) for i in range(rot.shape[1]-w+1)], axis=1)
                                    roti_cap = float(getattr(self.args, 'roti_cap_scale', 0.7)) * cap[:, :roti.shape[1], :]
                                    dyn_valid_roti_all.append((roti <= roti_cap).mean())
                except Exception:
                    pass

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        # For GIMtec/TEC datasets, align reporting with CSA pipeline: drop MAPE from logs
        disable_mape = (args.dataset.lower() in ['gimtec', 'tec'])

        def _corr_with_ratio(pred_t, true_t, mask_value=None):
            # Harmonize shapes like CORR_torch
            x, y = pred_t, true_t
            if x.dim() == 2:  # [B,N]
                x = x.unsqueeze(1).unsqueeze(1); y = y.unsqueeze(1).unsqueeze(1)
            elif x.dim() == 3:  # [B,T,N]
                x = x.transpose(1, 2).unsqueeze(1); y = y.transpose(1, 2).unsqueeze(1)
            elif x.dim() == 4:  # [B,T,N,D]
                x = x.transpose(2, 3); y = y.transpose(2, 3)
            dims = (0, 1, 2)
            xm = x.mean(dim=dims); ym = y.mean(dim=dims)
            xs = x.std(dim=dims); ys = y.std(dim=dims)
            numer = ((x - xm)*(y - ym)).mean(dim=dims)
            denom = (xs*ys)
            corr = numer / denom
            valid = torch.isfinite(corr) & torch.isfinite(denom) & (ys != 0)
            if valid.numel() == 0:
                return float('nan'), 0.0
            valid_vals = corr[valid]
            ratio = float(valid.float().mean().item())
            if valid_vals.numel() == 0:
                return float('nan'), ratio
            return float(torch.nanmean(valid_vals).item()), ratio

        tag = str(save_tag) if save_tag is not None else 'eval'
        # Always report only MAE/RMSE/R2/CORR
        corr_list = []
        r2_list = []
        for t in range(y_true.shape[1]):
            mae, _ = MAE_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
            rmse = RMSE_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
            corr_val, ratio = _corr_with_ratio(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
            # R2 in TECU domain via numpy
            try:
                import numpy as _np
                yt_np = y_true[:, t, ...].detach().cpu().numpy()
                yp_np = y_pred[:, t, ...].detach().cpu().numpy()
                m = _np.isfinite(yt_np) & _np.isfinite(yp_np)
                if m.any():
                    mu = float(_np.mean(yt_np[m]))
                    ss_res = float(_np.sum((yp_np[m] - yt_np[m]) ** 2))
                    ss_tot = float(_np.sum((yt_np[m] - mu) ** 2))
                    r2 = float('nan') if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
                else:
                    r2 = float('nan')
            except Exception:
                r2 = float('nan')
            corr_list.append(corr_val)
            r2_list.append(r2)
            logger.info("[{}] Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, R2:{:.4f}, CORR:{:.4f} (valid={:.1f}%)".format(
                tag, t + 1, mae, rmse, 0.0 if r2!=r2 else r2, 0.0 if corr_val!=corr_val else corr_val, ratio*100))
        mae, _ = MAE_torch(y_pred, y_true, args.mae_thresh)
        rmse = RMSE_torch(y_pred, y_true, args.mae_thresh)
        # overall r2 by numpy
        try:
            import numpy as _np
            yt = y_true.detach().cpu().numpy(); yp = y_pred.detach().cpu().numpy()
            valid = _np.isfinite(yt) & _np.isfinite(yp)
            if valid.any():
                ss_res = float(_np.sum((yp[valid]-yt[valid])**2))
                mu = float(_np.mean(yt[valid]))
                ss_tot = float(_np.sum((yt[valid]-mu)**2))
                r2_overall = float('nan') if ss_tot<=0 else (1.0 - ss_res/ss_tot)
            else:
                r2_overall = float('nan')
        except Exception:
            r2_overall = float('nan')
        corr_vals = [c for c in corr_list if not (c!=c)]
        corr_avg = sum(corr_vals)/len(corr_vals) if len(corr_vals)>0 else float('nan')
        logger.info("[{}] Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, R2:{:.4f}, CORR:{:.4f}".format(
                    tag, mae, rmse, 0.0 if r2_overall!=r2_overall else r2_overall, 0.0 if corr_avg!=corr_avg else corr_avg))

        # physics metrics (PINN) — only when enabled
        try:
            if bool(getattr(args, 'use_pinn', False)):
                phys = {}
                yp_np = y_pred.detach().cpu().numpy(); yt_np = y_true.detach().cpu().numpy()
                # nonneg_rate
                phys['nonneg_rate'] = float(np.mean(yp_np >= 0.0))
                # ROT/ROTI validity (TECU per minute)
                dt_min = float(getattr(args, 'interval', 120))
                if yp_np.shape[1] >= 2 and dt_min > 0:
                    rot = (yp_np[:, 1:, ...] - yp_np[:, :-1, ...]) / dt_min
                    rot_cap = float(getattr(args, 'rot_cap', 0.5))
                    phys['rot_validity'] = float(np.mean(np.abs(rot) <= rot_cap))
                    # roti: rolling std over window=3
                    if rot.shape[1] >= 3:
                        w = 3
                        # compute rolling std along time
                        roti = np.stack([rot[:, i:i+w, ...].std(axis=1) for i in range(rot.shape[1]-w+1)], axis=1)
                        roti_cap = float(getattr(args, 'roti_cap_scale', 0.7)) * rot_cap
                        phys['roti_validity'] = float(np.mean(roti <= roti_cap))
                    else:
                        phys['roti_validity'] = None
                else:
                    phys['rot_validity'] = None
                    phys['roti_validity'] = None
                # night monotonicity with cosSZA mask if drivers available and start_idx provided
                night_ratio = None
                try:
                    from lib.physics.drivers import DriversStore
                    # attempt per-batch evaluation to avoid large memory
                    store = DriversStore(self._drivers_path, device='cpu')
                    total_mask = 0; total_ok = 0
                    offset = 0
                    for b_idx, (sidx, T) in enumerate(zip(_batch_start_idx, _batch_T)):
                        if sidx is None or T is None or T < 2:
                            offset += (y_pred.shape[0] if b_idx==0 else 0)
                            continue
                        # slice within this batch range
                        B = 1  # per-sample aggregated later; approximate using broadcast
                        N = yp_np.shape[2]
                        d = store.slice_bt(int(sidx), int(T), int(N))
                        if 'cosSZA' in d:
                            cosz = d['cosSZA'].cpu().numpy()  # [1,T,N]
                            # take predicted for this batch-chunk (approx: use entire batch window)
                            yp_bt = yp_np[offset:offset+1, :T, :, :]
                            y_t = yp_bt[:, :-1, :, 0]
                            y_tp1 = yp_bt[:, 1:, :, 0]
                            night = (cosz[:, :-1, :] <= 0.0)
                            total_mask += int(night.sum())
                            total_ok += int(((y_tp1 - y_t) <= 0.0)[night].sum())
                        offset += 1
                    night_ratio = (float(total_ok) / float(total_mask)) if total_mask > 0 else None
                except Exception:
                    night_ratio = None
                phys['night_monotonicity'] = night_ratio
                # dynamic rot_cap series aggregation
                try:
                    if dyn_caps_all:
                        import numpy as _np
                        caps = _np.concatenate([c.reshape(1,-1) for c in dyn_caps_all], axis=0)  # [B_all, T-1]
                        caps_mean = caps.mean(axis=0).astype('float32')
                        np.save(os.path.join(args.log_dir, 'dynamic_rot_cap.npy'), caps_mean)
                        phys['dynamic_rot_cap_stats'] = {
                            'mean': float(caps_mean.mean()),
                            'min': float(caps_mean.min()),
                            'max': float(caps_mean.max()),
                            'unit': 'TECU_per_min',
                        }
                        phys['dynamic_rot_cap_path'] = 'dynamic_rot_cap.npy'
                    if dyn_valid_rot_all:
                        phys['rot_validity_dynamic'] = float(np.mean(dyn_valid_rot_all))
                    if dyn_valid_roti_all:
                        phys['roti_validity_dynamic'] = float(np.mean(dyn_valid_roti_all))
                except Exception:
                    pass
                import json, os
                with open(os.path.join(args.log_dir, 'physics_metrics.json'), 'w') as f:
                    json.dump(phys, f, indent=2)
        except Exception as e:
            logger.warning(f"physics_metrics.json skipped: {e}")

        # optionally save arrays for reproduction (float16 to reduce size)
        if save_arrays and log_filename:
            try:
                import os
                base_stem = os.path.splitext(log_filename)[0]
                np.save(os.path.join(args.log_dir, f"{base_stem}_{save_tag}_preds.npy"),
                        y_pred.detach().cpu().to(torch.float16).numpy())
                # Do not save ground-truth arrays by default since they are fully deterministic
                # and can be regenerated from the dataset split/windowing.
                logger.info(f"Saved arrays: {base_stem}_{save_tag}_preds.npy (true not saved)")
            except Exception as e:
                logger.warning(f"Failed to save arrays: {e}")
