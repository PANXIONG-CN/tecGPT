import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics, MAE_torch, RMSE_torch, CORR_torch

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

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        count_steps = 0
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

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(self.args.device, non_blocking=True)
                target = target.to(self.args.device, non_blocking=True)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                if self.args.mode == 'pretrain':
                    label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                else:
                    label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                with _amp_ctx(use_amp_flag, amp_dtype):
                    output, _, mask, _, _ = self.model(data, label=None)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label[..., :self.args.output_dim])
                with _amp_ctx(use_amp_flag, amp_dtype):
                    if self.args.mode == 'pretrain':
                        loss, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                    else:
                        res = self.loss(output, label[..., :self.args.output_dim])
                        loss = res[0] if isinstance(res, tuple) else res
                # accumulate only finite losses
                if torch.isfinite(loss).item():
                    total_val_loss += loss.item()
                    count_steps += 1
        denom = count_steps if count_steps > 0 else max(1, len(val_dataloader))
        val_loss = total_val_loss / denom
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))

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
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batch_seen += 1
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
                    out, out_time, mask, probability, eb = self.model(data, label, self.batch_seen, epoch)
                    loss_flow, loss_base = self.loss(out, label[..., :self.args.output_dim], mask)
                    if epoch > self.args.change_epoch:
                        loss_s = self.loss_kl(probability.log(), eb) * 0.1
                        loss = loss_flow + loss_s
                    else:
                        loss = loss_flow
            else:
                with _amp_ctx(use_amp_flag, amp_dtype):
                    out, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen)
                    res = self.loss(out, label[..., :self.args.output_dim])
                    # loss functions may return (mean_loss, elementwise) tuples (e.g., mask_mae)
                    loss = res[0] if isinstance(res, tuple) else res

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
            if batch_idx % self.args.log_step == 0:
                running_avg = (total_loss / max(1, count_steps)) if 'count_steps' in locals() else loss.item()
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} (running_avg: {:.6f})'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item(), running_avg))
        denom = count_steps if count_steps > 0 else self.train_per_epoch
        train_epoch_loss = total_loss/denom
        if self.args.mode == 'pretrain':
            train_epoch_flow_loss = total_flow_loss/self.train_per_epoch
            train_epoch_s_loss = total_s_loss / self.train_per_epoch
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} averaged Loss_s: {:.6f}'.format(epoch, train_epoch_flow_loss, train_epoch_s_loss))
        else:
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} (per-batch mean)'.format(epoch, train_epoch_loss))

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
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
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
        if disable_mape:
            # Log MAE/RMSE/CORR (nanmean) with valid ratio; values already in TECU.
            corr_list = []
            for t in range(y_true.shape[1]):
                mae, _ = MAE_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                rmse = RMSE_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                corr_val, ratio = _corr_with_ratio(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                corr_list.append(corr_val)
                logger.info("[{}] Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, CORR:{:.4f} (valid={:.1f}%)".format(
                    tag, t + 1, mae, rmse, 0.0 if corr_val!=corr_val else corr_val, ratio*100))
            mae, _ = MAE_torch(y_pred, y_true, args.mae_thresh)
            rmse = RMSE_torch(y_pred, y_true, args.mae_thresh)
            corr_vals = [c for c in corr_list if not (c!=c)]
            corr_avg = sum(corr_vals)/len(corr_vals) if len(corr_vals)>0 else float('nan')
            logger.info("[{}] Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, CORR:{:.4f}".format(
                        tag, mae, rmse, 0.0 if corr_avg!=corr_avg else corr_avg))
        else:
            corr_list = []
            for t in range(y_true.shape[1]):
                mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                    args.mae_thresh, args.mape_thresh)
                corr_val, ratio = _corr_with_ratio(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                corr_list.append(corr_val)
                logger.info("[{}] Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, CORR:{:.4f} (valid={:.1f}%)".format(
                    tag, t + 1, mae, rmse, mape*100, 0.0 if corr_val!=corr_val else corr_val, ratio*100))
            mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
            corr_vals = [c for c in corr_list if not (c!=c)]
            corr_avg = sum(corr_vals)/len(corr_vals) if len(corr_vals)>0 else float('nan')
            logger.info("[{}] Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
                        tag, mae, rmse, mape*100, 0.0 if corr_avg!=corr_avg else corr_avg))

        # optionally save arrays for reproduction (float16 to reduce size)
        if save_arrays and log_filename:
            try:
                import numpy as np, os
                base_stem = os.path.splitext(log_filename)[0]
                np.save(os.path.join(args.log_dir, f"{base_stem}_{save_tag}_preds.npy"),
                        y_pred.detach().cpu().to(torch.float16).numpy())
                # Do not save ground-truth arrays by default since they are fully deterministic
                # and can be regenerated from the dataset split/windowing.
                logger.info(f"Saved arrays: {base_stem}_{save_tag}_preds.npy (true not saved)")
            except Exception as e:
                logger.warning(f"Failed to save arrays: {e}")
