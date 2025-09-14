import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics, MAE_torch, RMSE_torch, CORR_torch

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
        # Always save best model to a fixed file name for consistency with Run.py test mode
        # Choose best model file name (pretrain: save named file + create symlink latest_pretrain.pth)
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if getattr(args, 'mode', '') == 'pretrain':
            ds = getattr(args, 'dataset', 'DATASET')
            arch = getattr(args, 'arch_tag', 'gptst')
            tgt = getattr(args, 'target_model', 'generic')
            gtag = getattr(args, 'graph_tag', 'na')
            # For pretrain, we will save only named file based on log filename; self.best_path points to that path
            base_stem = f"{ts}_{ds}_{arch}_pretrain_{tgt}_{gtag}"
            self.best_path = os.path.join(self.args.log_dir, f"{base_stem}.pth")
        else:
            self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
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

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        # val_pred = []
        # val_true = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(self.args.device, non_blocking=True)
                target = target.to(self.args.device, non_blocking=True)
                data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                if self.args.mode == 'pretrain':
                    label = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
                else:
                    label = target[..., :self.args.input_base_dim + self.args.input_extra_dim]
                output, _, mask, _, _ = self.model(data, label=None)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label[..., :self.args.output_dim])
                if self.args.mode == 'pretrain':
                    loss, loss_base = self.loss(output, label[..., :self.args.output_dim], mask)
                else:
                    res = self.loss(output, label[..., :self.args.output_dim])
                    loss = res[0] if isinstance(res, tuple) else res
                # accumulate only finite losses
                if torch.isfinite(loss).item():
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_flow_loss = 0
        total_s_loss = 0
        use_amp = getattr(self.args, 'amp', False)
        accum = max(1, int(getattr(self.args, 'accumulate_steps', 1)))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out, out_time, mask, probability, eb = self.model(data, label, self.batch_seen, epoch)
                    loss_flow, loss_base = self.loss(out, label[..., :self.args.output_dim], mask)
                    if epoch > self.args.change_epoch:
                        loss_s = self.loss_kl(probability.log(), eb) * 0.1
                        loss = loss_flow + loss_s
                    else:
                        loss = loss_flow
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out, out_time, mask, probability, eb2 = self.model(data, label, self.batch_seen)
                    res = self.loss(out, label[..., :self.args.output_dim])
                    # loss functions may return (mean_loss, elementwise) tuples (e.g., mask_mae)
                    loss = res[0] if isinstance(res, tuple) else res

            loss_to_back = loss / accum
            if use_amp:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            # add max grad clipping
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            elif self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == self.train_per_epoch:
                if use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
            total_loss += loss.item()
            # calculate total loss
            if self.args.mode == 'pretrain':
                total_flow_loss += loss_flow.item()
                if epoch > self.args.change_epoch:
                    total_s_loss += loss_s.item()
            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        if self.args.mode == 'pretrain':
            train_epoch_flow_loss = total_flow_loss/self.train_per_epoch
            train_epoch_s_loss = total_s_loss / self.train_per_epoch
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} averaged Loss_s: {:.6f}'.format(epoch, train_epoch_flow_loss, train_epoch_s_loss))
        else:
            self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

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
                    torch.save(target_state, self.best_path)
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
                    import json, os
                    # derive json filename from log filename to keep naming consistent
                    base_stem = os.path.splitext(self.log_filename)[0]
                    json_name = f"{base_stem}.json"
                    out_path = os.path.join(self.args.log_dir, json_name)
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
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

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

        if disable_mape:
            # Log MAE/RMSE/CORR only (align with original CSA repo). Values already in TECU.
            for t in range(y_true.shape[1]):
                mae, _ = MAE_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                rmse = RMSE_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                corr = CORR_torch(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh)
                logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, CORR:{:.4f}%".format(
                    t + 1, mae, rmse, corr))
            mae, _ = MAE_torch(y_pred, y_true, args.mae_thresh)
            rmse = RMSE_torch(y_pred, y_true, args.mae_thresh)
            corr = CORR_torch(y_pred, y_true, args.mae_thresh)
            logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, CORR:{:.4f}".format(mae, rmse, corr))
        else:
            for t in range(y_true.shape[1]):
                mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                       args.mae_thresh, args.mape_thresh)
                logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, CORR:{:.4f}%".format(
                    t + 1, mae, rmse, mape*100, corr))
            mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
            logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
                        mae, rmse, mape*100, corr))

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
