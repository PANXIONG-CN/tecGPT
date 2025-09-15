
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn as nn
import configparser
from model.Pretrain_model.GPTST import GPTST_Model as Network_Pretrain
from model.Model import Enhance_model as Network_Predict
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.Params_pretrain import parse_args
from lib.Params_predictor import get_predictor_params

# *************************************************************************#

# Mode = 'ori'            #ori, eval, pretrain
# DATASET = 'PEMS08'      #PEMS08, METR_LA, NYC_BIKE, NYC_TAXI
# model = 'MSDR'     # ASTGCN CCRNN DMVSTNET GWN MSDR MTGNN STWA STFGNN STGCN STGODE STMGCN STSGCN TGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

args = parse_args(device)
if args.mode !='pretrain':
    args_predictor = get_predictor_params(args)
    # Minimal override rule: if user explicitly passed certain single-hyphen flags
    # on the CLI, do NOT let predictor args overwrite them.
    cli_tokens = set(sys.argv[1:])
    protected_map = {
        '-epochs': 'epochs',
        '-batch_size': 'batch_size',
        '-val_ratio': 'val_ratio',
        '-test_ratio': 'test_ratio',
        '-amp': 'amp',
    }
    protected_keys = {v for k, v in protected_map.items() if k in cli_tokens}
    attr_list = []
    for arg in vars(args):
        attr_list.append(arg)
    for attr in attr_list:
        if hasattr(args, attr) and hasattr(args_predictor, attr):
            if attr in protected_keys:
                # keep user-specified main arg value
                continue
            setattr(args, attr, getattr(args_predictor, attr))
else:
    args_predictor = None

# print effective arguments
for arg in vars(args):
    print(arg, ':', getattr(args, arg))
print('==========')
if args_predictor is not None:
    for arg in vars(args_predictor):
        print(arg, ':', getattr(args_predictor, arg))
init_seed(args.seed, args.seed_mode)

model_label = 'gptst' if args.mode == 'pretrain' else args.model
print('mode: ', args.mode, '  model: ', model_label, '  dataset: ', args.dataset, '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)


# config log path
# - pretrain outputs are predictor-agnostic → save under Output/<DATASET>/pretrain
# - others keep Output/<DATASET>/<MODEL>
repo_root = file_dir
if args.mode == 'pretrain':
    # use explicit arch tag rather than overriding model name
    args.arch_tag = 'gptst'
    log_dir = os.path.join(repo_root, 'Output', args.dataset, 'pretrain')
else:
    log_dir = os.path.join(repo_root, 'Output', args.dataset, args.model)
Mkdir(log_dir)
args.log_dir = log_dir
args.load_pretrain_path = args.load_pretrain_path
args.save_pretrain_path = args.save_pretrain_path

# record command line for logging later
args.cmdline = f"python {os.path.basename(__file__)} " + " ".join(sys.argv[1:])

#load dataset
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, scaler_holiday = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
args.scaler_zeros = scaler_data.transform(0)
args.scaler_zeros_day = scaler_day.transform(0)
args.scaler_zeros_week = scaler_week.transform(0)
# args.scaler_zeros_holiday = scaler_holiday.transform(0)
#init model
if args.mode == 'pretrain':
    model = Network_Pretrain(args)
    model = model.to(args.device)
else:
    model = Network_Predict(args, args_predictor)
    model = model.to(args.device)

if args.xavier:
    for p in model.parameters():
        if p.requires_grad==True:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

print_model_parameters(model, only_num=False)

#init loss function, optimizer

def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

def scaler_huber_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

if args.mode == 'pretrain':
    # For pretrain, always use mask-aware loss with scaler; fallback to huber if requested.
    if args.loss_func == 'mask_huber':
        loss = scaler_huber_loss(scaler_data, mask_value=None)
        print('============================scaler_huber_loss (pretrain)')
    else:
        loss = scaler_mae_loss(scaler_data, mask_value=None)
        print('============================scaler_mae_loss (pretrain)')
else:
    if args.loss_func == 'mask_mae':
        loss = scaler_mae_loss(scaler_data, mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    elif args.loss_func == 'mask_huber':
        loss = scaler_huber_loss(scaler_data, mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError
loss_kl = nn.KLDivLoss(reduction='sum').to(args.device)

# optimizer
opt_name = getattr(args, 'optimizer', 'adam').lower()
weight_decay = getattr(args, 'weight_decay', 0.0)
if opt_name == 'adamw':
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                  weight_decay=weight_decay, amsgrad=False)
elif opt_name == 'sgd':
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr_init, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
else:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                 weight_decay=weight_decay, amsgrad=False)
#learning rate decay
lr_scheduler = None
if getattr(args, 'scheduler', 'none') == 'plateau':
    print('Applying ReduceLROnPlateau scheduler.')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
        threshold_mode=args.plateau_threshold_mode,
        cooldown=args.plateau_cooldown,
        min_lr=args.min_lr,
    )
elif getattr(args, 'scheduler', 'none') == 'cosine':
    print('Applying CosineAnnealingLR scheduler.')
    t_max = args.cosine_t_max if getattr(args, 'cosine_t_max', None) else args.epochs
    eta_min = args.lr_init * getattr(args, 'eta_min_factor', 0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=eta_min)
elif args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=lr_decay_steps,
        gamma=args.lr_decay_rate)

# #config log path
# current_dir = os.path.dirname(os.path.realpath(__file__))
# log_dir = os.path.join(current_dir,'SAVE', args.dataset)
# Mkdir(log_dir)
# args.log_dir = log_dir

#start training

trainer = Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'pretrain':
    trainer.train()
elif args.mode == 'eval':
    trainer.train()
elif args.mode == 'ori':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load(log_dir + '/best_model.pth'))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler_data, trainer.logger,
                 save_arrays=True, save_tag='test', log_filename=getattr(trainer,'log_filename',None))
    # Year-wise evaluation for GIMtec + CSA_WTConvLSTM to match original repo analysis
    try:
        if args.dataset.lower() in ['gimtec', 'tec']:
            # Run year-wise log for CSA (optional): keep compatibility
            if getattr(args, 'model', None) == 'CSA_WTConvLSTM':
                from lib.eval_gimtec_yearwise import evaluate_per_year
                evaluate_per_year(model, trainer)
            # Always produce JSON for GIMtec/TEC in test mode
            from lib.eval_gimtec_yearwise import compute_yearwise_metrics
            metrics = compute_yearwise_metrics(model, trainer)
            import json, os
            base_stem = os.path.splitext(trainer.log_filename)[0] if hasattr(trainer, 'log_filename') else 'eval'
            json_name = f"{base_stem}.json"
            out_path = os.path.join(args.log_dir, json_name)
            with open(out_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print('Saved JSON metrics to', out_path)
    except Exception as e:
        print('Year-wise evaluation skipped due to error:', e)
else:
    raise ValueError
# 确认 GIMtec/TEC 的时间分辨率（每帧≈2小时）
try:
    if str(getattr(args, 'dataset', '')).lower() in ['gimtec', 'tec']:
        args.interval = 120
except Exception:
    pass
