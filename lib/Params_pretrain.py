import argparse
import configparser
import os

def parse_args(device):


    # parser
    args = argparse.ArgumentParser(prefix_chars='-', description='pretrain_arguments')
    args.add_argument('-dataset', default='METR_LA', type=str, required=True)
    args.add_argument('-mode', default='ori', type=str, required=True)
    args.add_argument('-device', default=device, type=str, help='indices of GPUs')
    args.add_argument('-model', default='TGCN', type=str)
    args.add_argument('-cuda', default=True, type=bool)

    args_get, _ = args.parse_known_args()

    # get configuration
    # 兼容任意工作目录：基于当前文件定位 conf 目录
    _here = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(_here, '..', 'conf', 'GPTST_pretrain', f'{args_get.dataset}.conf')
    config_file = os.path.normpath(config_file)
    config = configparser.ConfigParser()
    read_ok = config.read(config_file)
    if not read_ok or 'data' not in config:
        raise FileNotFoundError(f'配置文件不存在或损坏: {config_file}')

    # data
    args.add_argument('-val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('-test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('-lag', default=config['data']['lag'], type=int)
    args.add_argument('-horizon', default=config['data']['horizon'], type=int)
    args.add_argument('-num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('-tod', default=config['data']['tod'], type=eval)
    args.add_argument('-normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('-column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('-default_graph', default=config['data']['default_graph'], type=eval)
    # model
    args.add_argument('-input_base_dim', default=config['model']['input_base_dim'], type=int)
    args.add_argument('-input_extra_dim', default=config['model']['input_extra_dim'], type=int)
    args.add_argument('-output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('-embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('-embed_dim_spa', default=config['model']['embed_dim_spa'], type=int)
    args.add_argument('-hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('-HS', default=config['model']['HS'], type=int)
    args.add_argument('-HT', default=config['model']['HT'], type=int)
    args.add_argument('-HT_Tem', default=config['model']['HT_Tem'], type=int)
    args.add_argument('-num_route', default=config['model']['num_route'], type=int)
    args.add_argument('-mask_ratio', default=config['model']['mask_ratio'], type=float)
    args.add_argument('-ada_mask_ratio', default=config['model']['ada_mask_ratio'], type=float)
    args.add_argument('-ada_type', default=config['model']['ada_type'], type=str)
    # train
    args.add_argument('-loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('-seed', default=config['train']['seed'], type=int)
    args.add_argument('-batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('-epochs', default=config['train']['epochs'], type=int)
    args.add_argument('-lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('-lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('-lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('-lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('-early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('-early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('-change_epoch', default=config['train']['change_epoch'], type=int)
    args.add_argument('-up_epoch', default=config['train']['up_epoch'], type=str)
    args.add_argument('-grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('-max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('-debug', default=config['train']['debug'], type=eval)
    args.add_argument('-debug_max_steps', default=1000, type=int, help='limit per-segment timesteps when debug=True')
    args.add_argument('-real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')
    # optimizer/scheduler options (to support VendorCode strategy)
    args.add_argument('-optimizer', default=config['train'].get('optimizer', fallback='adam'), type=str)
    args.add_argument('-weight_decay', default=float(config['train'].get('weight_decay', fallback='0.0')), type=float)
    # scheduler options (align CSA original ReduceLROnPlateau and Vendor cosine)
    args.add_argument('-scheduler', default=config['train'].get('scheduler', fallback='none'), type=str)
    args.add_argument('-plateau_factor', default=float(config['train'].get('plateau_factor', fallback='0.9')) , type=float)
    args.add_argument('-plateau_patience', default=int(config['train'].get('plateau_patience', fallback='20')), type=int)
    args.add_argument('-plateau_threshold', default=float(config['train'].get('plateau_threshold', fallback='1e-5')), type=float)
    args.add_argument('-plateau_threshold_mode', default=config['train'].get('plateau_threshold_mode', fallback='abs'), type=str)
    args.add_argument('-plateau_cooldown', default=int(config['train'].get('plateau_cooldown', fallback='10')), type=int)
    args.add_argument('-min_lr', default=float(config['train'].get('min_lr', fallback='0.0')), type=float)
    # cosine annealing
    args.add_argument('-cosine_t_max', default=int(config['train'].get('cosine_t_max', fallback='100')), type=int)
    args.add_argument('-eta_min_factor', default=float(config['train'].get('eta_min_factor', fallback='0.01')), type=float)
    # grad clipping (VendorCode uses 1.0)
    args.add_argument('-grad_clip', default=float(config['train'].get('grad_clip', fallback='0.0')), type=float)
    # early stop with min_delta
    args.add_argument('-early_stop_min_delta', default=float(config['train'].get('early_stop_min_delta', fallback='0.0')), type=float)
    # training enhancements (optional)
    args.add_argument('-accumulate_steps', default=1, type=int)
    args.add_argument('-amp', default=False, type=eval)
    args.add_argument('-seed_mode', default=config['train']['seed_mode'], type=eval)
    args.add_argument('-xavier', default=config['train']['xavier'], type=eval)
    args.add_argument('-load_pretrain_path', default=config['train']['load_pretrain_path'], type=str)
    args.add_argument('-save_pretrain_path', default=config['train']['save_pretrain_path'], type=str)
    # resume/init from checkpoint for supervised modes
    args.add_argument('-init_from', default='', type=str, help='path to model .pth (state_dict or raw) to initialize before training')
    # pretrain-specific dataloader options for GIMtec
    args.add_argument('-stride_horizon', default=False, type=eval, help='use stride=horizon when windowing')
    args.add_argument('-prefix_boundary', default=True, type=eval, help='prepend lag frames at year boundaries')
    # enable fixed year-based split for GIMtec/TEC in supervised modes as well
    args.add_argument('-year_split', default=False, type=eval, help='use fixed year-based split (GIMtec/TEC) for supervised modes')
    # naming tags for pretrain artifacts
    args.add_argument('-target_model', default='generic', type=str, help='downstream predictor tag for naming only')
    args.add_argument('-graph_tag', default='na', type=str, help='graph/adjacency tag for naming only')
    # test
    args.add_argument('-mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('-mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    args.add_argument('-log_dir', default='./', type=str)
    args.add_argument('-log_step', default=config['log']['log_step'], type=int)
    args.add_argument('-plot', default=config['log']['plot'], type=eval)
    # evaluation json output
    args.add_argument('-save_json', default=False, type=eval)
    args.add_argument('-json_name', default='eval_results.json', type=str)
    args, _ = args.parse_known_args()
    return args
