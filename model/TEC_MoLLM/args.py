import configparser


def parse_args(DATASET, parser):
    """
    TEC_MoLLM 预测器参数。
    - 遵循现有模式：从 conf/TEC_MoLLM/<DATASET>.conf 读取；找不到则给出合理默认。
    - 仅向下游传递必要字段，其他训练公共项由 lib/Params_predictor.py 追加。
    """
    cfg = configparser.ConfigParser()
    cfg.read(f"../conf/TEC_MoLLM/{DATASET}.conf")

    # data
    parser.add_argument('--num_nodes', type=int, default=int(cfg.get('data', 'num_nodes', fallback='5183')))
    parser.add_argument('--input_window', type=int, default=int(cfg.get('data', 'input_window', fallback='12')))
    parser.add_argument('--output_window', type=int, default=int(cfg.get('data', 'output_window', fallback='12')))

    # model
    parser.add_argument('--heads', type=int, default=int(cfg.get('model', 'heads', fallback='2')))
    parser.add_argument('--spatial_out', type=int, default=int(cfg.get('model', 'spatial_out', fallback='16')))
    parser.add_argument('--temporal_channels', type=str, default=cfg.get('model', 'temporal_channels', fallback='(64,128)'))
    parser.add_argument('--temporal_strides', type=str, default=cfg.get('model', 'temporal_strides', fallback='(2,2)'))
    parser.add_argument('--patch_len', type=int, default=int(cfg.get('model', 'patch_len', fallback='4')))
    parser.add_argument('--d_llm', type=int, default=int(cfg.get('model', 'd_llm', fallback='768')))
    parser.add_argument('--llm_layers', type=int, default=int(cfg.get('model', 'llm_layers', fallback='3')))
    parser.add_argument('--node_chunk', type=int, default=int(cfg.get('model', 'node_chunk', fallback='512')))
    parser.add_argument('--graph_tag', type=str, default=cfg.get('model', 'graph_tag', fallback='grid8'))
    # light spatio-temporal embeddings
    parser.add_argument('--use_node_tod_emb', type=eval, default=cfg.get('model', 'use_node_tod_emb', fallback='True'))
    parser.add_argument('--d_e', type=int, default=int(cfg.get('model', 'd_e', fallback='16')))
    parser.add_argument('--tod_bins', type=int, default=int(cfg.get('model', 'tod_bins', fallback='12')))
    # LoRA (optional)
    parser.add_argument('--use_lora', type=eval, default=cfg.get('model', 'use_lora', fallback='False'))
    parser.add_argument('--lora_r', type=int, default=int(cfg.get('model', 'lora_r', fallback='32')))
    parser.add_argument('--lora_alpha', type=int, default=int(cfg.get('model', 'lora_alpha', fallback='64')))
    parser.add_argument('--lora_targets', type=str, default=cfg.get('model', 'lora_targets', fallback='c_attn'))
    # dropout after LLM
    parser.add_argument('--dropout_after_llm', type=float, default=float(cfg.get('model', 'dropout_after_llm', fallback='0.1')))
    # fusion & HF
    parser.add_argument('--use_ln', type=eval, default=cfg.get('model', 'use_ln', fallback='False'))
    parser.add_argument('--hf_model_name', type=str, default=cfg.get('model', 'hf_model_name', fallback='gpt2-large'))
    parser.add_argument('--hf_cache_dir', type=str, default=cfg.get('model', 'hf_cache_dir', fallback='/root/autodl-tmp/cache'))

    # train (仅特定开关)
    parser.add_argument('--amp', type=eval, default=cfg.get('train', 'amp', fallback='True'))
    parser.add_argument('--scheduler', type=str, default=cfg.get('train', 'scheduler', fallback='plateau'))
    parser.add_argument('--plateau_factor', type=float, default=float(cfg.get('train', 'plateau_factor', fallback='0.5')))
    parser.add_argument('--plateau_patience', type=int, default=int(cfg.get('train', 'plateau_patience', fallback='2')))
    parser.add_argument('--plateau_threshold', type=float, default=float(cfg.get('train', 'plateau_threshold', fallback='0.001')))
    parser.add_argument('--plateau_threshold_mode', type=str, default=cfg.get('train', 'plateau_threshold_mode', fallback='rel'))
    parser.add_argument('--plateau_cooldown', type=int, default=int(cfg.get('train', 'plateau_cooldown', fallback='0')))
    parser.add_argument('--min_lr', type=float, default=float(cfg.get('train', 'min_lr', fallback='1e-5')))

    args, _ = parser.parse_known_args()
    return args
