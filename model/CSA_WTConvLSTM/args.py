import configparser


def parse_args(DATASET, parser):
    config_file = '../conf/CSA_WTConvLSTM/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--num_nodes', type=int, default=config['data'].getint('num_nodes'))
    parser.add_argument('--input_window', type=int, default=config['data'].getint('input_window'))
    parser.add_argument('--output_window', type=int, default=config['data'].getint('output_window'))
    # override normalizer and feature dims to mimic original CSA pipeline
    parser.add_argument('--normalizer', type=str, default=config['data'].get('normalizer', fallback='tec01'))
    parser.add_argument('--input_base_dim', type=int, default=config['data'].getint('input_base_dim', fallback=1))
    parser.add_argument('--input_extra_dim', type=int, default=config['data'].getint('input_extra_dim', fallback=0))

    # grid info (height/width maybe inferred from dataset plugin; keep as fallback here)
    parser.add_argument('--height', type=int, default=config['model'].getint('height'))
    parser.add_argument('--width', type=int, default=config['model'].getint('width'))

    # model specific
    parser.add_argument('--CSA_hidden_dim', type=list, default=eval(config['model']['CSA_hidden_dim']))
    parser.add_argument('--CSA_num_layers', type=int, default=config['model'].getint('CSA_num_layers'))
    parser.add_argument('--WTConvLSTM_hidden_dim', type=list, default=eval(config['model']['WTConvLSTM_hidden_dim']))
    parser.add_argument('--WT_num_layers', type=int, default=config['model'].getint('WT_num_layers'))
    parser.add_argument('--kernel_size', type=int, default=config['model'].getint('kernel_size'))
    parser.add_argument('--CA_bool', type=eval, default=config['model'].get('CA_bool'))
    parser.add_argument('--SA_bool', type=eval, default=config['model'].get('SA_bool'))
    parser.add_argument('--channel_second', type=eval, default=config['model'].get('channel_second'))

    # train
    parser.add_argument('--seed', type=int, default=config['train'].getint('seed'))
    parser.add_argument('--seed_mode', type=eval, default=config['train'].get('seed_mode'))
    parser.add_argument('--xavier', type=eval, default=config['train'].get('xavier'))
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])

    args, _ = parser.parse_known_args()
    return args
