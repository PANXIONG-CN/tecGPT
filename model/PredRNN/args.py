import argparse

def parse_args(DATASET, parser: argparse.ArgumentParser):
    parser.add_argument('--height', type=int, default=71)
    parser.add_argument('--width', type=int, default=73)
    parser.add_argument('--output_window', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--loss_func', type=str, default='mask_mae')
    args, _ = parser.parse_known_args()
    return args

