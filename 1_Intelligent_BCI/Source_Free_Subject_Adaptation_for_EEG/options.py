import argparse
import os

def parse_args():
    descript = 'Pytorch Implementation of \'Source-free Subject Adaptation for EEG-based Visual Recognition\''
    parser = argparse.ArgumentParser(description=descript)

    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--model_path', type=str, default='./models/source-free_mmd')
    parser.add_argument('--output_path', type=str, default='./outputs/source-free_mmd')
    parser.add_argument('--data_file', type=str, default='eeg_signals_raw_with_mean_std.pth')
    parser.add_argument('--split_file', type=str, default='data_split.pth')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--ckpt_path_source', type=str, default=None)
    parser.add_argument('--ckpt_path_g', type=str, default=None)
    parser.add_argument('--lr', type=str, default='[0.001]*200')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--target_subject', type=int, default=5)
    parser.add_argument('--mmd', action='store_true')
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
