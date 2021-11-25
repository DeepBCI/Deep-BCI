import os
import sys
import argparse
import datetime
import torch

from utils.utils import AttrDict, make_dir, print_dict
from utils.utils import str2list, str2list_int, band_list
from utils.utils import read_json, read_yaml


class Args:
    def __init__(self):
        self.args = self.make_args()
        if self.args.mode == 'test':
            keys = ['mode', 'load_path', 'device', 'all_subject']
            self.make_test_args(keys)
        self.set_target_subject()

    def make_args(self):
        parser = argparse.ArgumentParser()

        # Time
        now = datetime.datetime.now()
        parser.add_argument('--date', default=now.strftime('%Y-%m-%d'), help="Please do not enter any value.")
        parser.add_argument('--time', default=now.strftime('%H:%M:%S'), help="Please do not enter any value.")

        # Mode
        parser.add_argument('--mode', default="train", choices=['train', 'test'])
        parser.add_argument('--all_subject', action='store_true')

        # Model
        parser.add_argument('--model', help='model name')

        # Dataset
        parser.add_argument('--subject', type=int, default=1)
        parser.add_argument('--save_dir')
        parser.add_argument('--band', type=band_list, help="Range of bandpass filtering")
        parser.add_argument('--labels', default='all', type=str2list_int, help="Select classes")
        parser.add_argument('--start_time', type=float)
        parser.add_argument('--end_time', type=float)
        parser.add_argument('--window_size', type=int)
        parser.add_argument('--step', type=int)
        parser.add_argument('--verbose', action='store_true', help="On/Off of bandpass filtering log")

        # Train
        parser.add_argument('--criterion', default='CEE', help="Please enter loss function you want to use.")
        parser.add_argument('--opt', default='Adam', help="Please enter optimizer you want to use.")
        parser.add_argument('--metrics', default='loss,acc', type=str2list, help="Please connect it with a comma.")
        parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-04)
        parser.add_argument('--weight_decay', '-wd', dest='wd', type=float, default=0)
        parser.add_argument('--epochs', type=int, default=400)
        parser.add_argument('--batch_size', type=int, default=144)
        parser.add_argument('--scheduler', '-sch')
        if parser.parse_known_args()[0].scheduler == 'exp':
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'step':
            parser.add_argument('--step_size', type=int, required=True)
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'multi_step':
            parser.add_argument('--milestones', type=str2list_int, required=True)
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'plateau':
            parser.add_argument('--factor', type=float, required=True)
            parser.add_argument('--patience', type=int, required=True)
        elif parser.parse_known_args()[0].scheduler == 'cosine':
            parser.add_argument('--T_max', type=float, help='Max iteration number')
            parser.add_argument('--eta_min', type=float, help='minimum learning rate')

        # Test
        parser.add_argument('--load_path')

        # Miscellaneous
        parser.add_argument('--device', default=0, help="cpu or gpu number")
        parser.add_argument('--seed', type=int)
        parser.add_argument('--summary', action='store_true')

        # Parsing
        return parser.parse_args()

    def make_test_args(self, keys):
        args = {key: getattr(self.args, key) for key in keys}
        self.args = AttrDict(read_json(os.path.join(self.args.load_path, 'args.json')))
        self.args.update(args)
        self.args.cfg = AttrDict(self.args.cfg)

    def set_target_subject(self):
        # Single subject
        if not self.args.all_subject:
            self.args.target_subject = [self.args.subject]
        # All subject
        else:
            self.args.target_subject = list(range(1, 10))

    def preprocess(self):
        if self.args.mode == 'train':
            self.init_args()
            self.set_save_path()
            self.set_model_config()

    def set_save_path(self):
        # Make directory
        make_dir(f"./result/{self.args.save_dir}")
        # Set save path
        if not hasattr(self.args, 'save_path'):
            sub_dir = len(os.listdir(f"./result/{self.args.save_dir}"))
            self.args.save_path = f"./result/{self.args.save_dir}/{sub_dir}/{self.args.subject}"
        else:
            self.args.save_path = os.path.join(os.path.dirname(self.args.save_path), str(self.args.subject))
        # Make save directory
        make_dir(self.args.save_path)

    def set_model_config(self):
        self.args.cfg = AttrDict(read_yaml(f'configs/{self.args.model}_config.yaml'))

    def init_args(self):
        self.args.train_acc = 0.0
        self.args.val_acc = 0.0

    def print_info(self):
        print("")
        print("START".center(99, '='))
        print(f"PID: {os.getpid()}")
        print(f"Python version: {sys.version.split(' ')[0]}")
        print(f"Pytorch version: {torch.__version__}")
        print("")
        print_dict(vars(self.args))