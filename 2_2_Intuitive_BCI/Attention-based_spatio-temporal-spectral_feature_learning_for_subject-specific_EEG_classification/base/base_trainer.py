import os
import numpy as np
import torch
from torchinfo import summary

from utils.calculator import Calculator
from utils.utils import write_json, print_dict

torch.set_printoptions(linewidth=1000)


class BaseTrainer:
    def __init__(self, args, model, data, criterion, optimizer, scheduler, history):
        self.args = args
        self.model = model
        self.data = data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = history
        self.calculator = Calculator()

    def train(self):
        print("[Start Train]")
        self.save_options(train_phase='start')
        prev_epoch, total_epoch = self.set_epoch()
        for epoch in range(prev_epoch, total_epoch + 1):
            print(f"[{str(epoch).rjust(len(str(total_epoch)))}/{total_epoch}]", end=' ')
            self.train_epoch()

        print("[End Train]")
        self.save_options(train_phase='end')
        self.print_result()

    def train_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def set_epoch(self):
        prev_epoch = 1
        total_epoch = self.args.epochs
        return prev_epoch, total_epoch

    def save_options(self, train_phase='end'):
        self.args.seed_save = torch.initial_seed()
        self.args.cuda_seed_save = torch.cuda.initial_seed()
        if train_phase == 'end':
            self.args.train_acc = np.round(self.history["train_acc"][-1], 4)
            self.args.val_acc = np.round(self.history["val_acc"][-1], 4)
        write_json(os.path.join(self.args.save_path, "args.json"), vars(self.args))

    def model_summary(self, args, model):
        if args.summary:
            summary(model, args.cfg.input_shape, col_names=["kernel_size", "output_size", "num_params"],
                    device=model.device if not model.device == 'multi' else torch.device("cuda:0"))
            print("")

    def print_result(self):
        self.model_summary(self.args, self.model)
        print_dict(vars(self.args))
        print(f"Last checkpoint: {os.path.join(self.args.save_path, 'checkpoint', str(self.args.epochs) + '.tar')}")
