import os
from collections import defaultdict
import torch

from base.base_trainer import BaseTrainer
from utils.utils import make_dir, write_json


class Trainer(BaseTrainer):
    def __init__(self, args, model, data, criterion, optimizer=None, scheduler=None, history=None):
        super(Trainer, self).__init__(args, model, data, criterion, optimizer, scheduler, history)

    def train_epoch(self):
        history_mini_batch = defaultdict(list)
        # Train
        self.model.train()

        for i, data in enumerate(self.data.train_loader):
            # Update model per mini-batch
            inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Calculate log per mini-batch
            log = self.calculator.calculate(self.args.metrics, loss, labels, outputs, acc_count=True)

            # Record history per mini-batch
            self.record_history(log, history_mini_batch, phase='train')

        # Validation
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.data.val_loader):
                # Get output per mini-batch
                inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Calculate log per mini-batch
                log = self.calculator.calculate(self.args.metrics, loss, labels, outputs, acc_count=True)

                # Record history per mini-batch
                self.record_history(log, history_mini_batch, phase='val')

        # Write history per epoch
        self.write_history(history_mini_batch)

        # Print history
        self.print_history()

        # Save checkpoint
        self.save_checkpoint(epoch=len(self.history['train_loss']))

        # Update scheduler
        self.update_scheduler()

    def record_history(self, log, history, phase):
        for metric in log:
            history[f'{phase}_{metric}'].append(log[metric])

    def write_history(self, history):
        for metric in history:
            if metric.endswith('acc'):
                n_samples = len(getattr(self.data, f"{metric.split('_')[0]}_loader").dataset.y)
                self.history[metric].append((sum(history[metric]) / n_samples))
            else:
                self.history[metric].append(sum(history[metric]) / len(history[metric]))
        if self.args.mode == 'train':
            write_json(os.path.join(self.args.save_path, "history.json"), self.history)
        else:
            write_json(os.path.join(self.args.save_path, "history_test.json"), self.history)

    def print_history(self):
        print(f"S{self.args.subject:02}", end=' ')
        for metric in self.history:
            print(f"{metric}={self.history[metric][-1]:0.4f}", end=' ')
        if self.args.mode == 'train':
            print(f"lr={self.optimizer.state_dict()['param_groups'][0]['lr']:0.8f}", end=' ')
            print(f"wd={self.optimizer.state_dict()['param_groups'][0]['weight_decay']}", end=' ')
        print("")

    def save_checkpoint(self, epoch):
        make_dir(os.path.join(self.args.save_path, "checkpoints"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, os.path.join(self.args.save_path, f"checkpoints/{epoch}.tar"))
        if epoch >= 6:
            os.remove(os.path.join(self.args.save_path, f"checkpoints/{epoch - 5}.tar"))

    def update_scheduler(self):
        if self.scheduler:
            self.scheduler.step()

    def test(self):
        history_mini_batch = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.data.test_loader):
                # Get output per mini-batch
                inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Calculate log per mini-batch
                log = self.calculator.calculate(self.args.metrics, loss, labels, outputs, acc_count=True)

                # Record history per mini-batch
                self.record_history(log, history_mini_batch, phase='test')

        # Write test history
        self.write_history(history_mini_batch)

        # Print history
        self.print_history()