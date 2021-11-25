import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

from utils.utils import read_json, print_dict

np.set_printoptions(linewidth=np.inf)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 400)


class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def make_acc_table(self, target: dict, phase='test', digit=4, fmt='percentage') -> pd.DataFrame:
        table = pd.DataFrame(columns=[f"S{i:02}" for i in range(1, 10)])
        table.columns.name = 'Method'
        get_acc = lambda options: options[f"{phase}_acc"]
        # Make table
        for sub_dir, method in target.items():
            acc_tmp = []
            for subject in range(1, 10):
                try:
                    path = f"{self.save_dir}/{sub_dir}/{subject}"
                    options = read_json(f"{path}/args.json")
                    acc_tmp.append(get_acc(options))
                except Exception:
                    acc_tmp.append(0)
            table.loc[method] = np.array(acc_tmp)
        # Add mean column
        table['Mean'] = table.mean(axis=1)
        # Fit digit
        table = table.apply(lambda x: round(x, digit))
        # Formatting
        if fmt == 'percentage':
            table = table.apply(lambda x: x * 100)
        # Sub-dir column
        table['Sub-dir'] = target.keys()
        return table

    def get_options(self, sub_dir, subject=1, target_args='all'):
        path = f"{self.save_dir}/{sub_dir}/{subject}"
        options = read_json(f"{path}/args.json")
        if not target_args == 'all':
            options = {key: options[key] for key in target_args}
        return options

    @staticmethod
    def plot_slider(save_path, **kwargs):
        save_path_lst = [os.path.join(save_path, str(subject)) for subject in range(1, 10)]
        slider = widgets.IntSlider(min=1, max=9, step=1, value=1)
        interact(
            lambda idx: Logger.plot(save_path_lst[idx - 1], **kwargs),
            idx=slider)

    @staticmethod
    def plot(save_path: str, phase=["train", "val", "test"],
             title: str = " ", figsize: tuple = (15, 5), last_point=True):
        log_dict = read_json(os.path.join(save_path, "history.json"))
        phase = list(filter(lambda p: len(log_dict[f"{p}_loss"]) > 0, phase))
        metrics = sorted(list(set(map(lambda x: x.split('_')[-1], log_dict.keys()))))
        fig, axes = plt.subplots(ncols=len(metrics), figsize=figsize)
        # Draw result plot
        for col, metric in enumerate(metrics):
            for p in phase:
                axes[col].plot(log_dict[f"{p}_{metric}"], label=f"{p}")
            axes[col].set_xlabel("epoch", fontsize=15)
            axes[col].set_ylabel(f"{metric}", fontsize=15)
            axes[col].legend()
        fig.suptitle(f"{title}", fontsize=20)
        fig.show()
        if last_point:
            for p in phase:
                print(f'{p}_acc: {log_dict[f"{p}_acc"][-1]:0.4f}')
