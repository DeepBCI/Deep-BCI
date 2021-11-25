from collections import defaultdict
import numpy as np

from utils.utils import guarantee_numpy


class Calculator:
    def calculate(self, metrics, loss, y_true, y_pred, numpy=True, **kwargs):
        if numpy:
            y_true = guarantee_numpy(y_true)
            y_pred = guarantee_numpy(y_pred)

        history = defaultdict(list)
        for metric in metrics:
            history[metric] = getattr(self, f"get_{metric}")(loss=loss, y_true=y_true, y_pred=y_pred, **kwargs)
        return history

    def get_loss(self, loss, **kwargs):
        return float(loss)

    def get_acc(self, y_true, y_pred, argmax=True, acc_count=False, **kwargs):
        if argmax:
            y_pred = np.argmax(y_pred, axis=1)
        if acc_count:
            return sum(y_true == y_pred)
        else:
            return sum(y_true == y_pred) / len(y_true)
