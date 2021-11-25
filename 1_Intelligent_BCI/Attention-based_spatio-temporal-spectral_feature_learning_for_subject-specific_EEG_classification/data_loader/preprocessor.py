from functools import reduce
import numpy as np
import torch

from utils.utils import one_hot_encoding, transpose_tensor


class Preprocessor:
    @staticmethod
    def label_selection(X, y, target_labels, one_hot=False):
        if target_labels == 'all':
            return X, y
        else:
            if one_hot:
                y = np.argmax(y, axis=1)

            # Select labels
            labels = ((y == label) for label in target_labels)
            idx = reduce(lambda x, y: x | y, labels)
            X = X[idx]
            y = y[idx]

            # Mapping labels
            for mapping, label in enumerate(np.unique(y)):
                y[y == label] = mapping

            if one_hot:
                y = one_hot_encoding(y)

            return X, y

    @staticmethod
    def segment_tensor(tensor, window_size, step):
        """
        Parameters
        ----------
        tensor: [..., chans, times]
        window_size
        step
        Returns: [shape[0], segment, ..., chans, times]
        -------
        """
        if type(tensor) == torch.Tensor:
            segment = []
            times = torch.arange(tensor.size()[-1])
            start = times[::step]
            end = start + window_size
            for s, e in zip(start, end):
                if e > len(times):
                    break
                segment.append(tensor[..., s:e])
            return torch.stack(segment, dim=1)
        else:
            segment = []
            times = np.arange(tensor.shape[-1])
            start = times[::step]
            end = start + window_size
            for s, e in zip(start, end):
                if e > len(times):
                    break
                segment.append(tensor[..., s:e])
            segment = transpose_tensor(np.array(segment), [0, 1])
            return segment
