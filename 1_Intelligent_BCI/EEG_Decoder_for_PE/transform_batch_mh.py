import torch
import numpy as np


class BatchTransform(object):

    def __init__(self, only_trial_data=True):
        """
        Batch transforms are operations that are performed on trial tensors after being accumulated into batches via the
        :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution graph
        integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x, training=False):
        """
        Modifies a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor, tuple
            A batch of trial instance tensor. If initialized with `only_trial_data=False`, then this includes batches
            of all other loaded tensors as well.
        training: bool
                  Indicates whether this is a training batch or otherwise, allowing for alternate behaviour during
                  evaluation.

        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor batch, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()


class RandomTemporalCrop(BatchTransform):

    def __init__(self, max_crop_frac=0.25, temporal_axis=1):
        """
        Uniformly crops the time-dimensions of a batch.

        Parameters
        ----------
        max_crop_frac: float
                       The is the maximum fraction to crop off of the trial.
        """
        super(RandomTemporalCrop, self).__init__(only_trial_data=True)
        assert 0 < max_crop_frac < 1
        self.max_crop_frac = max_crop_frac
        self.temporal_axis = temporal_axis

    def __call__(self, x, training=False):
        if not training:
            return x

        trial_len = x.shape[self.temporal_axis]
        crop_len = np.random.randint(int((1 - self.max_crop_frac) * trial_len), trial_len)
        offset = np.random.randint(0, trial_len - crop_len)

        return x[:, offset:offset + crop_len, ...]

