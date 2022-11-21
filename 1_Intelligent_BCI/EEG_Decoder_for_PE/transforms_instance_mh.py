import warnings
import torch
import numpy as np
from typing import List
from scipy import signal
from dn3_utils_mh import min_max_normalize
from transforms_channels_mh import map_dataset_channels_deep_1010
from torch.nn.functional import interpolate

_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))
_EXTRA_CHANNELS = 5 #44 # TODO : 원래는 5

DEEP_1010_CHS_LISTING = [
    # EEG
    "NZ",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F9", *["F{}".format(n) for n in _LEFT_NUMBERS], "FZ", *["F{}".format(n) for n in _RIGHT_NUMBERS], "F10",

    "FT9", "FT7", *["FC{}".format(n) for n in _LEFT_NUMBERS[1:]], "FCZ",
    *["FC{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "FT8", "FT10",

    "T9", "T7", "T3", *["C{}".format(n) for n in _LEFT_NUMBERS[1:]], "CZ",
    *["C{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "T4", "T8", "T10",

    "TP9", "TP7", *["CP{}".format(n) for n in _LEFT_NUMBERS[1:]], "CPZ",
    *["CP{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "TP8", "TP10",

    "P9", "P7", "T5", *["P{}".format(n) for n in _LEFT_NUMBERS[1:]], "PZ",
    *["P{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "T6", "P8", "P10",

    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "IZ",
    # EOG
    "VEOGL", "VEOGR", "HEOGL", "HEOGR",

    # Ear clip references
    "A1", "A2", "REF",
    # SCALING
    "SCALE",
    # Extra
    *["EX{}".format(n) for n in range(1, _EXTRA_CHANNELS + 1)]
]

EEG_INDS = list(range(0, DEEP_1010_CHS_LISTING.index('VEOGL')))
EOG_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["VEOGL", "VEOGR", "HEOGL", "HEOGR"]]
REF_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["A1", "A2", "REF"]]
EXTRA_INDS = list(range(len(DEEP_1010_CHS_LISTING) - _EXTRA_CHANNELS, len(DEEP_1010_CHS_LISTING)))
SCALE_IND = -len(EXTRA_INDS) + len(DEEP_1010_CHS_LISTING)
_NUM_EEG_CHS = len(DEEP_1010_CHS_LISTING) - len(EOG_INDS) - len(REF_INDS) - len(EXTRA_INDS) - 1

# Not crazy about this approach..
from mne.utils._bunch import NamedInt
from mne.io.constants import FIFF
# Careful this doesn't overlap with future additions to MNE, might have to coordinate
DEEP_1010_SCALE_CH = NamedInt('DN3_DEEP1010_SCALE_CH', 3000)
DEEP_1010_EXTRA_CH = NamedInt('DN3_DEEP1010_EXTRA_CH', 3001)

DEEP_1010_CH_TYPES = ([FIFF.FIFFV_EEG_CH] * _NUM_EEG_CHS) + ([FIFF.FIFFV_EOG_CH] * len(EOG_INDS)) + \
                     ([FIFF.FIFFV_EEG_CH] * len(REF_INDS)) + [DEEP_1010_SCALE_CH] + \
                     ([DEEP_1010_EXTRA_CH] * _EXTRA_CHANNELS)

class InstanceTransform(object):

    def __init__(self, only_trial_data=True):
        """
        Trial transforms are, for the most part, simply operations that are performed on the loaded tensors when they are
        fetched via the :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution
        graph integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x):
        """
        Modifies a batch of tensors.
        Parameters
        ----------
        x : torch.Tensor, tuple
            The trial tensor, not including a batch-dimension. If initialized with `only_trial_data=False`, then this
            is a tuple of all ids, labels, etc. being propagated.
        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()

    def new_channels(self, old_channels):
        """
        This is an optional method that indicates the transformation modifies the representation and/or presence of
        channels.

        Parameters
        ----------
        old_channels : ndarray
                       An array whose last two dimensions are channel names and channel types.

        Returns
        -------
        new_channels : ndarray
                      An array with the channel names and types after this transformation. Supports the addition of
                      dimensions e.g. a list of channels into a rectangular grid, but the *final two dimensions* must
                      remain the channel names, and types respectively.
        """
        return old_channels

    def new_sfreq(self, old_sfreq):
        """
        This is an optional method that indicates the transformation modifies the sampling frequency of the underlying
        time-series.

        Parameters
        ----------
        old_sfreq : float

        Returns
        -------
        new_sfreq : float
        """
        return old_sfreq

    def new_sequence_length(self, old_sequence_length):
        """
        This is an optional method that indicates the transformation modifies the length of the acquired extracts,
        specified in number of samples.

        Parameters
        ----------
        old_sequence_length : int

        Returns
        -------
        new_sequence_length : int
        """
        return old_sequence_length

def same_channel_sets(channel_sets: list):
    """Validate that all the channel sets are consistent, return false if not"""
    for chs in channel_sets[1:]:
        if chs.shape[0] != channel_sets[0].shape[0] or chs.shape[1] != channel_sets[0].shape[1]:
            return False
        # if not np.all(channel_sets[0] == chs):
        #     return False
    return True


class To1020(InstanceTransform):

    # EEG_20_div = [
    #            'FP1', 'FP2',
    #     'F7', 'F3', 'FZ', 'F4', 'F8',
    #     'T7', 'C3', 'CZ', 'C4', 'T8',
    #     'T5', 'P3', 'PZ', 'P4', 'T6',
    #             'O1', 'O2'
    # ]

    EEG_20_div = [
                'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

    def __init__(self, only_trial_data=True, include_scale_ch=True, include_ref_chs=False):
        """
        Transforms incoming Deep1010 data into exclusively the more limited 1020 channel set.
        """
        super(To1020, self).__init__(only_trial_data=only_trial_data)
        self._inds_20_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in self.EEG_20_div]
        if include_ref_chs:
            self._inds_20_div.append([DEEP_1010_CHS_LISTING.index(ch) for ch in ['A1', 'A2']])
        if include_scale_ch:
            self._inds_20_div.append(SCALE_IND)

    def new_channels(self, old_channels):
        return old_channels[self._inds_20_div]

    def __call__(self, *x):
        x = list(x) # 90 by 768,  now : 90 by 15360 # 2번째
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][self._inds_20_div, ...]

        original_timepoint = x[0].shape[1] # 91000


        # # 1: for channel - freq 에한 time 방향
        # # (20, 13, 274)
        # f, t, Sxx = signal.spectrogram(x[0], nperseg=64, window='hamming', fs=256)
        # # f, t, Sxx = signal.spectrogram(x[0], nperseg=256, window='hamming', fs=256) # 0번째.. -> load # TODO : nperseg -> 218로 바꿔보기
        # # t: (406, ), f: (129, ), Sxx : (21, 129, 406) : (channel, freq, time) nperseg = 256
        # # t: (202, ), f: (257, ), Sxx : (21, 257, 202) : (channel, freq, time) nperseg = 512
        #
        # # pretrain :  t: (68, ), f: (129, ), Sxx : (20, 129, 68) : (channel, freq, time) nperseg = 256
        # # pretrain :  t: (137, ), f: (65, ), Sxx : (20, 65, 137) : (channel, freq, time) nperseg = 128
        # # pretrain :  t: (274, ), f: (33, ), Sxx : (20, 33, 274) : (channel, freq, time) nperseg = 64
        # e = np.where(f >= 50)
        # freq = f[0:e[0][0]]
        # tfreq = Sxx[:, 0:e[0][0], :]
        # # raw._data = tfreq.reshape(ch_n, -1) # (21, 20300)
        # x[0] = torch.from_numpy(tfreq.reshape(-1, len(t)))  # (260, 274) # TODO 221109 (260, 274)
        #
        # # 2: for freq - time 에한 channel 방향 # TODO 221117 (195, 20)
        # f, t, Sxx = signal.spectrogram(x[0], nperseg=64, window='hamming', fs=256)
        # # f, t, Sxx = signal.spectrogram(x[0], nperseg=256, window='hamming', fs=256) # 0번째.. -> load # TODO : nperseg -> 218로 바꿔보기
        # # t: (406, ), f: (129, ), Sxx : (21, 129, 406) : (channel, freq, time) nperseg = 256
        # # t: (202, ), f: (257, ), Sxx : (21, 257, 202) : (channel, freq, time) nperseg = 512
        #
        # # pretrain :  t: (68, ), f: (129, ), Sxx : (20, 129, 68) : (channel, freq, time) nperseg = 256
        # # pretrain :  t: (137, ), f: (65, ), Sxx : (20, 65, 137) : (channel, freq, time) nperseg = 128
        # # pretrain :  t: (274, ), f: (33, ), Sxx : (20, 33, 274) : (channel, freq, time) nperseg = 64
        # e = np.where(f >= 50)
        # freq = f[0:e[0][0]]
        # tfreq = Sxx[:, 0:e[0][0], :]
        #
        # chn_list = []
        # for ch_i in range(tfreq.shape[0]):
        #     chn_list.append(tfreq[0, :, :].flatten())
        #
        # b = np.array(list(zip(chn_list[0], chn_list[1],chn_list[2], chn_list[3],chn_list[4],chn_list[5],chn_list[6], chn_list[7], chn_list[8], chn_list[9],
        #                       chn_list[10], chn_list[11],chn_list[12], chn_list[13],chn_list[14],chn_list[15],chn_list[16], chn_list[17], chn_list[18], chn_list[19]))) #(195, 20)
        # x[0] = torch.from_numpy(b)
        # del b
        # del chn_list

        # 3: for channel - time 에한 freq 방향 # TODO 221117 (340, 232)
        f, t, Sxx = signal.spectrogram(x[0], nperseg=256*4, window='hamming', fs=256)
        # pretrain :  t: (68, ), f: (129, ), Sxx : (20, 129, 68) : (channel, freq, time) nperseg = 256 * 4
        # pretrain :  t: (68, ), f: (129, ), Sxx : (20, 129, 68) : (channel, freq, time) nperseg = 256
        # pretrain :  t: (137, ), f: (65, ), Sxx : (20, 65, 137) : (channel, freq, time) nperseg = 128
        # pretrain :  t: (274, ), f: (33, ), Sxx : (20, 33, 274) : (channel, freq, time) nperseg = 64
        e = np.where(f >= 110)
        freq = f[0:e[0][0]]
        tfreq = Sxx[:, 0:e[0][0], :]
        bb = []
        for ch_i in range(tfreq.shape[0]):
            for t_i in range(tfreq.shape[2]):
                bb.append(tfreq[ch_i, :, t_i])

        x[0] = torch.from_numpy(np.array(bb)) # *4-> (340, 440) or *4-> (340, 330) || (340, 232)-> 딱 20개...
        del bb

        # import matplotlib.pyplot as plt
        # plt.pcolormesh(t[750:1300], freq, tfreq[0, :, 750:1300], shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

        changed_timepoint = x[0].shape[1] # 20300
        # raw.n_times = raw.n_times * changed_tproimepoint / original_timepoint
        changed_ratio = changed_timepoint / original_timepoint

        return x # 20 by 768, 20 by 15360


class MappingDeep1010(InstanceTransform):
    """
    Maps various channel sets into the Deep10-10 scheme, and normalizes data between [-1, 1] with an additional scaling
    parameter to describe the relative scale of a trial with respect to the entire dataset.

    See https://doi.org/10.1101/2020.12.17.423197  for description.
    """
    def __init__(self, dataset, max_scale=None, return_mask=False):
        """
        Creates a Deep10-10 mapping for the provided dataset.

        Parameters
        ----------
        dataset : Dataset

        max_scale : float
                    If specified, the scale ind is filled with the relative scale of the trial with respect
                    to this, otherwise uses dataset.info.data_max - dataset.info.data_min.
        return_mask : bool
                      If `True` (`False` by default), an additional tensor is returned after this transform that
                      says which channels of the mapping are in fact in use.
        """
        super().__init__()
        self.mapping = map_dataset_channels_deep_1010(dataset.channels) # 60 by 90 -> 58 by 90
        # mapping_re = []  # TODO 221109
        # for map_i in self.mapping:
        #     for _ in range(50):
        #         mapping_re.append(map_i.tolist())
        # self.mapping_re = torch.from_numpy(np.array(mapping_re)).float()
        if max_scale is not None:
            self.max_scale = max_scale
        elif dataset.info is None or dataset.info.data_max is None or dataset.info.data_min is None:
            print(f"Warning: Did not find data scale information for {dataset}")
            self.max_scale = None
            pass
        else:
            self.max_scale = dataset.info.data_max - dataset.info.data_min
        self.return_mask = return_mask

    @staticmethod
    def channel_listing():
        return DEEP_1010_CHS_LISTING

    def __call__(self, x):
        if self.max_scale is not None: # 994000.0
            scale = 2 * (torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0) - 0.5) # -1 -> TODO : torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0)

        else:
            scale = 0
        # x = np.transpose(np.transpose(x, (2,1,0)) @ self.mapping, (2,1,0))

        # mapping_re = [] #
        # for map_i in self.mapping:
        #     for _ in range(50):
        #         mapping_re.append(map_i)
        # mapping_re = torch.from_numpy(np.array(mapping_re))
        # x = (x.transpose(1, 0) @ self.mapping_re).transpose(1, 0)

        x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0) # 1... (57, 66) @ (57, 90) 1번쩨
        # x: (58, 6, 535), self.mapping : (58, 90) -> mapping_Re : (58*50, 90)

        for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
            x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :]) # 다른 곳 min_max 없애기 ㅇㅇ

        used_channel_mask = self.mapping.sum(dim=0).bool()
        x[~used_channel_mask, :] = 0 # 안쓰는 채널 -> 0

        x[SCALE_IND, :] = scale

        if self.return_mask:
            return (x, used_channel_mask)
        else:
            return x

    def new_channels(self, old_channels: np.ndarray):
        channels = list()
        for row in range(self.mapping.shape[1]):
            active = self.mapping[:, row].nonzero().numpy()
            if len(active) > 0:
                channels.append("-".join([old_channels[i.item(), 0] for i in active]))
            else:
                channels.append(None)
        return np.array(list(zip(channels, DEEP_1010_CH_TYPES)))


class TemporalInterpolation(InstanceTransform):

    def __init__(self, desired_sequence_length, mode='nearest', new_sfreq=None):
        """
        This is in essence a DN3 wrapper for the pytorch function
        `interpolate() <https://pytorch.org/docs/stable/nn.functional.html>`_

        Currently only supports single dimensional samples (i.e. channels have not been projected into more dimensions)

        Warnings
        --------
        Using this function to downsample data below a suitable nyquist frequency given the low-pass filtering of the
        original data will cause dangerous aliasing artifacts that will heavily affect data quality to the point of it
        being mostly garbage.

        Parameters
        ----------
        desired_sequence_length: int
                                 The desired new sequence length of incoming data.
        mode: str
              The technique that will be used for upsampling data, by default 'nearest' interpolation. Other options
              are listed under pytorch's interpolate function.
        new_sfreq: float, None
                   If specified, registers the change in sampling frequency
        """
        super().__init__()
        self._new_sequence_length = desired_sequence_length
        self.mode = mode
        self._new_sfreq = new_sfreq

    def __call__(self, x):
        # squeeze and unsqueeze because these are done before batching
        if len(x.shape) == 2: # 90 by 3000 -> # 90 by 66
            return interpolate(x.unsqueeze(0), self._new_sequence_length, mode=self.mode).squeeze(0) # 90 by 768
        # Supports batch dimension
        elif len(x.shape) == 3:
            return interpolate(x, self._new_sequence_length, mode=self.mode) # 90 by 6 by 535 -> 90 by 6 by 15360
        else:
            raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

    def new_sequence_length(self, old_sequence_length):
        return self._new_sequence_length

    def new_sfreq(self, old_sfreq):
        if self._new_sfreq is not None:
            return self._new_sfreq
        else:
            return old_sfreq
