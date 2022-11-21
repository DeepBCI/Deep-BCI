import mne
import torch
import scipy
from collections import Iterable
from torch.utils.data.dataset import random_split
from random import seed as py_seed
from numpy.random import seed as np_seed
from torch import manual_seed
# from torch.cuda import manual_seed_all as gpu_seed
import numpy as np
import random
# for control the random seed
random_seed = 2022
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# From: https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothedCrossEntropyLoss(torch.nn.Module):
    """this loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # dim, index, src
            # writes all values from the tensor src into self at the indices index tensor
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def min_max_normalize(x: torch.Tensor, low=-1, high=1): # input값을 -1~1
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x # + 1e-9 # TODO 220927
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias ; 0~1 -> -0.5 ~ 0.5
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x


class DN3ConfigException(BaseException):
    """
    Exception to be triggered when DN3-configuration parsing fails.
    """
    pass


class DN3atasetException(BaseException):
    """
    Exception to be triggered when DN3-dataset-specific issues arise.
    """
    pass


class DN3atasetNanFound(BaseException):
    """
    Exception to be triggered when DN3-dataset variants load NaN data, or data becomes NaN when pushed through
    transforms.
    """
    pass



def rand_split(dataset, frac=0.75):
    if frac >= 1:
        return dataset
    samples = len(dataset)
    return random_split(dataset, lengths=[round(x) for x in [samples*frac, samples*(1-frac)]])


def unfurl(_set: set):
    _list = list(_set)
    for i in range(len(_list)):
        if not isinstance(_list[i], Iterable):
            _list[i] = [_list[i]]
    return tuple(x for z in _list for x in z)

def merging_events(eegfile, new_id, timing = [1, 2], id = 6, replace_events = False): # TODO : timing = [4, 5] for state time window, timing = [1, 2] for action
    events_from_annotation, event_dict = mne.events_from_annotations(eegfile)
    dd = mne.merge_events(events_from_annotation, timing, id, replace_events=replace_events)
    where = []
    # new_id = [6, 7, 6, 6]
    for i in dd:
        if i[2] == 6:
            where.append(i[0])

    chn = [0 for _ in range(len(where))]  # np.zeros(len(where))
    new_for_merge = []
    for pair in zip(where, chn, new_id):
        # print(pair)
        new_for_merge.append(pair)

    # epoch_data = mne.Epochs(eegfile, new_for_merge)

    return new_for_merge # event


def make_epochs_from_raw(raw: mne.io.Raw, tmin, tlen, event_ids=None, baseline=None, decim=1, filter_bp=None,
                         drop_bad=False, use_annotations=False, chunk_duration=None, name = None):
    sfreq = raw.info['sfreq']
    if filter_bp is not None:
        if isinstance(filter_bp, (list, tuple)) and len(filter_bp) == 2:
            raw.load_data()
            raw.filter(filter_bp[0], filter_bp[1])
        else:
            print('Filter must be provided as a two-element list [low, high]')

    try:
        if use_annotations and name == 'SPE':
            sess_i = raw.filenames[0].split('\\')[-2] # sess1
            subj_i = raw.filenames[0].split('\\')[-1][4:7] #subj001 -> 001
            # filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/median_spe/subj_' + str(subj_i) + '/' + sess_i + '_spe_label.mat'# SPE_label.mat'
            # filepath = './EEG_preprocessed_mh/median_spe/subj_' + str(
            #     subj_i) + '/' + sess_i + '_spe_label.mat'
            # filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/median_spe_1_per_subj/subj_' + str(
            #     subj_i) + '/' + sess_i + '_spe_label.mat'  # SPE_label.mat'
            filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/median_spe_per_subj/subj_' + str(
                subj_i) + '/' + sess_i + '_spe_label.mat'  # SPE_label.mat'

            data_mat = scipy.io.loadmat(filepath)
            new_ids = data_mat.get('spe_label')[0]
            events = merging_events(raw, new_ids, timing = [1, 2]) # [1])
            # events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]

            # # for delete R time # 221004 masking try.... ㅎ
            # events_from_annotation_, event_dict_ = mne.events_from_annotations(raw)
            # where_r = []
            # where_s1 = []
            # for i in events_from_annotation_:
            #     if i[2] == event_dict_['rewd']:# ==3
            #         where_r.append(i[0]) # 맨 마지막꺼 빼고
            #     elif i[2] == event_dict_['sta1']: # ==4
            #         where_s1.append(i[0]) # 맨 첫번째꺼 빼고
            # # r~s1 시간 0으로 masking
            # where_r_post = where_r[:-1]
            # where_s1_post = where_s1[1:]
            # for start_i in range(len(where_r_post)) :
            #     raw._data[:, where_r_post[start_i]:where_s1_post[start_i]] = 0
            # # raw._data = 0  # 64 by 396359 == channel by time_point

        elif use_annotations and name == 'RPE':
            sess_i = raw.filenames[0].split('\\')[-2]  # sess1
            subj_i = raw.filenames[0].split('\\')[-1][4:7]  # subj001 -> 001
            # filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/median_rpe/subj_' + str(
            #     subj_i) + '/' + sess_i + '_rpe_label.mat'
            # filepath = './EEG_preprocessed_mh/median_rpe/subj_' + str(
            #     subj_i) + '/' + sess_i + '_rpe_label.mat'
            # filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/median_rpe_1_per_subj/subj_' + str(
            #     subj_i) + '/' + sess_i + '_rpe_label.mat'  # RPE_label.mat'
            filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/median_rpe_per_subj/subj_' + str(
                subj_i) + '/' + sess_i + '_rpe_label.mat'  # RPE_label.mat'
            data_mat = scipy.io.loadmat(filepath)
            new_ids = data_mat.get('rpe_label')[0]
            events = merging_events(raw, new_ids, timing = [1, 2]) # [2])
            # events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]
        elif use_annotations and name == 'mbmf':
            sess_i = raw.filenames[0].split('\\')[-2]  # sess1
            subj_i = raw.filenames[0].split('\\')[-1][5:7]  # subj001 -> 001
            filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/active_model/subj_' + str(
                subj_i) + '/' + sess_i + '_active_model.mat'
            # filepath = './EEG_preprocessed_mh/active_model/subj_' + str(
            #     subj_i) + '/' + sess_i + '_active_model.mat'
            data_mat = scipy.io.loadmat(filepath)
            new_ids = data_mat.get('active_model')[0]
            events = merging_events(raw, new_ids)
            # events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]
        elif use_annotations and name == 'behavior':
            sess_i = raw.filenames[0].split('\\')[-2]  # sess1
            subj_i = raw.filenames[0].split('\\')[-1][4:7]  # subj001 -> 001
            filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/behavior_label/subj_' + str(
                subj_i) + '/' + sess_i + '_behavior_label.mat'
            # filepath = './EEG_preprocessed_mh/behavior_label/subj_' + str(
            #     subj_i) + '/' + sess_i + '_behavior_label.mat'
            data_mat = scipy.io.loadmat(filepath)
            new_ids = data_mat.get('act_total')[0]
            events = merging_events(raw, new_ids)
            # events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]
        elif use_annotations and name == 'pmb': # 221024
            sess_i = raw.filenames[0].split('\\')[-2]  # sess1
            subj_i = raw.filenames[0].split('\\')[-1][5:7]  # subj001 -> 01
            filepath = 'Y:/Research/EEG_kdj/EEG_preprocessed_mh/p_mb/subj_' + str(
                subj_i) + '/' + sess_i + '_pmb.mat'
            # filepath = './EEG_preprocessed_mh/behavior_label/subj_' + str(
            #     subj_i) + '/' + sess_i + '_behavior_label.mat'
            data_mat = scipy.io.loadmat(filepath)
            new_ids = data_mat.get('p_mb')[0]
            events = merging_events(raw, new_ids)
        elif use_annotations:
            events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]
        elif isinstance(raw, mne.io.edf.edf.RawGDF) : # TODO
            events = mne.events_from_annotations(raw)[0]

        else:
            events = mne.find_events(raw)
            events = events[[i for i in range(len(events)) if events[i, -1] in event_ids.keys()], :]
    except ValueError as e:
        raise DN3ConfigException(*e.args)


    return mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / sfreq, preload=True, decim=decim,
                      baseline=baseline, reject_by_annotation=drop_bad, event_repeated= 'error') # TODO drop... 원래는 'error'


def skip_inds_from_bad_spans(epochs: mne.Epochs, bad_spans: list): # 해당 epoch 자체를 제외..
    if bad_spans is None:
        return None

    start_times = epochs.events[:, 0] / epochs.info['sfreq']
    end_times = start_times + epochs.tmax - epochs.tmin # start time + epoch 길이

    skip_inds = list()
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        for bad_start, bad_end in bad_spans:
            if bad_start <= start < bad_end or bad_start < end <= bad_end:
                skip_inds.append(i)
                break

    return skip_inds
