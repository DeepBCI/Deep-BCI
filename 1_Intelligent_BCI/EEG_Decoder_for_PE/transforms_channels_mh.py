import torch
import numpy as np
from collections import OrderedDict

_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))

_EXTRA_CHANNELS = 5 #44  # TODO 원래는 5

HGSN64to1020_dict_ = {'10': 'FP1', '5': 'FP2', '18': 'F7', '12': 'F3', '6' : 'FZ', '60': 'F4', '58': 'F8', '20': 'C3',
                     '50': 'C4', '24': 'T7', '52': 'T8', '29': 'TP7', '28': 'P3', '42': 'P4', '47': 'TP8', '30': 'P7', '44': 'P8',
                     '37': 'OZ', '35': 'O1', '39': 'O2'} #, '7': 'FC1', '4': 'FCZ', '54': 'FC2', '36' : 'POZ'} # 24개 -> 20 개

# HGSN64to1020_dict = {'10': 'FP1', '5': 'FP2', '18': 'F7', '12': 'F3', '6' : 'FZ', '60': 'F4', '58': 'F8', '20': 'C3',
#                      '50': 'C4', '24': 'T7', '52': 'T8', '29': 'TP9', '28': 'P3', '42': 'P4', '47': 'TP10', '30': 'P7', '44': 'P8',
#                      '37': 'OZ', '35': 'O1', '39': 'O2',
#                      '1': 'F10', '2': 'AF4', '3': 'F2', '4': 'FCZ', '7': 'FC1', '8' : 'AFZ', '9': 'F1', '11': 'AF3', '13': 'F5',
#                      '14': 'FC5', '15': 'FC3', '16': 'C1', '17': 'F9', '19': 'FT7', '21': 'CP1', '22': 'C5', '23': 'T9',
#                      '25': 'TP7', '26': 'CP5', '31': 'P3', '33': 'PO3', '34': 'PZ', '36': 'POZ', '38': 'PO4', '40': 'P4',
#                      '41': 'CP2', '46': 'CP6', '48': 'TP8', '49': 'C6', '51': 'C2', '53': 'FC4', '54': 'FC2', '55': 'T10',
#                      '56': 'FT8', '57': 'FC6', '59': 'F6', '43': 'P10', '32': 'P9', '27': 'CP5', '45': 'CP6'} # TODO

HGSN64to1020_dict = {'10': 'FP1', '5': 'FP2', '18': 'F7', '12': 'F3', '6' : 'FZ', '60': 'F4', '58': 'F8', '20': 'C3',
                     '50': 'C4', '24': 'T7', '52': 'T8', '29': 'TP9', '28': 'P3', '42': 'P4', '47': 'TP10', '30': 'P7', '44': 'P8',
                     '37': 'OZ', '35': 'O1', '39': 'O2',
                     '1': 'F10', '2': 'AF4', '3': 'F2', '4': 'FCZ', '7': 'FC1', '8' : 'AFZ', '9': 'F1', '11': 'AF3', '13': 'F5',
                     '14': 'FC5', '15': 'FC3', '16': 'C1', '17': 'F9', '19': 'FT7', '21': 'CP1', '22': 'C5', '23': 'T9',
                     '25': 'TP7', '26': 'CP5', '33': 'PO3', '34': 'PZ', '36': 'POZ', '38': 'PO4',
                     '41': 'CP2', '46': 'CP6', '48': 'TP8', '49': 'C6', '51': 'C2', '53': 'FC4', '54': 'FC2', '55': 'T10',
                     '56': 'FT8', '57': 'FC6', '59': 'F6'}

HGSN128to1020_dict = {'22': 'FP1', '9': 'FP2', '33': 'F7', '24': 'F3', '11': 'FZ', '124': 'F4', '122': 'F8',
                      '40': 'T7', '36': 'C3', '104': 'C4', '109': 'T8', '58': 'T5', '52': 'P3', '62': 'PZ',
                      '92': 'P4', '96': 'T6', '70': 'O1', '83': 'O2', '45': 'T3', '108': 'T4'} # TODO 221103

bcito1020_dict = {'1': 'FZ', '2': 'F7', '3': 'F3', '4': 'FCZ', '5': 'F4', '6': 'F8', '7': 'T7', '8': 'C3', '9': 'C1', '10': 'CZ',
                  '11': 'C2', '12': 'C4', '13': 'T8', '14': 'P7', '15': 'P3', '16': 'CPZ', '17': 'P4', '18': 'P8', '19': 'O1', '20': 'PZ', '21': 'O2',
                  '22': 'OZ'}

"""
EEG_20_div = [
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                'O1', 'O2'
    ]
  T5 -> P7, T6 -> P8  
"""
# CZ, T5, T6 - 가 메인으로 안들어가 있음.

# 27, 32, 43, 45, 61, 62, 63, 64
# PO8


# 40... - P4, 31... - P3

# + 43 - P10, 32 - P9
# + 27 - CP5, 45 - CP6


# LM - TP7, TP9 ; RM - TP8, TP10
# '7': 'FC1', '4': 'FCZ', '54': 'FC2', '36' : 'POZ'

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

def stringify_channel_mapping(original_names: list, mapping: np.ndarray): #names, deep_mapping
    result = ''
    heuristically_mapped = list()

    def match_old_new_idx(old_idx, new_idx_set: list):
        # try:
        new_names = [DEEP_1010_CHS_LISTING[i] for i in np.nonzero(mapping[old_idx, :])[0] if i in new_idx_set]
        # except:
        #     print("a")
        return ','.join(new_names)

    for inds, label in zip([list(range(0, _NUM_EEG_CHS)), EOG_INDS, REF_INDS, EXTRA_INDS],
                           ['EEG', 'EOG', 'REF', 'EXTRA']):
        result += "{} (original(new)): ".format(label)
        for idx, name in enumerate(original_names):

            news = match_old_new_idx(idx, inds)

            if len(news) > 0:
                result += '{}({}) '.format(name, news)
                if news != name.upper():
                    heuristically_mapped.append('{}({}) '.format(name, news))
        result += '\n'

    result += 'Heuristically Assigned: ' + ' '.join(heuristically_mapped)

    return result


def _deep_1010(map, names, eog, ear_ref, extra):

    for i, ch in enumerate(names):
        if ch not in eog and ch not in ear_ref and ch not in extra:
            try:
                map[i, DEEP_1010_CHS_LISTING.index(str(ch).upper())] = 1.0
            except ValueError:
                print("Warning: channel {} not found in standard layout. Skipping...".format(ch))
                continue

    # Normalize for when multiple values are mapped to single location
    summed = map.sum(axis=0)[np.newaxis, :]
    mapping = torch.from_numpy(np.divide(map, summed, out=np.zeros_like(map), where=summed != 0)).float()
    mapping.requires_grad_(False)
    return mapping


def _check_num_and_get_types(type_dict: OrderedDict):
    type_lists = list()
    for ch_type, max_num in zip(('eog', 'ref'), (len(EOG_INDS), len(REF_INDS))):
        channels = [ch_name for ch_name, _type in type_dict.items() if _type == ch_type]

        for name in channels[max_num:]:
            print("Losing assumed {} channel {} because there are too many.".format(ch_type, name))
            type_dict[name] = None
        type_lists.append(channels[:max_num])
    return type_lists[0], type_lists[1]



def _valid_character_heuristics(name, informative_characters):
    possible = ''.join(c for c in name.upper() if c in informative_characters).replace(' ', '')
    if possible == "":
        print("Could not use channel {}. Could not resolve its true label, rename first.".format(name))
        return None
    return possible



def _heuristic_eog_resolution(eog_channel_name):
    return _valid_character_heuristics(eog_channel_name, "VHEOGLR")


def _heuristic_ref_resolution(ref_channel_name: str):
    ref_channel_name = ref_channel_name.replace('EAR', '')
    ref_channel_name = ref_channel_name.replace('REF', '')
    if ref_channel_name.find('A1') != -1:
        return 'A1'
    elif ref_channel_name.find('A2') != -1:
        return 'A2'

    if ref_channel_name.find('L') != -1:
        return 'A1'
    elif ref_channel_name.find('R') != -1:
        return 'A2'
    return "REF"


def _heuristic_eeg_resolution(eeg_ch_name: str):
    eeg_ch_name = eeg_ch_name.upper()
    # remove some common garbage
    eeg_ch_name = eeg_ch_name.replace('EEG', '')
    eeg_ch_name = eeg_ch_name.replace('REF', '')
    informative_characters = set([c for name in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS] for c in name])
    return _valid_character_heuristics(eeg_ch_name, informative_characters)


def _likely_eeg_channel(name):
    if name is not None:
        for ch in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS]:
            if ch in name.upper():
                return True
    return False


def _heuristic_resolution(old_type_dict: OrderedDict):
    resolver = {'eeg': _heuristic_eeg_resolution, 'eog': _heuristic_eog_resolution, 'ref': _heuristic_ref_resolution,
                'extra': lambda x: x, None: lambda x: x}

    new_type_dict = OrderedDict()

    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue

        new_name = resolver[ch_type](old_name)
        if new_name is None:
            new_type_dict[old_name] = None
        else:
            while new_name in new_type_dict.keys():
                print('Deep1010 Heuristics resulted in duplicate entries for {}, incrementing name, but will be lost '
                      'in mapping'.format(new_name))
                new_name = new_name + '-COPY'
            new_type_dict[new_name] = old_type_dict[old_name]

    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict


def map_named_channels_deep_1010(channel_names: list, EOG=None, ear_ref=None, extra_channels=None):
    """
    Maps channel names to the Deep1010 format, will automatically map EOG and extra channels if they have been
    named according to standard convention. Otherwise provide as keyword arguments.

    Parameters
    ----------
    channel_names : list
                   List of channel names from dataset
    EOG : list, str
         Must be a single channel name, or left and right EOG channels, optionally vertical L/R then horizontal
         L/R for four channels.
    ear_ref : Optional, str, list
               One or two channels to be used as references. If two, should be left and right in that order.
    extra_channels : list, None
                     Up to 6 extra channels to include. Currently not standardized, but could include ECG, respiration,
                     EMG, etc.

    Returns
    -------
    mapping : torch.Tensor
              Mapping matrix from previous channel sequence to Deep1010.
    """
    map = np.zeros((len(channel_names), len(DEEP_1010_CHS_LISTING)))
    # map = np.zeros((len(channel_names), 20))
    if isinstance(EOG, str):
        EOG = [EOG] * 4
    elif len(EOG) == 1:
        EOG = EOG * 4
    elif EOG is None or len(EOG) == 0:
        EOG = []
    elif len(EOG) == 2:
        EOG = EOG * 2
    else:
        assert len(EOG) == 4
    for eog_map, eog_std in zip(EOG, EOG_INDS):
        try:
            map[channel_names.index(eog_map), eog_std] = 1.0
        except ValueError:
            raise ValueError("EOG channel {} not found in provided channels.".format(eog_map))

    if isinstance(ear_ref, str):
        ear_ref = [ear_ref] * 2
    elif ear_ref is None:
        ear_ref = []
    else:
        assert len(ear_ref) <= len(REF_INDS)
    for ref_map, ref_std in zip(ear_ref, REF_INDS):
        try:
            map[channel_names.index(ref_map), ref_std] = 1.0
        except ValueError:
            raise ValueError("Reference channel {} not found in provided channels.".format(ref_map))

    if isinstance(extra_channels, str):
        extra_channels = [extra_channels]
    elif extra_channels is None:
        extra_channels = []
    assert len(extra_channels) <= _EXTRA_CHANNELS
    for ch, place in zip(extra_channels, EXTRA_INDS):
        if ch is not None:
            map[channel_names.index(ch), place] = 1.0
    # TODO for before 20


    return _deep_1010(map, channel_names, EOG, ear_ref, extra_channels)



def map_dataset_channels_deep_1010(channels: np.ndarray, exclude_stim=True):
    """
    Maps channels as stored by a :any:`DN3ataset` to the Deep1010 format, will automatically map EOG and extra channels
    by type.

    Parameters
    ----------
    channels : np.ndarray
               Channels that remain a 1D sequence (they should not have been projected into 2 or 3D grids) of name and
               type. This means the array has 2 dimensions:
               ..math:: N_{channels} \by 2
               With the latter dimension containing name and type respectively, as is constructed by default in most
               cases.
    exclude_stim : bool
                   This option allows the stim channel to be added as an *extra* channel. The default (True) will not do
                   this, and it is very rare if ever where this would be needed.

    Warnings
    --------
    If for some reason the stim channel is labelled with a label from the `DEEP_1010_CHS_LISTING` it will be included
    in that location and result in labels bleeding into the observed data.

    Returns
    -------
    mapping : torch.Tensor
              Mapping matrix from previous channel sequence to Deep1010.
    """
    if len(channels.shape) != 2 or channels.shape[1] != 2:
        raise ValueError("Deep1010 Mapping: channels must be a 2 dimensional array with dim0 = num_channels, dim1 = 2."
                         " Got {}".format(channels.shape))
    channel_types = OrderedDict()

    # Use this for some semblance of order in the "extras"
    extra = [None for _ in range(_EXTRA_CHANNELS)]
    extra_idx = 0

    flg_hgsn128 = False # TODO 221102
    flg_hgsn64 = False
    if channels.shape[0] == 20:
        flg_hgsn128 = True
        if len(channels[0][0])>3 and channels[0][0][0:3] == 'EEG':
            flg_hgsn128 = False

    elif channels[0][0][0] == 'E':
        flg_hgsn64 = True
        if len(channels[0][0])>3 and channels[0][0][0:3] == 'EEG':
            flg_hgsn64 = False

    for name, ch_type in channels:
        # Annoyingly numpy converts them to strings...
        if flg_hgsn128:
            name_ind = str(name[1:]) # TODO 221102
            try :
                name = HGSN128to1020_dict[name_ind]
            except:
                pass

        if flg_hgsn64:
            name_ind = str(name[1:])
            try:
                name = HGSN64to1020_dict[name_ind]
            except:
                pass

        ch_type = int(ch_type)
        if ch_type == FIFF.FIFFV_EEG_CH and _likely_eeg_channel(name): # 2
            channel_types[name] = 'eeg'
        elif ch_type == FIFF.FIFFV_EOG_CH or name in [DEEP_1010_CHS_LISTING[idx] for idx in EOG_INDS]: # 202
            channel_types[name] = 'eog'
        elif ch_type == FIFF.FIFFV_STIM_CH: # 3
            if exclude_stim:
                channel_types[name] = None
                continue
            # if stim, always set as last extra
            channel_types[name] = 'extra'
            extra[-1] = name
        elif 'REF' in name.upper() or 'A1' in name.upper() or 'A2' in name.upper() or 'EAR' in name.upper():
            channel_types[name] = 'ref'
        else:
            if extra_idx == _EXTRA_CHANNELS - 1 and not exclude_stim:
                print("Stim channel overwritten by {} in Deep1010 mapping.".format(name))
            elif extra_idx == _EXTRA_CHANNELS:
                print("No more room in extra channels for {}".format(name))
                continue
            channel_types[name] = 'extra'
            extra[extra_idx] = name
            extra_idx += 1

    revised_channel_types = _heuristic_resolution(channel_types) # 59
    eog, ref = _check_num_and_get_types(revised_channel_types)
    if len(eog) == 3 :
        channel_types['VEOGR'] = 'eog'
        revised_channel_types = _heuristic_resolution(channel_types)
        eog, ref = _check_num_and_get_types(revised_channel_types)
        print("transformers_channel_mh_355 : EOG editted arbitrary..")

    return map_named_channels_deep_1010(list(revised_channel_types.keys()), eog, ref, extra)
