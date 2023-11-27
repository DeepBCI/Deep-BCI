# Code modify by hn_jo@korea.ac.kr 
# last update 2023.11.27
# Input : raw data
# Output : connectivity map by region (N by N matrix, N is number of region)

import numpy as np
import mne_connectivity

def get_connectivity(data, method):
    con = mne_connectivity.spectral_connectivity_epochs(
        data,
        method=method,
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        tmin=tmin,
        mt_adaptive=False,
        n_jobs=1,
    )
    return con


def connectivity_mean(con):
    con_mean = []
    con_mean = con[0].get_data("dense")[:, :, 0].copy()
    for sub in range(1, len(con)):
        con_mean = np.add(con_mean, con[sub].get_data("dense")[:, :, 0].copy())
    con_mean = np.divide(con_mean, len(con))
    return con_mean


def ch_to_idx(ch_names, ch_list):
    ch_idx = []
    for ch in ch_list:
        ch_idx.append(ch_names.index(ch))
    return ch_idx


def get_roi_con(con, region1, region2):
    roi_con = []
    for i in region1:
        for j in region2:
            if i != j:
                if i < j:
                    roi_con.append(con[j][i])
                else:
                    roi_con.append(con[i][j])
    return np.mean(roi_con)


def get_roi_map(con, ch):
    roi_map = np.zeros((5, 5))
    roi_list = [frontal, central, temporal, parietal, occipital]
    for i in range(0, 5):
        for j in range(0, 5):
            roi_map[i][j] = get_roi_con(
                con, ch_to_idx(ch, roi_list[i]), ch_to_idx(ch, roi_list[j])
            )
    return roi_map


# example 
# This example based on BrainVision data
# You can modify this code for your own data

sfreq = 100 # sampling frequency
fmin = 8 # frequency min
fmax = 12 # frequency max
tmin = 0 # time min

ground = "Fpz"
reference = "Cz"
frontal = [
    "Fp1",
    "Fp2",
    "AFz",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "Fz",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
]
central = [
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "FC5",
    "FC6",
    "Cz",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
]
temporal = ["FT7", "FT8", "FT9", "FT10", "T7", "T8", "TP7", "TP8", "TP9", "TP10"]
parietal = [
    "CPz",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "Pz",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
]
occipital = ["POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2", "Iz"]

baseline = [] # baseline connectivity data
g1 = [] # group 1 connectivity data
g2 = [] # group 2 connectivity data
g3 = [] # group 3 connectivity data
group_list = [baseline, g1, g2, g3]
data_list = [baseline_data, g1_data, g2_data, g3_data] # raw data list
method = "pli"

for idx, con_value in enumerate(group_list):
    for sub in range(len(data_list[idx])):
        con_value.append(get_connectivity(data_list[idx][sub], method))

for idx, value in enumerate(group_list):
    mean = []
    mean = connectivity_mean(value)
    roi = get_roi_map(mean, ch_names) # ch_names is channel name list from raw data
