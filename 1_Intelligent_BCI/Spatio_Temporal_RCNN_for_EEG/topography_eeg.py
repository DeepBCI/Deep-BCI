import numpy as np
import matplotlib.pyplot as plt
from mne.channels import read_montage
from mne.channels.layout import make_eeg_layout
from mne import EvokedArray, create_info
from mne.viz import plot_evoked_topomap
from mne.viz import plot_topomap
from mne import set_eeg_reference
import scipy.io
import setting as st



# covdata = np.load('covdata.npy')
# covpred = np.load('covpred.npy')
# inv_covpred = 1 / covpred
weights = np.load('spatialconvweight.npy')
weights = np.reshape(weights, (22,1,4096))
weights = np.squeeze(weights, axis=1)
# weights = np.matmul(covdata, weights)###
# weights = inv_covpred * weights####
# clusteringW = scipy.io.loadmat('rcl_filter_clustered_prototype.mat')["avg_spatial_filters"]

# channels you selected
# channels = ['Fp1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
#             'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
#             'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7',
#             'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
# channels_type = ['eeg'] * 32

channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2',
           'FC4', 'C5', 'C3', 'C1', 'Cz',
           'C2', 'C4', 'C6', 'CP3', 'CP1',
           'CPz', 'CP2', 'CP4', 'P3', 'Pz',
           'P4', 'POz']
channels_type = ['eeg'] * 22


# read EEG standard montage from mne
montage = read_montage('standard_1020', channels)

# make your own info
sampling_freq = 250
info_custom = create_info(ch_names=channels, sfreq=sampling_freq, ch_types=channels_type, montage=montage)
info_custom['description'] = 'My custom dataset'
# print(info_custom)


# your_data_array = np.transpose(weights)
your_data_array = weights
# clusteredData = np.transpose(clusteringW)
maxvalue = np.max(your_data_array)
# maxvalue_cluster = np.max(clusteredData)
minvalue = np.min(your_data_array)
# minvalue_cluster = np.min(clusteredData)
compare = np.fabs([maxvalue, minvalue])
maxabsolute = np.max(compare)

evoked = EvokedArray(your_data_array, info=info_custom, tmin=0.)
# evoked = EvokedArray(clusteredData, info=info_custom, tmin=0.)


layout_custom = make_eeg_layout(info=info_custom)
# print (maxvalue, minvalue)
#

num_prototypes = 4096 #st.num_feature_map
for idx in range(num_prototypes):
    f = plot_topomap(data = your_data_array[:, idx],pos=evoked.info,show=False, vmax=maxvalue, vmin=minvalue)
    # plt.subplot(np.ceil(np.sqrt(num_prototypes)), np.ceil(np.sqrt(num_prototypes)), idx+1)   aaaa
    # cur_axes = plt.gca(axes)
    # f = plot_topomap(data=clusteredData[:, idx], pos=evoked.info, axes=cur_axes, show=False, vmax=maxvalue_cluster, vmin=minvalue_cluster)
    # im, cont = plot_topomap(data=clusteredData[:, idx], pos=evoked.info, show=False, vmax=maxvalue_cluster, vmin=minvalue_cluster)
    im, cont = plot_topomap(data=your_data_array[:, idx], pos=evoked.info, show=False, vmax=maxvalue, vmin=minvalue)

   # print(idx+1, cont, im)
    # fig.colorbar(axes)
    # if idx == 5:
    # plt.colorbar(cont)
    plt.colorbar(im)

    plt.savefig(st.Topography_path + "cluster_topo_%d.png" % (idx+1))
    plt.close()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88) aaaa
plt.show()