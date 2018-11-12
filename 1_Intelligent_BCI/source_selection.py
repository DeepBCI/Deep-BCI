import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from fooof import FOOOF
import utils as ut

dat = np.empty(shape=(9,4,22,72,126))
for sbj in range(9):
    for cls in range(4):
        dat[sbj,cls,:,:,:], freqs = ut.freq_domain(sbj+1, cls, "RESTING", "training")

dat = np.mean(dat, axis=1)
dat = np.mean(dat, axis=2)

f_range = [8,30]
fm = FOOOF(peak_width_limits = [0.5,12.0], max_n_peaks=5, min_peak_amplitude=0, peak_threshold=1.0)

######
# First, you need to change fm.fit function in order to get corrected curve.
######

correct = np.empty(shape=(9,22,23))
bg = np.empty(shape=(9,22,23))
for sbj in range(9):
    for ch in range(22):
        correct[sbj,ch,:], bg[sbj,ch,:] = fm.fit(freqs, dat[sbj, ch, :], f_range) #We changed this function.

final = np.empty(shape=(9, 22, 23))
for sbj in range(9):
    for ch in range(22):
        final[sbj, ch, :] = correct[sbj, ch, :] - bg[sbj, ch, :]

final2 = np.swapaxes(final,axis1=1, axis2=2)
final3 = np.empty(shape=(9,23*22))

for sbj in range(9):
    for ch in range(22):
        final3[sbj,ch*23:23*(ch+1)] = final2[sbj,:,ch]

#Hierarchical clustering
link = linkage(final3, method="average", metric="cosine")
plt.figure(figsize=(8,6))
sublist = []
for s in range(1,10):
    tmp = str(s)
    sublist.append(tmp)

ax = plt.gca()
plt.title("Clustering", fontsize=25)
plt.ylabel("Distance", fontsize=20)
plt.xlabel("Subject", fontsize=20)
ax.tick_params(axis="y", which="major", labelsize=15)
dn = dendrogram(link, labels=sublist, leaf_font_size=15)
plt.savefig("BCIC_clustering11",dpi=300)
plt.show()