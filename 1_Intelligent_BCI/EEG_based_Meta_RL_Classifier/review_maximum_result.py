import numpy as np
import scipy
import fnmatch
import os
import csv

direc_root = 'chan_by_chan_ebe_sta12_MAXREL'


def ext_files(direc='6CH', crite='MBMF', fsmethod='MI'):
    lom = []  # list of matrix.
    for file in os.listdir(direc):
        if fnmatch.fnmatch(file, '*_{0}_{1}.npy'.format(crite, fsmethod)):
            lom.append(file)
    lom.sort()
    return lom

criteria = ['spec' + 'flex']
fsmethods = [ 'MI']

for crite in criteria:
    mat = np.zeros((18,10))
    chanind = np.zeros((18,))
    timewind = np.zeros((58,))
    for fsmethod in fsmethods:
        with open(direc_root + '/' + crite + '_' + fsmethod + '.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for i in range(64):
                direc = direc_root + '/channel_{0}'.format(i)
                lom = ext_files(direc, crite, fsmethod)
                subinx = 0
                for subi in lom:
                    for j in range(58):
                        base = np.load(direc + '/' + subi)
                        means = base[j].mean()
                        stds = base[j].std()
                        if mat[subinx,:].mean() < base[j].mean():
                            mat[subinx, :] = base[j]
                            chanind[subinx] = i
                            timewind[subinx] = j
                    print(subi)
                    subinx += 1
            subinx = 0
            for subi in lom:
                spamwriter.writerow([subi[:5]] + [str(mat[subinx,:].mean())] + [str(mat[subinx,:].std())] + [str(chanind[subinx]+1)] + [str(mat[subinx,:].tolist())])
                subinx += 1
