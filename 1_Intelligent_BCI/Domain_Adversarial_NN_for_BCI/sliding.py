import numpy as np
import scipy.io as sio
import setting as st

def rolling_window(input, window):
    def rolling_window_lastaxis(input, window):
        if window < 1:
            raise ValueError("`Window` must be at least 1.")
        if window > input.shape[-1]:
            raise ValueError("`Window` is too long.")
        shape = input.shape[:-1] + (input.shape[-1] - window + 1, window)
        strides = input.strides + (input.strides[-1],)
        return np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)

    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(input, window)
    for i, win in enumerate(window):
        if win > 1:
            input = input.swapaxes(i, -1)
            input = rolling_window_lastaxis(input, win)
            output = input.swapaxes(-2, i)
    return output

# Training data
def load(sb):
    dat = np.array([sio.loadmat(st.data_path + ('A%02dTClass1.mat')%sb)['Class1'],
                    sio.loadmat(st.data_path + ('A%02dTClass2.mat')%sb)['Class2'],
                    sio.loadmat(st.data_path + ('A%02dTClass3.mat')%sb)['Class3'],
                    sio.loadmat(st.data_path + ('A%02dTClass4.mat')%sb)['Class4']])
    return dat

def tr_roll_and_save(data, sb):
    dat = np.empty(shape=(0, st.channel_cnt, st.window_size, 1), dtype=np.float32)
    lbl = np.empty(shape=0, dtype=np.uint8)

    for cnt, cur_dat in enumerate(data):
        cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]
        rolled_dat = rolling_window(cur_dat, (1, st.window_size))
        rolled_dat = rolled_dat.reshape(-1, st.channel_cnt, st.window_size)[..., None]

        dat = np.concatenate((dat, rolled_dat), axis=0)
        lbl = np.concatenate((lbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0)

    np.save(st.data_path + ('Rolled_Tdat_%d.npy') % sb, dat)
    np.save(st.data_path + ('Rolled_Tlbl_%d.npy') % sb, lbl)
    print("Sliding and save for training data of subject %d" % sb)

def cv_roll_and_save(data, sb):
    dat = np.empty(shape=(0, st.channel_cnt, st.window_size, 1), dtype=np.float32)
    lbl = np.empty(shape=0, dtype=np.uint8)

    for cnt, cur_dat in enumerate(data):
        cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]
        rolled_dat = rolling_window(cur_dat, (1, st.window_size))
        rolled_dat = rolled_dat.reshape(-1, st.channel_cnt,st. window_size)[..., None]

        dat = np.concatenate((dat, rolled_dat), axis=0)
        lbl = np.concatenate((lbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0)

    np.save(st.data_path + ('Rolled_Vdat_%d.npy') % sb, dat)
    np.save(st.data_path + ('Rolled_Vlbl_%d.npy') % sb, lbl)
    print("Sliding and save for validation data of subject %d" % sb)


for sbj in range(1,3):
    dat = load(sbj)
    cvdat = dat[:,:,:,:22]
    trdat = dat[:,:,:,22:]

    tr_roll_and_save(trdat, sbj)
    cv_roll_and_save(cvdat, sbj)

# Test data
def test_roll_and_save(sb):
    data = np.array(sio.loadmat(st.data_path + ('A%02dE.mat') % sb)['data'])
    temp = np.zeros(shape=(54432, 23, 512, 1), dtype=np.int8)
    for i in range(0, data.shape[-1]):
        for j in range(0, 189):
            temp[189*i + j, :, :, 0] = data[:, j:j+512, i]
    data = temp[:, :-1, :, :]
    label = temp[:, -1, 0, 0] - 1

    np.save(st.data_path + ('Rolled_Edat_%d.npy') % sb,data)
    np.save(st.data_path + ('Rolled_Elbl_%d.npy') % sb,label)
    print("Sliding and save for test data of subject %d" % sb)

for sbj in range(1,3):
    test_roll_and_save(sbj)