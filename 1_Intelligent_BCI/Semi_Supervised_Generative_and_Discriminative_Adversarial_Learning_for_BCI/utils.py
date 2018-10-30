import numpy as np
import scipy.io
import setting as st
import tensorflow as tf

def rolling_window(a, window):
    def rolling_window_lastaxis(a, window):
        if window < 1:
            raise ValueError("`window` must be at least 1.")
        if window > a.shape[-1]:
            raise ValueError("`window` is too long.")
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a


def load_data_v2(sbj, training):
    path = st.data_path
    if training == True:
        data = np.squeeze(np.array([scipy.io.loadmat(path + "/A%02dT" % sbj)["data"]])) #(22, 1250, 4 X 72)
        data = data[:, 500:, :] #(22, 750, 288) To reject resting state signals.
        tmp = np.zeros(shape=72)
        label = np.concatenate((tmp, tmp+1, tmp+2, tmp+3)) #(288,)
        return data, label
    else:
        data = np.squeeze(np.array([scipy.io.loadmat(path + "/A%02dE" % sbj)["data"]])) #(22, 1250, 4 X 72)
        data = data[:, 500:, :] #(22, 750, 288)
        label = np.squeeze(np.array([scipy.io.loadmat(path + "/A%02dE_label" % sbj)["label"]])) #(288,)
        return data, label


def load_data(sbj, training):
    path = st.data_path
    if training == True:
        data = np.array([scipy.io.loadmat(path + "/A%02dTClass1" % sbj)["Class1"],
                       scipy.io.loadmat(path + "/A%02dTClass2" % sbj)["Class2"],
                       scipy.io.loadmat(path + "/A%02dTClass3" % sbj)["Class3"],
                       scipy.io.loadmat(path + "/A%02dTClass4" % sbj)["Class4"]]) #(4, 23, 700, 72)

        cvdata = data[:, :, :, :7] #(4, 23, 700, 7)
        cvdat = np.empty(shape=(0, 22, 512, 1), dtype=np.float32)
        cvlbl = np.empty(shape=0, dtype=np.uint8)
        for cnt, cur_dat in enumerate(cvdata):
            cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]

            rolled_dat = rolling_window(cur_dat, (1, 512))

            rolled_dat = rolled_dat.reshape(-1, 22, 512)[..., None]

            cvdat = np.concatenate((cvdat, rolled_dat), axis=0) #(10584, 22, 512, 1)
            cvlbl = np.concatenate((cvlbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0) #(10584,)

        data = data[:,:,:,7:]

        dat = np.empty(shape=(0, 22, 512, 1), dtype=np.float32)
        lbl = np.empty(shape=0, dtype=np.uint8)

        for cnt, cur_dat in enumerate(data):
            cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]

            rolled_dat = rolling_window(cur_dat, (1, 512))

            rolled_dat = rolled_dat.reshape(-1, 22, 512)[..., None]

            dat = np.concatenate((dat, rolled_dat), axis=0) #(43848, 22, 512, 1)
            lbl = np.concatenate((lbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0) #(43848,)
        return dat, lbl, cvdat, cvlbl
    else:
        data = np.array(scipy.io.loadmat(path + "/A%02dE"% sbj)["data"]) #(23, 700, 288)
        temp = np.zeros(shape=(54432, 23, 512, 1))
        for i in range(0, data.shape[-1]):
            for j in range(0, 189):
                temp[189*i + j, :, :, 0] = data[:, j:j+512, i] #(54432, 23, 512, 1)
        data = temp[:, :-1, :, :] #(54432, 22, 512, 1)
        label = temp[:, -1, 0, 0] - 1 #(54432,)

        return data, label

def load_small_data(sbj, proportion, training):
    path = st.data_path
    if training == True:
        data = np.array([scipy.io.loadmat(path + "/A%02dTClass1" % sbj)["Class1"],
                       scipy.io.loadmat(path + "/A%02dTClass2" % sbj)["Class2"],
                       scipy.io.loadmat(path + "/A%02dTClass3" % sbj)["Class3"],
                       scipy.io.loadmat(path + "/A%02dTClass4" % sbj)["Class4"]]) #(4, 23, 700, 72)

        tmp = int(data.shape[-1] * proportion)

        data = data[:,:,:,:tmp]

        dat = np.empty(shape=(0, 22, 512, 1), dtype=np.float32)
        lbl = np.empty(shape=0, dtype=np.uint8)

        for cnt, cur_dat in enumerate(data):
            cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]

            rolled_dat = rolling_window(cur_dat, (1, 512))

            rolled_dat = rolled_dat.reshape(-1, 22, 512)[..., None]

            dat = np.concatenate((dat, rolled_dat), axis=0)
            lbl = np.concatenate((lbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0) #(43848,)
        return dat, lbl
    else:
        data = np.array(scipy.io.loadmat(path + "/A%02dE"% sbj)["data"]) #(23, 700, 288)
        temp = np.zeros(shape=(54432, 23, 512, 1))
        for i in range(0, data.shape[-1]):
            for j in range(0, 189):
                temp[189*i + j, :, :, 0] = data[:, j:j+512, i] #(54432, 23, 512, 1)
        data = temp[:, :-1, :, :] #(54432, 22, 512, 1)
        label = temp[:, -1, 0, 0] - 1 #(54432,)
        return data, label

def load_unlabeld_data(sbj, labeled_proportion, training):
    path = st.data_path
    if training == True:
        data = np.array([scipy.io.loadmat(path + "/A%02dTClass1" % sbj)["Class1"],
                         scipy.io.loadmat(path + "/A%02dTClass2" % sbj)["Class2"],
                         scipy.io.loadmat(path + "/A%02dTClass3" % sbj)["Class3"],
                         scipy.io.loadmat(path + "/A%02dTClass4" % sbj)["Class4"]])  # (4, 23, 700, 72)

        prop = int(data.shape[-1] * labeled_proportion)

        unldata = data[:, :, :, prop:]
        unldat = np.empty(shape=(0, 22, 512, 1), dtype=np.float32)
        unllbl = np.empty(shape=0, dtype=np.uint8)
        for cnt, cur_dat in enumerate(unldata):
            cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]

            rolled_dat = rolling_window(cur_dat, (1, 512))

            rolled_dat = rolled_dat.reshape(-1, 22, 512)[..., None]

            unldat = np.concatenate((unldat, rolled_dat), axis=0)  # (10584, 22, 512, 1)
            unllbl = np.concatenate((unllbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)),
                                   axis=0)  # (10584,)

        data = data[:, :, :, :prop]

        dat = np.empty(shape=(0, 22, 512, 1), dtype=np.float32)
        lbl = np.empty(shape=0, dtype=np.uint8)

        for cnt, cur_dat in enumerate(data):
            cur_dat = np.swapaxes(cur_dat, 0, 2)[..., :-1]

            rolled_dat = rolling_window(cur_dat, (1, 512))

            rolled_dat = rolled_dat.reshape(-1, 22, 512)[..., None]

            dat = np.concatenate((dat, rolled_dat), axis=0)  # (43848, 22, 512, 1)
            lbl = np.concatenate((lbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)),
                                 axis=0)  # (43848,)
        return dat, lbl, unldat
    else:
        data = np.array(scipy.io.loadmat(path + "/A%02dE" % sbj)["data"])  # (23, 700, 288)
        temp = np.zeros(shape=(54432, 23, 512, 1))
        for i in range(0, data.shape[-1]):
            for j in range(0, 189):
                temp[189 * i + j, :, :, 0] = data[:, j:j + 512, i]  # (54432, 23, 512, 1)
        data = temp[:, :-1, :, :]  # (54432, 22, 512, 1)
        label = temp[:, -1, 0, 0] - 1  # (54432,)

        return data, label

def get_noise(batch_size, n_noise):
    gauss = np.random.normal(loc=0, scale=1, size=[batch_size, n_noise])
    return gauss

def randomize_dataset(data, label):
    rand_idx = np.random.permutation(data.shape[0])
    tmp_dat = np.zeros(shape=data.shape)
    tmp_lbl = np.zeros(shape=label.shape)
    for idx in range(rand_idx.shape[0]):
        tmp_dat[idx, :, :, :] = data[rand_idx[idx], :, :, :]
        tmp_lbl[idx] = label[rand_idx[idx]]
    return tmp_dat, tmp_lbl

def sigmoid(input):
    return 1/(1+np.exp(-input))

def calculate_loss_baseline(logits, labels):
    label = tf.one_hot(tf.cast(labels, tf.int64), depth=4)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    return loss

def calculate_loss_at(logit_real, logit_fake, features_real, features_fake, labels):
    epsilon = 1e-8 # To avoid NAN loss
    label = tf.one_hot(tf.cast(labels, tf.int64), depth=4)
    loss_D1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit_real[:, :-1]))
    prob_fake = tf.nn.softmax(logit_fake)[:, -1]
    loss_D2 = -1 * tf.reduce_mean(tf.log(prob_fake + epsilon))
    loss_D = loss_D1 + loss_D2

    loss_G1 = -1 * tf.reduce_mean(tf.log(1 - prob_fake + epsilon))
    # Feature matching
    tmp1, tmp2 = tf.reduce_mean(features_real), tf.reduce_mean(features_fake)
    loss_G2 = tf.reduce_mean(tf.square(tmp1 - tmp2))
    loss_G = loss_G1 + loss_G2
    return loss_D, loss_G

def calculate_loss_at_for_unlab(logit_real, logit_unsup, logit_fake, features_real, features_fake, labels):
    epsilon = 1e-8 # To avoid NAN loss
    label = tf.one_hot(tf.cast(labels, tf.int64), depth=4)
    loss_D_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit_real[:, :-1]))
    prob_real = tf.nn.softmax(logit_unsup)[:, -1]
    prob_fake = tf.nn.softmax(logit_fake)[:, -1]
    loss_D_unsup1 = -1 * tf.reduce_mean(tf.log(1 - prob_real + epsilon))
    loss_D_unsup2 = -1 * tf.reduce_mean(tf.log(prob_fake + epsilon))
    loss_D = loss_D_sup + loss_D_unsup1 + loss_D_unsup2

    loss_G1 = -1 * tf.reduce_mean(tf.log(1 - prob_fake + epsilon))
    # Feature matching
    tmp1, tmp2 = tf.reduce_mean(features_real), tf.reduce_mean(features_fake)
    loss_G2 = tf.reduce_mean(tf.square(tmp1 - tmp2))
    loss_G = loss_G1 + loss_G2
    return loss_D, loss_G

