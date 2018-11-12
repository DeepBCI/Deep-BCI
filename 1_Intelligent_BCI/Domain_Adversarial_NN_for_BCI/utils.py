import numpy as np
import tensorflow as tf
import mne as mne
import scipy.io as sio
import setting as st


def get_params(name, shape):
    w = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    n_filter = shape[-1]
    b = tf.Variable(name=name + "b", initial_value=tf.constant(0.1, shape=[n_filter], dtype=tf.float32))
    return w, b

def conv(input, w, b, stride):
    conv = tf.nn.conv2d(input=input, filter=w, strides=(1, 1, stride, 1), padding="SAME")
    return tf.nn.bias_add(conv, b)

def batchnorm(input):
    return tf.nn.batch_normalization(x=input, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-8)

def dropout(input, keep_prob):
    return tf.nn.dropout(x=input, keep_prob=keep_prob)

def relu(input):
    return tf.nn.relu(input)

def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)

def decision(val_lbl_all, val_pred_all, class_number):
    val_lbl_all = np.array(val_lbl_all).flatten()
    val_pred_all = np.array(val_pred_all).flatten()

    original_lbl_all = np.zeros(shape=(int(val_lbl_all.shape[0] / st.window_number)), dtype=np.int8)
    original_pred_all = np.zeros(shape=(int(val_pred_all.shape[0] / st.window_number)), dtype=np.int8)

    for i in range(0, int(val_lbl_all.shape[0] / st.window_number) - 1):
        counts1 = np.bincount(val_lbl_all[st.window_number * i:st.window_number * (i + 1)], minlength=class_number)
        counts2 = np.bincount(val_pred_all[st.window_number * i:st.window_number * (i + 1)], minlength=class_number)
        ind1 = np.argmax(counts1)
        ind2 = np.argmax(counts2)

        original_lbl_all[i] = ind1
        original_pred_all[i] = ind2

    return original_lbl_all, original_pred_all

def time_domain(sbj, cls, status, session):
    if session == "training":
        data = np.array(sio.loadmat(st.data_path + "/A%02dT.mat" % sbj)["data"])
    elif session == "test":
        data = np.array(sio.loadmat(st.data_path + "/A%02dE.mat" % sbj)["data"])
    data = data[:, :, cls * 72:(cls + 1) * 72]
    data2 = data.astype(np.float64)
    data2 = np.swapaxes(data2, axis1=2, axis2=1)

    filtered = mne.filter.filter_data(data2, sfreq=250, l_freq=st.low, h_freq=st.high)

    if status == "MI":
        baseline = filtered[:, :, :250]
        baseline = np.expand_dims(np.mean(baseline, axis=2), axis=2)
        baseline = np.repeat(baseline, filtered.shape[2], 2)
        output = filtered - baseline
        output = output[:, :, 500:]
        return output
    else:
        baseline = filtered[:, :, :250]
        return baseline


def freq_domain(sbj, cls, status, session):
    data = time_domain(sbj, cls, status, session)
    if session == "test":
        data = data[:-1, :, :]
    psds = np.empty(shape=(data.shape[0], data.shape[1], 126))
    for ch in range(22):
        for tr in range(72):
            psds[ch, tr, :], freqs = mne.time_frequency.psd_array_welch(data[ch, tr, :], sfreq=250, n_fft=250, n_overlap=0, verbose=False)
    return psds, freqs


def cal_cos(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

def cal_dis(cos):
    return 1-cos