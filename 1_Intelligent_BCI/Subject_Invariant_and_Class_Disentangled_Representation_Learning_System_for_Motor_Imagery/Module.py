import tensorflow as tf
import numpy as np
import Setting as st

kernel_initializer = tf.glorot_normal_initializer()
regul = tf.contrib.layers.l2_regularizer(scale=st.alpha)

def load(sbj, fold):
    tr = np.load(st.PSD_data_path + "PSD_splitted/5-fold/CV%d_tr_sub%02d.npy" % (fold, sbj))
    vl = np.load(st.PSD_data_path + "PSD_splitted/5-fold/CV%d_vl_sub%02d.npy" % (fold, sbj))
    ts = np.load(st.PSD_data_path + "PSD_splitted/5-fold/CV%d_ts_sub%02d.npy" % (fold, sbj))

    return tr, vl, ts


def Gaussian_normalization(data, mean, std, train = True):
    # data: [channel, freq, trials]

    if train == True:
        mean_ch = np.empty(shape=(st.n_ch))
        std_ch = np.empty(shape=(st.n_ch))

        for cha in range(st.n_ch):
            mean_ch[cha] = data[cha,:,:].mean()
            std_ch[cha] = data[cha,:,:].std()

        mean = mean_ch
        std = std_ch

    for ch in range(st.n_ch):
        data[ch,:,:] = (data[ch,:,:]-mean[ch])/(std[ch]+0.00000001)
    return data, mean, std


def get_var(name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name), tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)


def _batch_norm(inputs, name=None, momentum=0.9, epsilon=1e-5, freeze=False, reuse=None):
    if not st.bn:
        return inputs

    if freeze or not st.is_train:
        trainable = False
        training = False

    else:
        training = True
        trainable = st.is_train

    return tf.layers.batch_normalization(inputs=inputs, axis=-1, momentum=momentum, epsilon=epsilon, name = name + "_bn", training=training, trainable=trainable, reuse=reuse)


def _conv2d(inputs, filters, n_kernel, strides=(1,1), padding="VALID", regularizer=None, activation=None, use_bias=True, name=None, freeze=False, reuse=None):

    is_train = st.is_train
    if freeze:
        is_train = False

    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=n_kernel, strides=strides, padding = padding, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=regularizer, trainable=is_train, name=name+"_conv2d", reuse=reuse)


def _deconv2d(inputs, filters, n_kernel, strides=(1,1), padding="VALID", regularizer=None, activation=None, use_bias=True, name=None, freeze=False, reuse=None):
    is_train = st.is_train
    if freeze:
        is_train = False

    return tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=n_kernel, strides=strides, padding=padding, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=regularizer, trainable=is_train, name=name+"_deconv2d", reuse=reuse)


def _dense(inputs, units, regularizer= None, activation=None, use_bias=True, name=None, freeze=False, reuse=None):
    is_train = st.is_train
    if freeze:
        is_train = False

    return tf.layers.dense(inputs=inputs, units=units, activation=activation, use_bias=use_bias,
                           kernel_initializer=kernel_initializer, kernel_regularizer=regularizer, trainable=is_train, name=name + "_dense", reuse=reuse)

def _dropout(inputs, rate=0.5, name=None, freeze=False):
    if not st.drop:
        return inputs
    training = True
    if freeze:
        training = False

    return tf.layers.dropout(inputs=inputs, rate=rate, training=training, name=name + "_dropout")

def _leaky_relu(inputs, alpha=st.lrelu_alpha, name=None):
    return tf.nn.leaky_relu(features=inputs, alpha=alpha, name=name + "_lrelu")

def _elu(inputs, name=None):
    return tf.nn.elu(features=inputs, name=name + "_elu")

def _relu(inputs, name=None):
    return tf.nn.relu(features=inputs, name=name+"_relu")


def _maxpooling(inputs, size, name, strides=(1,1), padding="VALID"):
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=size, strides=strides, padding=padding, name=name +"_maxpool")


def _avgpooling(inputs, size, name, strides=(1,1), padding="VALID"):
    return tf.layers.average_pooling2d(inputs=inputs, pool_size=size, strides=strides, padding=padding, name=name +"_avgpool")



def conv2d_layer(i, f, k, do=0., bn=False, s=1, pad='valid', reg=None, act=None, bias=True, name=None,
                 freeze=False, reuse=None):
    a = i
    out = i
    if do != 0.:
        out = _dropout(inputs=out, rate=do, name=name, freeze=freeze)
    out = _conv2d(inputs=out, filters=f, n_kernel=k, strides=s, padding=pad, regularizer=reg, activation=None, use_bias=bias,
                       name=name, freeze=freeze, reuse=reuse)
    if bn:
        out = _batch_norm(inputs=out, name=name, freeze=freeze, reuse=reuse)

    if act == "relu":
        out = _relu(inputs=out, name=name)
    elif act == "lrelu":
        out = _leaky_relu(inputs=out, name=name)
    elif act == "elu":
        out = _elu(inputs=out, name=name)

    else:
        out = out
    return out


def deconv2d_layer(i, f, k, do=0., bn=False, s=1, pad="valid", reg=None, act=None, bias=True, name=None, freeze=False, reuse=None):

    a= i
    out = i
    if do !=0:
        out = _dropout(inputs=out, rate=do, name=name, freeze=freeze)
    out = _deconv2d(inputs=out, filters=f, n_kernel=k, strides=s, padding=pad, regularizer=reg, activation=None, use_bias=bias, name=name, freeze=freeze, reuse=reuse)

    if bn:
        out = _batch_norm(inputs=out, name=name, freeze=freeze, reuse=reuse)

    if act == "relu":
        out = _relu(inputs=out, name=name)
    elif act=="lrelu":
        out = _leaky_relu(inputs=out, name=name)
    elif act == "elu":
        out = _elu(inputs=out, name=name)

    else:
        out = out
    return out


def dense_layer(i, u, do=0., bn=False, act=None, reg=None, bias=True, name=None, freeze=False, reuse=None):
    out = i

    if do != 0.:
        out = _dropout(inputs=out, rate=do, name=name, freeze=freeze)
    out = _dense(inputs=out, units=u, activation=None, regularizer=reg, use_bias=bias, name=name, freeze=freeze, reuse=reuse)
    if bn:
        out = _batch_norm(inputs=out, name=name, freeze=freeze, reuse=reuse)
    if act == "relu":
        out = _relu(inputs=out, name=name)
    elif act == "lrelu":
        out = _leaky_relu(inputs=out, name=name)
    elif act == "elu":
        out = _elu(inputs=out, name=name)
    else:
        out = out
    return out


class upper_enc:
    def __init__(self, name, inputs, freeze=False):
        self.name = name
        self.inputs = inputs
        self.freeze = freeze

    def _build_net(self, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            out = conv2d_layer(i=self.inputs, f=st.nfeat1, k=[1, st.n_freq], bn=st.bn, act=st.act, reg=regul, name="conv1", freeze=self.freeze, reuse=reuse)
        return out

class lower_enc():
    def __init__(self, name, inputs, freeze=False):
        self.name = name
        self.inputs = inputs
        self.freeze = freeze

    def _build_net(self, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            out = conv2d_layer(i=self.inputs, f=st.nfeat2, k=[st.n_ch, 1], bn=st.bn, act=st.act, reg=regul, name="conv2", freeze=self.freeze, reuse=reuse)

        return out

def classifier(latent, freez=False, reuse=None):
    with tf.variable_scope("classifier", reuse=reuse):
        in1 = dense_layer(i=latent, u=100, do=st.drop, bn=st.bn, act=st.act, reg=regul, name="cls_fc1", freeze=freez, reuse=reuse)
        in2 = dense_layer(i=in1, u=2, do=False, bn=False, act="None", reg=regul, name="cls_fc2", freeze=freez, reuse=reuse)

    return in2

class global_disc:
    def __init__(self, name, localf, nfeat2, globalf, freeze=False):
        self.name = name
        self.localf = localf
        self.nfeat2 = nfeat2
        self.globalf = globalf
        self.freeze = freeze

    def _build_net(self, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            local_conv = conv2d_layer(i=self.localf, f=self.nfeat2, k=[st.n_ch, 1], bn=False, act=st.act, name="conv3", freeze=self.freeze, reuse=reuse)

            concat = tf.concat(values=[local_conv, self.globalf], axis=-1, name="global_disc_concat")

            out = dense_layer(i=concat, u=int(self.nfeat2/2), do=st.drop, bn=False, act=st.act, name="global_disc_fc1", freeze=self.freeze, reuse=reuse)
            out = dense_layer(i=out, u=1, do=st.drop, bn=False, act="None", name="global_disc_fc2", freeze=self.freeze, reuse=reuse)

        return out


class local_disc:
    def __init__(self, name, localf, globalf, freeze=False):
        self.name = name
        self.localf = localf
        self.globalf = globalf
        self.freeze = freeze

    def _build_net(self, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            local_conv = conv2d_layer(i=self.localf, f=st.nfeat3, k=[1, 1], bn=False, act=st.act, name="conv4",
                                      freeze=self.freeze, reuse=reuse)

            local_conv = tf.squeeze(local_conv, axis=2)
            globalf = dense_layer(i=self.globalf, u=st.nfeat3, do=st.drop, bn=False, act=st.act, name="local_disc_fc1",
                                  freeze=self.freeze, reuse=reuse)
            globalf = tf.squeeze(globalf, axis=2)
            tile = tf.tile(input=globalf, multiples=(1, tf.shape(local_conv)[1], 1), name="tile")

            dot = tf.matmul(local_conv, tf.transpose(tile, perm=[0, 2, 1]), name="dot")

        return dot




class Class_disentanglement:
    def __init__(self, freeze=False):
        self.freeze = freeze

    def classifier(self, latent, freez, reuse=None):
        with tf.variable_scope("classifier", reuse=reuse):
            in1 = dense_layer(i=latent, u=st.nfeat2, do=st.drop, bn=st.bn, act=st.act, reg=regul, name="cls_fc1", freeze=freez, reuse=reuse)
            in2 = dense_layer(i=in1, u=2, do=False, bn=False, act="None", reg=regul, name="cls_fc2", freeze=freez, reuse=reuse)

        return in2

    def MINE(self, latent1, latent2, freez, reuse=None):
        with tf.variable_scope("MINE", reuse=reuse):
            in11 = dense_layer(i=latent1, u=20, do=st.drop, bn=True, act="None", name="MINE1_r", freeze=freez, reuse=reuse)
            in12 = dense_layer(i=latent2, u=20, do=st.drop, bn=True, act="None", name="MINE1_i", freeze=freez, reuse=reuse)

            in1 = _elu(in11+in12, name="sum")

            in2 = dense_layer(i=in1, u=20, do=st.drop, bn=False, act=st.act, name="MINE2", freeze=freez, reuse=reuse)
            out = dense_layer(i=in2, u=1, do=st.drop, bn=False, act="None", name="MINE3", freeze=freez, reuse=reuse)

        return out

    def spatial_deconv(self, latent, freez, reuse=None):
        with tf.variable_scope("spatial_deconv", reuse=reuse):
            out = deconv2d_layer(i=latent, f=st.nfeat1, k=[st.n_ch, 1], bn=st.bn, act=st.act, reg=regul, name="deconv1", freeze=freez, reuse=reuse)
        return out

    def spectral_deconv(self, latent,freez,reuse=None):
        with tf.variable_scope("spectral_deconv", reuse=reuse):
            out = deconv2d_layer(i=latent, f=1, k=[1,st.n_freq], bn=False, act="None", reg=regul, name="deconv2", freeze=freez, reuse=reuse)
        return out