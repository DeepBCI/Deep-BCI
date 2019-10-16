import tensorflow as tf
import numpy as np
import os
import Setting as st
import Modules as md
import argparse
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--gpu", type=int, default=7)

ARGS = parser.parse_args()
fi = ARGS.fold
gpu = ARGS.gpu

all = np.arange(1,53)
bad = np.array([29,34])
sources = np.setdiff1d(all, bad)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


""" Placeholding """
dat = tf.placeholder(tf.float32, shape=(None, st.n_ch, st.n_freq, 1), name="X")
lbl = tf.placeholder(tf.float32, shape=(None, st.n_cl), name="Y")

dat_val = tf.placeholder(tf.float32, shape=(None, st.n_cl, st.n_freq, 1), name="X_val")
lbl_val = tf.placeholder(tf.float32, shape=(None, st.n_cl), name="Y_val")


class DeepInfoMaxLoss():
    def __init__(self, name, localf, globalf, labels, freez=None, reus=None):
        self.name = name
        self.freez = freez
        self.reus = reus
        self.localf = localf
        self.globalf = globalf
        self.global_d = md.global_disc
        self.local_d = md.local_disc
        self.labels = labels

    def _build_net(self):
        with tf.variable_scope(self.name):
            # global_discriminator
            rand_idx1 = tf.random_shuffle(
                tf.range(tf.div(tf.size(self.globalf), tf.shape(self.globalf)[-1]))[:, None])
            rand_idx2 = tf.random_shuffle(
                tf.range(tf.div(tf.size(self.globalf), tf.shape(self.globalf)[-1]))[:, None])
            shuffle1 = tf.gather_nd(self.localf, rand_idx1)
            shuffle2 = tf.gather_nd(self.localf, rand_idx2)

            output1 = self.global_d(name="global_discriminator", localf=self.localf, globalf=self.globalf, freeze=self.freez)._build_net(reuse=self.reus)
            output2 = self.global_d(name="global_discriminator", localf=shuffle1, globalf=self.globalf, freeze=True)._build_net(reuse=tf.AUTO_REUSE)
            g_joint = tf.reduce_mean(tf.log(2.) - tf.nn.softplus(tf.negative(output1)))
            g_margi = tf.reduce_mean(tf.nn.softplus(tf.negative(output2)) + output2 - tf.log(2.))
            loss_global = (g_joint - g_margi)

            # local_discriminator
            output3 = self.local_d(name="local_discriminator", localf=self.localf, globalf=self.globalf, freeze=self.freez)._build_net(reuse=self.reus)
            output4 = self.local_d(name="local_discriminator", localf=shuffle2, globalf=self.globalf, freeze=True)._build_net(reuse=tf.AUTO_REUSE)

            l_joint_ch = tf.reduce_mean((tf.log(2.) - tf.nn.softplus(tf.negative(output3))), axis=1)
            l_margi_ch = tf.reduce_mean((tf.nn.softplus(tf.negative(output4)) + output4 - tf.log(2.)), axis=1)

            loss_local = tf.reduce_mean((l_joint_ch - l_margi_ch))

            loss_mi = loss_global + loss_local

        return loss_mi, loss_global, loss_local, g_joint, g_margi


class EMCLoss():
    def __init__(self, name, input, label, freez=False, reuse=None):
        self.name = name
        self.freez = freez
        self.reuse = reuse
        self.input = input
        self.label = label
        self.DE = md.Class_disentanglement(freeze=self.freez)

    def _build_net(self):
        with tf.variable_scope(self.name):
            """ Split the global feature """
            globalf = tf.squeeze(self.input, axis=1)
            globalf = tf.squeeze(globalf, axis=1)
            re, irre = tf.split(globalf, [int(st.nfeat2/2), int(st.nfeat2/2)], axis=-1)

            """ Classifier """
            logit_re = self.DE.classifier(latent=re, freez=self.freez, reuse=self.reuse)
            loss_re = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit_re, labels=self.label))
            pred_re = tf.argmax(tf.sigmoid(logit_re), -1)

            """ MINE """
            ridx = tf.random_shuffle(tf.range(tf.div(tf.size(globalf), tf.shape(globalf)[-1]))[:, None])
            shuffle = tf.gather_nd(re, ridx)
            output1 = self.DE.MINE(latent1=re, latent2=irre, freez=self.freez, reuse=self.reuse)
            output2 = self.DE.MINE(latent1=shuffle, latent2=irre, freez=True, reuse=tf.AUTO_REUSE)
            m_joint = tf.reduce_mean(tf.log(2.) - tf.nn.softplus(tf.negative(output1)))
            m_margi = tf.reduce_mean(tf.nn.softplus(tf.negative(output2)) + output2 - tf.log(2.))
            loss_mine = (m_joint - m_margi)

            """ Concat the class-relevant feature and class-irrelevant feautre """
            latents = tf.concat([re, irre], axis=-1)
            latents = tf.expand_dims(latents, axis=1)
            latents = tf.expand_dims(latents, axis=1)

            localfp = self.DE.spatial_deconv(latents, freez=self.freez, reuse=self.reuse)
            recon = self.DE.spectral_deconv(localfp, freez=self.freez, reuse=self.reuse)

        return loss_mine, loss_re, pred_re, recon




""" Load data """
alltr = np.empty(shape=(st.n_ch, st.n_freq, 0), dtype=np.float32)
allvl = np.empty(shape=(st.n_ch, st.n_freq, 0), dtype=np.float32)
allts = np.empty(shape=(st.n_ch, st.n_freq, 0), dtype=np.float32)

lbltr = np.empty(shape=(0), dtype=np.int32)
lblvl = np.empty(shape=(0), dtype=np.int32)
lblts = np.empty(shape=(0), dtype=np.int32)

test_trials = []

for ia in sources:
    trdat, vldat, tsdat = md.load(ia, "5fold", fi)

    trlbl = np.concatenate((np.zeros(shape=(int(trdat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(trdat.shape[-1] / 2)), dtype=np.int32)), axis=-1)
    vllbl = np.concatenate((np.zeros(shape=(int(vldat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(vldat.shape[-1] / 2)), dtype=np.int32)), axis=-1)
    tslbl = np.concatenate((np.zeros(shape=(int(tsdat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(tsdat.shape[-1] / 2)), dtype=np.int32)), axis=-1)

    alltr = np.concatenate((alltr, trdat), axis=-1)
    allvl = np.concatenate((allvl, vldat), axis=-1)
    allts = np.concatenate((allts, tsdat), axis=-1)

    lbltr = np.concatenate((lbltr, trlbl), axis=-1)
    lblvl = np.concatenate((lblvl, vllbl), axis=-1)
    lblts = np.concatenate((lblts, tslbl), axis=-1)

    test_trials.append(tsdat.shape[-1])

test_trials = np.array(test_trials)

""" Gaussian normalization """
train, mean, std = md.Gaussian_normalization(alltr, mean=0, std=1, train=True)
valid, _, _ = md.Gaussian_normalization(allvl, mean, std, False)
test, _, _ = md.Gaussian_normalization(allts, mean, std, False)

num_tr = train.shape[-1]
num_vl = valid.shape[-1]
num_ts = test.shape[-1]



""" Define the whole network """
# Training
out1 = md.upper_enc(name="Upper_enc", inputs=dat)._build_net()
out2 = md.lower_enc(name="Lower_enc", inputs=out1)._build_net()
loss_mi, loss_g, loss_l, gjoint, gmargi = DeepInfoMaxLoss(name="DeepInfoMax", localf=out1, globalf=out2, labels=lbl)._build_net()
loss_dmi, loss_re, pred_re, recon = EMCLoss(name="EMC", input=out2, label=lbl)._build_net()
loss_recon = tf.reduce_mean(tf.squared_difference(dat, recon))
# l2 regularization
loss_l2 = tf.losses.get_regularization_loss()
loss_re += loss_l2


# Validation
out1_val = md.upper_enc(name="Upper_enc", inputs=dat_val, freeze=True)._build_net(reuse=tf.AUTO_REUSE)
out2_val = md.lower_enc(name="Lower_enc", inputs=out1_val, freeze=True)._build_net(reuse=tf.AUTO_REUSE)
loss_mi_val, loss_g_val, loss_l_val, gjoint_val, gmargi_val = DeepInfoMaxLoss(name="DeepInfoMax", localf=out1_val, globalf=out2_val, labels=lbl_val, freez=True, reus=tf.AUTO_REUSE)._build_net()
loss_dmi_val, loss_re_val, pred_re_val, recon_val = EMCLoss(name="EMC", input=out2_val, label=lbl_val, freez=True, reuse=tf.AUTO_REUSE)._build_net()
loss_recon_val = tf.reduce_mean(tf.squared_difference(dat_val, recon_val))



""" Collect paramters """
param_ue, bn_ue = md.get_var("Upper_enc")
param_le, bn_le = md.get_var("Lower_enc")
param_g, bn_g = md.get_var("DeepInfoMax/global_discriminator")
param_l, bn_l = md.get_var("DeepInfoMax/local_discriminator")

param_ad, bn_ad = md.get_var("EMC/spatial_deconv")
param_ed, bn_ed = md.get_var("EMC/spectral_deconv")
param_cl, bn_cl = md.get_var("EMC/classifier")
param_mi, bn_mi = md.get_var("EMC/MINE")

parameters_mi = [param_ue, param_le, param_g, param_l]
parameters_c = [param_ue, param_le, param_cl]
parameters_en = [param_ue, param_le, param_ad, param_ed]
parameters_dmi = [param_ue, param_le, param_mi]

"""
Deep InfoMax
"""
bns = bn_ue + bn_le + bn_g + bn_l
with tf.control_dependencies(bns):
    optim_mi = tf.train.MomentumOptimizer(learning_rate=st.learning_rate, momentum=0.9).minimize(-loss_mi, var_list=parameters_mi)

""" 
Update classifier, lower encoder, and upper encoder by minimizing classification loss
"""
bn_c = bn_ue + bn_le + bn_cl
with tf.control_dependencies(bn_c):
    optim_dc = tf.train.MomentumOptimizer(learning_rate=st.learning_rate, momentum=0.9).minimize(loss_re, var_list=parameters_c)

"""
Update MINE by maximizing MINE loss
Update upper encoder and lower encoder by multiplying a negative value to MINE loss 
"""
bn_dmi = bn_mi + bn_le + bn_ue
with tf.control_dependencies(bn_dmi):
    optim_dmi = tf.train.MomentumOptimizer(learning_rate=st.learning_rate, momentum=0.9)
    vars = optim_dmi.compute_gradients(-loss_dmi, var_list=parameters_dmi)

    name_grad = {}
    name_var = {}

    for grad, var in vars:
        temp_grad = grad
        if grad is not None:

            # Gradient Clipping
            if var.name in name_grad:
                temp_grad = tf.stack((grad, name_grad[var.name]), axis=0)
                norm_grad = tf.norm(temp_grad, axis=(-1, -1))
                clip_grad = tf.reduce_mean(norm_grad)
                temp_grad = tf.clip_by_norm(grad, clip_norm=clip_grad)

            # Gradient reversal layer
            if "EMC/MINE" not in var.name:
                temp_grad = tf.negative(temp_grad)
            if var.name not in name_grad:
                name_grad[var.name] = temp_grad
            else:
                name_grad[var.name] += temp_grad
            name_var[var.name] = var

    grad_vars = [(name_grad[name], name_var[name]) for name in name_grad.keys()]
    optim_dmi = optim_dmi.apply_gradients(grad_vars)

"""
Reconstruction
"""
bn_de = bn_ue + bn_le + bn_ad + bn_ed
with tf.control_dependencies(bn_de):
    optim_de = tf.train.MomentumOptimizer(learning_rate=st.learning_rate, momentum=0.9).minimize(loss_recon, var_list=parameters_en)




init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    init.run()
    saver_tr = tf.train.Saver(keep_checkpoint_every_n_hours=12, max_to_keep=100000)
    writer = tf.summary.FileWriter("./logs_GIST/PSD_All/Momentum/DIM_Disentangle_Allsubs_%dfold" % (fi))

    aa = 0
    bb = 0
    total_batch = int((num_tr) / st.bs)
    train = np.swapaxes(train, 0, -1)
    train = np.swapaxes(train, 1, 2)

    valid = np.swapaxes(valid, 0, -1)
    valid = np.swapaxes(valid, 1, 2)
    valid2 = np.expand_dims(valid, axis=3)

    tst = np.swapaxes(test, 0, -1)
    tst = np.swapaxes(tst, 1, 2)
    tst = np.expand_dims(tst, axis=3)

    for epoch in range(st.n_epoch):
        """Randomize"""
        rnd_i = np.random.permutation(num_tr)
        tmp = np.zeros(shape=(train.shape))
        tmpl = np.zeros(shape=(lbltr.shape), dtype=np.int64)

        for ii in range(rnd_i.shape[0]):
            tmp[ii, :, :] = train[rnd_i[ii], :, :]
            tmpl[ii] = lbltr[rnd_i[ii]]

        train1 = tmp
        train1 = np.expand_dims(train1, axis=3)
        label1 = tmpl

        pred_all_tr = np.empty(shape=(0), dtype=np.float32)
        for batch in range(total_batch):
            batch_x = train1[batch * st.bs: (batch + 1) * st.bs, :, :, :]
            batch_y = label1[batch * st.bs: (batch + 1) * st.bs]
            batch_y = np.eye(st.n_cl)[batch_y]

            hgamma = 1
            halpha = 1
            hbeta = 1

            feed_dict = {dat: batch_x, lbl: batch_y}

            # Deep InfoMax loss
            _, dim_var, glo_var, loc_var = sess.run([optim_mi, loss_mi, loss_g, loss_l], feed_dict=feed_dict)

            # classification loss
            _, re_var= sess.run([optim_dc, loss_re], feed_dict=feed_dict)

            # MINE loss
            _, dmi_var = sess.run([optim_dmi, loss_dmi], feed_dict=feed_dict)

            # Reconstruction loss
            _, rec_var = sess.run([optim_de, loss_recon], feed_dict=feed_dict)


            aa = aa + 1
            print("All subjects: %04dth epoch, %04dth iter, %dth fold" % (epoch + 1, aa, fi))

        if (epoch + 1) % st.eval_epoch == 0:
            print("All variables are saved in path", saver_tr.save(sess, (
                    st.model_path + "All_DIM_Disentangle_%dfold_Momentum_%02dth_epoch.ckpt") % (fi, (epoch + 1))))




        # Validation
        batch_x_val = valid2
        batch_y_val = lblvl
        batch_y_val = np.eye(st.n_cl)[batch_y_val]

        feed_dict_val = {dat_val: batch_x_val, lbl_val: batch_y_val}
        re_var_val, pred_val = sess.run([loss_re_val, pred_re_val], feed_dict=feed_dict_val)
        val_acc = accuracy_score(lblvl, pred_val)