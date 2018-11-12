import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import setting as st
import utils as ut

dat = tf.placeholder(tf.float32, shape=(None, st.channel_cnt, st.window_size, 1))
lbl = tf.placeholder(tf.float32, shape=(None))
dom = tf.placeholder(tf.float32, shape=(None))
hyper_lambda = tf.placeholder(tf.float32)
keep = tf.placeholder(tf.float32)

def Model(data, dom_l, cls_l, keep=0.5, reuse=None):
    with tf.variable_scope("feature_extractor") as scope_f:

        # List to store the output of each CNN path
        filter_output = []

        # CNNs with small filter size at the first layer
        ws1, bs1 = ut.get_params(name="sconv1", shape=[1, st.fs / 2, 1, 64])
        sconv1 = ut.conv(input=data, w=ws1, b=bs1, stride=st.fs / 16)
        sconv1 = ut.relu(ut.batchnorm(sconv1))
        sconv1 = tf.nn.max_pool(sconv1, ksize=[1, 1, 8, 1], strides=[1, 1, 8, 1], padding="SAME")
        sconv1 = ut.dropout(sconv1, keep_prob=keep)

        # CNNs with small filter size at the second layer
        ws2, bs2 = ut.get_params(name="sconv2", shape=[1, 8, 64, 128])
        sconv2 = ut.conv(input=sconv1, w=ws2, b=bs2, stride=1)
        sconv2 = ut.relu(ut.batchnorm(sconv2))

        # CNNs with small filter size at the third layer
        ws3, bs3 = ut.get_params(name="sconv3", shape=[1, 8, 128, 128])
        sconv3 = ut.conv(input=sconv2, w=ws3, b=bs3, stride=1)
        sconv3 = ut.relu(ut.batchnorm(sconv3))

        # CNNs with small filter size at the fourth layer
        ws4, bs4 = ut.get_params(name="sconv4", shape=[1, 8, 128, 128])
        sconv4 = ut.conv(input=sconv3, w=ws4, b=bs4, stride=1)
        sconv4 = ut.relu(ut.batchnorm(sconv4))
        sconv4 = tf.nn.max_pool(sconv4, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding="SAME")

        # Spatial Convolution
        wssp, bssp = ut.get_params(name="sspconv", shape=[22, 1, 128, 128])
        sspconv = tf.nn.bias_add(tf.nn.conv2d(input=sconv4, filter=wssp, strides=[1, 1, 1, 1], padding="VALID"), bssp)
        sspconv = ut.relu(ut.batchnorm(sspconv))

        sf = tf.reshape(sspconv, shape=(-1, 1024))

        ###################################################
        # CNNs with large filter size at the first layer
        wl1, bl1 = ut.get_params(name="lconv1", shape=[1, st.fs * 2, 1, 64])
        lconv1 = ut.conv(input=data, w=wl1, b=bl1, stride= st.fs / 2)
        lconv1 = ut.relu(ut.batchnorm(lconv1))
        lconv1 = tf.nn.max_pool(lconv1, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding="SAME")
        lconv1 = ut.dropout(lconv1, keep_prob=keep)

        # CNNs with large filter size at the second layer
        wl2, bl2 = ut.get_params(name="lconv2", shape=[1, 6, 64, 128])
        lconv2 = ut.conv(input=lconv1, w=wl2, b=bl2, stride=1)
        lconv2 = ut.relu(ut.batchnorm(lconv2))

        # CNNs with large filter size at the third layer
        wl3, bl3 = ut.get_params(name="lconv3", shape=[1, 6, 128, 128])
        lconv3 = ut.conv(input=lconv2, w=wl3, b=bl3, stride=1)
        lconv3 = ut.relu(ut.batchnorm(lconv3))

        # CNNs with large filter size at the fourth layer
        wl4, bl4 = ut.get_params(name="lconv4", shape=[1, 6, 128, 128])
        lconv4 = ut.conv(input=lconv3, w=wl4, b=bl4, stride=1)
        lconv4 = ut.relu(ut.batchnorm(lconv4))
        lconv4 = tf.nn.max_pool(lconv4, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

        # Spatial Convolution
        wlsp, blsp = ut.get_params(name="lspconv", shape=[22, 1, 128, 128])
        lspconv = tf.nn.bias_add(tf.nn.conv2d(input=lconv4, filter=wlsp, strides=[1, 1, 1, 1], padding="VALID"), blsp)
        lspconv = ut.relu(ut.batchnorm(lspconv))

        lf = tf.reshape(lspconv, shape=(-1, 512))  # small filter's feature

        # Concatenate sf and lf
        filter_output.append(sf)
        filter_output.append(lf)

        feature = tf.concat(values=filter_output, axis=1)

    with tf.variable_scope("domain_classifier") as scope_c:

        fc1 = tf.contrib.layers.fully_connected(inputs=feature, num_outputs=512, activation_fn=None)
        fc1 = ut.leaky_relu(ut.batchnorm(fc1))

        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=128, activation_fn=None)
        fc2 = ut.leaky_relu(ut.batchnorm(fc2))

        fc3 = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=32, activation_fn=None)
        fc3 = ut.leaky_relu(ut.batchnorm(fc3))

        output_d = tf.contrib.layers.fully_connected(inputs=fc3, num_outputs=2, activation_fn=None)
        domain_cast_label = tf.cast(dom_l, tf.int64)
        domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(domain_cast_label, depth=2), logits=output_d))

    with tf.variable_scope("label_predictor_target") as scope_pt:

        fc4 = tf.contrib.layers.fully_connected(inputs=feature, num_outputs=512, activation_fn=None)
        fc4 = ut.leaky_relu(ut.batchnorm(fc4))

        fc5 = tf.contrib.layers.fully_connected(inputs=fc4, num_outputs=128, activation_fn=None)
        fc5 = ut.leaky_relu(ut.batchnorm(fc5))

        fc6 = tf.contrib.layers.fully_connected(inputs=fc5, num_outputs=32, activation_fn=None)
        fc6 = ut.leaky_relu(ut.batchnorm(fc6))

        output_l1 = tf.contrib.layers.fully_connected(inputs=fc6, num_outputs=4, activation_fn=None)
        class_cast_label = tf.cast(cls_l, tf.int64)
        label_loss1 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(class_cast_label, depth=4), logits=output_l1))
        label_loss1 = label_loss1 * tf.cast(tf.equal(dom_l, 0), tf.float32)
        label_pred1 = tf.argmax(tf.sigmoid(output_l1), -1)

    with tf.variable_scope("label_predictor_source") as scope_ps:

        fc7 = tf.contrib.layers.fully_connected(inputs=feature, num_outputs=512, activation_fn=None)
        fc7 = ut.leaky_relu(ut.batchnorm(fc7))

        fc8 = tf.contrib.layers.fully_connected(inputs=fc7, num_outputs=128, activation_fn=None)
        fc8 = ut.leaky_relu(ut.batchnorm(fc8))

        fc9 = tf.contrib.layers.fully_connected(inputs=fc8, num_outputs=32, activation_fn=None)
        fc9 = ut.leaky_relu(ut.batchnorm(fc9))

        output_l2 = tf.contrib.layers.fully_connected(inputs=fc9, num_outputs=4, activation_fn=None)
        label_loss2 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(class_cast_label, depth=4), logits=output_l2))
        label_loss2 = label_loss2 * tf.cast(tf.equal(dom_l, 1), tf.float32)
        label_pred2 = tf.argmax(tf.sigmoid(output_l2), -1)

    return domain_loss, label_loss1, label_pred1, label_loss2, label_pred2


loss_d, loss_l1, pred_l1, loss_l2, pred_l2 = Model(data=dat, dom_l = dom, cls_l = lbl)
loss_l = (loss_l1 + loss_l2) * 0.5
loss_f = loss_l - hyper_lambda * loss_d

var_f = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="feature_extractor")
var_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="domain_classifier")
var_y1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor_target")
var_y2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor_source")

train_y1 = tf.train.AdagradOptimizer(learning_rate=st.learning_rate).minimize(loss_l1, var_list=var_y1) # target
train_y2 = tf.train.AdagradOptimizer(learning_rate=st.learning_rate).minimize(loss_l2, var_list=var_y2) # source
train_d = tf.train.AdagradOptimizer(learning_rate=st.learning_rate).minimize(loss_d, var_list=var_d)
train_f = tf.train.AdagradOptimizer(learning_rate=st.learning_rate).minimize(loss_f, var_list=var_f)

X_tgt = np.load(st.data_path + ('Rolled_Tdat_%d.npy') % st.tgt)
Y_tgt = np.load(st.data_path + ('Rolled_Tlbl_%d.npy') % st.tgt)
X_src = np.load(st.data_path + ('Rolled_Tdat_%d.npy') % st.src)
Y_src = np.load(st.data_path + ('Rolled_Tlbl_%d.npy') % st.src)

val_X_tgt = np.load(st.data_path + 'Rolled_Vdat_%d.npy' % st.tgt)
val_Y_tgt = np.load(st.data_path + 'Rolled_Vlbl_%d.npy' % st.tgt)
val_X_src = np.load(st.data_path + 'Rolled_Vdat_%d.npy' % st.src)
val_Y_src = np.load(st.data_path + 'Rolled_Vlbl_%d.npy' % st.src)

X_train = np.concatenate((X_tgt, X_src), axis=0)
Y_train = np.concatenate((Y_tgt, Y_src), axis=0)
D_train = np.concatenate((np.tile([1], Y_tgt.shape[0]), np.tile([0], Y_src.shape[0])), axis=0)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver_tr = tf.train.Saver(keep_checkpoint_every_n_hours=12, max_to_keep=1000)

print("Start Training!!")
total_batch = int((X_train.shape[0]) / st.batch_size_tr) #2314
total_batch_val = int((val_X_tgt.shape[0])/ st.batch_size_val) # 88
for epoch in range(st.total_epoch):
    temp_data_all = np.zeros(shape=X_train.shape, dtype=np.float32)
    temp_label_all = np.zeros(shape=Y_train.shape, dtype=np.float32)
    temp_domain_all = np.zeros(shape=D_train.shape, dtype=np.float32)

    rnd_idx = np.random.permutation(X_train.shape[0])
    for b in range(rnd_idx.shape[0]):
        temp_data_all[b, :, :, :] = X_train[rnd_idx[b], :, :, :]
        temp_label_all[b] = Y_train[rnd_idx[b]]
        temp_domain_all[b] = D_train[rnd_idx[b]]

    X_train = temp_data_all
    Y_train = temp_label_all
    D_train = temp_domain_all

    for batch in range(total_batch):
        batch_x = X_train[batch * st.batch_size_tr:(batch + 1) * st.batch_size_tr, :, :, :]
        batch_y = Y_train[batch * st.batch_size_tr:(batch + 1) * st.batch_size_tr]
        batch_d = D_train[batch * st.batch_size_tr:(batch + 1) * st.batch_size_tr]

        p = float((st.batch_size_tr * epoch) + batch) / (total_batch * st.total_epoch)
        hlambda = (2. / (1. + np.exp(-10. * p))) - 1
        _, _, loss_var_y = sess.run([train_y1, train_y2, loss_l], feed_dict={dat: batch_x, lbl: batch_y, dom: batch_d, hyper_lambda: hlambda, keep:0.5})
        _, loss_var_d= sess.run([train_d, loss_d], feed_dict={dat: batch_x, lbl: batch_y, dom: batch_d, hyper_lambda: hlambda, keep:0.5})
        _, loss_var_f = sess.run([train_f, loss_f], feed_dict={dat: batch_x, lbl: batch_y, dom: batch_d, hyper_lambda: hlambda, keep:0.5})


    # Calculate validation error per subject
    val_batch_lbl_all_1 = []
    val_batch_lbl_all_2 = []

    val_pred_lbl_all_1 = []
    val_pred_lbl_all_2 = []

    for batch_val in range(total_batch_val):
        batch_x_val_1 = val_X_tgt[batch_val * st.batch_size_val: (batch_val + 1) * st.batch_size_val, :, :, :]
        batch_y_val_1 = val_Y_tgt[batch_val * st.batch_size_val: (batch_val + 1) * st.batch_size_val]
        val_pred_lbl_1 = sess.run([pred_l1], feed_dict={dat: batch_x_val_1, lbl: batch_y_val_1, keep: 1.0})

        batch_x_val_2 = val_X_src[batch_val * st.batch_size_val: (batch_val + 1) * st.batch_size_val, :, :, :]
        batch_y_val_2 = val_Y_src[batch_val * st.batch_size_val: (batch_val + 1) * st.batch_size_val]
        val_pred_lbl_2 = sess.run([pred_l2], feed_dict={dat: batch_x_val_2, lbl: batch_y_val_2, keep: 1.0})

        val_batch_lbl_all_1.append(batch_y_val_1)
        val_batch_lbl_all_2.append(batch_y_val_2)

        val_pred_lbl_all_1.append(val_pred_lbl_1)
        val_pred_lbl_all_2.append(val_pred_lbl_2)

    ori_lbl_all_1, ori_pred_all_1 = ut.decision(val_batch_lbl_all_1, val_pred_lbl_all_1, class_number=4)
    ori_lbl_all_2, ori_pred_all_2 = ut.decision(val_batch_lbl_all_2, val_pred_lbl_all_2, class_number=4)

    val_acc_lbl_1 = accuracy_score(ori_lbl_all_1, ori_pred_all_1)
    val_acc_lbl_2 = accuracy_score(ori_lbl_all_2, ori_pred_all_2)

    print("%dth Epoch over." % (epoch + 1))
    if (epoch+1) % st.eval_epoch == 0:
        print("All variables are saved in path", saver_tr.save(sess, (st.model_path + "CNN2path_tgt%d_src%d_bs%d_te%d_%02dth_epoch.ckpt") % (st.tgt, st.src, st.batch_size_tr, st.total_epoch, epoch+1)))
print("Optimization finished!!")