import tensorflow as tf
import numpy as np
import utils as ut
import network as nt
import setting as st

from sklearn.metrics import accuracy_score, cohen_kappa_score

## Placeholding
X = tf.placeholder(dtype=tf.float32, shape=[st.batch_size, 22, 512, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[st.batch_size])
Z = tf.placeholder(dtype=tf.float32, shape=[st.batch_size, st.n_noise])

X_valid = tf.placeholder(dtype=tf.float32, shape=[239, 22, 512, 1])
Y_valid = tf.placeholder(dtype=tf.float32, shape=[239])

## Load dataset
sbj = st.subject
data, label, data_valid, label_valid = ut.load_data(sbj=sbj, training=True) #(22, 750, 288), (288,)
data_test, label_test = ut.load_data(sbj=sbj, training=False) #(22, 750, 288), (288,)

## Adversarial Training
G, sptial_weight_g = nt.DeepDeconvNet(input=Z)
logit_real, _, feature_real, _ = nt.DeepConvNet(input=X, adversarial=True)
logit_fake, _, feature_fake, _ = nt.DeepConvNet(input=G, adversarial=True, reuse=True)
loss_d, loss_g = ut.calculate_loss_at(logit_real=logit_real, logit_fake=logit_fake,
                                      features_real=feature_real, features_fake=feature_fake, labels=Y)

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DeepConvNet")
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DeepDeconvNet")

logit_test, _, feature_tsne, sptial_weight_d = nt.DeepConvNet(input=X_valid, adversarial=True, reuse=True)

learning_rate = st.learning_rate
d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_d, var_list=D_vars)
g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_g, var_list=G_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=1000)

rest_point = data.shape[1] - st.window_size + 1
total_batch = int((data.shape[-1] * rest_point)/st.batch_size)
loss_d_, loss_g_ = 0, 0

print("Adversarially Trained Results")
print("Subject %d" % sbj)
for epoch in range(st.total_epoch):
    # Randomize the dataset
    rand_idx = np.random.permutation(rest_point * data.shape[-1])  # 68832

    # Feed dictionary
    for batch in range(total_batch):
        batch_x = np.empty(shape=(st.batch_size, 22, st.window_size, 1))
        batch_y = np.empty(shape=(st.batch_size))
        for i in range(st.batch_size):
            position = np.unravel_index(indices=rand_idx[epoch * st.batch_size + i], dims=(rest_point, data.shape[-1]))
            batch_x[i, :, :, 0] = data[:, position[0]:position[0] + st.window_size, position[1]]
            batch_y[i] = label[position[1]]
        batch_z = ut.get_noise(st.batch_size, st.n_noise)

        _, loss_g_ = sess.run([g_optimizer, loss_g], feed_dict={X:batch_x, Y:batch_y, Z:batch_z})
        _, loss_d_ = sess.run([d_optimizer, loss_d], feed_dict={X:batch_x, Y:batch_y, Z:batch_z})
    print("%04dth Epoch, Discriminator training Loss: %04f, Generator training Loss: %04f" % (epoch + 1, loss_d_, loss_g_))

    # Validation
    prediction = np.zeros(shape=(288))
    grount_truth = np.zeros(shape=(288))
    for trials in range(0, 288):
        batch_x = np.empty(shape=(rest_point, 22, st.window_size, 1))
        for batch in range(rest_point):
            batch_x[batch, :, :, 0] = data_test[:, batch:batch + st.window_size, trials]
        pred, feature = sess.run([logit_test, feature_tsne], feed_dict={X_valid:batch_x})
        pred = np.argmax(np.bincount(np.argmax(ut.sigmoid(np.squeeze(np.asarray(pred))[:, :-1]), -1)))
        grount_truth[trials] = label_test[trials]
        prediction[trials] = pred
        np.save(st.tsne_path + "/adv%04d_gt%d.npy" % (epoch, grount_truth[trials]), feature)

    print("Test accuracy: %f Kappa value: %f"
          % (accuracy_score(y_true=grount_truth, y_pred=prediction), cohen_kappa_score(y1=grount_truth, y2=prediction)))

    sample, wd, wg = sess.run([G, sptial_weight_d, sptial_weight_g], feed_dict={Z:batch_z})
    saver.save(sess, st.path + "/adversarial_model%sbj%epoch.ckpt" %(sbj, epoch))

