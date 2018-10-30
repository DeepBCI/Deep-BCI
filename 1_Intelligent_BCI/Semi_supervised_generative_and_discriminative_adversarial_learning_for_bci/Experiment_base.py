import tensorflow as tf
import numpy as np
import utils as ut
import network as nt
import setting as st

from sklearn.metrics import accuracy_score, cohen_kappa_score

## Placeholding
X = tf.placeholder(dtype=tf.float32, shape=[st.batch_size, 22, 512, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[st.batch_size])

X_valid = tf.placeholder(dtype=tf.float32, shape=[239, 22, 512, 1])
Y_valid = tf.placeholder(dtype=tf.float32, shape=[239])

## Load dataset
sbj = st.subject
data, label, data_valid, label_valid = ut.load_data(sbj=sbj, training=True) #(22, 750, 288), (288,)
data_test, label_test = ut.load_data(sbj=sbj, training=False) #(22, 750, 288), (288,)

## Baseline
logits, _, _, _ = nt.DeepConvNet(input=X, adversarial=False)
loss = ut.calculate_loss_baseline(logits=logits, labels=Y)
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DeepConvNet")

_, output, feature_tsne, sptial_weight = nt.DeepConvNet(input=X_valid, adversarial=False, reuse=True)

learning_rate = st.learning_rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=1000)

rest_point = data.shape[1] - st.window_size + 1
total_batch = int((data.shape[-1] * rest_point)/st.batch_size)
loss_ = 0

print("Baseline Results")
print("Subject %d" % sbj)
for epoch in range(st.total_epoch):
    # Randomize the dataset
    rand_idx = np.random.permutation(rest_point * data.shape[-1]) # 68832

    # Feed dictionary
    for batch in range(total_batch):
        batch_x = np.empty(shape=(st.batch_size, 22, st.window_size, 1))
        batch_y = np.empty(shape=(st.batch_size))
        for i in range(st.batch_size):
            position = np.unravel_index(indices=rand_idx[epoch * st.batch_size + i], dims=(rest_point, data.shape[-1]))
            batch_x[i, :, :, 0] = data[:, position[0]:position[0] + st.window_size, position[1]]
            batch_y[i] = label[position[1]]

        _, loss_ = sess.run([optimizer, loss], feed_dict={X:batch_x, Y:batch_y})
    print("%04dth Epoch, Training Loss: %04f" % (epoch + 1, loss_))


    # Validation
    prediction = np.zeros(shape=(288))
    grount_truth = np.zeros(shape=(288))
    for trials in range(0, 288):
        batch_x = np.empty(shape=(rest_point, 22, st.window_size, 1))
        for batch in range(rest_point):
            batch_x[batch, :, :, 0] = data_test[:, batch:batch+st.window_size, trials]
        pred, feature = sess.run([output, feature_tsne], feed_dict={X_valid:batch_x})
        grount_truth[trials] = label_test[trials]
        prediction[trials] = np.argmax(np.bincount(np.squeeze(np.asarray(pred))))
        np.save(st.tsne_path + "/common%04d_gt%d.npy" % (epoch, grount_truth[trials]), feature)

    print("Validation accuracy: %f Kappa value: %f"
          % (accuracy_score(y_true=grount_truth, y_pred=prediction), cohen_kappa_score(y1=grount_truth, y2=prediction)))

    saver.save(sess, st.path + "/model%sbj%epoch.ckpt" %(sbj, epoch))