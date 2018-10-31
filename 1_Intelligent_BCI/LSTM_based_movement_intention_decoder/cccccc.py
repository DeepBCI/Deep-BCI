import numpy as np
import tensorflow as tf

x_train = np.random.rand(1000, 64, 61, 61)
x_train[:500, :, :, :] = x_train[:500, :, :, :] + 2
y_train = np.ones((1000,))
y_train[:500] *= -1

x_tt = np.zeros((x_train.shape[0], 61 * 64, 61, 1))
for i in range(x_train.shape[3]):
    x_tt[:, :, i, :] = x_train[:, :, :, i].reshape((-1, x_train.shape[2] * x_train.shape[1], 1))

# conv network
x_in = tf.placeholder(shape=[None, 3904, 61, 1], dtype=tf.float32)
w1 = tf.Variable(tf.truncated_normal(shape=(61 * 64, 7, 1, 16), stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[16]))
conv1 = tf.nn.conv2d(x_in, w1, strides=[1, 1, 1, 1], padding="SAME")
conv1_out = tf.nn.relu(conv1 + bias1)
pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[32]))
conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME")
conv2_out = tf.nn.relu(conv2 + bias2)
pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

pool2_ = tf.reshape(pool2, shape=(-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]))

fc_w = tf.Variable(tf.truncated_normal(shape=(976*31*32, 61*61), stddev=0.1))
fc_b = tf.Variable(tf.constant(0.1, shape=[61*61]))

init_out = tf.nn.relu(tf.matmul(pool2_, fc_w) + fc_b)

weight_ = []
bias_ = []
layer_pr = [61 * 61, 100, 15, 2]
layer_pr = [61 * 61, 25, 2]
for i in range(len(layer_pr) - 1):
    weight_.append(tf.Variable(tf.zeros([layer_pr[i], layer_pr[i + 1]]), dtype=tf.float32))
    bias_.append(tf.Variable(tf.zeros([layer_pr[i + 1]]), dtype=tf.float32))

y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])

outlayer = []
outlayer.append(init_out)
for i in range(len(layer_pr) - 2):
    temp_ = tf.nn.relu(tf.matmul(outlayer[-1], weight_[i]) + bias_[i])
    outlayer.append(temp_)

# predict
y = tf.matmul(outlayer[-1], weight_[len(layer_pr) - 2]) + bias_[len(layer_pr) - 2]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(pool1, feed_dict={x_in: x_tt.astype(np.float32)})