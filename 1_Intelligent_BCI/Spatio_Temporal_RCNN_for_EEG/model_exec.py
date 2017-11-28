import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import tensorflow as tf
import scipy.io
import setting as st
from sklearn.preprocessing import normalize
from sklearn.metrics import cohen_kappa_score as kappa

class model_exec:
    def __init__(self):
        self.epochs = 50
        self.eval_freq = 500
        self.init_lr = 0.003
        self.final_lr = 0.0003

    def train_RCNN(self, ce, sm, dn, ln, channel_cnt, time_cnt):
        data = [scipy.io.loadmat(st.RCNN_separated_class + "A13789TClass1.mat")["Class1"],
                scipy.io.loadmat(st.RCNN_separated_class + "A13789TClass2.mat")["Class2"],
                scipy.io.loadmat(st.RCNN_separated_class + "A13789TClass3.mat")["Class3"],
                scipy.io.loadmat(st.RCNN_separated_class + "A13789TClass4.mat")["Class4"]]
        dat = np.empty(shape=(0, channel_cnt, time_cnt, 1), dtype=np.float32)
        lbl = np.empty(shape=0, dtype=np.uint8)

        for cnt, cur_dat in enumerate(data):
            cur_dat = np.swapaxes(cur_dat, 0,2)[...,:-1]

            rolled_dat = self.rolling_window(cur_dat, (1,time_cnt))

            rolled_dat = rolled_dat.reshape(-1, channel_cnt, time_cnt)[...,None]

            dat = np.concatenate((dat, rolled_dat), axis=0)
            lbl = np.concatenate((lbl, np.full(shape=rolled_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0)

        train_size = dat.shape[0]
        batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
        learning_rate = tf.train.exponential_decay(learning_rate=self.init_lr, global_step=batch * st.batch_size,
                                                   decay_steps=train_size, staircase=True,
                                                   decay_rate=np.power(self.final_lr / self.init_lr,
                                                                       np.float(1) / self.epochs))
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(ce, global_step=batch)
        predict = tf.to_double(100) * (
            tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(sm, ln, 1))))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.summary.scalar("error", predict)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(st.RCNN_summary_path, sess.graph)
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=8, max_to_keep=100)
            start_time = time.time()
            cur_epoch = 0
            rand_idx = np.random.permutation(train_size)
            print("Variable Initialized")
            for step in range(int(self.epochs * train_size) // st.batch_size):
                offset = (step * st.batch_size) % (train_size - st.batch_size)
                batch_data = dat[rand_idx[offset:offset + st.batch_size]]
                batch_labels = lbl[rand_idx[offset:offset + st.batch_size]]

                feed_dict = {dn: batch_data, ln: batch_labels}
                _, summary_out = sess.run([optimizer, summary_op], feed_dict=feed_dict)

                summary_writer.add_summary(summary_out, global_step=step * st.batch_size)
                if step % self.eval_freq == 0:
                    l, lr, predictions = sess.run(
                        [ce, learning_rate, predict],
                        feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print(
                        'Step %d (epoch %.2f), %.2f s' % (
                          step, float(step) * st.batch_size / train_size, elapsed_time))
                    print('Minibatch loss: %.3f, learning rate: %.9f' % (l, lr))
                    print('Minibatch error: %.1f%%' % predictions)

                if cur_epoch != int((step * st.batch_size) / train_size):
                    print("Saved in path", saver.save(sess, st.RCNN_model_path + "%02d.ckpt" % (cur_epoch)))
                    rand_idx = np.random.permutation(train_size)
                cur_epoch = int((step * st.batch_size) / train_size)
            print("Saved in path", saver.save(sess, st.RCNN_model_path + "savedmodel_final.ckpt"))
        tf.reset_default_graph()

    def rolling_window(self, a, window):
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


    def test_RCNN1(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A01E.mat")["data"]
        print("Subject 1")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)
                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN2(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A02E.mat")["data"]
        print("Subject 2")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN3(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A03E.mat")["data"]
        print("Subject 3")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        # data = normalize(data) #Gaussian Normalization
        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN4(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A04E.mat")["data"]
        print("Subject 4")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN5(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A05E.mat")["data"]
        print("Subject 5")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN6(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A06E.mat")["data"]
        print("Subject 6")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))
        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN7(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A07E.mat")["data"]
        print("Subject 7")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        # data = normalize(data) #Gaussian Normalization
        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN8(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A08E.mat")["data"]
        print("Subject 8")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
    def test_RCNN9(self, sm, dn, wsc, bsc, time_cnt):
        data1 = scipy.io.loadmat(st.RCNN_separated_class + "A09E.mat")["data"]
        print("Subject 9")
        data1 = np.swapaxes(data1, 0,2)
        label = data1[...,-1]-1
        data1 = data1[...,:-1]
        rolled_data = self.rolling_window(data1, (1, time_cnt))

        # data = normalize(data) #Gaussian Normalization
        for i in range(00, 47):
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                saver.restore(sess, st.RCNN_model_path + "%02d.ckpt" %(i))
                w, b = sess.run([wsc, bsc])


                np.save(st.path + "spatialconvweight.npy", w)
                # scipy.io.savemat('/home/a/PycharmProjects/RCNN_BCI/conv1weight.mat', {'w': w})
                # data = np.pad(data, ((0, time_cnt - 1), (0, 0)), mode="edge")
                # rolled_data = self.rolling_window(data, (time_cnt, 1))

                acc = 0
                results4 = np.empty([288, 1])
                mean_prediction = np.empty([288, 189])##
                gt1 = np.empty([288, 1])
                for cnt, dat in enumerate(rolled_data):
                    results = sess.run(sm, feed_dict={dn: dat[...,None]})
                    results1 = np.argmax(results, axis=-1)
                    mean_prediction[cnt] = results1###
                    results2 = np.bincount(results1, minlength=4)
                    results3 = np.argmax(results2)
                    results4[cnt] = results3
                    gt = np.unique(label[cnt])
                    gt1[cnt] = gt
                # print(gt==results3, gt, results3)
                    if gt == results3:
                        acc = acc + 1


                mean_prediction = np.mean(mean_prediction, axis=0)##
                # cov_pred = np.cov(mean_prediction)##
                # np.save(st.path + 'covdata.npy', cov_data)##
                # np.save(st.path + 'covpred.npy', cov_pred)##
                acc = acc / 288
                print("%02d th Accuracy: %.3f" % (i, acc))
                print("%02d th kappa coefficient: %.3f" % (i,kappa(gt1, results4)))
        tf.reset_default_graph()
