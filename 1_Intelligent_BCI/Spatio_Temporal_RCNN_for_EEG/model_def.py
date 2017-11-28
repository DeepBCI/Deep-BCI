import numpy as np
import tensorflow as tf
import setting as st

class model_def:
    def __init__(self):
        self.fm = 64
        self.fcnode = 64
    # Modules
    def init_weight_bias(self, name, shape, filtercnt, trainable):
        weights = tf.get_variable(name=name + "w", shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  dtype=tf.float32, trainable=trainable)
        biases = tf.Variable(initial_value=tf.constant(0.1, shape=[filtercnt], dtype=tf.float32),
                             name=name + "b", trainable=trainable)
        return weights, biases
    def conv_layer(self, data, weight, bias, padding):
        conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
        return tf.nn.relu(tf.nn.bias_add(conv, bias))
    def batch_norm(self, data):
        return tf.nn.batch_normalization(x=data, mean=0, variance=1, offset=None, scale=None, variance_epsilon=0.00000001)
    def dropout(self, data, dropout):
        return tf.nn.dropout(data, dropout)
    def pool_layer(self, data):
        return tf.nn.max_pool(value=data, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding="VALID")
    def fc_layer(self, data, weight, bias):
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]
        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
        hidden = tf.nn.relu(hidden)
        return hidden
    def output_layer(self, data, weight, bias, label):
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]
        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden, labels=label)), tf.nn.softmax(hidden)

    # Models
    def RCNN(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        #RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d3, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #Fully Connected layer
        wfc, bfc = self.init_weight_bias(name="fclayer", shape=[8 * self.fm, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer = self.fc_layer(data=bnsc, weight=wfc, bias=bfc)
        bnfclayer = self.batch_norm(fclayer)
        dfclayer = self.dropout(bnfclayer, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def Inception_RCNN(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
        else:
            batch_size = 700-time_cnt+1

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)


        #Inception_RCL1
        rcl1w1a, rcl1b1a = self.init_weight_bias(name="rcl1conv1a", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv1a = self.conv_layer(data=data_node, weight=rcl1w1a, bias=rcl1b1a, padding="SAME")

        rcl1w1b, rcl1b1b = self.init_weight_bias(name="rlc1conv1b", shape=[1, 3, 1, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv1b = self.conv_layer(data=data_node, weight=rcl1w1b, bias=rcl1b1b, padding="SAME")

        rcl1w1c, rcl1b1c = self.init_weight_bias(name="rcl1conv1c", shape=[1, 5, 1, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv1c = self.conv_layer(data=data_node, weight=rcl1w1c, bias=rcl1b1c, padding="SAME")

        # rcl1w1d, rcl1b1d, = self.init_weight_bias(name="rcl1conv1d", shape=[1, 7, 1, self.fm], filtercnt=self.fm, trainable=train)
        # rcl1conv1d = self.conv_layer(data=data_node, weight=rcl1w1d, bias=rcl1b1d, padding="SAME")
        rcl1sum1 = rcl1conv1a + rcl1conv1b + rcl1conv1c# + rcl1conv1d

        rcl1w1, rcl1b1 = self.init_weight_bias(name="rcl1conv1", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)###
        rcl1conv1 = self.conv_layer(data=rcl1sum1, weight=rcl1w1, bias=rcl1b1, padding="SAME")####

        rcl1w2a, rcl1b2a = self.init_weight_bias(name="rcl1conv2a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv2a = self.conv_layer(data=rcl1conv1, weight=rcl1w2a, bias=rcl1b2a, padding="SAME")

        rcl1w2b, rcl1b2b = self.init_weight_bias(name="rcl1conv2b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv2b = self.conv_layer(data=rcl1conv1, weight=rcl1w2b, bias=rcl1b2b, padding="SAME")

        rcl1w2c, rcl1b2c = self.init_weight_bias(name="conv2c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv2c = self.conv_layer(data=rcl1conv1, weight=rcl1w2c, bias=rcl1b2c, padding="SAME")

        # rcl1w2d, rcl1b2d, = self.init_weight_bias(name="rcl1conv2d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl1conv2d = self.conv_layer(data=rcl1conv1, weight=rcl1w2d, bias=rcl1b2d, padding="SAME")
        rcl1sum2 = rcl1conv2a + rcl1conv2b + rcl1conv2c# + rcl1conv2d
        rcl1sum2 = rcl1sum1 + rcl1sum2

        rcl1w2, rcl1b2 = self.init_weight_bias(name="rcl1conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)#######
        rcl1conv2 = self.conv_layer(data=rcl1sum2, weight=rcl1w2, bias=rcl1b2, padding="SAME")########

        rcl1w3a, rcl1b3a = self.init_weight_bias(name="rcl1conv3a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv3a = self.conv_layer(data=rcl1conv2, weight=rcl1w3a, bias=rcl1b3a, padding="SAME")

        rcl1w3b, rcl1b3b = self.init_weight_bias(name="rcl1conv3b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv3b = self.conv_layer(data=rcl1conv2, weight=rcl1w3b, bias=rcl1b3b, padding="SAME")

        rcl1w3c, rcl1b3c = self.init_weight_bias(name="rcl1conv3c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv3c = self.conv_layer(data=rcl1conv2, weight=rcl1w3c, bias=rcl1b3c, padding="SAME")

        # rcl1w3d, rcl1b3d, = self.init_weight_bias(name="rcl1conv3d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl1conv3d = self.conv_layer(data=rcl1conv2, weight=rcl1w3d, bias=rcl1b3d, padding="SAME")
        rcl1sum3 = rcl1conv3a + rcl1conv3b + rcl1conv3c# + rcl1conv3d
        rcl1sum3 = rcl1sum1 + rcl1sum3

        rcl1w3, rcl1b3 = self.init_weight_bias(name="rcl1conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)  #######
        rcl1conv3 = self.conv_layer(data=rcl1sum3, weight=rcl1w3, bias=rcl1b3, padding="SAME")  ########



        rcl1w4a, rcl1b4a = self.init_weight_bias(name="rcl1conv4a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv4a = self.conv_layer(data=rcl1conv3, weight=rcl1w4a, bias=rcl1b4a, padding="SAME")

        rcl1w4b, rcl1b4b = self.init_weight_bias(name="rcl1conv4b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv4b = self.conv_layer(data=rcl1conv3, weight=rcl1w4b, bias=rcl1b4b, padding="SAME")

        rcl1w4c, rcl1b4c = self.init_weight_bias(name="rcl1conv4c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl1conv4c = self.conv_layer(data=rcl1conv3, weight=rcl1w4c, bias=rcl1b4c, padding="SAME")

        # rcl1w4d, rcl1b4d, = self.init_weight_bias(name="rcl1conv4d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl1conv4d = self.conv_layer(data=rcl1conv3, weight=rcl1w4d, bias=rcl1b4d, padding="SAME")
        rcl1sum4 = rcl1conv4a + rcl1conv4b + rcl1conv4c# + rcl1conv4d
        rcl1sum4 = rcl1sum1 + rcl1sum4

        rcl1w4, rcl1b4 = self.init_weight_bias(name="rcl1conv4", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)  #######
        rcl1conv4 = self.conv_layer(data=rcl1sum4, weight=rcl1w4, bias=rcl1b4, padding="SAME")  ########

        rcl1bn1 = self.batch_norm(rcl1conv4)
        rcl1p1 = self.pool_layer(rcl1bn1)
        rcl1d1 = self.dropout(rcl1p1, dropout=self.dr)





        #Inception_RCL2
        rcl2w1a, rcl2b1a = self.init_weight_bias(name="rcl2conv1a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv1a = self.conv_layer(data=rcl1d1, weight=rcl2w1a, bias=rcl2b1a, padding="SAME")

        rcl2w1b, rcl2b1b = self.init_weight_bias(name="rlc2conv1b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv1b = self.conv_layer(data=rcl1d1, weight=rcl2w1b, bias=rcl2b1b, padding="SAME")

        rcl2w1c, rcl2b1c = self.init_weight_bias(name="rcl2conv1c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv1c = self.conv_layer(data=rcl1d1, weight=rcl2w1c, bias=rcl2b1c, padding="SAME")

        # rcl2w1d, rcl2b1d, = self.init_weight_bias(name="rcl2conv1d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl2conv1d = self.conv_layer(data=rcl1d1, weight=rcl2w1d, bias=rcl2b1d, padding="SAME")
        rcl2sum1 = rcl2conv1a + rcl2conv1b + rcl2conv1c# + rcl2conv1d




        rcl2w1, rcl2b1 = self.init_weight_bias(name="rcl2conv1", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)  #######
        rcl2conv1 = self.conv_layer(data=rcl2sum1, weight=rcl2w1, bias=rcl2b1, padding="SAME")  ########



        rcl2w2a, rcl2b2a = self.init_weight_bias(name="rcl2conv2a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv2a = self.conv_layer(data=rcl2conv1, weight=rcl2w2a, bias=rcl2b2a, padding="SAME")

        rcl2w2b, rcl2b2b = self.init_weight_bias(name="rcl2conv2b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv2b = self.conv_layer(data=rcl2conv1, weight=rcl2w2b, bias=rcl2b2b, padding="SAME")

        rcl2w2c, rcl2b2c = self.init_weight_bias(name="rcl2conv2c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv2c = self.conv_layer(data=rcl2conv1, weight=rcl2w2c, bias=rcl2b2c, padding="SAME")

        # rcl2w2d, rcl2b2d, = self.init_weight_bias(name="rcl2conv2d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl2conv2d = self.conv_layer(data=rcl2conv1, weight=rcl2w2d, bias=rcl2b2d, padding="SAME")
        rcl2sum2 = rcl2conv2a + rcl2conv2b + rcl2conv2c #+ rcl2conv2d
        rcl2sum2 = rcl2sum1 + rcl2sum2




        rcl2w2, rcl2b2 = self.init_weight_bias(name="rcl2conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)  #######
        rcl2conv2 = self.conv_layer(data=rcl2sum2, weight=rcl2w2, bias=rcl2b2, padding="SAME")  ########







        rcl2w3a, rcl2b3a = self.init_weight_bias(name="rcl2conv3a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv3a = self.conv_layer(data=rcl2conv2, weight=rcl2w3a, bias=rcl2b3a, padding="SAME")

        rcl2w3b, rcl2b3b = self.init_weight_bias(name="rcl2conv3b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv3b = self.conv_layer(data=rcl2conv2, weight=rcl2w3b, bias=rcl2b3b, padding="SAME")

        rcl2w3c, rcl2b3c = self.init_weight_bias(name="rcl2conv3c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv3c = self.conv_layer(data=rcl2conv2, weight=rcl2w3c, bias=rcl2b3c, padding="SAME")

        # rcl2w3d, rcl2b3d, = self.init_weight_bias(name="rcl2conv3d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl2conv3d = self.conv_layer(data=rcl2conv2, weight=rcl2w3d, bias=rcl2b3d, padding="SAME")
        rcl2sum3 = rcl2conv3a + rcl2conv3b + rcl2conv3c# + rcl2conv3d
        rcl2sum3 = rcl2sum1 + rcl2sum3






        rcl2w3, rcl2b3 = self.init_weight_bias(name="rcl2conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)  #######
        rcl2conv3 = self.conv_layer(data=rcl2sum3, weight=rcl2w3, bias=rcl2b3, padding="SAME")  ########







        rcl2w4a, rcl2b4a = self.init_weight_bias(name="rcl2conv4a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv4a = self.conv_layer(data=rcl2conv3, weight=rcl2w4a, bias=rcl2b4a, padding="SAME")

        rcl2w4b, rcl2b4b = self.init_weight_bias(name="rcl2conv4b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv4b = self.conv_layer(data=rcl2conv3, weight=rcl2w4b, bias=rcl2b4b, padding="SAME")

        rcl2w4c, rcl2b4c = self.init_weight_bias(name="rcl2conv4c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        rcl2conv4c = self.conv_layer(data=rcl2conv3, weight=rcl2w4c, bias=rcl2b4c, padding="SAME")

        # rcl2w4d, rcl2b4d, = self.init_weight_bias(name="rcl2conv4d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # rcl2conv4d = self.conv_layer(data=rcl2conv3, weight=rcl2w4d, bias=rcl2b4d, padding="SAME")
        rcl2sum4 = rcl2conv4a + rcl2conv4b + rcl2conv4c# + rcl2conv4d
        rcl2sum4 = rcl2sum1 + rcl2sum4





        rcl2w4, rcl2b4 = self.init_weight_bias(name="rcl2conv4", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)  #######
        rcl2conv4 = self.conv_layer(data=rcl2sum4, weight=rcl2w4, bias=rcl2b4, padding="SAME")  ########


        rcl2bn1 = self.batch_norm(rcl2conv4)
        rcl2p1 = self.pool_layer(rcl2bn1)
        rcl2d1 = self.dropout(rcl2p1, dropout=self.dr)



        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=rcl2d1, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        psc = self.pool_layer(bnsc)


       # #Fully Connected layer
       # wfc, bfc = self.init_weight_bias(name="fclayer", shape=[8 * self.fm, self.fcnode], filtercnt=self.fcnode, trainable=train)
       # fclayer = self.fc_layer(data = psc, weight=wfc, bias=bfc)
       # bnfclayer = self.batch_norm(fclayer)
       # dfclayer = self.dropout(bnfclayer, dropout=self.dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[8 * self.fm, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(psc, weight=wo, bias=bo, label=label_node)
        # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_out, labels=label_node))
        # soft_max = tf.nn.softmax(l_out)
        return cross_entropy, soft_max, data_node, label_node, wsc ,bsc
    def Inception_RCNN2(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
        else:
            batch_size = 700-time_cnt+1

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)


        #RCL1
        w1, b1, = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        w1d, b1d = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        sum1d = conv1+conv1d
        bn1d = self.batch_norm(sum1d)
        # p1 = self.pool_layer(bn1d)
        # d1 = self.dropout(p1, dropout=self.dr)


        # First inception module
        win1a, bin1a = self.init_weight_bias(name="convin1a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin1a = self.conv_layer(data=bn1d, weight=win1a, bias=bin1a, padding="SAME")

        win1b, bin1b = self.init_weight_bias(name="convin1b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin1b = self.conv_layer(data=bn1d, weight=win1b, bias=bin1b, padding="SAME")

        win1c, bin1c = self.init_weight_bias(name="convin1c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin1c = self.conv_layer(data=bn1d, weight=win1c, bias=bin1c, padding="SAME")

        win1d, bin1d = self.init_weight_bias(name="convin1d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin1d = self.conv_layer(data=bn1d, weight=win1d, bias=bin1d, padding="SAME")
        sumin1 = convin1a + convin1b + convin1c + convin1d
        bnin1 = self.batch_norm(sumin1)
        pin1 = self.pool_layer(bnin1)
        din1 = self.dropout(pin1, dropout=self.dr)



        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=din1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        sum2d = conv2 + conv2d
        bn2d = self.batch_norm(sum2d)
        # p2 = self.pool_layer(bn2d)
        # d2 = self.dropout(p2, dropout=self.dr)


        # Second inception module
        win2a, bin2a = self.init_weight_bias(name="convin2a", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin2a = self.conv_layer(data=bn2d, weight=win2a, bias=bin2a, padding="SAME")

        win2b, bin2b = self.init_weight_bias(name="convin2b", shape=[1, 3, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin2b = self.conv_layer(data=bn2d, weight=win2b, bias=bin2b, padding="SAME")

        win2c, bin2c = self.init_weight_bias(name="convin2c", shape=[1, 5, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin2c = self.conv_layer(data=bn2d, weight=win2c, bias=bin2c, padding="SAME")

        win2d, bin2d = self.init_weight_bias(name="convin2d", shape=[1, 7, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        convin2d = self.conv_layer(data=bn2d, weight=win2d, bias=bin2d, padding="SAME")
        sumin2 = convin2a + convin2b + convin2c + convin2d
        bnin2 = self.batch_norm(sumin2)
        pin2 = self.pool_layer(bnin2)
        din2 = self.dropout(pin2, dropout=self.dr)




        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=din2, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        psc = self.pool_layer(bnsc)




        #Output layer
        wf, bf = self.init_weight_bias(name="fc", shape=[1 * 8 * self.fm, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(psc, weight=wf, bias=bf, label=label_node)
        # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_out, labels=label_node))
        # soft_max = tf.nn.softmax(l_out)
        return cross_entropy, soft_max, data_node, label_node, w1,b1
    def RCNN2(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
        else:
            batch_size = 700-time_cnt+1

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)


        #RCL1
        #channel_cnt(22) X time_cnt    @  feature_map(64)
        w1, b1, = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=self.dr)




        # RCL2
        #channel_cnt(22) X time_cnt/4    @  feature_map(64)
        w2, b2, = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm,
                                        trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm,
                                         trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm,
                                         trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm,
                                         trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=self.dr)



        #Spatial Convolutional layer
        #channel_cnt(22) X time_cnt/16    @  feature_map(64)
        w3, b3 = self.init_weight_bias(name="conv3", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="VALID")
        bn3 = self.batch_norm(conv3)
        p3 = self.pool_layer(bn3)
        d3 = self.dropout(p3, dropout=self.dr)



        #Sum feature map
        #1 X time_cnt/16    @  1
        w4, b4 = self.init_weight_bias(name="conv4", shape=[1, 1, self.fm, 1], filtercnt=1, trainable=train)
        conv4 = self.conv_layer(data = d3, weight=w4, bias=b4, padding="SAME")
        bn4 = self.batch_norm(conv4)
        d4 = self.dropout(bn4, dropout=self.dr)


        #Fully connected layer
        #1 X time_cnt/16    @  1
        wfc, bfc = self.init_weight_bias(name="fclayer", shape=[1 * 8 * 1, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fc = self.fc_layer(data=d4, weight=wfc, bias=bfc)
        bnfc = self.batch_norm(fc)



        #Output layer
        wf, bf = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(bnfc, weight=wf, bias=bf, label=label_node)
        # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_out, labels=label_node))
        # soft_max = tf.nn.softmax(l_out)
        return cross_entropy, soft_max, data_node, label_node, w1,b1
    def RCNN3(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0


        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)


        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)



        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)



        #RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        d3 = self.dropout(p3, dropout=dr)




        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d3, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #1st Fully Connected layer
        wfc1, bfc1 = self.init_weight_bias(name="fclayer1", shape=[8 * self.fm, self.fcnode1], filtercnt=self.fcnode1, trainable=train)
        fclayer1 = self.fc_layer(data=bnsc, weight=wfc1, bias=bfc1)
        bnfclayer1 = self.batch_norm(fclayer1)
        dfclayer1 = self.dropout(bnfclayer1, dropout=dr)


        #2nd Fully Connected layer
        wfc2, bfc2 = self.init_weight_bias(name="fclayer2", shape=[self.fcnode1, self.fcnode2], filtercnt=self.fcnode2, trainable=train)
        fclayer2 = self.fc_layer(data=dfclayer1, weight=wfc2, bias=bfc2)
        bnfclayer2 = self.batch_norm(fclayer2)
        dfclayer2 = self.dropout(bnfclayer2, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode2, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer2, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, w1, b1
    def RCNN4(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        #RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[3, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d3, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #Fully Connected layer
        wfc, bfc = self.init_weight_bias(name="fclayer", shape=[8 * 20 * self.fm, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer = self.fc_layer(data=bnsc, weight=wfc, bias=bfc)
        bnfclayer = self.batch_norm(fclayer)
        dfclayer = self.dropout(bnfclayer, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def RCNN5(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        #RCL3
        # w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        # bn3 = self.batch_norm(conv3)
        #
        # w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        # sum3a = conv3+conv3a
        # bn3a = self.batch_norm(sum3a)
        #
        # w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        # sum3b = conv3+conv3b
        # bn3b = self.batch_norm(sum3b)
        #
        # w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        # sum3c = conv3+conv3c
        # bn3c = self.batch_norm(sum3c)
        # p3 = self.pool_layer(bn3c)
        # d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d2, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #Fully Connected layer
        wfc, bfc = self.init_weight_bias(name="fclayer", shape=[8 * 4 * self.fm, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer = self.fc_layer(data=bnsc, weight=wfc, bias=bfc)
        bnfclayer = self.batch_norm(fclayer)
        dfclayer = self.dropout(bnfclayer, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def RCNN6(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        #RCL3
        # w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        # bn3 = self.batch_norm(conv3)
        #
        # w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        # sum3a = conv3+conv3a
        # bn3a = self.batch_norm(sum3a)
        #
        # w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        # sum3b = conv3+conv3b
        # bn3b = self.batch_norm(sum3b)
        #
        # w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        # sum3c = conv3+conv3c
        # bn3c = self.batch_norm(sum3c)
        # p3 = self.pool_layer(bn3c)
        # d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d2, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #Fully Connected layer1
        wfc1, bfc1 = self.init_weight_bias(name="fclayer1", shape=[8 * 4 * self.fm, self.fcnode * 2], filtercnt=self.fcnode * 2, trainable=train)
        fclayer1 = self.fc_layer(data=bnsc, weight=wfc1, bias=bfc1)
        bnfclayer1 = self.batch_norm(fclayer1)
        dfclayer1 = self.dropout(bnfclayer1, dropout=dr)


        #Fully Connected layer2
        wfc2, bfc2 = self.init_weight_bias(name="fclayer2", shape=[self.fcnode * 2, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer2 = self.fc_layer(data=dfclayer1, weight=wfc2, bias=bfc2)
        bnfclayer2 = self.batch_norm(fclayer2)
        dfclayer2 = self.dropout(bnfclayer2, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer2, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def RCNN7(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        # w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        # conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        # bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, 1, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=data_node, weight=w1a, bias=b1a, padding="SAME")
        # sum1a = conv1+conv1a
        bn1a = self.batch_norm(conv1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1a+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1a+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        # RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d3, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #Fully Connected layer1
        wfc1, bfc1 = self.init_weight_bias(name="fclayer1", shape=[8 * self.fm, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer1 = self.fc_layer(data=bnsc, weight=wfc1, bias=bfc1)
        bnfclayer1 = self.batch_norm(fclayer1)
        dfclayer1 = self.dropout(bnfclayer1, dropout=dr)


        #Fully Connected layer2
        # wfc2, bfc2 = self.init_weight_bias(name="fclayer2", shape=[self.fcnode * 2, self.fcnode], filtercnt=self.fcnode, trainable=train)
        # fclayer2 = self.fc_layer(data=dfclayer1, weight=wfc2, bias=bfc2)
        # bnfclayer2 = self.batch_norm(fclayer2)
        # dfclayer2 = self.dropout(bnfclayer2, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer1, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def RCNN8(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=data_node, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1 = self.conv_layer(data=bnsc, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        # RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        # d3 = self.dropout(p3, dropout=dr)


        #Fully Connected layer1
        wfc1, bfc1 = self.init_weight_bias(name="fclayer1", shape=[8 * self.fm, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer1 = self.fc_layer(data=p3, weight=wfc1, bias=bfc1)
        bnfclayer1 = self.batch_norm(fclayer1)
        dfclayer1 = self.dropout(bnfclayer1, dropout=dr)


        #Fully Connected layer2
        # wfc2, bfc2 = self.init_weight_bias(name="fclayer2", shape=[self.fcnode * 2, self.fcnode], filtercnt=self.fcnode, trainable=train)
        # fclayer2 = self.fc_layer(data=dfclayer1, weight=wfc2, bias=bfc2)
        # bnfclayer2 = self.batch_norm(fclayer2)
        # dfclayer2 = self.dropout(bnfclayer2, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer1, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def RCNN9(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, self.fm/4, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1a = self.conv_layer(data=bn1, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm/4, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm/4, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm/4, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm/2, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm/2, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm/2, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        #RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm/2, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm * 2], filtercnt=self.fm*2, trainable=train)
        spatialconv = self.conv_layer(data=d3, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #Fully Connected layer
        wfc, bfc = self.init_weight_bias(name="fclayer", shape=[8 * self.fm*2, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer = self.fc_layer(data=bnsc, weight=wfc, bias=bfc)
        bnfclayer = self.batch_norm(fclayer)
        dfclayer = self.dropout(bnfclayer, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
    def RCNN10(self, train, channel_cnt, time_cnt):
        if train:
            batch_size = st.batch_size
            dr = 0.5
        else:
            batch_size = 700-time_cnt+1
            dr = 1.0

        data_node = tf.placeholder(tf.float32, shape=(batch_size, channel_cnt, time_cnt, 1))
        label_node = tf.placeholder(tf.int64, shape=batch_size)

        #RCL1
        #w1, b1 = self.init_weight_bias(name="conv1", shape=[1, 1, 1, self.fm], filtercnt=self.fm, trainable=train)
        #conv1 = self.conv_layer(data=data_node, weight=w1, bias=b1, padding="SAME")
        #bn1 = self.batch_norm(conv1)

        w1a, b1a = self.init_weight_bias(name="conv1a", shape=[1, 9, 1, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1a = self.conv_layer(data=data_node, weight=w1a, bias=b1a, padding="SAME")
        sum1a = conv1a #conv1+conv1a
        bn1a = self.batch_norm(sum1a)

        w1b, b1b = self.init_weight_bias(name="conv1b", shape=[1, 9, self.fm/4, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1b = self.conv_layer(data=bn1a, weight=w1b, bias=b1b, padding="SAME")
        sum1b = sum1a + conv1b #conv1+conv1b
        bn1b = self.batch_norm(sum1b)

        w1c, b1c = self.init_weight_bias(name="conv1c", shape=[1, 9, self.fm/4, self.fm/4], filtercnt=self.fm/4, trainable=train)
        conv1c = self.conv_layer(data=bn1b, weight=w1c, bias=b1c, padding="SAME")
        sum1c = sum1a + conv1c #conv1+conv1c
        bn1c = self.batch_norm(sum1c)

        # w1d, b1d, = self.init_weight_bias(name="conv1d", shape=[1,9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv1d = self.conv_layer(data=bn1c, weight=w1d, bias=b1d, padding="SAME")
        # sum1d = conv1+conv1d
        # bn1d = self.batch_norm(sum1d)
        p1 = self.pool_layer(bn1c)
        d1 = self.dropout(p1, dropout=dr)


        # RCL2
        w2, b2 = self.init_weight_bias(name="conv2", shape=[1, 1, self.fm/4, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2 = self.conv_layer(data=d1, weight=w2, bias=b2, padding="SAME")
        bn2 = self.batch_norm(conv2)

        w2a, b2a = self.init_weight_bias(name="conv2a", shape=[1, 9, self.fm/2, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2a = self.conv_layer(data=bn2, weight=w2a, bias=b2a, padding="SAME")
        sum2a = conv2 + conv2a
        bn2a = self.batch_norm(sum2a)

        w2b, b2b = self.init_weight_bias(name="conv2b", shape=[1, 9, self.fm/2, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2b = self.conv_layer(data=bn2a, weight=w2b, bias=b2b, padding="SAME")
        sum2b = conv2 + conv2b
        bn2b = self.batch_norm(sum2b)

        w2c, b2c = self.init_weight_bias(name="conv2c", shape=[1, 9, self.fm/2, self.fm/2], filtercnt=self.fm/2, trainable=train)
        conv2c = self.conv_layer(data=bn2b, weight=w2c, bias=b2c, padding="SAME")
        sum2c = conv2 + conv2c
        bn2c = self.batch_norm(sum2c)

        # w2d, b2d, = self.init_weight_bias(name="conv2d", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        # conv2d = self.conv_layer(data=bn2c, weight=w2d, bias=b2d, padding="SAME")
        # sum2d = conv2 + conv2d
        # bn2d = self.batch_norm(sum2d)
        p2 = self.pool_layer(bn2c)
        d2 = self.dropout(p2, dropout=dr)


        #RCL3
        w3, b3 = self.init_weight_bias(name="conv3", shape=[1, 1, self.fm/2, self.fm], filtercnt=self.fm, trainable=train)
        conv3 = self.conv_layer(data=d2, weight=w3, bias=b3, padding="SAME")
        bn3 = self.batch_norm(conv3)

        w3a, b3a = self.init_weight_bias(name="conv3a", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3a = self.conv_layer(data=bn3, weight=w3a, bias=b3a, padding="SAME")
        sum3a = conv3+conv3a
        bn3a = self.batch_norm(sum3a)

        w3b, b3b = self.init_weight_bias(name="conv3b", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3b = self.conv_layer(data=bn3a, weight=w3b, bias=b3b, padding="SAME")
        sum3b = conv3+conv3b
        bn3b = self.batch_norm(sum3b)

        w3c, b3c = self.init_weight_bias(name="conv3c", shape=[1, 9, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        conv3c = self.conv_layer(data=bn3b, weight=w3c, bias=b3c, padding="SAME")
        sum3c = conv3+conv3c
        bn3c = self.batch_norm(sum3c)
        p3 = self.pool_layer(bn3c)
        d3 = self.dropout(p3, dropout=dr)


        #Spatial Convolutional layer
        wsc, bsc = self.init_weight_bias(name="spatialconv", shape=[channel_cnt, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        spatialconv = self.conv_layer(data=d3, weight=wsc, bias=bsc, padding="VALID")
        bnsc = self.batch_norm(spatialconv)
        # psc = self.pool_layer(bnsc)


        #1X1 Conv
        wc, bc = self.init_weight_bias(name="finalconv", shape=[1, 1, self.fm, self.fm], filtercnt=self.fm, trainable=train)
        finalconv = self.conv_layer(data=bnsc, weight=wc, bias=bc, padding="SAME")
        bnfinal = self.batch_norm(finalconv)


        #Fully Connected layer1
        wfc1, bfc1 = self.init_weight_bias(name="fclayer1", shape=[8 * self.fm, 8 * self.fcnode], filtercnt=8 * self.fcnode, trainable=train)
        fclayer1 = self.fc_layer(data=bnfinal, weight=wfc1, bias=bfc1)
        bnfclayer1 = self.batch_norm(fclayer1)
        dfclayer1 = self.dropout(bnfclayer1, dropout=dr)



        #Fully Connected layer2
        wfc2, bfc2 = self.init_weight_bias(name="fclayer2", shape=[8 * self.fcnode, self.fcnode], filtercnt=self.fcnode, trainable=train)
        fclayer2 = self.fc_layer(data=dfclayer1, weight=wfc2, bias=bfc2)
        bnfclayer2 = self.batch_norm(fclayer2)
        dfclayer2 = self.dropout(bnfclayer2, dropout=dr)


        #Output layer
        wo, bo = self.init_weight_bias(name="output", shape=[self.fcnode, 4], filtercnt=4, trainable=train)
        cross_entropy, soft_max = self.output_layer(dfclayer2, weight=wo, bias=bo, label=label_node)
        return cross_entropy, soft_max, data_node, label_node, wsc, bsc
