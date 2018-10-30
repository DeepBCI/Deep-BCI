import tensorflow as tf

def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)

def get_params(name, shape, n_filter):
    w = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    b = tf.Variable(name=name + "b", initial_value=tf.constant(0.1, shape=[n_filter], dtype=tf.float32))
    return w, b

def conv(input, w, b, strides, padding):
    conv = tf.nn.conv2d(input=input, filter=w, strides=strides, padding=padding)
    return tf.nn.bias_add(conv, b)

def maxpool(input, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1]):
    return tf.nn.max_pool(value=input, ksize=ksize, strides=strides, padding="VALID")

def deconv(input, newshape, w, b, padding):
    input = tf.image.resize_bilinear(images=input, size=newshape)
    deconv = tf.nn.conv2d(input=input, filter=w, strides=[1, 1, 1, 1], padding=padding)
    return tf.nn.bias_add(deconv, b)

def batchnorm(input):
    return tf.nn.batch_normalization(x=input, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-8)

def dropout(input, keep):
    return tf.nn.dropout(x=input, keep_prob=keep)

def RSTNN(input, adversarial, keep=0.5, num_class=4, reuse=None):
    with tf.variable_scope("RSTNN") as scope:
        if reuse:
            scope.reuse_variables()
            keep = 1.0

        # 1st RCL
        w1, b1 = get_params(name="rcl1", shape=(1, 9, 1, 16), n_filter=16)
        rcl1 = tf.nn.relu(batchnorm(conv(input=input, w=w1, b=b1, strides=[1, 1, 1, 1], padding="SAME"))) #(batch, 22, 512, 16)
        w1a, b1a = get_params(name="rcl1a", shape=(1, 9, 16, 16), n_filter=16)
        rcl1a = tf.nn.relu(batchnorm(conv(input=rcl1, w=w1a, b=b1a, strides=[1, 1, 1, 1], padding="SAME")))
        rcl1a = rcl1 + rcl1a
        w1b, b1b = get_params(name="rcl1b", shape=(1, 9, 16, 16), n_filter=16)
        rcl1b = tf.nn.relu(batchnorm(conv(input=rcl1a, w=w1b, b=b1b, strides=[1, 1, 1, 1], padding="SAME")))
        rcl1b = rcl1 + rcl1b
        rcl1b = dropout(maxpool(rcl1b, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1]), keep) #(batch, 22, 128, 16)

        # 2nd RCL
        w2, b2 = get_params(name="rcl2", shape=(1, 1, 16, 32), n_filter=32)
        rcl2 = tf.nn.relu(batchnorm(conv(input=rcl1b, w=w2, b=b2, strides=[1, 1, 1, 1], padding="SAME")))
        w2a, b2a = get_params(name="rcl2a", shape=(1, 9, 32, 32), n_filter=32)
        rcl2a = tf.nn.relu(batchnorm(conv(input=rcl2, w=w2a, b=b2a, strides=[1, 1, 1, 1], padding="SAME")))
        rcl2a = rcl2 + rcl2a
        w2b, b2b = get_params(name="rcl2b", shape=(1, 9, 32, 32), n_filter=32)
        rcl2b = tf.nn.relu(batchnorm(conv(input=rcl2a, w=w2b, b=b2b, strides=[1, 1, 1, 1], padding="SAME")))
        rcl2b = rcl2 + rcl2b
        w2c, b2c = get_params(name="rcl2c", shape=(1, 9, 32, 32), n_filter=32)
        rcl2c = tf.nn.relu(batchnorm(conv(input=rcl2b, w=w2c, b=b2c, strides=[1, 1, 1, 1], padding="SAME")))
        rcl2c = rcl2 + rcl2c
        rcl2c = dropout(maxpool(rcl2c, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1]), keep) #(batch, 22, 32, 32)

        # 3rd RCL
        w3, b3 = get_params(name="rcl3", shape=(1, 1, 32, 64), n_filter=64)
        rcl3 = tf.nn.relu(batchnorm(conv(input=rcl2c, w=w3, b=b3, strides=[1, 1, 1, 1], padding="SAME")))
        w3a, b3a = get_params(name="rcl3a", shape=(1, 9, 64, 64), n_filter=64)
        rcl3a = tf.nn.relu(batchnorm(conv(input=rcl3, w=w3a, b=b3a, strides=[1, 1, 1, 1], padding="SAME")))
        rcl3a = rcl3 + rcl3a
        w3b, b3b = get_params(name="rcl3b", shape=(1, 9, 64, 64), n_filter=64)
        rcl3b = tf.nn.relu(batchnorm(conv(input=rcl3a, w=w3b, b=b3b, strides=[1, 1, 1, 1], padding="SAME")))
        rcl3b = rcl3 + rcl3b
        w3c, b3c = get_params(name="rcl3c", shape=(1, 9, 64, 64), n_filter=64)
        rcl3c = tf.nn.relu(batchnorm(conv(input=rcl3b, w=w3c, b=b3c, strides=[1, 1, 1, 1], padding="SAME")))
        rcl3c = rcl3 + rcl3c
        rcl3c = dropout(maxpool(rcl3c, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1]), keep) #(batch, 22, 8, 64)

        # Spatial feature extraction
        ws, bs = get_params(name="spatial", shape=(22, 1, 64, 64), n_filter=64)
        convsp = tf.nn.relu(batchnorm(conv(input=rcl3c, w=ws, b=bs, strides=[1, 1, 1, 1], padding="VALID"))) #(batch, 1, 8, 64)
        ws1, bs1 = get_params(name="spatial1_1", shape=(1, 1, 64, 64), n_filter=64)
        convsp1 = tf.nn.relu(batchnorm(conv(input=convsp, w=ws1, b=bs1, strides=[1, 1, 1, 1], padding="VALID"))) #(batch, 1, 8, 64)

        # Classifier
        batch = convsp1.get_shape().as_list()[0]
        convsp1 = tf.reshape(convsp1, shape=(batch, -1))
        num_hidden = convsp1.get_shape().as_list()[-1]
        wfc1, bfc1 = get_params(name="fc1", shape=(num_hidden, num_hidden), n_filter=num_hidden)
        fc1 = dropout(tf.nn.bias_add(tf.matmul(convsp1, wfc1), bfc1), keep)
        wfc2, bfc2 = get_params(name="fc2", shape=(num_hidden, num_hidden/8), n_filter=num_hidden/8)
        fc2 = dropout(tf.nn.bias_add(tf.matmul(fc1, wfc2), bfc2), keep)

        if adversarial == True:
            wfc3, bfc3 = get_params(name="fc3", shape=(num_hidden/8, num_class+1), n_filter=num_class+1)
            logits = tf.nn.bias_add(tf.matmul(fc2, wfc3), bfc3)
            output = tf.argmax(tf.sigmoid(logits), -1)
        else:
            wfc3, bfc3 = get_params(name="fc3", shape=(num_hidden/8, num_class), n_filter=num_class)
            logits = tf.nn.bias_add(tf.matmul(fc2, wfc3), bfc3)
            output = tf.argmax(tf.sigmoid(logits), -1)
    return logits, output, convsp1, rcl3c, ws #logit, prediction, spatial convolution inputs, and spatial convolution weights


def DeepConvNet(input, adversarial, keep=0.5, num_class = 4, reuse=None):
    with tf.variable_scope("DeepConvNet") as scope:
        if reuse:
            scope.reuse_variables()
            keep = 1.0

        # 1st Temporal Convolution, Linear activation
        w1, b1 = get_params(name="deep1", shape=(1, 10, 1, 25), n_filter=25)
        conv1 = conv(input=input, w=w1, b=b1, strides=[1, 1, 1, 1], padding="VALID") #(batch, 22, 168, 25)

        # 2nd Spatial Convolution, ReLU activation
        w2, b2 = get_params(name="deep2", shape=(22, 1, 25, 25), n_filter=25)
        conv2 = conv(input=conv1, w=w2, b=b2, strides=[1, 1, 1, 1], padding="VALID")
        conv2 = batchnorm(conv2)
        conv2 = tf.nn.elu(conv2)
        # conv2 = tf.nn.relu(conv2)  #(batch, 1, 168, 25)
        conv2 = maxpool(conv2) #(batch, 1, 167, 25)
        # conv2 = dropout(input=conv2, keep=keep)

        # 3rd Temporal Convolution, ReLU activation
        w3, b3 = get_params(name="deep3", shape=(1, 10, 25, 50), n_filter=50)
        conv3 = conv(input=conv2, w=w3, b=b3, strides=[1, 1, 1, 1], padding="VALID")
        conv3 = batchnorm(conv3)
        conv3 = tf.nn.elu(conv3)
        # conv3 = tf.nn.relu(conv3)  #(batch, 1, 53, 50)
        conv3 = maxpool(conv3) #(batch, 1, 52, 50)
        # conv3 = dropout(input=conv3, keep=keep)

        # 4th Temporal Convolution, ReLU activation
        w4, b4 = get_params(name="deep4", shape=(1, 10, 50, 100), n_filter=100)
        conv4 = conv(input=conv3, w=w4, b=b4, strides=[1, 1, 1, 1], padding="VALID")
        conv4 = batchnorm(conv4)
        conv4 = tf.nn.elu(conv4)
        # conv4 = tf.nn.relu(conv4) #(batch, 1, 15, 100)
        conv4 = maxpool(conv4) #(batch, 1, 14, 100)
        # conv4 = dropout(input=conv4, keep=keep)

        # 5th Temporal Convolution, ReLU activation
        w5, b5 = get_params(name="deep5", shape=(1, 10, 100, 200), n_filter=200)
        conv5 = conv(input=conv4, w=w5, b=b5, strides=[1, 1, 1, 1], padding="VALID")
        conv5 = batchnorm(conv5)
        conv5 = tf.nn.elu(conv5)
        # conv5 = tf.nn.relu(conv5)
        conv5 = maxpool(conv5) #(batch, 1, 2, 200)
        # conv5 = dropout(input=conv5, keep=keep)

        if adversarial == True:
            # Linear Classification, Softmax activation, k+1 class output
            batch = conv5.get_shape().as_list()[0]
            conv5 = tf.reshape(conv5, shape=(batch, -1)) #(batch, 400)
            num_hidden = conv5.get_shape().as_list()[-1]
            w6, b6 = get_params(name="deepout", shape=(num_hidden, num_class + 1), n_filter=num_class + 1)
            logits = tf.nn.bias_add(tf.matmul(conv5, w6), b6)
            output = tf.argmax(tf.sigmoid(logits), -1)
        else:
            # Linear Classification, Softmax activation, k class output
            batch = conv5.get_shape().as_list()[0]
            conv5 = tf.reshape(conv5, shape=(batch, -1)) #(batch, 400)
            num_hidden = conv5.get_shape().as_list()[-1]
            w6, b6 = get_params(name="deepout", shape=(num_hidden, num_class), n_filter=num_class)
            logits = tf.nn.bias_add(tf.matmul(conv5, w6), b6)
            output = tf.argmax(tf.sigmoid(logits), -1)
        return logits, output, conv5, w2 # For output, feature matching technique, and activation pattern

def DeepDeconvNet(input):
    batch = input.get_shape().as_list()[0]
    with tf.variable_scope("DeepDeconvNet"):
        # Reshaping
        z = tf.reshape(input, shape=(batch, 1, 2, -1))  #SHOULD (batch, 1, 2, 200)

        # 1st Temporal Deconvolution, Leaky ReLU activation
        wd1, bd1 = get_params(name="deepde1", shape=(1, 10, 200, 100), n_filter=100)
        deconv1 = deconv(input=z, newshape=(1, 14), w=wd1, b=bd1, padding="SAME")
        deconv1 = leaky_relu(deconv1)

        # 2nd Temporal Deconvolution, leaky ReLU activation
        wd2, bd2 = get_params(name="deepde2", shape=(1, 10, 100, 50), n_filter=50)
        deconv2 = deconv(input=deconv1, newshape=(1, 52), w=wd2, b=bd2, padding="SAME")
        deconv2 = leaky_relu(deconv2)

        # 3rd Temporal Deconvolution, leaky ReLU activation
        wd3, bd3 = get_params(name="deepde3", shape=(1, 10, 50, 25), n_filter=25)
        deconv3 = deconv(input=deconv2, newshape=(1, 167), w=wd3, b=bd3, padding="SAME")
        deconv3 = leaky_relu(deconv3)

        # 4th Spatial Deconvolution, leaky ReLU activation
        wd4, bd4 = get_params(name="deepde4", shape=(22, 1, 25, 25), n_filter=25)
        deconv4 = deconv(input=deconv3, newshape=(22, 168), w=wd4, b=bd4, padding="SAME")
        deconv4 = leaky_relu(deconv4)

        # 5th Temporal Deconvolution, tanh activation
        wd5, bd5 = get_params(name="deepde5", shape=(1, 10, 25, 1), n_filter=1)
        deconv5 = deconv(input=deconv4, newshape=(22, 512), w=wd5, b=bd5, padding="SAME")
        deconv5 = 3 * tf.nn.tanh(deconv5)

        return deconv5, wd4 # For data generation and activation pattern

def ShallowConvNet(input, adversarial, keep=0.5, num_class = 4, reuse=None):
    with tf.variable_scope("ShallowConvNet") as scope:
        if reuse:
            scope.reuse_variables()
            keep = 1.0

        # 1st Temporal Convolution, Linear activation
        w1, b1 = get_params(name="shallow1", shape=(1, 25, 1, 40), n_filter=40)
        conv1 = conv(input=input, w=w1, b=b1, strides=[1, 1, 1, 1], padding="VALID") #(batch, 22, 488, 40)

        # 2nd Spatial Convolution, Squaring activation
        w2, b2 = get_params(name="shallow2", shape=(22, 1, 40, 40), n_filter=40)
        conv2 = conv(input=conv1, w=w2, b=b2, strides=[1, 1, 1, 1], padding="VALID")
        conv2 = batchnorm(conv2)
        conv2 = tf.square(conv2)
        # conv2 = tf.nn.relu(conv2)  #(batch, 1, 488, 40)
        conv2 = dropout(input=conv2, keep=keep)

        # Mean pooling, logarithm activation
        conv2 = tf.nn.pool(input=conv2, window_shape=[1, 75], pooling_type="AVG", padding="VALID", strides=[1, 15])
        conv2 = tf.log(conv2)  #(batch, 1, 28, 40)

        if adversarial == True:
            # Linear Classification, Softmax activation, k+1 class output
            batch = conv2.get_shape().as_list()[0]
            conv2 = tf.reshape(conv2, shape=(batch, -1)) #(batch, 1120)
            num_hidden = conv2.get_shape().as_list()[-1]
            w6, b6 = get_params(name="shllowout", shape=(num_hidden, num_class + 1), n_filter=num_class + 1)
            logits = tf.nn.bias_add(tf.matmul(conv2, w6), b6)
            output = tf.argmax(tf.sigmoid(logits), -1)
        else:
            # Linear Classification, Softmax activation, k class output
            batch = conv2.get_shape().as_list()[0]
            conv2 = tf.reshape(conv2, shape=(batch, -1)) #(batch, 1120)
            num_hidden = conv2.get_shape().as_list()[-1]
            w6, b6 = get_params(name="shllowout", shape=(num_hidden, num_class), n_filter=num_class)
            logits = tf.nn.bias_add(tf.matmul(conv2, w6), b6)
            output = tf.argmax(tf.sigmoid(logits), -1)
        return logits, output, conv2, w2 # For output, feature matching technique, and activation pattern

def ShallowDeconvNet(input):
    batch = input.get_shape().as_list()[0]
    with tf.variable_scope("ShallowDeconvNet"):
        # Reshaping
        z = tf.reshape(input, shape=(batch, 1, -1, 40))  # SHOULD (batch, 1, 28, 40)

        # 1st Spatial Deconvolution, Leaky ReLU activation
        wd1, bd1 = get_params(name="shallowde1", shape=(22, 1, 40, 40), n_filter=40)
        deconv1 = deconv(input=z, newshape=(22, 488), w=wd1, b=bd1, padding="SAME")
        deconv1 = leaky_relu(deconv1)

        # 2nd Temporal Deconvolution, leaky ReLU activation
        wd2, bd2 = get_params(name="shallowde2", shape=(1, 25, 40, 1), n_filter=1)
        deconv2 = deconv(input=deconv1, newshape=(22, 512), w=wd2, b=bd2, padding="SAME")
        deconv2 = 3 * tf.nn.tanh(deconv2)
        return deconv2, wd1  # For data generation and activation pattern

