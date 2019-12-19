# Import APIs
import tensorflow as tf

def Band_Spatial_Module(input, output_band, output_spatial, kernel_band, kernel_spatial, regularizer, initializer):
    # Band Module, the number of parameters: input filter * output filter + kernel parameters
    tmp = tf.layers.separable_conv2d(inputs=input, filters=output_band, kernel_size=kernel_band, padding="same",
                                     activation=tf.nn.leaky_relu,
                                     depthwise_initializer=initializer, pointwise_initializer=initializer,
                                     depthwise_regularizer=regularizer, pointwise_regularizer=regularizer)
    tmp = tf.layers.batch_normalization(inputs=tmp)

    # Spatial Module
    feature = tf.layers.conv2d(inputs=tmp, filters=output_spatial, kernel_size=kernel_spatial, activation=tf.nn.leaky_relu,
                                kernel_initializer=initializer, kernel_regularizer=regularizer)

    feature = tf.layers.batch_normalization(inputs=feature)
    return tmp, feature

def MSNN(eeg, label, num_channel, num_output, sampling_freq, SSVEP_mode=False, reuse=False):
    with tf.variable_scope("MSNN", reuse=reuse):
        if SSVEP_mode == False:
            k1, F1 = (1, 100), 16
            k2, F2 = (1, 60), 32
            k3, F3 = (1, 20), 64
        else:
            # Like EEGNet, we used different kernel sizes and feature map dimensions for the SSVEP experiment.
            k1, F1 = (1, 15), 32
            k2, F2 = (1, 10), 32
            k3, F3 = (1, 5), 32

        # We used L1-L2 regularizer, Xavier Initializer for all convolutional layers.
        regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.01, scale_l2=0.001)
        initializer = tf.contrib.layers.xavier_initializer()

        # Temporal convolution, the number of parameters: 1 * 4 * 256 = 1024
        hidden = tf.layers.conv2d(inputs=eeg, filters=4, kernel_size=(1, sampling_freq/2), activation=tf.nn.leaky_relu,
                                  kernel_initializer=initializer, kernel_regularizer=regularizer)
        hidden = tf.layers.batch_normalization(inputs=hidden)
        # hidden = tf.layers.max_pooling2d(inputs=hidden, pool_size=(1, 2), strides=(1, 2))

        # 1st B-S Module, the number of parameters: (4 * 16 + 50) + (16 * 16 + 64) = 114 + 320
        hidden, feature_low = Band_Spatial_Module(input=hidden, output_band=F1, output_spatial=F1, kernel_band=k1,
                                                  kernel_spatial=(num_channel, 1), regularizer=regularizer,
                                                  initializer=initializer)

        # 2nd B-S Module, the number of parameters: (16 * 32 + 30) + (32 * 32 + 64) = 542 + 1088
        hidden, feature_mid = Band_Spatial_Module(input=hidden, output_band=F2, output_spatial=F2, kernel_band=k2,
                                                  kernel_spatial=(num_channel, 1), regularizer=regularizer,
                                                  initializer=initializer)

        # 3rd B-S Module, the number of parameters: (32 * 64 + 10) + (64 * 64 + 64) = 2058 + 4160
        hidden, feature_high = Band_Spatial_Module(input=hidden, output_band=F3, output_spatial=F3, kernel_band=k3,
                                                   kernel_spatial=(num_channel, 1), regularizer=regularizer,
                                                   initializer=initializer)  # (batch, 1, 384 timepoint, 64)

        # Feature concatenation and flattening
        features = tf.concat((feature_low, feature_mid, feature_high), axis=-1)
        # GAP
        features = tf.reduce_mean(features, -2)
        features = tf.layers.flatten(inputs=features)
        # Classifier
        output = tf.layers.dense(inputs=features, units=num_output, activation=None,
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))####
        prediction = tf.math.round(tf.math.softmax(output))
    return loss, prediction
