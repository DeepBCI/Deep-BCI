from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m

import scipy.io
import theano
import theano.tensor as T

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from utils import augment_EEG, cart2sph, pol2cart

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

tf.reset_default_graph()
readdataN = 2660
#readdataN = 128



def azim_proj(pos):
    """
    computes the azimuthal equidistant projection of input point in 3d cartesian coordinates.
    imagine a plane being placed against (tangent to) a globe. if
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3d cartesian coordinates
    :return: projected coordinates using azimuthal equidistant projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    generates eeg images given electrode locations in 2d space and multiple feature values for each electrode

    :param locs: an array with shape [n_electrodes, 2] containing x, y
                        coordinates for each electrode.
    :param features: feature matrix as [n_samples, n_features]
                                features are as columns.
                                features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: number of pixels in the output images
    :param normalize:   flag for whether to normalize each band over all samples
    :param augment:     flag for generating augmented images
    :param pca:         flag for pca based data augmentation
    :param std_mult     multiplier for std of added noise
    :param n_components: number of components in pca to retain for augmentation
    :param edgeless:    if true generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=false).
    :return:            tensor of size [samples, colors, w, h] containing generated
                        images.
    """
    global readdataN


    feat_array_temp = []
    nelectrodes = locs.shape[0]     # number of electrodes
    # test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nelectrodes == 0
    n_colors = features.shape[1] / nelectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nelectrodes : nelectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_eeg(feat_array_temp[c], std_mult, pca=true, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_eeg(feat_array_temp[c], std_mult, pca=false, n_components=n_components)
    nsamples = features.shape[0]

    #temp deifinition for fast debuging
    nsamples = readdataN

    # interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nsamples, n_gridpoints, n_gridpoints]))
    # generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nsamples, 4)), axis=1)
    # interpolating
    for i in xrange(nsamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('interpolating {0}/{1}\r'.format(i+1, nsamples), end='\r')
    # normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    
    temp1 = np.swapaxes(np.asarray(temp_interp), 0, 1)
    temp2 = np.swapaxes(temp1, 1, 2)
    temp3 = np.swapaxes(temp2, 2, 3)
    return temp3

# Params
learning_rate = 0.001
training_iters = 200000
batch_size = 32
display_step = 10

# Network Params
n_input = 32*32 
n_classes = 4 
dropout = 0.5 


# define initial weight & variable for tf
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# define
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x
    #return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

#create model cnn
def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3, filtersize=(3, 3)):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param w_init: Initial weight values
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers in each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """

    count = 0
    n_classes = 4
    #x = tf.placeholder("float", shape=[None, imsize*imsize, n_colors])
    #x = tf.placeholder("float", shape=[None, imsize*imsize, n_colors])
    if input_var is None:
        x = tf.placeholder(tf.float32, shape=[None, imsize, imsize, n_colors])
    else:
        x = input_var
    #y = tf.placeholder(tf.float32, shape=[None, n_classes])
    x_image = tf.reshape(x, [-1, imsize, imsize, n_colors])

    inkernel = 3
    outkernel = n_filters_first
    
    #If w_init is not...
    #Define Weight, Bias using weight_variable & bias variable
    if w_init is None:
        weights = []
        for i, s in enumerate(n_layers):
            w_tmp = []
            h_tmp = []
            for l in range(s):
                w_tmp.append(weight_variable([filtersize[0], filtersize[1], inkernel, outkernel]))
                inkernel = outkernel
            outkernel = outkernel*2
            weights.append(w_tmp)

        #fully connected layer's weight
        weights.append(weight_variable([outkernel, 512])) 
        weights.append(weight_variable([512, n_classes]))
    else:
        weights = w_init
    
    inkernel = 3
    outkernel = n_filters_first
    
    bias = []
    for i, s in enumerate(n_layers):
        b_tmp = []
        for l in range(s):
            b_tmp.append(bias_variable([outkernel]))
            inkernel = outkernel
        outkernel = outkernel*2
        bias.append(b_tmp)
    #fully connected layer's weight
    bias.append(bias_variable([512]))
    bias.append(bias_variable([n_classes]))


    h_conv = []  # system function
    h_pool = []
    x_datain = x_image

    print("convlayer monitor")
    #define convolutional network
    for i, s in enumerate(n_layers):
        h_conv_tmp = []
        for l in range(s):
            #define h
            h_convout = tf.nn.relu(conv2d(x_datain, weights[i][l], bias[i][l]))
            h_conv_tmp.append(h_convout)
            x_datain = h_convout
        h_conv.append(h_conv_tmp) #save result
        #max pooling layer
        h_convout = maxpool2d(x_datain, 2)
        h_pool.append(h_convout) #save reuslt
        x_datain = h_convout
    print("cnn out shape")
    print(h_convout.shape)
    return h_convout, weights
    


def build_convpool_conv1d(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=3, n_filter=64, save_v={}):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None

    print('Build conv 1d')
    print('inputvar shape', input_vars.shape)
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            print("inputvar:", input_vars[:, i].shape)
            convnet, w_init = build_cnn(input_vars[:, i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[:, i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        print(i, ':', convnet.shape) #7, 4, 4, 128
        convnets.append(tf.contrib.layers.flatten(convnet))
    
    #convpool = tf.concat(convnets, 1)
    convpool = tf.concat([tf.expand_dims(t, 1) for t in convnets], 1)
    print('1.concat:', convpool.shape) #dataN, winN, features (?, 3, 2048)
    featureN = convpool.shape[2]

    print('2.reshape:', convpool.shape) #?, 3, 2048
    #define 1d conv
    filter_ = tf.get_variable("conv_filter", shape = [n_timewin, featureN, n_filter])
    convpool = tf.nn.conv1d(convpool, filter_, stride=1, padding='VALID')#, data_format = "NCHW"))
    #convpool = tf.nn.dropout(convpool, 0.5)
    print('4.1dconv:', convpool.shape) #?, 64, 1
    convpool = tf.reshape(convpool, [-1, n_filter])
    print('5.ho1:', convpool.shape) #?, 64. 1
    #print(weight_variable([n_filter, 512]))
    #print(bias_variable([512]))
    
    
    #weight init.
    if 'conv1d_w0' not in save_v:
        save_v['conv1d_v0'] = weight_variable([n_filter, 512])
    if 'conv1d_w1' not in save_v:
        save_v['conv1d_w1'] = weight_variable([512, nb_classes])
    
    #bias init.
    if 'conv1d_b0' not in save_v:
        save_v['conv1d_b0'] = bias_varibble([512])
    if 'conv1d_b1' not in save_v:
        save_v['conv1d_b1'] = bias_variable([nb_classes])

    convpool = tf.add(tf.matmul(convpool, save_v['conv1d_w0']), save_v['conv1d_b0'])
    convpool = tf.nn.relu(convpool)
    convpool = tf.nn.dropout(convpool, 0.5)

    convpool = tf.add(tf.matmul(convpool, save_v['conv1d_w1']), save_v['conv1d_b1'])

    print('final shape:', convpool.shape)


    return convpool, save_v


def build_convpool_lstm(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3, save_v={}):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 128
    
    print('check input lstm:', input_vars.shape)
    
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[:, i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[:, i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        #convnets.append(FlattenLayer(convnet))
        
        convnets.append(tf.contrib.layers.flatten(convnet))
    print('convnet', convnet.shape) #?, 4, 4, 128
    print('0:', convnets[0].shape) #?, 2048
    convpool = tf.concat([tf.expand_dims(t, 1) for t in convnets], 1)
    print('1. concat:', convpool.shape) #dataN, winN, features #?, 3, 2048
    featureN = convpool.shape[2]
  
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    n_hidden = 128
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse = tf.get_variable_scope().reuse)
    with tf.variable_scope('lstm'):
        #lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=True)
        convpool, states = tf.nn.dynamic_rnn(lstm_cell, convpool, dtype=tf.float32, scope = 'lstm')
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse = tf.get_variable_scope().reuse)
    #convpool, states = tf.nn.dynamic_rnn(lstm_cell, convpool, dtype=tf.float32)

    print('state', states)
    print('3.lstm shape:', convpool.shape)
    print('4.slice',convpool[:, -1].shape)


    #weight init.
    if 'lstm_w0' not in save_v:
        save_v['lstm_v0'] = weight_variable([n_hidden, 256])
    if 'lstm_w1' not in save_v:
        save_v['lstm_w1'] = weight_variable([256, nb_classes])


    #bias init.
    if 'lstm_b0' not in save_v:
        save_v['lstm_b0'] = bias_variable([256])
    if 'lstm_b1' not in save_v:
        save_v['lstm_b1'] = bias_variable([nb_classes])


    convpool = tf.add(tf.matmul(convpool[:, -1], save_v['lstm_w0']), save_v['lstm_b0']) #slice layer
    convpool = tf.nn.relu(convpool)
    convpool = tf.nn.dropout(convpool, 0.5)

    convpool = tf.add(tf.matmul(convpool, save_v['lstm_w1']), save_v['lstm_b1'])
    return convpool, save_v


def build_convpool_mix(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3, n_filter=64):
    """
    Builds the complete network with LSTM and 1D-conv layers combined

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 128
    
    print('check input lstm:', input_vars.shape)
    
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[:, i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[:, i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        #convnets.append(FlattenLayer(convnet))
        
        convnets.append(tf.contrib.layers.flatten(convnet))
    print('convnet', convnet.shape) #?, 4, 4, 128
    print('0:', convnets[0].shape) #?, 2048
    #convpool = tf.concat(convnets, 1)
    convpool = tf.concat([tf.expand_dims(t, 1) for t in convnets], 1)
    print('1. concat:', convpool.shape) #dataN, winN, features #?, 3, 2048
    featureN = convpool.shape[2]
    
    # we want the shape to be [n_samples, features, numTinewin]
    #define 1d conv
    filter_ = tf.get_variable("conv_filter1", shape = [n_timewin, featureN, n_filter])
    cnn_out = tf.nn.conv1d(convpool, filter_, stride=1, padding='VALID')#, data_format = "NCHW"))
    #convpool = tf.nn.dropout(convpool, 0.5)
    print('2.1dconv:', cnn_out.shape) #?, 64, 1
    cnn_out = tf.reshape(cnn_out, [-1, n_filter])
    
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    n_hidden = 128

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse = tf.get_variable_scope().reuse)
    with tf.variable_scope('mixlstm'):
        #lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=True)
        lstm_out, states = tf.nn.dynamic_rnn(lstm_cell, convpool, dtype=tf.float32, scope = 'mixlstm')
    
    print('3.lstm', lstm_out.shape)
    
    # Merge!
    dense_input = tf.concat([cnn_out, lstm_out[:, -1]], 1)

    print('4. dense', dense_input.shape)
    #denselayer
    convpool = tf.add(tf.matmul(dense_input, weight_variable([192, 512])), bias_variable([512])) #slice layer
    convpool = tf.nn.relu(convpool)
    convpool = tf.nn.dropout(convpool, 0.5)

    convpool = tf.add(tf.matmul(convpool, weight_variable([512, nb_classes])), bias_variable([nb_classes]))

    return convpool


def build_convpool_max(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=3):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    convnets = []
    w_init = None

    print('Build max')
    print('inputvar shape', input_vars.shape)
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            print("inputvar:", input_vars[:, i].shape)
            convnet, w_init = build_cnn(input_vars[:, i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[:, i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        print(i, ':', convnet.shape) #7, 4, 4, 128
        convnets.append(tf.contrib.layers.flatten(convnet))

    convpool = tf.add(tf.matmul(convpool, weight_variable([n_filter, 512])), bias_variable([512]))
    convpool = tf.nn.relu(convpool)
    convpool = tf.nn.dropout(convpool, 0.5)

    convpool = tf.add(tf.matmul(convpool, weight_variable([512, nb_classes])), bias_variable([nb_classes]))


    

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    print("here is batch!!!")
    
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        inputs = np.swapaxes(inputs, 0, 1)
        input_len = inputs.shape[0]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[excerpt], targets[excerpt]


#train definition
def train(images, labels, fold, model_type, batch_size=32, num_epochs=5, n_layer = (4, 2, 1)):
    print('model type:', model_type)    
    #for data parsing
    num_classes = 4

    if len(images.shape) == 4:
        sampleN, images_height, images_width, colorN = images.shape
        print('4:', images.shape)
        input_var = tf.placeholder(tf.float32, [None, images_height, images_width, colorN])
        output_var = tf.placeholder(tf.float32, [None, num_classes])


    elif len(images.shape) == 5:
        windowN, sampleN, images_height, images_width, colorN = images.shape
        #print('5:', images.shape)
        #images = np.swapaxes(images, 0, 1)
        print('5:', images.shape)
        input_var = tf.placeholder(tf.float32, [None, windowN, images_height, images_width, colorN])
        output_var = tf.placeholder(tf.float32, [None, num_classes])
    else:
        print('warning!')
        return False
    
    #data devide
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)

    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    keep_prob = tf.placeholder(tf.float32)

    #accordance to ... model.
    if model_type == '1dconv':
        #network = build_convpool_conv1d(input_var, num_classes, n_timewin = windowN)
        network = build_convpool_conv1d(input_var, num_classes)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output_var))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    elif model_type == 'maxpool':
        network = build_convpool_max(input_var, num_classes)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output_var))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    elif model_type == 'mix':
        network = build_convpool_mix(input_var, num_classes, 100)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output_var))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    elif model_type == 'lstm':
        network = build_convpool_lstm(input_var, num_classes, 100)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output_var))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    elif model_type == 'cnn':
        network, _ = build_cnn(input_var, n_layers = n_layer)
        #define fully connected network

        network = tf.reshape(network, [-1, 4*4*128])
        network = tf.add(tf.matmul(network, weight_variable([4*4*128, 512])), bias_variable([512]))
        network = tf.nn.relu(network)
        network = tf.nn.dropout(network, 0.5)
        
        #define out layer
        network = tf.add(tf.matmul(network, weight_variable([512, num_classes])),bias_variable([num_classes]))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output_var))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        
    else:
        raise ValueError("Model not supported ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']")

    #for monitor final network
    print("here is our network shape")
    print(network.shape)
    print(output_var.shape)

    #create a loss expression for training, i.e. ascalar objective we want
    #to minimize (for our multi-class problem, .... o
  

    #evaluate model
    correct_pred = tf.equal(tf.argmax(network, 1), tf.argmax(output_var, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #MODEL Def end


    #Traingning & varlidation..
    print("start training")
    # Initializing the variables
    init = tf.global_variables_initializer()
    display_step = 1
    xin = input_var
    yout = output_var

    #for saving weight
    saver = tf.train.Saver()
    filepath = "./tmp/" 
    filesave_path = filepath + model_type + ".ckpt"

    # gpu config
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #run session
    with tf.Session(config = config) as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(init)
        best_val_acc = 0
        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            train_batches = 0
            start_time = time.time()
            train_step = 1
            val_step = 1
            val_batches = 0
            sum_train_acc = 0
            sum_train_loss = 0
            sum_val_acc = 0
            sum_val_loss = 0

            # train accuracy
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets = batch
                targets = np.eye(num_classes)[targets.reshape(-1)]
                
                #check for monitor size
                if train_step ==1:
                    print("batch info is.. batch, inputs.shape, targets.shape")
                    print(X_train.shape)
                    print(y_train.shape)
                    print("batch info")
                    print(inputs.shape)
                    print(targets.shape)
                    print(xin.shape)
                    print(yout.shape)
                
                sess.run(optimizer, feed_dict = {xin: inputs, yout: targets, keep_prob: 0.5})
                
                if train_step % display_step == 0:
                    #calculate batch loss & accuracy
                    train_loss, train_acc = sess.run([cost, accuracy], feed_dict = {xin: inputs, yout: targets, keep_prob: 1.0})
                    print("Iter " + str(train_step*batch_size) + ", Minibatch Loss= " + \
                            "{:.6f}".format(train_loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(train_acc))
                    
                    sum_train_loss += train_loss
                    sum_train_acc += train_acc
                    print(train_loss, sum_train_loss, train_acc, sum_train_acc)
                    save_path = saver.save(sess, filesave_path, global_step = train_step)

                train_step += 1

            print("train Opt fin")
            # validation accuracy
            for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                inputs, targets = batch
                targets = np.eye(num_classes)[targets.reshape(-1)]
                sess.run(optimizer, feed_dict = {xin: inputs, yout: targets, keep_prob: 0.5})
                
                if val_step % display_step == 0:
                    #calculate batch loss & accuracy
                    val_acc = sess.run(accuracy, feed_dict = {xin: inputs, yout: targets, keep_prob: 1.0})
                    print("Iter " + str(val_step*batch_size) + ", Minibatch Loss= " + \
                            ", cross validatino  Accuracy= " + "{:.5f}".format(val_acc))

                    #sum_val_loss += val_loss
                    sum_val_acc += val_acc

                    print(val_acc, sum_val_acc)
                val_step += 1

            print("Validation Opt fin")
            
            #calculate average loss & accuracy
            av_train_loss = sum_train_loss/train_step*display_step
            av_train_acc = sum_train_acc/train_step*display_step
            #av_val_loss = sum_val_loss/val_step*display_step
            av_val_acc = sum_val_acc/val_step*display_step

            print(av_train_loss, av_train_acc, av_val_acc)

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(av_train_loss))
            #print("  validation loss:\t\t{:.6f}".format(av_val_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))

            if av_val_acc > best_val_acc:
                best_val_acc = av_val_acc
                # After training, we compute & print the test error
                sum_test_err = 0
                sum_test_acc = 0
                test_batches = 0
                
                for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                    inputs, targets = batch
                    targets = np.eye(num_classes)[targets.reshape(-1)]

                    test_acc = sess.run(accuracy, feed_dict={xin: inputs, yout: targets, keep_prob: 1.0})
                    print(test_acc)
                    sum_test_acc += test_acc
                    test_batches += 1

                av_test_acc = sum_test_acc / test_batches
                print("Final results:")
                #print("  test loss:\t\t\t{:.6f}".format(av_test_err))
                print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
                
                # Dump the network weights to a file like this:
                #tf.saver.save(sess, 'my-model')
                # `save` method will call `export_meta_graph` implicitly.
                # you will get saved graph files:my-model.meta
        print('-'*50)
        print("Best validation accuracy:\t\t{:.2f} %".format(best_val_acc * 100))
        print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
        writer = tf.summary.FileWriter('logs', sess.graph)


        #modelsave part
        save_path = saver.save(sess, filesave_path)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    from utils import reformatInput

    # Load electrode locations
    print('Loading data...')
    locs = scipy.io.loadmat('../Sample data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    feats = scipy.io.loadmat('../Sample data/FeatureMat_timeWin.mat')['features']
    subj_nums = np.squeeze(scipy.io.loadmat('../Sample data/trials_subNums.mat')['subjectNum'])
    
    #for testing
    fold_pairs_test = []
    
    cnt = 0
    for i in np.unique(subj_nums):
        cnt +=1
        tmprandomlist = np.arange(readdataN)
        np.random.shuffle(tmprandomlist)
        fold_pairs_test.append((tmprandomlist[0:(readdataN/10*9)], tmprandomlist[(readdataN/10*9):]))
    
    print(cnt)


    # lEAVE-sUBJECT-oUT CROSS VALIDTION
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)  # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))

    
    #print(fold_pairs)
    # CNN Mode
    print('Generating images...')
    # Find the average response over time windows
    av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(feats.shape[1] / 192)])
    av_feats = av_feats / (feats.shape[1] / 192)
    images = gen_images(np.array(locs_2d),
                                  av_feats,
                                  32, normalize=False)
    print('\n')

    #print(images)
    print('image shape :')
    print(images.shape)


    # Class labels should start from 0
    print('Training the CNN Model...')
    #train(images, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'cnn')
    #train(images, np.squeeze(feats[0:readdataN, -1]) - 1, fold_pairs_test[2], 'cnn')

    # Conv-LSTM Mode
    print('Generating images for all time windows...')
    images_timewin = np.array([gen_images(np.array(locs_2d),
                                                    feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
                                         range(feats.shape[1] / 192)
                                         ])
    #print(images_timewin)
    print(images_timewin.shape)


    print('\n')
    print('Training the traing 1d ')
    print(feats[0:readdataN, -1].shape)
   
    #train(images_timewin, np.squeeze(feats[0:readdataN, -1]) - 1, fold_pairs_test[2], '1dconv')
    #train(images_timewin, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], '1dconv')

    print('\n')
    print('Training the traing lstm ')
    
    train(images_timewin, np.squeeze(feats[0:readdataN, -1]) - 1, fold_pairs_test[2], 'lstm')
    #train(images_timewin, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], '1dconv')

    

    print('\n')
    print('Training the LSTM-CONV Model...')
    train(images_timewin, np.squeeze(feats[0:readdataN, -1]) - 1, fold_pairs_test[2], 'mix')
    #train(images_timewin, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'mix')

    print('Done!')





