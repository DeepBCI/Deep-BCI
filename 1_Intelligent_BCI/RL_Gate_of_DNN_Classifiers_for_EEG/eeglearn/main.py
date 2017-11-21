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
from utils_genimg import azim_proj, gen_images, iterate_minibatches

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

#import for rl
import tensorflow as tf
import scipy.ndimage
import sys
import os
import time
import csv
import argparse
import download


import random
import time

#from game import import Game
from model import DQN


# Default base-directory for the checkpoints and log-files.
# The environment-name will be appended to this.
checkpoint_base_dir = 'checkpoints_tutorial16/'
# Combination of base-dir and environment-name.
checkpoint_dir = None
# Full path for the log-file for rewards.
log_reward_path = None
# Full path for the log-file for Q-values.
log_q_values_path = None


def update_path(env_name):
    pass

class Log:
    """
    Base-class for logging data to a text-file during training.
    It is possible to use TensorFlow / TensorBoard for this,
    but it is quite awkward to implement, as it was intended
    for logging variables and other aspects of the TensorFlow graph.
    We want to log the reward and Q-values which are not in that graph.
    """

    def __init__(self, file_path):
        """Set the path for the log-file. Nothing is saved or loaded yet."""

        # Path for the log-file.
        self.file_path = file_path

        # Data to be read from the log-file by the _read() function.
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, count_states, msg):
        """
        Write a line to the log-file. This is only called by sub-classes.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.
        :param count_states: 
            Counter for the number of states processed during training.
        :param msg:
            Message to write in the log.
        """

        with open(file=self.file_path, mode='a', buffering=1) as file:
            msg_annotated = "{0}\t{1}\t{2}\n".format(count_episodes, count_states, msg)
            file.write(msg_annotated)

    def _read(self):
        """
        Read the log-file into memory so it can be plotted.
        It sets self.count_episodes, self.count_states and self.data
        """

        # Open and read the log-file.
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            self.count_episodes, self.count_states, data = zip(*reader)

        # Convert the remaining log-data to a NumPy float-array.
        self.data = np.array(data, dtype='float')


class LogReward(Log):
    """Log the rewards obtained for episodes during training."""

    def __init__(self):
        # These will be set in read() below.
        self.episode = None
        self.mean = None

        # Super-class init.
        Log.__init__(self, file_path=log_reward_path)

    def write(self, count_episodes, count_states, reward_episode, reward_mean):
        """
        Write the episode and mean reward to file.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.
        :param count_states: 
            Counter for the number of states processed during training.
        :param reward_episode:
            Reward for one episode.
        :param reward_mean:
            Mean reward for the last e.g. 30 episodes.
        """
        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)


class LogQValues(Log):
    """Log the Q-Values during training."""
    def __init__(self):
        # These will be set in read() below.
        self.min = None
        self.mean = None
        self.max = None
        self.std = None

        # Super-class init.
        Log.__init__(self, file_path=log_q_values_path)

    def write(self, count_episodes, count_states, q_values):
        """
        Write basic statistics for the Q-values to file.
        :param count_episodes:
            Counter for the number of episodes processed during training.
        :param count_states: 
            Counter for the number of states processed during training.
        :param q_values:
            Numpy array with Q-values from the replay-memory.
        """
        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(np.min(q_values), np.mean(q_values),
                                                          np.max(q_values), np.std(q_values))
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.
        It sets self.count_episodes, self.count_states, self.min / mean / max / std.
        """

        # Read the log-file using the super-class.
        self._read()
        # Get the logged statistics for the Q-values.
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]



def print_progress(msg):
    """
    Print progress on a single line and overwrite the line.
    Used during optimization.
    """

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


#############################################################
# Height of each image-frame in the state.
state_height = 32*7
state_width = 32

# Size of each image in the state.
state_img_size = np.array([state_height, state_width])
state_channels = 3

# Shape of the state-array.
state_shape = [state_height, state_width, state_channels]
#############################################################



def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3):
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
    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    #print(w_init)
    if w_init is None:
        w_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
        #print("i'm here ! w_init was None!")
    # Input layer

    network = InputLayer(shape=(None, n_colors, imsize, imsize),
                                        input_var=input_var)
    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3),
                          W=w_init[count], pad='same')
            count += 1
            weights.append(network.W)
        network = MaxPool2DLayer(network, pool_size=(2, 2))
    return network, weights



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
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(convnet)
    # convpooling using Max pooling over frames
    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def build_convpool_conv1d(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=3):
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
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    #print('1.concat:', convpool.output_shape) #None, 6144
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    #print('2.reshape:', convpool.output_shape) #(None, 3, 2048)
    convpool = DimshuffleLayer(convpool, (0, 2, 1))
    #print('3.Dimshuffle:', convpool.output_shape)  #(none, 2048, 3)
    
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    convpool = Conv1DLayer(convpool, 64, 3) #incoming, num_filters, filter_size, pad = 0(Valid) 
    #print('4.1dconv:', convpool.output_shape)  #(none, 64, 1)
    
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    #print('ho1', convpool.output_shape)  #(none, 512)
    
    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    #print('ho2', convpool.output_shape)  #(none, 4)
    return convpool


def build_convpool_lstm(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
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
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))

    #print(convnet.output_shape) #None, 128, 4, 4
    #print('0.:', convnets[0].output_shape) #None, 2048... 128*4*4
    
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    #print('1.concat:', convpool.output_shape) #None, 6144
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))
    
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features) 
    #print('2.Reshape:', convpool.output_shape) #None, 3, 2048
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.
    #print('3.LSTM:', convpool.output_shape) #None, 3, 128
    convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
    
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    #print('4.slice:', convpool.output_shape) #None, 128
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def build_convpool_mix(input_vars, nb_classes, grad_clip=110, imsize=32, n_colors=3, n_timewin=3):
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
    # Build 7 parallel CNNs with shared weights
    for i in range(n_timewin):
        if i == 0:
            convnet, w_init = build_cnn(input_vars[i], imsize=imsize, n_colors=n_colors)
        else:
            convnet, _ = build_cnn(input_vars[i], w_init=w_init, imsize=imsize, n_colors=n_colors)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    convpool = ReshapeLayer(convpool, ([0], n_timewin, get_output_shape(convnets[0])[1]))

    #print('1.convpool:', convpool.shape) #[0], 3, 2048
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))
    #print('1.5. convpool reshape:', reformConvpool.output_shape) #None 2048, 3
    
    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    #print('2. conv_out shape:', conv_out.output_shape) #None, 64, 1
    conv_out = FlattenLayer(conv_out)
    #print('2.5. conv_out shape:', conv_out.output_shape) #None, 64
    
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
    #print('3 lstm:', lstm.output_shape) #None, 3, 128
    lstm_out = SliceLayer(lstm, -1, 1)
    #print('3.5 lstmout:', lstm_out.output_shape) #None, 128
    
    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out]) #None, 192
    #print('4 dense:', dense_input.output_shape)
    
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(convpool,
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool






###
#def predcict
def predict(image, label, model_type):
    # Prepare Theano variables for inputs and targets
    input_var = T.TensorType('floatX', ((False,) * 5))()
    target_var = T.ivector('targets')
    num_classes = 4

    #print('img:', image.shape)
    #print('tmp:', _Xtmp.shape, _y.shape)
    if image.ndim == 4:
        _Xtmp, _y = image, np.array([label])
        _X = np.expand_dims(_Xtmp, axis=1)
        dataN = 1
    
    elif image.ndim == 5:
        #image = np.swapaxes(image, 0, 1)
        _Xtmp, _y = image, np.array([label])
        dataN = len(_y[0])
        _X = _Xtmp
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...%s", model_type)   
    print('curshape:', _X.shape, _y.shape)
    # Building the appropriate model
    if model_type == '1dconv':
        network = build_convpool_conv1d(input_var, num_classes)
    elif model_type == 'maxpool':
        network = build_convpool_max(input_var, num_classes)
    elif model_type == 'lstm':
        network = build_convpool_lstm(input_var, num_classes, 100)
    elif model_type == 'mix':
        network = build_convpool_mix(input_var, num_classes, 100)
    elif model_type == 'cnn':
        input_var = T.tensor4('inputs')
        network, _ = build_cnn(input_var)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                             num_units=256,
                             nonlinearity=lasagne.nonlinearities.rectify)
        network = DenseLayer(lasagne.layers.dropout(network, p=.5),
                             num_units=num_classes,
                             nonlinearity=lasagne.nonlinearities.softmax)
    else:
        raise ValueError("Model not supported ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']")
    

    #load model & weights
    with np.load('weights_lasg_{0}.npz'.format(model_type)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)
    
    #create expression for testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # as a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    test_result = T.eq(T.argmax(test_prediction, axis=1), target_var)

    # compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    # compile a second function computing the validation loss and accuracy:
    
    
    val_fn = theano.function([input_var, target_var], [test_loss, test_result])
    val_fn_mean = theano.function([input_var, target_var], [test_loss, test_acc])

    pred_loss = np.zeros(dataN)
    result = np.zeros(dataN)
    for i in range(dataN):
        #print(_Xext.shape, _y[:, i].shape)
        #print(_X.shape) 
        if image.ndim == 4:
            _Xext = _X
            pred_loss[i], result[i] = val_fn(_Xext, _y)
        elif image.ndim == 5:
            _Xext = np.expand_dims(_X[i], 1)
            pred_loss[i], result[i] = val_fn(_Xext, _y[:, i])
    
    print('inputsize:', _Xext.shape)
    print('for data', dataN, model_type, result)
    return result, pred_loss

model_list =  ['1dconv', 'maxpool', 'lstm', 'mix']


def predict_all(image, label):
    dict_result = {}
    dict_loss = {}
    result = []
    for model in model_list:
        result, pred_loss = predict(image, label, model)
        dict_result.update({model: result})
        dict_loss.update({model: pred_loss})
        #result.append(predict(image, label, model))
    #result = np.array(result)
    #print(result)
    #print(dict_result)
    return dict_result, dict_loss



flags = tf.app.flags
#flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
#flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')



flags.FLAGSMAX_EPISODE = 10000
TARGET_UPDATE_INTERVAL = 1000
TRAIN_INTERVAL = 4 #train per n frame
OBSERVE = 100 #After n data stack, start train

n_act = len(model_list)
SCREEN_WIDTH = 32
SCREEN_HEIGHT = 32

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")



config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train_rl(images, targets, folds, stochastic = False, test = False, base_rand = False): 
    print('start train rl')


    #print(images.shape)
    #(X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput_rl(images, targets, fold)

    #X_train = X_train.astype("float32", casting='unsafe')
    #X_val = X_val.astype("float32", casting='unsafe')
    #X_test = X_test.astype("float32", casting='unsafe')
    
    #print('check')
    #print(X_train.shape)
    with tf.Session() as sess:
        #onfig = get_config(FLAGS) or FLAGS
       
        model = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, n_act)
        
        rewards = tf.placeholder(tf.float32, [None])
        tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter('logs', sess.graph)
        summary_merged = tf.summary.merge_all()
        
        print('total %s folds', len(folds))
        
        #(X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput_rl(images, targets, fold)
#        X_train = X_train.astype("float32", casting='unsafe')
#        X_val = X_val.astype("float32", casting='unsafe')
#        X_test = X_test.astype("float32", casting='unsafe')

        ###

        # init target network
        model.update_target_network()
        
        # get next action from DQN
        epsilon = 1.0
        # def frame N
        t_step = 0
        tot_reward_list = []


        MAX_EPISODE = 10000
        n_img = len(targets)
        
        n_epi = n_img
        if stochastic: n_epi = MAX_EPISODE
        

        # call pred & loss 
        n_test = 3 
        if test:  #for debugging
            pred_all, loss_all = predict_all(images[0:n_test, :], targets[0:test, :])
            if not stochastic: n_epi = n_test
        else: pred_all, loss_all = predict_all(images, targets)
        
        #pred_all_train, loss_all_train = predict_all(X_train, y_train)

        #print(pred_all)

        # run simulation
        pred_rl = []
        for epi in range(n_epi):
            terminal = False
            tot_reward = 0

            #init game & get current state
            
            #state parsing
            state = np.expand_dims(images[epi], 0)
            #state = np.expand_dims(X_train[epi], 0)
            model.init_state(state)

            if np.random.rand() < epsilon:
                act = random.randrange(n_act)
            else:
                act = model.get_action()

            if epi > OBSERVE: epsilon -= 1/100
            if base_rand: act = random.randrange(n_act)
            
            #stochastic define
            if stochastic:
                ii = random.randrange(n_img)
                state = np.expand_dims(images[ii], 0)  
                #state = np.expand_dims(X_train[ii], 0)
                state_i = ii

            else:
                state = np.expand_dims(images[epi], 0)
                #state = np.expand_dims(X_train[epi], 0)
                state_i = epi
            
            # get model str by act
            choosen_model = model_list[act]
            
            # reward function
            if pred_all[choosen_model][state_i] == 1:
                reward = 1
                pred_rl.append(1)
            else:
                reward = -2
                pred_rl.append(0)


            tot_reward += reward

            model.remember(state, act, reward, terminal)

            if t_step > OBSERVE and t_step % TRAIN_INTERVAL == 0:
                # DQN train
                model.train()

            if t_step % TARGET_UPDATE_INTERVAL == 0:
                # target update
                model.update_target_network()

            t_step += 1

            print('epi: %d score: %d' % ((epi+1), tot_reward))

            tot_reward_list.append(tot_reward)

            if epi % 10 == 0:
                summary = sess.run(summary_merged, feed_dict={rewards: tot_reward_list})
                writer.add_summary(summary, t_step)
                tot_reward_list = []

            if epi % 100 == 0:
                saver.save(sess, 'model/dqn.ckpt', global_step=t_step)

        return tot_reward_list, pred_rl, pred_all



readdataN = 2670
if __name__ == '__main__':
    from utils import reformatInput_rl, reformatInput_for_predict
    import pickle

    #Load electrode locations    
    genimg_mark = False


    locs = scipy.io.loadmat('../Sample data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    feats = scipy.io.loadmat('../Sample data/FeatureMat_timeWin.mat')['features']
    subj_nums = np.squeeze(scipy.io.loadmat('../Sample data/trials_subNums.mat')['subjectNum'])
    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)  # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))



    if genimg_mark:
        # Load electrode locations
        print('Loading data...')
                # CNN Mode
        print('Generating images...')
        # Find the average response over time windows
        av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(feats.shape[1] / 192)])
        av_feats = av_feats / (feats.shape[1] / 192)
        images = gen_images(np.array(locs_2d),
                                    av_feats,
                                    32, normalize=False)
        print('\n')
        
        # Conv-LSTM Mode
        print('Generating images for all time windows...')
        images_timewin = np.array([gen_images(np.array(locs_2d),
                                                        feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
                                            range(feats.shape[1] / 192)
                                            ])
        print('\n')

        with open('objs.pickle', 'w') as f:
            pickle.dump([images, images_timewin], f)
    else:
        # Getting back the objects:
        with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
            images, images_timewin = pickle.load(f) 

    #print(images_timewin.shape)

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

    inputs = np.swapaxes(images_timewin, 0, 1)

    outputs = (np.squeeze(feats[:, -1]) - 1).astype(np.int32)
    #pred, loss = predict(inputs, outputs, '1dconv') 
    #pred_all, loss_all = predict_all(inputs[0], outputs[0])
    
    reward_lists = []
    pred_rls = []
    pred_alls = []


    reward_list, pred_rl, pred_all = train_rl(inputs, outputs, fold_pairs)
    reward_lists.append(reward_list)
    pred_rls.append(pred_rl)
    pred_alls.append(pred_all)
        
#print(pred, loss)
#print(pred_all, loss_all)


    n_data = float(len(pred_rl))
    sum_dic = {}
    acc_dic = {}
    for model in model_list:
        sum_dic[model] = sum(pred_all[model])
        acc_dic[model] = sum_dic[model]/n_data
    sum_rl = sum(pred_rl)
    acc_rl = sum(pred_rl) / n_data

    print("summary of models")
    print(sum_dic)
    print(sum_rl)
    print(acc_dic)
    print(acc_rl)





