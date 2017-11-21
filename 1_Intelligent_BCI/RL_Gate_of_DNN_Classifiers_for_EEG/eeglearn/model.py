import tensorflow as tf
import numpy as np
import random
np.random.seed(1234)
from collections import deque

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


def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v



class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.total_reward = 0
        self.current_reward = 0
    


class DQN:
    REPLAY_MEMORY = 10000
    BATCH_SIZE = 1
    GAMMA = 0.99
    #STATE_LEN = 1 #for consider past state
    
    n_window = 7
    n_color = 3
    def __init__(self, session, width, height, n_act):
        self.session = session
        self.n_act = n_act
        self.width = width
        self.height = height
        self.memory = deque()
        self.state = None

        self.input_X = tf.placeholder(tf.float32, [None, self.n_window, width, height, self.n_color])
        self.input_A = tf.placeholder(tf.int64, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])

        #self.Q = self._build_network('main')
        self.Q = self._build_1dcnn('main')
        self.cost, self.train_op = self._build_op()

        #self.target_Q = self._build_network('target')
        self.target_Q = self._build_1dcnn('target')

    def _build_cnn(self, input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3, filtersize=(3, 3)):
        count = 0
        n_classes = self.n_act
        if input_var is None:
            x = tf.placeholder(tf.float32, shape=[None, imsize, imsize, n_colors])
        else:
            x = input_var
        x_image = tf.reshape(x, [-1, imsize, imsize, n_colors])

        inkernel = 3
        outkernel = n_filters_first
        
        #If w_init is not... #Define Weight, Bias using weight_variable & bias variable
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
            h_pool.append(h_convout) #save result
            x_datain = h_convout
        return h_convout, weights


    def _build_1dcnn(self, name, n_colors=3, n_timewin=3, n_filter=64, save_v={}):
        with tf.variable_scope(name):
            convnets = []
            w_init = None

            # Build 7 parallel CNNs with shared weights
            for i in range(n_timewin):
                if i == 0:
                    convnet, w_init = self._build_cnn(self.input_X[:, i], n_colors=n_colors)
                else:
                    convnet, _ = self._build_cnn(self.input_X[:, i], w_init=w_init, n_colors=n_colors)
                convnets.append(tf.contrib.layers.flatten(convnet))
            
            convpool = tf.concat([tf.expand_dims(t, 1) for t in convnets], 1) #dataN, winN, features (?, 3, 2048)
            featureN = convpool.shape[2]

            #define 1d conv
            filter_ = get_scope_variable("filter", "conv_filter", shape = [n_timewin, featureN, n_filter])
            convpool = tf.nn.conv1d(convpool, filter_, stride=1, padding='VALID') #, data_format = "NCHW")) #?, 64, 1
            convpool = tf.reshape(convpool, [-1, n_filter]) #?, 64. 1
             
            #weight init.
            if 'conv1d_w0' not in save_v:
                save_v['conv1d_w0'] = weight_variable([n_filter, 512])
            if 'conv1d_w1' not in save_v:
                save_v['conv1d_w1'] = weight_variable([512, self.n_act])
            
            #bias init.
            if 'conv1d_b0' not in save_v:
                save_v['conv1d_b0'] = bias_variable([512])
            if 'conv1d_b1' not in save_v:
                save_v['conv1d_b1'] = bias_variable([self.n_act])

            #dense layer
            convpool = tf.add(tf.matmul(convpool, save_v['conv1d_w0']), save_v['conv1d_b0'])
            convpool = tf.nn.relu(convpool)
            convpool = tf.nn.dropout(convpool, 0.5)

            convpool = tf.add(tf.matmul(convpool, save_v['conv1d_w1']), save_v['conv1d_b1'], name='network')

            return convpool


    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=None)
        return Q


    def _build_op(self):
        one_hot = tf.one_hot(self.input_A, self.n_act, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op


    # refer: https://github.com/hunkim/ReinforcementZeroToAll/
    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q,
                                   feed_dict={self.input_X: [self.state]})
        action = np.argmax(Q_value[0])
        return action


    def init_state(self, state):
        #state = [state for _ in range(self.STATE_LEN)]
        #self.state = np.stack(state, axis=2)

        state = np.swapaxes(state, 2, 3)
        state = np.swapaxes(state, 3, 4)
        state = np.squeeze(state)
        self.state = state
 
    def remember(self, state, action, reward, terminal):
        #next_state = np.append(self.state[:, :, 1:], next_state, axis=2)
        
        #print('remember')
        #print(state.shape)
        state = np.swapaxes(state, 2, 3)
        #print(state.shape)
        state = np.swapaxes(state, 3, 4)
        state = np.squeeze(state)
        #print(state.shape)
        next_state = state
        #print('rememberend')
        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        
        # print(sample_memory)
        


        sample_memory = random.sample(self.memory, self.BATCH_SIZE)
        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]


        #return state, action, reward, terminal
        return state, next_state, action, reward, terminal
    
    
    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q,
                                          feed_dict={self.input_X: next_state})

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op,
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
})
