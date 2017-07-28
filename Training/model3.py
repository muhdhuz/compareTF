import numpy as np
import tensorflow as tf
from params import *

if FRE_ORIENTATION == "2D":
    k_height = K_HEIGHT
    k_inputChannels = NUM_CHANNELS
    k_ConvRows = 3 #conv kernel height
    k_ConvCols = 3 #conv kernel width
    k_poolRows = 2
    k_downsampledHeight = -(-k_height//4)
    k_downsampledWidth = -(-K_NUMFRAMES//4)

elif FRE_ORIENTATION == "1D":
    k_height = K_HEIGHT
    k_inputChannels = NUM_CHANNELS
    k_ConvRows = 1 #conv kernel height
    k_ConvCols = 3 #conv kernel width
    k_poolRows = 1
    k_downsampledHeight = 1
    k_downsampledWidth = -(-K_NUMFRAMES//4)

# model params common to both 1D and 2D
K_NUMCONVLAYERS = 3
k_ConvStrideRows = 1 #kernel horizontal stride
k_ConvStrideCols = 1 #kernel vertical stride
k_poolStrideRows = k_poolRows

###########################

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k_h=k_poolRows, k_w=2):
    # MaxPool2D wrapper
    # ksize = [batch, height, width, channels]
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.dropout(conv1, dropout)
    
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    
    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    return out
    
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 180 outputs
    'wc1': tf.Variable(tf.truncated_normal([k_ConvRows, k_ConvCols, k_inputChannels, L1_CHANNELS], stddev=0.1), name='wc1'),
    # 5x5 conv, 128 inputs, 180 outputs
    'wc2': tf.Variable(tf.truncated_normal([k_ConvRows, k_ConvCols, L1_CHANNELS, L2_CHANNELS], stddev=0.1), name='wc2'),
    # 5x5 conv, 128 inputs, 180 outputs
    'wc3': tf.Variable(tf.truncated_normal([k_ConvRows, k_ConvCols, L2_CHANNELS, L3_CHANNELS], stddev=0.1), name='wc3'),
    # fully connected, (37//4=10)*(50//4=13)*180 inputs, 1200 outputs
    'wd1': tf.Variable(tf.truncated_normal([k_downsampledHeight*k_downsampledWidth*L3_CHANNELS, FC_SIZE], stddev=0.1), name='wd1'),
    # 1200 inputs, 50 outputs (class prediction)
    'wout': tf.Variable(tf.truncated_normal([FC_SIZE, N_LABELS], stddev=0.01), name='wout')
}

biases = {
    'bc1': tf.Variable(tf.zeros([L1_CHANNELS]), name='bc1'),
    'bc2': tf.Variable(tf.constant(1.0,shape=[L2_CHANNELS]), name='bc2'),
    'bc3': tf.Variable(tf.constant(1.0,shape=[L3_CHANNELS]), name='bc3'),
    'bd1': tf.Variable(tf.constant(1.0,shape=[FC_SIZE]), name='bd1'),
    'bout': tf.Variable(tf.constant(1.0,shape=[N_LABELS]), name='bout')
}