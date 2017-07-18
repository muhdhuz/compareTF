import numpy as np
import tensorflow as tf
from params import *

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, dropout)
    
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
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
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 180 outputs
    'wc1': tf.Variable(tf.truncated_normal([IMAGE_HEIGHT, 3, 1, 24], stddev=0.1)),
    # 5x5 conv, 128 inputs, 180 outputs
    'wc2': tf.Variable(tf.truncated_normal([-(-IMAGE_HEIGHT//2), 3, 24, 48], stddev=0.1)),
    # fully connected, (37//4=10)*(50//4=13)*180 inputs, 1200 outputs
    # 5x5 conv, 128 inputs, 180 outputs
    'wc3': tf.Variable(tf.truncated_normal([-(-IMAGE_HEIGHT//4), 3, 48, 96], stddev=0.1)),
    # fully connected, (37//4=10)*(50//4=13)*180 inputs, 1200 outputs
    'wd1': tf.Variable(tf.truncated_normal([-(-IMAGE_HEIGHT//4)*-(-IMAGE_WIDTH//4)*96, 800], stddev=0.1)),
    # 1200 inputs, 50 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([800, N_LABELS], stddev=0.01))
}

biases = {
    'bc1': tf.Variable(tf.zeros([24])),
    'bc2': tf.Variable(tf.constant(1.0,shape=[48])),
    'bc3': tf.Variable(tf.constant(1.0,shape=[96])),
    'bd1': tf.Variable(tf.constant(1.0,shape=[800])),
    'out': tf.Variable(tf.constant(1.0,shape=[N_LABELS]))
}