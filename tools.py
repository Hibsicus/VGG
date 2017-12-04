# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#%%
def conv(layer_name, x, out_channel, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    '''
    Args:
        layer_name: conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels:number of output channels or comvolutional kernels
        kernel_size: VGG paper used:[3, 3]
        stride: VGG paper used[1, 1, 1, 1]
    Returns:
        4D tensor
    '''
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channel],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channel],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

#%%
    
def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    '''
    Args:
        x: input tensor
        kernel: 2x2
        stride:
        is_max_pool:
            if True max
            else avg
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
#%%
def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)

#%%
def FC_layer(layer_name, x, out_nodes):
    '''
    Args:
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''    
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
        
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) #flatten inti 1D
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x
#%%
def loss(logits, labels):
    '''
    Args:
        logits: [batch_size, n_classes]
        labels: one-hot labels
    '''    
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss
    
#%%
def accuracy(logits, labels):
    '''
    Args:
        logits: float - [batch_size, num_classes]
        labels: labels tensor
    Returns:
        accuracy
    '''
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
        return accuracy
#%%
def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimize.minimize(loss, global_step=global_step)
        return train_op
#%%