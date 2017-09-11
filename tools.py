# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 06:51:16 2017

@author: User
"""

import tensorflow as tf
import numpy as np

#%%
def conv(layer_name, x, out_channels, kernel_size=[3, 3],strude = [1, 1, 1, 1], is_pretrain=True):
    '''
    Args:    
        Layer_name: conv1, pool1 ...
        x: input tensor, [batch_size, height, width, channels]
        out_channles: number of output channels(or convolutional kernels)
        kernel_size: the layer of convolution kernel, VGG paper used: [3, 3]
        stride: A list of ints. 1-D of Length 4. VGG paper used: [1, 1, 1, 1]
    Returns:
        4D tensor
    '''
    in_channels = x.get_shap()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
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
        kernel: pooling kernel, VGG paper used[1, 2, 2, 1], the size of kernel is 2x2
        stride: stride size, VGG paper used[1, 2, 2, 1]
        padding:
        is_max_pool: boolen
            if true: use max pooling
            else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x
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
    return x
#%%
def FC_layer(layer_name, x, out_nodes):
    '''
    Args:
        Layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes:number of neurons for current FC layer
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
        flat_x = tf.reshape(x, [-1, size])
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x
#%%
def loss(logits, labels):
    '''
    Args:
        Logits: Logits tensor, [batch_size, n_classes]
        Labels: one_hot, labels
    '''
    with tf.name_scope('Loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels==labels, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss)
        return loss
    
#%%

def accuracy(logits, lables):
    '''
    Args:
        Logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        Labels: Labels tensor
    Returns:
        A scalar int32 tensor with the number of examples(out of batch_size)
        that were predicted correctly.
    '''
    with tf.name_scope("accuracy") as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
        return accuracy
#%%
def optimize(loss, learning_rate, global_step):
    '''Gradient Descent
    '''
    with tf.name_scope('optimizer'):
        optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimize.minimize(loss, global_step=global_step)
        return train_op
#%%
