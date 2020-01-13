import tensorflow as tf
import numpy as np
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from staircase import clip_to_staircase
bitwidth_A = 1
bitwidth_W = 3
weight_decay = 5e-4


def quantize(x, alpha, is_training):

    shape = x.get_shape()
    mask = tf.ones(shape)
    thre_x = tf.reduce_max(tf.abs(x)) * 0.02
    thre_y = tf.reduce_max(tf.abs(x)) * 0.07
    thre_z = tf.reduce_max(tf.abs(x)) * 0.14
    mask_x = tf.where((x > -thre_x) & (x < thre_x), tf.zeros(shape), mask * 2 ** (-2))
    mask_y = tf.where((x > -thre_y) & (x < thre_y), mask_x, mask * 2 ** (-1))
    mask_z = tf.where((x > -thre_z) & (x < thre_z), mask_y, mask)

    if is_training:
        return x - (x - tf.sign(x) * mask_z) * alpha
    else:
        return tf.sign(x) * mask_z


def QuantizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='BinarizedSpatialConvolution'):
    def b_conv2d(x, alpha, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w, -1, 1)
            bin_w = quantize(w, alpha, is_training, bitwidth_W)
            bin_x = quantize(x, alpha, is_training, bitwidth_A)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane], initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d


def QuantizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, alpha, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weights', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w, -1, 1)
            bin_w = quantize(w, alpha, is_training)
            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('biases', [nOutputPlane], initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d


def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='SpatialConvolution'):
    def conv2d(x, alpha, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weights', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(weight_decay)(w))
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('biases', [nOutputPlane], initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d


def LastConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='SpatialConvolution'):
    def conv2d(x, alpha, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weights', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(weight_decay)(w))
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('biases', [nOutputPlane], initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return tf.reshape(out, [out.get_shape().as_list()[0], -1])
    return conv2d


def Affine(nOutputPlane, bias=True, name='Affine', reuse=False):
    def affineLayer(x, alpha, is_training=True):
        with tf.variable_scope(name, None, [x], reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(weight_decay)(w))
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane], initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer


def QuantizedAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def b_affineLayer(x, alpha, is_training=True):
        with tf.variable_scope(name, 'Affine', [x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = quantize(x, alpha, is_training, bitwidth_A)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            w = tf.clip_by_value(w, -1, 1)
            bin_w = quantize(w, alpha, is_training, bitwidth_W)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane], initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer


def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)


def wrapNN(f, *args, **kwargs):
    def layer(x, alpha, scope='', is_training=True):
        return f(x, *args, **kwargs)
    return layer


def Dropout(p, name='Dropout'):
    def dropout_layer(x, alpha, is_training=True):
        with tf.variable_scope(name, None, [x]):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x, p)
            else:
                return x
    return dropout_layer


def ReLU(name='ReLU'):
    def layer(x, alpha, is_training=True):
        with tf.variable_scope(name, None, [x]):
            return tf.nn.relu(x)
    return layer


def HardTanh(name='HardTanh'):
    def layer(x, alpha, is_training=True):
        with tf.variable_scope(name, None, [x]):
            return tf.clip_by_value(x, -1, 1)
    return layer


def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH

    def max_pool(x, alpha, is_training=True):
        with tf.variable_scope(name, None, [x]):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool


def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH

    def avg_pool(x, alpha, is_training=True):
        with tf.variable_scope(name, None, [x]):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool


def BatchNormalization(name='bn'):
    def layer(x, alpha, is_training=True):
        return tf.layers.batch_normalization(x,
                                             momentum=0.9,
                                             epsilon=1e-5,
                                             scale=True,
                                             training=True,
                                             name=name)
    return layer


def Sequential(moduleList):
    def model(x, alpha, is_training=True):
    # Create model
        output = x
        #with tf.variable_scope(name,None,[x]):
        for i, m in enumerate(moduleList):
            output = m(output, alpha=alpha, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model


def Residual(moduleList):
    m = Sequential(moduleList)
    def model(x, alpha, is_training=True):
    # Create model
        output = tf.add(m(x, alpha=alpha, is_training=is_training), x)
        return output
    return model
