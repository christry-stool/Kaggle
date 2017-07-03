import tensorflow as tf
import numpy as np


def conv(layer_name, input_tensor, out_channels, kernel_size, strides, padding='SAME', train=True):
    in_channels = input_tensor.shape[-1]
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name="weight",
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  trainable=train)
        biases = tf.get_variable(name="bias",
                                 shape=[out_channels],
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=train)

        output = tf.nn.conv2d(input_tensor, weights, strides=strides, padding=padding, name='conv')
        output = tf.nn.relu(tf.nn.bias_add(output, biases, name='bias_add'), name='relu')
        return output


def pool(layer_name, input_tensor, kernel_size, strides, padding='SAME', by_max=True):
    if by_max:
        with tf.name_scope(layer_name):
            output = tf.nn.max_pool(input_tensor, ksize=kernel_size, strides=strides, padding=padding, name='pool')
    else:
        with tf.name_scope(layer_name):
            output = tf.nn.avg_pool(input_tensor, ksize=kernel_size, strides=strides, padding=padding, name='pool')
    return output


def full_connect(layer_name, input_tensor, out_nodes, regularizer=None):
    shape = input_tensor.get_shape()
    if len(shape) == 4:
        in_nodes = shape[1].value * shape[2].value * shape[3].value
    else:
        in_nodes = shape[-1].value
    reshape_input = tf.reshape(input_tensor, [-1, in_nodes])

    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name='weight',
                                  shape=[in_nodes, out_nodes],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        biases = tf.get_variable(name='bias',
                                 shape=[out_nodes],
                                 initializer=tf.constant_initializer(0.1))
        output = tf.nn.relu(tf.matmul(reshape_input, weights) + biases)
        return output


def full_connect_not_relu(layer_name, input_tensor, out_nodes, regularizer=None):
    shape = input_tensor.get_shape()
    if len(shape) == 4:
        in_nodes = shape[1].value * shape[2].value * shape[3].value
    else:
        in_nodes = shape[-1].value
    reshape_input = tf.reshape(input_tensor, [-1, in_nodes])

    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name='weight',
                                  shape=[in_nodes, out_nodes],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        biases = tf.get_variable(name='bias',
                                 shape=[out_nodes],
                                 initializer=tf.constant_initializer(0.1))
        output = tf.matmul(reshape_input, weights) + biases
        return output


def dropout(input_tensor, value=0.5):
    output = tf.nn.dropout(input_tensor, value)
    return output


def batch_norm(input_tensor):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
    output = tf.nn.batch_normalization(input_tensor,
                                       mean=batch_mean,
                                       variance=batch_var,
                                       offset=None,
                                       scale=None,
                                       variance_epsilon=epsilon)
    return output
