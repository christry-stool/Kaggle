import tensorflow as tf
import numpy as np


def loss(logits, labels, regularizer=None):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        if regularizer is not None:
            cross_entropy_mean = tf.add(tf.reduce_mean(cross_entropy), tf.add_n(tf.get_collection('losses')), name='loss')
        else:
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', cross_entropy_mean)
        return cross_entropy_mean


def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        # correct_prediction = tf.nn.in_top_k(logits, labels, 1)
        # correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(scope+'/accuracy', accuracy_value)
        return accuracy_value


def optimizer(losses, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=losses, global_step=global_step)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        #     .minimize(loss=losses, global_step=global_step)
        return train_step


def moving_average(decay, global_step):
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        return variable_averages_op
