import tensorflow as tf
import numpy as np
import os

import input_data
import model
import tools

TRAIN_PATH = 'datasets/train.csv'
LOG_TRAIN_PATH = 'log/train'
LOG_VAL_PATH = 'log/val'
MODEL_SAVE_PATH = 'models/'
MODEL_NAME = 'model.ckpt'

WIDTH = 28
HEIGHT = 28
CHANNEL = 1
N_CLASSES = 10

VALIDATION_PERCENTAGE = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def train():
    train_image, train_label, val_image, val_label = input_data.get_train_and_val(TRAIN_PATH, VALIDATION_PERCENTAGE)
    train_pointer = [0]
    val_pointer = [0]

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WIDTH, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = model.LeNet_5(input_tensor=x, n_classes=N_CLASSES, regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)

    variables_averages_op = tools.moving_average(MOVING_AVERAGE_DECAY, global_step)
    train_loss = tools.loss(logits=y, labels=y_, regularizer=regularizer)
    train_acc = tools.accuracy(logits=y, labels=y_)
    train_step = tools.optimizer(losses=train_loss, learning_rate=LEARNING_RATE, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        val_writer = tf.summary.FileWriter(LOG_VAL_PATH, sess.graph)

        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = input_data.get_batch(train_image, train_label, WIDTH, HEIGHT, CHANNEL, BATCH_SIZE, train_pointer)
            _, loss_value, acc_value, step = sess.run([train_op, train_loss, train_acc, global_step],
                                                      feed_dict={x: xs, y_: ys})
            if i % 50 == 0:
                print("After %d training step(s), loss on training batch is %g, accuracy is %g" %
                      (step, loss_value, acc_value))
                summary_str = sess.run(summary_op, feed_dict={x: xs, y_: ys})
                train_writer.add_summary(summary_str, step)

            if i % 200 == 0:
                val_xs, val_ys = input_data.get_batch(val_image, val_label, WIDTH, HEIGHT, CHANNEL, BATCH_SIZE,
                                                      val_pointer)
                val_loss_value, val_acc_value = sess.run([train_loss, train_acc], feed_dict={x: val_xs, y_: val_ys})
                print("After %d training step(s), valuation loss is %g, accuracy is %g" %
                      (step, val_loss_value, val_acc_value))
                summary_str = sess.run(summary_op, feed_dict={x: val_xs, y_: val_ys})
                val_writer.add_summary(summary_str, step)

            if i % 1000 == 0 or i+1 == TRAINING_STEPS:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def valuate():
    train_image, train_label, val_image, val_label = input_data.get_train_and_val(TRAIN_PATH, VALIDATION_PERCENTAGE)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WIDTH, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES], name='y-input')

    y = model.LeNet_5(input_tensor=x, n_classes=N_CLASSES)
    acc = tools.accuracy(logits=y, labels=y_)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)

    with tf.Session() as sess:
        xs = val_image
        reshaped_xs = np.reshape(xs, (-1, WIDTH, HEIGHT, CHANNEL))
        ys = val_label

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_score = 0.
            for i in np.arange(100, ys.shape[0], 100):
                validate_feed = {x: reshaped_xs[i - 100:i], y_: ys[i - 100:i]}
                accuracy_score = sess.run(acc, feed_dict=validate_feed)
                global_score += accuracy_score
                if i+100 > ys.shape[0]:
                    global_score /= i/100.
            print("After %s training step(s), validation accuracy = %g" % (global_step, global_score))
        else:
            print('No checkpoint file found')
