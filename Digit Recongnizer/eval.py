import tensorflow as tf
import numpy as np
import pandas as pd

import train
import model
import input_data

TEST_PATH = 'datasets/test.csv'


def evaluate():
    image = input_data.get_test(TEST_PATH)

    x = tf.placeholder(tf.float32, [train.BATCH_SIZE, train.WIDTH, train.HEIGHT, train.CHANNEL], name='x-input')

    y = model.LeNet_5(x, train.N_CLASSES)
    predict_y = tf.argmax(y, 1)

    variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)

    with tf.Session() as sess:
        xs = image
        reshaped_xs = np.reshape(xs, (-1, train.WIDTH, train.HEIGHT, train.CHANNEL))

        ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            num_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print(num_step)
            conv_y_predict = []
            for i in np.arange(100, 28001, 100):
                validate_feed = {x: reshaped_xs[i - 100:i]}
                y_predict = sess.run(predict_y, feed_dict=validate_feed)
                conv_y_predict = np.append(conv_y_predict, y_predict)
            conv_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': np.int32(conv_y_predict)})
            conv_submission.to_csv('datasets/conv_submission.csv', index=False)

        else:
            print('No checkpoint file found')

evaluate()

