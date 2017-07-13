import tensorflow as tf
import numpy as np
import pandas as pd

import train
import model
import input_data

TEST_PATH = 'datasets_test/'


def evaluate():
    image_file, label = input_data.get_test_file(TEST_PATH)
    image_batch, label_batch = input_data.get_batch(image_files=image_file,
                                                    labels=label,
                                                    weight=train.WEIGHT,
                                                    height=train.HEIGHT,
                                                    channel=train.CHANNEL,
                                                    batch_size=train.BATCH_SIZE,
                                                    capacity=train.CAPACITY,
                                                    num_epochs=1,
                                                    shuffle=False)

    x = tf.placeholder(tf.float32, [train.BATCH_SIZE, train.WEIGHT, train.HEIGHT, train.CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [train.BATCH_SIZE, train.NUM_CLASSES], name='y-input')

    y = model.inference(x, train.NUM_CLASSES)
    predict_y = tf.argmax(y, 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            conv_y_predict = []
            try:
                for i in range(400):
                    if coord.should_stop():
                        break
                    print(i)
                    xs, ys = sess.run([image_batch, label_batch])
                    print(ys[:10])
                    yy, y_predict = sess.run([y, predict_y], feed_dict={x: xs, y_: ys})
                    conv_y_predict = np.append(conv_y_predict, y_predict)

            except tf.errors.OutOfRangeError:
                conv_submission = pd.DataFrame({'id': range(1, 12501), 'label': np.int32(conv_y_predict)})
                conv_submission.to_csv('conv_submission.csv', index=False)
                print('Done training -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)


evaluate()
