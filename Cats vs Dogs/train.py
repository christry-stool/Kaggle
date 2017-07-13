import tensorflow as tf

import model
import tools
import input_data
import os

FILE_PATH = 'datasets/'
LOG_TRAIN_PATH = 'log/train'
LOG_VAL_PATH = 'log/val'
MODEL_SAVE_PATH = 'models/'
MODEL_NAME = 'model.ckpt'

WEIGHT = 208
HEIGHT = 208
CHANNEL = 3
NUM_CLASSES = 2

LEARNING_RATE = 0.001
REGULARAZTION_RATE = 0.0001

BATCH_SIZE = 50
CAPACITY = 500
TRAINING_STEPS = 30000
VALIDATION_PERCENTAGE = 20


def train():
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

    train_image_file, train_label, val_image_file, val_label = input_data.get_file(FILE_PATH, VALIDATION_PERCENTAGE)

    train_image_batch, train_label_batch = input_data.get_batch(train_image_file,
                                                                train_label,
                                                                weight=WEIGHT,
                                                                height=HEIGHT,
                                                                channel=CHANNEL,
                                                                batch_size=BATCH_SIZE,
                                                                capacity=CAPACITY)

    val_image_batch, val_label_batch = input_data.get_batch(val_image_file,
                                                            val_label,
                                                            weight=WEIGHT,
                                                            height=HEIGHT,
                                                            channel=CHANNEL,
                                                            batch_size=BATCH_SIZE,
                                                            capacity=CAPACITY)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WEIGHT, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='y-input')

    if ckpt and ckpt.model_checkpoint_path:
        num_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        num_step = int(num_step)
        print(num_step)
        global_step = tf.Variable(num_step, trainable=False)
    else:
        global_step = tf.Variable(0, trainable=False)
#    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = model.inference(x, NUM_CLASSES)
    train_loss = tools.loss(logits=y, labels=y_)
    train_acc = tools.accuracy(logits=y, labels=y_)
    train_op = tools.optimizer(train_loss, LEARNING_RATE, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        val_writer = tf.summary.FileWriter(LOG_VAL_PATH, sess.graph)

        try:
            for i in range(TRAINING_STEPS):
                if coord.should_stop():
                    break

                xs, ys = sess.run([train_image_batch, train_label_batch])
                _, loss_value, acc_value, step = sess.run([train_op, train_loss, train_acc, global_step],
                                                          feed_dict={x: xs, y_: ys})

                if i % 1 == 0:
                    print("After %d training step(s), loss on training batch is %g, accuracy is %g" %
                          (step, loss_value, acc_value))

                if i % 50 == 0:
                    summary_str = sess.run(summary_op, feed_dict={x: xs, y_: ys})
                    train_writer.add_summary(summary_str, step)

                if i % 200 == 0:
                    val_xs, val_ys = sess.run([val_image_batch, val_label_batch])
                    val_loss_value, val_acc_value = sess.run([train_loss, train_acc], feed_dict={x: val_xs, y_: val_ys})
                    print("After %d training step(s), valuation loss is %g, accuracy is %g" %
                          (step, val_loss_value, val_acc_value))
                    summary_str = sess.run(summary_op, feed_dict={x: val_xs, y_: val_ys})
                    val_writer.add_summary(summary_str, step)

                if i % 200 == 0 or step + 1 == TRAINING_STEPS:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            coord.request_stop()

        coord.join(threads)


def validate():
    train_image_file, train_label, val_image_file, val_label = input_data.get_file(FILE_PATH, VALIDATION_PERCENTAGE)

    val_image_batch, val_label_batch = input_data.get_batch(val_image_file,
                                                            val_label,
                                                            weight=WEIGHT,
                                                            height=HEIGHT,
                                                            channel=CHANNEL,
                                                            batch_size=BATCH_SIZE,
                                                            capacity=CAPACITY,
                                                            num_epochs=1)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WEIGHT, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='y-input')

    y = model.inference(x, NUM_CLASSES, evaluate=False)
    predict_y = tf.argmax(y, 1)
    val_loss = tools.loss(logits=y, labels=y_)
    val_acc = tools.accuracy(logits=y, labels=y_)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            global_score = 0.
            num_step = 0
            try:
                for i in range(TRAINING_STEPS):
                    if coord.should_stop():
                        break

                    val_xs, val_ys = sess.run([val_image_batch, val_label_batch])
                    yy, loss_value, acc_value = sess.run([predict_y, val_loss, val_acc],
                                                         feed_dict={x: val_xs, y_: val_ys})
                    global_score += acc_value
                    num_step += 1
                    print("%d:After %s training step(s), validation accuracy = %g" % (i, global_step, acc_value))

            except tf.errors.OutOfRangeError:
                print("global accuracy = %g" % (global_score / num_step))
                print('Done training -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)

# train()
validate()
