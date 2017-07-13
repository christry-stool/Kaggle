import tensorflow as tf
import numpy as np
import math
import os


def get_file(file_path, validation_percentage):
    if os.path.exists('train_label.txt'):
        print('exist')
        return get_file_from_txt()

    file_list = []
    label_list = []
    for file in os.listdir(file_path):
        name = file.split(sep='.')
        if name[0] == 'cat':
            file_list.append(file_path + file)
            label_list.append(0)
        else:
            file_list.append(file_path + file)
            label_list.append(1)
    examples = np.array([file_list, label_list])
    examples = examples.transpose()
    np.random.shuffle(examples)
    image_file = examples[:, 0]
    labels = np.array(examples[:, 1], dtype=np.int32)
    n_examples = len(labels)
    n_val = math.ceil(n_examples*validation_percentage/100.)
    n_train = n_examples - n_val
    train_image_file = image_file[0:n_train]
    train_label = labels[0:n_train]
    val_image_file = image_file[n_train:-1]
    val_label = labels[n_train:-1]

    output = open('train_image_file.txt', 'w')
    for item in train_image_file:
        output.write(item)
        output.write('\n')
    output.close()

    output = open('train_label.txt', 'w')
    for item in train_label:
        item = str(item)
        output.write(item)
        output.write('\n')
    output.close()

    output = open('val_image_file.txt', 'w')
    for item in val_image_file:
        output.write(item)
        output.write('\n')
    output.close()

    output = open('val_label.txt', 'w')
    for item in val_label:
        item = str(item)
        output.write(item)
        output.write('\n')
    output.close()

    return train_image_file, train_label, val_image_file, val_label


def get_file_from_txt():
    train_image_file = []
    train_label = []
    val_image_file = []
    val_label = []

    with open('train_image_file.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            train_image_file.append(line)

    with open('train_label.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            train_label.append(line)

    with open('val_image_file.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            val_image_file.append(line)

    with open('val_label.txt', 'r') as file:
        for line in file:
            line = line.strip('\n')
            val_label.append(line)

    train_image_file = np.array(train_image_file)
    val_image_file = np.array(val_image_file)
    train_label = np.array(train_label, dtype=np.int32)
    val_label = np.array(val_label, dtype=np.int32)

    return train_image_file, train_label, val_image_file, val_label


def get_batch(image_files, labels, weight, height, channel, batch_size, capacity, num_epochs=None, shuffle=True):
    image_files = tf.cast(image_files, dtype=tf.string)
    labels = tf.cast(labels, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([image_files, labels], shuffle=shuffle, num_epochs=num_epochs)
    image_file = input_queue[0]
    label = input_queue[1]
    image_contents = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image_contents, channels=channel)

    image = tf.image.resize_image_with_crop_or_pad(image, height, weight)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=50,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, dtype=tf.float32)
    label_batch = tf.one_hot(label_batch, depth=2)
    label_batch = tf.reshape(label_batch, [batch_size, 2])
    label_batch = tf.cast(label_batch, dtype=tf.float32)
    return image_batch, label_batch


def get_test_file(file_path):
    file_list = []
    label_list = []
    t = 1
    for file in os.listdir(file_path):
        file_list.append(file_path + file)
        label_list.append(t)
        t = 1-t
    examples = np.array([file_list, label_list])
    examples = examples.transpose()
    image_file = examples[:, 0]
    labels = np.array(examples[:, 1], dtype=np.int32)
    print(labels[:10])
    return image_file, labels



