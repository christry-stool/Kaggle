import pandas as pd
import numpy as np
import math


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def get_train_and_val(file_path, validation_percentage):
    data = pd.read_csv(file_path)
    label = data['label']
    image = data.drop('label', 1)
    label = np.array(label)
    image = np.array(image, np.float32)
    image /= 255.
    n_examples = image.shape[0]
    print(n_examples)
    n_val = math.ceil(n_examples*validation_percentage/100.)
    n_train = n_examples-n_val
    train_image = image[:n_train]
    train_label = label[:n_train]
    train_label = dense_to_one_hot(train_label, 10)
    val_image = image[n_train:-1]
    val_label = label[n_train:-1]
    val_label = dense_to_one_hot(val_label, 10)
    print(train_image.shape, " ", val_image.shape)
    return train_image, train_label, val_image, val_label


def get_test(file_path):
    data = pd.read_csv(file_path)
    image = np.array(data, np.float32)
    image /= 255.
    print(image.shape)
    return image


def get_batch(image, label, width, height, channel, batch_size, pointer):
    n_examples = label.shape[0]
    if pointer[0] > n_examples:
        pointer[0] = 0
    pi = pointer[0]
    if pi + batch_size < n_examples:
        batch_image = image[pi: pi+batch_size]
        batch_image = np.reshape(batch_image, [-1, width, height, channel])
        batch_label = label[pi: pi+batch_size]
        pi += batch_size
        pointer[0] = pi
        return batch_image, batch_label
    else:
        batch_image = image[pi: n_examples]
        batch_label = label[pi: n_examples]
        n_then = batch_size - batch_image.shape[0]
        batch_image_then = image[:n_then]
        batch_label_then = label[:n_then]
        batch_image = np.concatenate((batch_image, batch_image_then), axis=0)
        batch_image = np.reshape(batch_image, [-1, width, height, channel])
        batch_label = np.concatenate((batch_label, batch_label_then), axis=0)
        pi = n_then
        pointer[0] = pi
        return batch_image, batch_label



