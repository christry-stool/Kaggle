import tensorflow as tf

import layers


def inference(input_tensor, n_classes, train=True, regularizer=None, evaluate=False):
    input_tensor = layers.conv('conv1', input_tensor, out_channels=16, kernel_size=[3, 3], strides=[1, 1, 1, 1], train=train)
    input_tensor = layers.batch_norm(input_tensor)
    input_tensor = layers.pool('pool1', input_tensor, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], by_max=True)

    input_tensor = layers.conv('conv2', input_tensor, out_channels=16, kernel_size=[3, 3], strides=[1, 1, 1, 1], train=train)
    input_tensor = layers.batch_norm(input_tensor)
    input_tensor = layers.pool('pool2', input_tensor, kernel_size=[1, 2, 2, 1], strides=[1, 1, 1, 1], by_max=True)

    input_tensor = layers.full_connect('fc1', input_tensor, out_nodes=128, regularizer=regularizer)
    input_tensor = layers.full_connect('fc2', input_tensor, out_nodes=128, regularizer=regularizer)
    if not evaluate:
        input_tensor = layers.dropout(input_tensor)
    output_tensor = layers.full_connect_not_relu('fc3', input_tensor, out_nodes=n_classes, regularizer=regularizer)

    return output_tensor

