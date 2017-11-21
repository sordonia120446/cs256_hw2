"""
CNN classifier of Zener cards.

:authors Jason, Nick, Sam
"""

import argparse
import glob
import math
import os
import sys
import time
import shutil
import logging

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import init_data


tf.logging.set_verbosity(tf.logging.INFO)
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

modes = ['cross', 'cross-l1', 'cross-l2', 'ctest']


def load_data(args):
    """
    Iterate through each file in data_folder and construct
    the input data features.
    """
    
    data = init_data(args)
    x = data['X_plus'] + data['X_minus']
    y = data['Y_plus'] + data['Y_minus']

    x = data['X_minus'] + data['X_plus']
    y = data['Y_minus'] + data['Y_plus']

    ret = []

    for img, label in zip(x, y):
        #img = img.reshape(32, 32).astype('uint8')
        img = img.reshape(32, 32)

        # im = Image.fromarray(img.astype('uint8'))
        # im.show()
        #Image.fromarray(img).show()
        # sys.exit(0)
        ret.append({
            'x': img,
            'y': label
        })

    return ret


def k_fold_split(input_data, iteration, k=5):
    """Split data into k-subsets, with k-th one being the test."""

    x = [d['x'] for d in input_data]  # imgs
    y = [d['y'] for d in input_data]  # indices of labels

    total_size = len(x)
    subset_size = math.floor(total_size / 5)
    # break_ind = total_size - subset_size
    right_ind = (iteration + 1) * subset_size
    left_ind = iteration * subset_size
    current_ind = 0

    train_data = []
    test_data = []
    for x, y in zip(x, y):
        if current_ind < left_ind or current_ind >= right_ind:
            train_data.append({
                'x': x,
                'y': y
            })
        else:
            test_data.append({
                'x': x,
                'y': y
            })

        current_ind += 1
    return train_data, test_data


def numpyize_inputs(input_data):
    """Convert features & labels into numpy arrays for input into tf."""
    features = np.asarray([d['x'] for d in input_data], dtype=np.float32)
    labels = np.asarray([d['y'] for d in input_data], dtype=np.float32)

    return features, labels


def get_confusion_matrix(labels, predictions):
    '''
    Generate a confusion matrix for evaluation.

    Note for rows, columns 0...5 0=NONSTICK, 1=12-STICKY...up to 5=STICK_PALINDROME

    :param labels: A tensor containing actual labels for each example
    :param predictions: A tensor containing the label predicted by the model
    :return: Tuple containing matrix and update op for use in eval_metric_ops
    '''
    with tf.variable_scope('get_confusion_matrix'):
        matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=6)
        matrix_sum = tf.Variable(tf.zeros(shape=(6,6), dtype=tf.int32),
                                 trainable=False,
                                 name='confusion_matrix',
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES])

        # Update matrix_sum by adding matrix to it
        update = tf.assign_add(matrix_sum, matrix)

        # Return confusion matrix and update op
        return tf.convert_to_tensor(matrix_sum), update


def cnn_model_fn(features, labels, mode, params):
    """Builds a CNN model roughly like LeNet-5."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Zener images are 32x32 pixels, and have one color channel
    input_layer = tf.reshape(features['x'], [-1, 32, 32, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 1]
    # Output Tensor Shape: [batch_size, 32, 32, 6]
    # First conv layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[5, 5],
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, 6]
    # Output Tensor Shape: [batch_size, 16, 16, 6]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 16, 16, 6]
    # Output Tensor Shape: [batch_size, 16, 16, 16]
    conv2 = tf.layers.conv2d(  # Third conv layer
        inputs=pool1,
        filters=16,
        kernel_size=[5, 5],
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 16, 16, 16]
    # Output Tensor Shape: [batch_size, 8, 8, 16]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 8, 8, 16]
    # Output Tensor Shape: [batch_size, 5 * 5 * 16]
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    # Dense Layer
    # Densely connected layer with 256 neurons
    # Input Tensor Shape: [batch_size, 5 * 5 * 16]
    # Output Tensor Shape: [batch_size, 256]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense1, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Dense Layer
    # Densely connected layer with 40 neurons
    # Input Tensor Shape: [batch_size, 256]
    # Output Tensor Shape: [batch_size, 40]
    dense2 = tf.layers.dense(inputs=dropout, units=40, activation=tf.nn.relu)    
    
    # Logits layer
    # Input Tensor Shape: [batch_size, 256]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=dense2, units=5)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    labels = tf.cast(labels, tf.int64)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'))

    # check if l1 or l2 regularization is used
    if params['cost'] == 'cross-l1':
        l1_regularizer = tf.contrib.layers.l1_regularizer(
            scale=0.005, scope=None
        )
        weights = tf.trainable_variables()  # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        loss = loss + regularization_penalty  # this loss needs to be minimized
    elif params['cost'] == 'cross-l2':
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=0.005, scope=None
        )
        weights = tf.trainable_variables()  # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
        loss = loss + regularization_penalty  # this loss needs to be minimized

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(args):
    # check the value of cost parameter
    if args.cost not in modes:
        print('Invalid cost parameter:', args.cost)
        sys.exit(0)

    # call function based on mode
    input_data = load_data(args)

    # Args specify training
    if args.cost != 'ctest':
        # perform 5fold cross validation
        validation_cost_total = 0

        for iteration in range(5):
            # Clear previous k-fold run
            try:
                shutil.rmtree(args.model_file_name, ignore_errors=False, onerror=None)
            except:
                pass

            # Create the Estimator
            model_params = {'cost': args.cost}
            zener_classifier = tf.estimator.Estimator(
                model_fn=cnn_model_fn, model_dir=args.model_file_name, params=model_params)

            # Set up logging for predictions
            # Log the values in the "Softmax" tensor with label "probabilities"
            tensors_to_log = {"probabilities": "softmax_tensor"}
            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=50)

            # K-fold split
            train_data, test_data = k_fold_split(input_data, iteration, k=5)

            # Finish preprocess of data into numpy arrs for feeding to tf
            train_features, train_labels = numpyize_inputs(train_data)
            test_features, test_labels = numpyize_inputs(test_data)

            # training the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': train_features},
                y=train_labels,
                batch_size=args.batch_size,
                num_epochs=args.max_updates,
                shuffle=True)
            zener_classifier.train(
                input_fn=train_input_fn,
                steps=None,
                hooks=[logging_hook])

            # Testing the model
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': test_features},
                y=test_labels,
                num_epochs=10,
                shuffle=False)
            eval_results = zener_classifier.evaluate(input_fn=eval_input_fn)
            # add loss to total
            validation_cost_total += float(eval_results['loss'])
            print(eval_results)

        print('Average validation cost: ' + str(validation_cost_total / 5.0))
    # Testing existing model
    else:
        # Create the Estimator
        model_params = {'cost': args.cost}
        zener_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=args.model_file_name, params=model_params)

        # Finish preprocess of data into numpy arrs for feeding to tf
        features, labels = numpyize_inputs(input_data)

        # Evaluate the model
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            num_epochs=10,
            shuffle=False)
        eval_results = zener_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


"""CLARGS"""
parser = argparse.ArgumentParser(
    description='Train and test CNN to classify Zener shapes',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)


parser.add_argument(
    'cost',
    help='The type of loss function to use with the CNN [cross | cross-l1 | cross-l2]'
)
parser.add_argument(
    'network_description',
    help='Path to a file containing a description of the CNN architecture'
)
parser.add_argument(
    'epsilon',
    type=float,
    help='Epsilon error tolerance.'
)
parser.add_argument(
    'max_updates',
    type=int,
    help='Training steps/epochs.'
)
parser.add_argument(
    'class_letter',
    help='Specify the class letter [P, W, Q, S].'
)
parser.add_argument(
    'model_file_name',
    help='Filename to output trained model.'
)
parser.add_argument(
    'train_folder_name',
    help='Locating of training data.'
)

# Optional args
parser.add_argument(
    '--batch-size',
    default=100,
    type=int,
    help='Training steps/epochs.'
)
parser.add_argument(
    '--learning-rate',
    default=0.0001,
    type=float,
    help='Training steps/epochs.'
)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

