"""
CNN classifier of Zener cards.

:authors Jason, Nick, Sam
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import tensorflow as tf

from utils import init_data


tf.logging.set_verbosity(tf.logging.INFO)


def load_data(args):
    """
    Iterate through each file in data_folder and construct
    the input data features (40-len DNA).

    :param data_folder: path to folder of data
    :param exclude: kth file to exclude in cross-validation
    :returns type list: list of dicts for DNA-arr & label
    """
    
    data = init_data(args)
    x = data['X_plus'] + data['X_minus']
    y = data['Y_plus'] + data['Y_minus']

    ret = []

    for img, label in zip(x, y):
        ret.append({
            'x': img,
            'y': label
        })

    return ret


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


def cnn_model_fn(features, labels, mode):
        ''' Builds a CNN model roughly like LeNet-5'''
        input = tf.reshape(features['x'], [-1, 32, 32, 1])  # Input layer 32x32, 1 channel

        # First conv layer
        c1 = tf.layers.conv2d(
            inputs=input,
            filters=6,
            kernel_size=[5, 5],
            activation=tf.nn.relu
        )

        # Second pooling layer
        s2 = tf.layers.max_pooling2d(  # Outputs 6 14x14 feature maps
            inputs=c1,
            pool_size=[2, 2],
            strides=2
        )

        # Third conv layer
        c3 = tf.layers.conv2d(  # Third conv layer
            inputs=s2,
            filters=16,
            kernel_size=[5, 5],
            activation=tf.nn.relu
        )

        # Fourth pooling layer
        s4 = tf.layers.max_pooling2d(
            inputs=c3,
            pool_size=[2, 2],
            strides=2
        )

        # Fifth conv layer
        c5 = tf.layers.conv2d( # Outputs 6 5x5 feature maps
            inputs=s4,
            filters=120,
            kernel_size=[1, 1],
            padding='same',
            activation=tf.nn.relu
        )

        c5_flat = tf.reshape(c5, [-1, 120]) # Flatten output to connect with dense layer

        # Sixth dense layer
        f6 = tf.layers.dense(inputs=c5_flat, units=84, activation=tf.nn.relu)

        # Output layer (binary classifier)
        logits = 



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
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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


def train_mode(data, model_name):
    # Init estimator
    sticky_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_name
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=1000
    )

    features = np.asarray([d['x'] for d in data])
    labels = np.asarray([d['y'] for d in data], dtype=np.float32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': features},
        y=labels,
        batch_size=MINIBATCH_SIZE,
        num_epochs=None,
        shuffle=True
    )

    start = time.perf_counter()
    sticky_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )
    train_time = time.perf_counter() - start
    print(f'Trained for {train_time:.3f} seconds')


def test_mode(data, model_name):
    # Init estimator
    sticky_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_name
    )
    features = np.asarray([d['x'] for d in data])
    labels = np.asarray([d['y'] for d in data], dtype=np.float32)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        y=labels,
        num_epochs=1,
        shuffle=False
    )

    start = time.perf_counter()
    eval_results = sticky_classifier.evaluate(input_fn=eval_input_fn)
    test_time = time.perf_counter() - start
    print(f'Tested for {test_time:.3f} seconds')
    return eval_results['accuracy']


def main(args):
    # call function based on mode
    train_data = load_data(args)

    features = np.asarray([d['x'] for d in train_data])
    labels = np.asarray([d['y'] for d in train_data], dtype=np.float32)

    # Create the Estimator
    zener_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=args.model_file_name)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': features},
        y=labels,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)
    zener_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
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

