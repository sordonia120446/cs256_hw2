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
from PIL import Image

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
    # Input Tensor Shape: [batch_size, 28, 28, 6]
    # Output Tensor Shape: [batch_size, 14, 14, 6]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 6]
    # Output Tensor Shape: [batch_size, 10, 10, 16]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[5, 5],
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 10, 10, 16]
    # Output Tensor Shape: [batch_size, 5, 5, 16]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # Convolutional Layer #3
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 5, 5, 16]
    # Output Tensor Shape: [batch_size, 1, 1, 120]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=120,
        kernel_size=[5, 5],
        activation=tf.nn.relu
    )

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 1, 1, 120]
    # Output Tensor Shape: [batch_size, 120]
    conv3_flat = tf.reshape(conv3, [-1, 120])

    # Dense Layer
    # Densely connected layer with 256 neurons
    # Input Tensor Shape: [batch_size, 120]
    # Output Tensor Shape: [batch_size, 84]
    dense = tf.layers.dense(inputs=conv3_flat, units=84, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 256]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=dense, units=5)

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
    # call function based on mode
    train_data = load_data(args)

    features = np.asarray([d['x'] for d in train_data], dtype=np.float32)
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

