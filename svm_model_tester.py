"""
Test an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import os
import argparse
import glob
import pickle

from sk_train import poly_kernel, rep_data


def load_data(args):
    '''
    Loads the trained model, training data and testing data, if they exist.

    :param args: Command line arguments
    '''

    # Load trained SVM model
    model_file_name = args.model_file_name

    file_ext = os.path.splitext(model_file_name)[1]
    if file_ext != '.txt':
        raise Exception('MODEL FILE IS NOT OF THE CORRECT FORMAT')

    try:
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)
    except IOError:
        raise Exception('CAN\'T FIND MODEL FILE')

    # Load the training data
    training_data_path = os.path.join(os.getcwd(), args.train_folder_data)

    if not os.path.exists(training_data_path):
        raise Exception('Training data folder not found')

    training_data = []
    for f_name in glob.glob(os.path.join(training_data_path, '*.png')):
        x_test = rep_data(f_name)
        print type(x_test)
        training_data.append(x_test)

    if not training_data:
        raise Exception('NO TRAINING DATA')

    # Load the test data
    testing_data_path = os.path.join(os.getcwd(), args.test_folder_data)

    if not os.path.exists(testing_data_path):
        raise Exception('Testing data folder not found')

    testing_data = []
    for f in glob.glob(os.path.join(testing_data_path, '*.png')):
        testing_data.append(f)

    if not testing_data:
        raise Exception('NO TESTING DATA')

    return model, training_data, testing_data


def test_SVM(p, training_data, x):
    """
    Computes g(x) from the lecture notes.

    :param p: Params for the trained model (alphas, lambda, A~C)
    :training_data: Training set which maximal separator is based upon
    :param x: test vector
    :returns: True if g >= 0; otherwise, False
    """
    
    # Compare  to positive ex's
    alpha_i = p['alpha_i']
    X_plus = p['X_plus']
    print x

    sum_plus = sum(
        [ai*poly_kernel(xi, x) for ai, xi in zip(alpha_i, X_plus)]
    )

    # Compare to negative ex's
    alpha_j = p['alpha_j']
    X_minus = p['X_minus']
    sum_minus = sum(
        [-aj*poly_kernel(xj, x) for aj, xj in zip(alpha_j, X_minus)]
    )

    sum_total = sum_plus + sum_minus

    A = p['A']
    B = p['B']
    g = sum_total + 0.5*(B - A)

    return True if g >= 0 else False


# CLARGS
parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

parser.add_argument(
    'model_file_name',
    help='Path of the file containing the trained model.'
)

parser.add_argument(
    'train_folder_data',
    help='Path of the folder containing the training data.'
)

parser.add_argument(
    'test_folder_data',
    help='Path of the folder containing the testing data.'
)

if __name__ == '__main__':
    args = parser.parse_args()

    model, training_data, testing_data = load_data(args)
    for input_test in testing_data:
        print type(input_test)
        test_SVM(model, training_data, testing_data)

    print 'Tests complete'

