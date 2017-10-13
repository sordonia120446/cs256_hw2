"""
Test an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import os
import argparse
import glob
import pickle


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
    for f in glob.glob(os.path.join(training_data_path, '*.png')):
        training_data.append(f)

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


def test_SVM(model, training_data, testing_data):
    pass

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

    load_data(args)