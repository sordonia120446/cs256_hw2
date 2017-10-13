"""
Test an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import os
import argparse
import glob
import pickle

from sk_train import poly_kernel, rep_data


testing_class = 'W'

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

    #################################################################
    # Why is this here???

    # Load the training data
    training_data_path = os.path.join(os.getcwd(), args.train_folder_data)

    if not os.path.exists(training_data_path):
        raise Exception('Training data folder not found')

    training_data = []
    for f_name in glob.glob(os.path.join(training_data_path, '*.png')):
        x_train = rep_data(f_name)
        training_data.append(x_train)
        # to score... jason chee

    if not training_data:
        raise Exception('NO TRAINING DATA')
    #################################################################

    # Load the test data
    testing_data_path = os.path.join(os.getcwd(), args.test_folder_data)

    if not os.path.exists(testing_data_path):
        raise Exception('Testing data folder not found')

    testing_data = []
    I_plus = []
    for f_name in glob.glob(os.path.join(testing_data_path, '*.png')):
        x_test = rep_data(f_name)
        testing_data.append(x_test)
        # to score... jason chee
        f_name = os.path.splitext(os.path.basename(f_name))

        ind, letter = f_name[0].split('_')
        if letter.upper() == testing_class:
            I_plus.append(ind)

    if not testing_data:
        raise Exception('NO TESTING DATA')
    print I_plus
    return model, testing_data, I_plus


def test_SVM(p, x):
    """
    Computes g(x) from the lecture notes.

    :param p: Params for the trained model (alphas, lambda, A~C)
    :training_data: Training set which maximal separator is based upon
    :param x: test vector
    :returns: True if g >= 0; otherwise, False
    """
    
    # Compare  to positive ex's alpha_i = p['alpha_i']
    alpha_i = p['alpha_i']
    X_plus = p['X_plus']

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
    print g
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

    # Read inputs
    model, testing_data, I_plus = load_data(args)

    # Compare
    results = {'Correct': 0,
               'False Positive': 0,
               'False Negative': 0}

    for i, input_test in enumerate(testing_data):
        g = test_SVM(model, input_test)
        if g:
            if str(i) in I_plus:
                print str(i+1) + ' Correct'
                results['Correct'] += 1
            else:
                print str(i+1) + ' False Positive'
                results['False Positive'] += 1
        else:
            if str(i) not in I_plus:
                print str(i+1) + ' Correct'
                results['Correct'] += 1
            else:
                print str(i) + ' False Negative'
                results['False Negative'] += 1

    # for debugging --
    print 'Num Correct: ' + str(results['Correct'])
    print 'num positives: ' + str(len(I_plus))
    print 'out of: ' + str(len(testing_data))
    # ----------------

    for result in results:
        results[result] /= float(len(testing_data))



    print 'Fraction Correct: ' + str(results['Correct'])
    print 'Fraction False Positive: ' + str(results['False Positive'])
    print 'Fraction False Negative: ' + str(results['False Negative'])
    # print 'Tests complete'

