"""
Train an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import argparse
import glob
import math
import os

import numpy as np
from PIL import Image


############################################################
#Calculations
############################################################


def poly_kernel(x, x_i, p=4, c=1):
    """
    :param x_t: training input
    :param x_i: input vector
    :returns type int:
    """
    x_t = np.transpose(x)
    return (np.dot(x_t, x_i) + c)**p


def calc_lambda(X_plus, X_minus):
    """
    Calculate scaling factor (lambda) of convex hull.

    :param X_plus: the list of numpy vectors that are positive training examples
    :param X_minus: the list of numpy vectors that are negative training examples
    :returns type <float>:
    """

    m_plus = np.array([])  # positive centroid
    m_minus = np.array([])  # negative centroid

    # calculate m_plus and m_minus
    for i in xrange(25 * 25):
        m_plus_i = 0
        for X_plus_i in X_plus:
            m_plus_i += X_plus_i

        m_minus_i = 0
        for X_minus_i in X_minus:
            m_minus_i += X_minus_i

        m_plus_i /= len(X_plus)
        m_minus_i /= len(X_minus)
        m_plus = m_plus.append(m_plus, m_plus_i)
        m_minus = m_minus.append(m_minus, m_minus_i)

    # calculate r from m_plus and m_minus
    r = np.linalg.norm(m_plus - m_minus)

    # calculate r_plus
    r_pluses = []
    for X_plus_i in X_plus:
        r_pluses.append(np.linalg.norm(X_plus_i - m_plus))
    r_plus = max(r_pluses)

    # calculate r_minus
    r_minuses = []
    for X_minus_i in X_minus:
        r_minuses.append(np.linalg.norm(X_minus_i - m_minus))
    r_minus = max(r_minuses)

    return 0.5 * r / (r_plus + r_minus)  # lambda <= r / (r+ + r-)


def calc_mi(x_k, p):
    """
    Calculate the m_i to find the one closest to being within epsilon
    of the correct side of the hyperplane.  Important for stop condition.

    This is for the positive examples I_plus.

    :param x_k: the k-th elem of x (pos ex)
    :param p: params dict of alphas & letters
    :returns type float: the m_i value
    """

    D_i = poly_kernel(p['x_i'], x_k)
    E_i = poly_kernel(p['x_j'], x_k)

    m_i_num = float(D_i - E_i + p['B'] - p['C'])
    try:
        m_i_denom = math.sqrt(p['A'] + p['B'] -2*p['C'])
    except ValueError:
        raise Exception('Check the stop condition denom for m_i')

    return m_i_num/m_i_denom


def calc_mj(x_k, p):
    """
    Calculate the m_j to find the one closest to being within epsilon
    of the correct side of the hyperplane.  Important for stop condition.

    This is for the negative examples I_minus.

    :param x_k: the k-th elem of x (neg ex)
    :param p: params dict of alphas & letters
    :returns type float: the m_i value
    """

    D_i = poly_kernel(p['x_i'], x_k)
    E_i = poly_kernel(p['x_j'], x_k)

    m_i_num = float(-D_i + E_i + p['A'] - p['C'])
    try:
        m_i_denom = math.sqrt(p['A'] + p['B'] -2*p['C'])
    except ValueError:
        raise Exception('Check the stop condition denom for m_i')

    return m_i_num/m_i_denom


############################################################
#Preliminaries
############################################################


def init_data(args):
    """
    Initialize the preliminaries for S-K algo learning of SVM

    :param args: the CLARGS from user input
    :returns type dict: The dict of X's, I's, Y's (all +/-'s)
    """

    img_dir = os.path.join(args.train_folder_name, '*.png')

    X_plus = []
    X_minus = []
    I_plus = []
    I_minus = []

    for img_path in glob.glob(img_dir):
        f_name = os.path.splitext(os.path.basename(img_path))
        ind, letter = f_name[0].split('_')

        if letter.upper() == args.class_letter.upper():
            X_plus.append(rep_data(img_path))
            I_plus.append(ind)
        else:
            X_minus.append(rep_data(img_path))
            I_minus.append(ind)

    if len(X_plus) < 1:
        raise Exception('NO DATA')

    if len(X_plus) != len(I_plus) or len(X_minus) != len(I_minus):
        raise Exception('[ERROR] Init filter is not working')

    ret = {
        'X_plus': X_plus,
        'X_minus': X_minus,
        'I_plus': I_plus,
        'I_minus': I_minus
    }

    print 'Data inputs initialized'

    return ret


############################################################
#S-K Algo Core Logic
############################################################


def sk_init(data, i=0):
    """
    Step 1: Initialization of s-k algo for kernel version.
    Defines alpha_i & alpha_j, along with A~E.

    :param input_data: the dict input data for +/-'s.
    :returns type dict: alphas & letters
    """
    ret = {}

    # Define alpha
    alpha_i = np.zeros(len(data['X_plus']), dtype=np.int)
    alpha_j = np.zeros(len(data['X_minus']), dtype=np.int)

    # Positive ex
    x_i1 = data['X_plus'][i]
    i1 = data['I_plus'][i]

    # Negative ex
    x_j1 = data['X_minus'][i]
    j1 = data['I_minus'][i]

    # Set alpha's to one for support vector "guesses"
    alpha_i[i] = 1
    alpha_j[i] = 1

    # Define A~C
    A = poly_kernel(x_i1, x_i1)
    B = poly_kernel(x_j1, x_j1)
    C = poly_kernel(x_i1, x_j1)

    # Add to dict
    ret = {
        'x_i': x_i1,
        'i': i1,
        'x_j': x_j1,
        'j': j1,
        'alpha_i': alpha_i,
        'alpha_j': alpha_j,
        'A': A,
        'B': B,
        'C': C
    }

    print ret

    return ret


def should_stop(d, p, epsilon):
    """
    Determine whether to stop or continue.

    :param d: the input data dict of X's & I's
    :param p: dict of alphas & letters
    :epsilon: error tolerance defined in CLARGS
    :returns type bool: True if stop condition met; otherwise, False
    """

    """
    Get min vals
    NB: m_is & m_js are indexed by their m-values and store
    their corresponding indices as well as x_vector.
    """
    m_is = {}
    for pos_ex, pos_ind in zip(d['X_plus'], d['I_plus']):
        m_is[(calc_mi(pos_ex, p))] = {
            'ind': pos_ind,
            'x': pos_ex
        }

    m_js = {}
    for neg_ex, neg_ind in zip(d['X_minus'], d['I_minus']):
        m_js[calc_mj(neg_ex, p)] = {
            'ind': neg_ind,
            'x': neg_ex
        }

    m_i_min = min(m_is.keys())
    m_j_min = min(m_js.keys())

    # Define x_t and its corresponding metadata
    if m_is[m_i_min] < m_js[m_j_min]:
        ret = {
            'category': 'pos',  # positive category
            'm_t': m_i_min,  # see calc_mi
            't_ind': m_is[m_i_min]['ind'],  # index val of min
            'x_t': m_is[m_i_min]['x']  # support vector
        }
    else:
        ret = {
            'category': 'neg',  # negative category
            'm_t': m_j_min,  # see calc_mi
            't': m_js[m_j_min],  # index val of min
            'x_t': m_js[m_j_min]['x']  # support vector
        }

    # Calc deltas
    err_msg = 'Attempted negative sqrt for {} ex stop condition check'
    try:
        m_delta = math.sqrt(p['A'] + p['B'] - 2*p['C']) - ret['m_t']
    except ValueError:
        raise Exception(err_msg.format(ret))

    # Compare to epsilon
    if m_delta < epsilon:
        print 'Stop condition met!:  {}'.format(m_delta)
        return True, ret

    return False, ret


def adapt(d, p, x_t):
    """
    :param d: input data dict of X's & I's from sample space
    :param p: params dict of alphas & letters
    :returns type dict: new dict of alphs & letters params
    """

    D_i = poly_kernel(p['x_i'], x_t['x_t'])
    E_i = poly_kernel(p['x_j'], x_t['x_t'])

    if x_t['category'] == 'pos':
        # logic for positive ex
        q = 1
    elif x_t['category'] == 'neg':
        # logic for negative ex
        q = 1

    return p


def sk_algorithm(input_data, args):
    """
    Find support vectors of scaled convex hulls for X+ & X-.

    :param x: the input numpy vector from an img
    :args: the CLARGS from user input
    :returns type dict: final dict of alphas and letters
    """
    # TODO implement scaling logic

    # Initialization
    params = sk_init(input_data)

    for i in xrange(int(args.max_updates)):

        # Print alphas & letters on every 1000th step
        if i % 1000 == 0:
            print '\nOn training step {}'.format(i)
            print params

        # Check for stop condition
        is_done, x_t = should_stop(input_data, params, args.epsilon)
        if is_done:
            return params

        params = adapt(input_data, params, x_t)

    print '\nTrained for {}'.format(args.max_updates)

    return params


############################################################
#Reading in the data
############################################################


def rep_data(img_path):
    """
    The contents of this image as a sequence object containing pixel values. The sequence object is flattened, so that values for line one follow directly after the values of line zero, and so on.

    Note that the sequence object returned by this method is an internal PIL data type, which only supports certain sequence operations. To convert it to an ordinary sequence (e.g. for printing), use list(im.getdata()).

    :param img_path: the path to image file
    :returns numpy arrays: A vector representation of the image.
    """
    img = Image.open(img_path)
    arr = np.array(list(img.getdata()), int)

    return arr/255 # normalize to 1's for white; 0's otherwise


def classify_pixels(img_arr):
    """
    Pixel value 255 corresponds to white.

    :param img_arr type <numpy arr> 
    :returns: two numpy vectors of white & non-white pixels
    """

    white_pixels = []
    nonwhite_pixels = []
    for (x, y), value in np.ndenumerate(img_arr):
        if value == 255:
            white_pixels.add((x,y))
        else:
            nonwhite_pixels.add((x, y))

    return white_pixels, nonwhite_pixels


############################################################
#CLARGS
############################################################
parser = argparse.ArgumentParser(
    description='S-K Learning algo for SVM',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Add CLARGS
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


if __name__ == '__main__':
    args = parser.parse_args()

    # Init
    input_data = init_data(args)  # dict

    # Run algo
    params = sk_algorithm(input_data, args)

    print '\n Final output:  '

    for k, v in params.items():
        print '{}: {}'.format(k, v)

