"""
Train an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import argparse
import glob
import math
import os
import pickle

import numpy as np
from PIL import Image


############################################################
#Calculations
############################################################


def poly_kernel(x, x_i, p=4, c=1):
    """
    :param x_t: training input
    :param x_i: input vector
    :returns type int: The kernel output if successful; otherwise, 0
    """
    x_t = np.transpose(x)
    try:
        return (np.dot(x_t, x_i) + c)**p
    except ValueError:
        print 'Input: {}'.format(x_t.shape)
        print 'Input: {}'.format(x_i)
        raise Exception('Kernel fnc cannot dot vectors of diff dims')

    return 0


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
            m_plus_i += X_plus_i[i]

        m_minus_i = 0
        for X_minus_i in X_minus:
            m_minus_i += X_minus_i[i]

        m_plus = np.append(m_plus, m_plus_i)
        m_minus = np.append(m_minus, m_minus_i)

    m_plus /= float(len(X_plus))
    m_minus /= float(len(X_minus))

    # print 'm_plus: ' + str(m_plus) + ' len: ' + str(len(m_plus))
    # print 'm_minus: ' + str(m_minus) + ' len: ' + str(len(m_minus))

    # calculate r from m_plus and m_minus (Euclidean distance between centroids)
    r = np.linalg.norm(m_plus - m_minus)
    # print 'r: ' + str(r)

    # calculate r_plus (radius of positive convex hull)
    r_plus = 0
    for X_plus_i in X_plus:
        r_plus = max(r_plus, np.linalg.norm(X_plus_i - m_plus))

    # calculate r_minus (radius of negative convex hull)
    r_minus = 0
    for X_minus_i in X_minus:
        r_minus = max(r_minus, np.linalg.norm(X_minus_i - m_minus))

    lam = (0.5 * r) / (r_plus + r_minus)
    print 'lambda = {}'.format(lam)

    # return lambda and the centroids
    return lam, m_plus, m_minus


def scale_inputs(X_plus, X_minus):
    lam, m_plus, m_minus = calc_lambda(X_plus, X_minus)
    for i, x_i in enumerate(X_plus):
        X_plus[i] = lam * x_i + (1 - lam) * m_plus
    for j, x_j in enumerate(X_minus):
        X_minus[j] = lam * x_j + (1 - lam) * m_minus

    # print 'xplus' + str(X_plus)
    # print 'xminus' + str(X_minus)
    return X_plus, X_minus


def calc_mi(x_k, p, ind):
    """
    Calculate the m_i to find the one closest to being within epsilon
    of the correct side of the hyperplane.  Important for stop condition.

    This is for the positive examples I_plus.

    :param x_k: the k-th elem of x (pos ex)
    :param p: params dict of alphas & letters
    :returns type float: the m_i value
    """

    D_i = p['D'][ind]
    E_i = p['E'][ind]

    m_i_num = float(D_i - E_i + p['B'] - p['C'])
    try:
        m_i_denom = math.sqrt(p['A'] + p['B'] -2*p['C'])
    except ValueError:
        raise Exception('Check the stop condition denom for m_i')

    return m_i_num/m_i_denom


def calc_mj(x_k, p, ind):
    """
    Calculate the m_j to find the one closest to being within epsilon
    of the correct side of the hyperplane.  Important for stop condition.

    This is for the negative examples I_minus.

    :param x_k: the k-th elem of x (neg ex)
    :param p: params dict of alphas & letters
    :returns type float: the m_i value
    """

    #D_i = poly_kernel(p['x_i'], x_k)
    #E_i = poly_kernel(p['x_j'], x_k)
    D_i = p['D'][ind]
    E_i = p['E'][ind]

    m_i_num = float(-D_i + E_i + p['A'] - p['C'])
    try:
        m_i_denom = math.sqrt(p['A'] + p['B'] -2*p['C'])
    except ValueError:
        raise Exception('Check the stop condition denom for m_j')

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

    if len(X_plus) < 1 or len(X_minus) < 1:
        raise Exception('NO DATA')

    if len(X_plus) != len(I_plus) or len(X_minus) != len(I_minus):
        raise Exception('[ERROR] Init filter is not working')

    # TODO scaled to lambda
    X_plus, X_minus = scale_inputs(X_plus, X_minus)

    ret = {
        'X_plus': X_plus,
        'X_minus': X_minus,
        'I_plus': I_plus,
        'I_minus': I_minus
    }

    print 'Data inputs initialized'

    return ret  # Vectors in X by class and index


############################################################
#S-K Algo Core Logic
############################################################


def sk_init(data, i=0):
    """
    Step 1: Initialization of s-k algo for kernel version.
    Defines alpha_i & alpha_j, along with A~E.

    :param input_data: the dict input data for +/-'s.
    :returns type dict: pos_ex, neg_ex, alphas, & letters
    """
    ret = {}

    # Define alpha (alpha_i = pos weights, alpha_j = neg weights)
    alpha_i = np.zeros(len(data['X_plus']), dtype=np.int)
    alpha_j = np.zeros(len(data['X_minus']), dtype=np.int)

    # Positive ex (any vector in X+, default is index 0)
    x_i1 = data['X_plus'][i]
    i1 = data['I_plus'][i]

    # Negative ex (any vector in X-, default is index 0)
    x_j1 = data['X_minus'][i]
    j1 = data['I_minus'][i]

    # Set alpha's to one for support vector "guesses"
    alpha_i[i] = 1
    alpha_j[i] = 1

    # Define A~C
    A = poly_kernel(x_i1, x_i1)
    B = poly_kernel(x_j1, x_j1)
    C = poly_kernel(x_i1, x_j1)

    # Define D & E for all i in I, x_i in X
    D = {}
    E = {}
    for ind, x_i in zip(data['I_plus'], data['X_plus']):
        D[ind] = poly_kernel(x_i, x_i1)
        E[ind] = poly_kernel(x_i, x_j1)

    for ind, x_i in zip(data['I_minus'], data['X_minus']):
        D[ind] = poly_kernel(x_i, x_i1)
        E[ind] = poly_kernel(x_i, x_j1)

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
        'C': C,
        'D': D,
        'E': E
    }

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
        m_i = calc_mi(pos_ex, p, pos_ind)
        m_is[m_i] = {
            'ind': pos_ind,
            'x': pos_ex
        }

    m_js = {}
    for neg_ex, neg_ind in zip(d['X_minus'], d['I_minus']):
        m_j = calc_mj(neg_ex, p, neg_ind)
        m_js[m_j] = {
            'ind': neg_ind,
            'x': neg_ex
        }

    m_i_min = min(m_is.keys())
    m_j_min = min(m_js.keys())

    # Define x_t (vector closest to hyperplane) and its corresponding metadata
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
            't_ind': m_js[m_j_min]['ind'],  # index val of min
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
        print 'Stop condition met for tolerance: {}'.format(m_delta)
        return True, ret

    return False, ret


def adapt(d, p, x_t):
    """
    :param d: input data dict of X's & I's from sample space
    :param p: params dict of alphas & letters
    :returns type dict: new dict of alphs & letters params
    """

    A = p['A']
    B = p['B']
    C = p['C']
    D = p['D']
    E = p['E']

    t = x_t['t_ind'] # Index of vector closest to hyperplane

    try:
        D_t = D[t]
        E_t = E[t]
    except KeyError:
        raise Exception('FATAL ERROR! CHECK YOUR INPUT LOGIC!!')

    delta_i_t = lambda i, t: 1 if i == t else 0

    if x_t['category'] == 'pos':
        # logic for positive ex, i.e. if x_t is from positive examples
        q_num = float( A - D_t + E_t - C)
        q_denom = A + poly_kernel(x_t['x_t'], x_t['x_t']) - 2 * (D_t - E_t)
        q = q_num/q_denom

        # Adapt positive alphas (coefficients)
        old_alpha = p['alpha_i'] 
        new_alpha = np.zeros(old_alpha.shape)

        for i, a_i in enumerate(old_alpha):
            new_alpha[i] = (1 - q) * old_alpha[i] + q * delta_i_t(i, t)

        # Update alpha_i
        p['alpha_i'] = new_alpha

        # Update kernel functions
        p['A'] = A * (1 - q)**2 + 2 * (1 - q) * q * D_t + q**2 * poly_kernel(x_t['x_t'], x_t['x_t'])
        p['C'] = (1 - q) * C + q * E_t

        # Update D and add back to params dict
        for ind, D_i in D.items():
            D[ind] = (1 - q)*D_i + q*poly_kernel(x_t['x_t'], x_t['x_t'])

        p['D'] = D


    elif x_t['category'] == 'neg':
        # logic for negative ex, i.e. if x_t is from negative examples
        q_num = float(B - E_t + D_t - C)
        q_denom = B + poly_kernel(x_t['x_t'], x_t['x_t']) - 2 * (E_t - D_t)
        q = q_num/q_denom

        # Adapt positive alphas (coefficients)
        old_alpha = p['alpha_j'] 
        new_alpha = np.zeros(old_alpha.shape)

        for j, a_i in enumerate(old_alpha):
            new_alpha[j] = (1 - q) * old_alpha[j] + q * delta_i_t(j, t)

        # Update alpha_i
        p['alpha_j'] = new_alpha

        # Update kernel functions
        p['B'] = B * (1 - q)**2 + 2 * (1 - q) * q * E_t + q**2 * poly_kernel(x_t['x_t'], x_t['x_t'])
        p['C'] = (1 - q) * C + q * D_t

        # Update E
        for ind, E_i in E.items():
            E[ind] = (1 - q)*E_i + q*poly_kernel(x_t['x_t'], x_t['x_t'])

        p['E'] = E

    return p


def sk_algorithm(input_data, args):
    """
    Find support vectors of scaled convex hulls for X+ & X-.

    :param x: the input numpy vector from an img
    :args: the CLARGS from user input
    :returns type dict: final dict of alphas and letters
    """
    # TODO implement scaling logic (need to return lambda and centroids for serialization)

    # Initialization
    params = sk_init(input_data)

    for i in xrange(int(args.max_updates)): # If max num of updates reached before err < epsilon, stop

        # Print alphas & letters on every 1000th step
        if i % 1000 == 0:
            print '\nOn training step {}'.format(i)
            #print params

        # Check for stop condition
        is_done, x_t = should_stop(input_data, params, args.epsilon)
        if is_done:
            print 'Completed training at step {step}'.format(step=i)
            return params

        params = adapt(input_data, params, x_t)

    print '\nTrained for {}'.format(args.max_updates)

    return params


def serialize_model(params, input_data, filename):
    '''
    Serialize the model generated from training as a text file

    :param params: Dictionary containing trained class, centroids, lambda and weights
    :param filename: Name of file to save model in
    :returns type bool: True if write succeeds; otherwise, False
    '''

    model = params

    for k, v in input_data.items():
        model[k] = v

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        return True

    return False


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
    input_data = init_data(args)  # dict of input data

    # Run algo
    params = sk_algorithm(input_data, args)  # dict of model params

    # Write model to file
    if serialize_model(params, input_data, args.model_file_name):
        print 'Model saved to {}'.format(args.model_file_name)

    print '\n Final output:  '


