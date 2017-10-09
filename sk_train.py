"""
Train an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import argparse
import os

import numpy as np
from PIL import Image


def poly_kernel(x, x_i, p=4, c=1):
    """
    :param x_t: training input
    :param x_i: input vector
    :returns type int:
    """
    x_t = np.transpose(x)
    return (np.dot(x_t, x_i) + c)**p


def calculate_x_prime(x):
    x_prime = x
    lam = 0
    return x_prime, lam


def sk_algorithm(x):
    # x is the set of input vectors
    # x prime is scaled input vectors with lambda
    x_prime, lam = calculate_x_prime(x)

    error = 0.1  # placeholder
    num_updates = 0  # placeholder
    epsilon = 0.01  # from input
    max_updates = 1000  # from input
    
    while error < epsilon or num_updates is max_updates:
        # TODO add weight update
        continue

    # TODO feed vector input here instead of iterating through it
    g = 0
    for i in xrange(len(x)):
        y = 0
        if x[i] in x_prime:
            y = 0
        alpha = 1
        A = 0  # placeholder
        B = 0  # placeholder
        g += alpha * y * poly_kernel(x[i], x_prime[i]) + (B - A) / 2


def rep_data(img_path):
    """
    The contents of this image as a sequence object containing pixel values. The sequence object is flattened, so that values for line one follow directly after the values of line zero, and so on.

    Note that the sequence object returned by this method is an internal PIL data type, which only supports certain sequence operations. To convert it to an ordinary sequence (e.g. for printing), use list(im.getdata()).

    :param img_path: the path to image file
    :returns numpy arrays: A vector representation of the image.
    """
    img = Image.open(img_path)
    arr = np.array(list(img.getdata()))

    return arr


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
    description='One-off Bioinformatics Analytics Scripts:\n\
    Calcs:\n\
        1) Calculate absolute genus distribution counts\n\
        2) Calculate percent genus distribution counts + genus-counts details',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Add CLARGS
parser.add_argument(
    'epsilon',
    help='Epsilon error tolerance.'
)
parser.add_argument(
    'max_updates',
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

    # For testing image -> numpy arr
    img_folder = 'zener_shapes'

    img_path = os.path.join(img_folder, 'S.jpg')
    img_v = rep_data(img_path)

