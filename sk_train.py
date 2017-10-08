"""
Train an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""

import sys
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
        sk_training_step()

    g = 0
    for i in range(len(x)):
        y = 0
        if x[i] in x_prime:
            y = 0
        alpha = 1
        A = 0  # placeholder
        B = 0  # placeholder
        g += alpha * y * poly_kernel(x[i], x_prime[i]) + (B - A) / 2


def sk_training_step():
    # from slide
    # S-K Algorithm - Kernel Version
    pass


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
    for (x, y), value in np.ndenumerate(arr):
        if value == 255:
            white_pixels.add((x,y))
        else:
            nonwhite_pixels.add((x, y))

    return white_pixels, nonwhite_pixels


def main():
    num_args = len(sys.argv)
    if num_args != 6:
        print 'INCORRECT PARAMETERS'
        print sys.argv
        return

    epsilon = sys.argv[1]
    max_updates = sys.argv[2]
    class_letter = sys.argv[3]
    folder_name = sys.argv[4]
    num_examples = sys.argv[5]

    # check if folder_name has valid images
    # check if num_examples is a positive int


if __name__ == '__main__':
    main()

    # For testing image -> numpy arr
    img_folder = 'zener_shapes'

    img_path = os.path.join(img_folder, 'S.jpg')
    img_v = rep_data(img_path)

