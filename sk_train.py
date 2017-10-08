"""
Train an SVM on a polynomial-kernel transformed dataset.
:authors Jason, Nick, Sam
"""


import sys
import numpy as np


def poly_kernel(x, x_i, p=4, c=1):
    """
    :param x_t: training input
    :param x_i: input vector
    :returns type int:
    """
    x_t = np.transpose(x)
    return (np.dot(x_t, x_i) + c)**p


def calculate_x_prime(x):
    x_prime = []
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


if __name__ == "__main__":
    main()