"""
Utils module for handling data.

:author Sam O
"""
import glob
import os

import numpy as np
from PIL import Image


shapes = ['O', 'P', 'Q', 'S', 'W']

# Map letter to index in one hot vector
letter_to_ind = {
    shapes[0]: 0,
    shapes[1]: 1,
    shapes[2]: 2,
    shapes[3]: 3,
    shapes[4]: 4,    
}


def init_data(args, as_PIL=False):
    """
    Initialize the preliminaries for S-K algo learning of SVM

    :param args: the CLARGS from user input
    :returns type dict: The dict of X's, I's, Y's (all +/-'s)
    """

    img_dir = os.path.join(args.train_folder_name, '*.png')

    X_plus = []
    X_minus = []
    Y_plus = []
    Y_minus = []
    I_plus = []
    I_minus = []

    for img_path in glob.glob(img_dir):
        f_name = os.path.splitext(os.path.basename(img_path))
        ind, letter = f_name[0].split('_')

        if letter.upper() == args.class_letter.upper():
            X_plus.append(rep_data(img_path, as_PIL))
            # Y_plus.append(ord(letter.upper()))
            I_plus.append(ind)
            Y_plus.append(letter_to_ind[letter.upper()])
        else:
            X_minus.append(rep_data(img_path, as_PIL))
            # Y_minus.append(ord(letter.upper()))
            I_minus.append(ind)
            Y_minus.append(letter_to_ind[letter.upper()])            

    if len(X_plus) < 1 or len(X_minus) < 1:
        raise Exception('NO DATA')

    if len(X_plus) != len(I_plus) or len(X_minus) != len(I_minus):
        raise Exception('[ERROR] Init filter is not working')

    # X_plus, X_minus = scale_inputs(X_plus, X_minus)

    ret = {
        'X_plus': X_plus,
        'X_minus': X_minus,
        'Y_plus': Y_plus,
        'Y_minus': Y_minus,
        'I_plus': I_plus,
        'I_minus': I_minus
    }

    print('Data inputs initialized')

    return ret  # Vectors in X by class and index


def rep_data(img_path, as_PIL=False):
    """
    The contents of this image as a sequence object containing pixel values. The sequence object is flattened, so that values for line one follow directly after the values of line zero, and so on.

    Note that the sequence object returned by this method is an internal PIL data type, which only supports certain sequence operations. To convert it to an ordinary sequence (e.g. for printing), use list(im.getdata()).

    :param img_path: the path to image file
    :returns numpy arrays: A vector representation of the image.
    """
    img = Image.open(img_path)
    if as_PIL:
        return img

    arr = np.array(list(img.getdata()), int)

    # arr = (arr/255 + 1) % 2
    #  black is 1, 0 is white

    for i, element in np.ndenumerate(arr):
        if element == 0:
            arr[i] = -1
        else:
            arr[i] = 1

    return arr
