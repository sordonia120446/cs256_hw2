import argparse




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