import argparse


def print_str(s):
    print(f'\n ~~ {s} ~~ \n')


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_train_samples', type=int)
    parser.add_argument('n_test_samples', type=int)
    return parser.parse_args()
