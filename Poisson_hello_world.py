import numpy as np
import argparse

def gen_poisson_val(l, n):
    s = np.random.poisson(l, n)
    print('Sample Set :', s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lambda_val', type=float, help='Lambda value')
    parser.add_argument('-n', '--Events', type=int, help='Count of Events')
    args = parser.parse_args()
    gen_poisson_val(args.lambda_val, args.Events)

if __name__ == '__main__':
    main()