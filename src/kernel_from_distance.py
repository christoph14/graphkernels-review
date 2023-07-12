#!/usr/bin/env python3

import argparse
import os

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DATA', type=str, help='Input file')
    parser.add_argument(
        '-a', '--algorithm',
        nargs='+',
        default=[],
        type=str,
        help='Indicates which algorithms to run'
    )
    parser.add_argument(
        '-k', '--kernel',
        type=str,
        default='rbf',
        help='The kernel type to be used'
    )
    parser.add_argument(
        '-g', '--gamma',
        type=float,
        default=0.1,
        help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’'
    )
    parser.add_argument(
        '-f', '--force', action='store_true',
        default=False,
        help='If specified, overwrites data'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='Output directory'
    )

    args = parser.parse_args()

    param_grid = {
        'rbf': [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1],  # $\gamma$ = gamma
    }

    matrices = dict()
    for algo in args.algorithm:
        distances = np.loadtxt(f"{args.DATA}/{algo}.csv")
        y = np.loadtxt(f"{args.DATA}/labels.csv")

        if args.kernel == 'rbf':
            f = lambda gamma: np.exp(-gamma * distances)
        else:
            raise ValueError(f"The given kernel ´{args.kernel}´ is not supported.")

        matrices = {
            str(param): f(gamma=param)
            for param in param_grid[args.kernel]
        }
        matrices['y'] = y

        os.makedirs(f"{args.output}", exist_ok=True)
        np.savez(f"{args.output}/{algo}.npz", **matrices)
