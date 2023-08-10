#!/usr/bin/env python3

import argparse
import os

import numpy as np
from sklearn.model_selection import ParameterGrid


def check_distance_matrix(distances, log=False):
    # Check distance matrix, TODO remove after integration in main repo
    n_errors = np.count_nonzero(np.isnan(distances) | np.isinf(distances))
    if n_errors > 0:
        if log: print(f'Warning: {n_errors} NaNs/infs in distance matrix.')
        if (np.isnan(distances) | np.isinf(distances)).all():
            return np.ones_like(distances)
    distances = np.nan_to_num(distances, nan=np.nanmax(distances))

    # Ensure that the distance matrix is non-negative
    if np.min(distances) < 0:
        if log: print(f'Warning: negative values in distance matrix.')
        distances -= np.min(distances)
    return distances


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
        'gamma': np.logspace(-10, 10, 21),
        'epsilon': np.array([0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        # 'epsilon': np.array([0.002, 0.004, 0.006, 0.008, 0.01, 0.06, 0.08, 0.1]),
    }
    grid = ParameterGrid(param_grid)

    matrices = dict()
    for algo in args.algorithm:
        y = np.loadtxt(f"{args.DATA}/labels.csv")

        if args.kernel == 'rbf':
            def f(gamma, epsilon):
                distances = np.loadtxt(f"{args.DATA}/{algo}-{epsilon}-approx.csv")
                distances = check_distance_matrix(distances, log=False)
                return np.exp(-gamma * distances)
        else:
            raise ValueError(f"The given kernel ´{args.kernel}´ is not supported.")

        matrices = {
            '#'.join(map(str, param.values())): f(gamma=param['gamma'], epsilon=param['epsilon'])
            for param in grid
        }
        matrices['y'] = y

        os.makedirs(f"{args.output}", exist_ok=True)
        np.savez(f"{args.output}/{algo}.npz", **matrices)
