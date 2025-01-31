#!/usr/bin/env python3
#
# create_kernel_matrices.py: Given a set of graphs, applies a number of
# graph kernels and stores the resulting kernel matrices. Moreover, the
# script also stored the labels inside each set of matrices (under `y`)
# in order to make the output self-contained.

import argparse
import logging
import os
import random
import sys
import traceback

import graphkernels.kernels as gk
import igraph as ig
import numpy as np

from timeit import time
from tqdm import tqdm


def preprocess(graph):
    '''
    Performs pre-processing on a graph in order for the `graphkernels`
    package to work properly. At present, the following two steps will
    be performed:

    1. Deleting of edge labels and edge attributes
    2. Creating a degree-based vertex label for unlabelled graphs

    :param graph: Graph to operate on
    :return: Copy of pre-processed graph
    '''

    # Remove all edge attributes (including labels)
    for name in graph.es.attributes():
        del graph.es[name]

    # Check whether we need to create a vertex attribute
    if 'label' not in graph.vs.attributes():
        # This skips one step: we could also add a uniform label here
        # but then WL will need an additional step to work as expected
        graph.vs['label'] = graph.degree()

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str, help='Input file(s)')
    parser.add_argument(
        '-a', '--algorithm',
        nargs='+',
        default=[],
        type=str,
        help='Indicates which algorithms to run'
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
    parser.add_argument(
        '-t', '--timing', action='store_true',
        default=False,
        help='If specified, only stores timing information'
    )
    parser.add_argument(
        '--same_size',
        action='store_const', const=True, default=False,
        help='todo'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    logging.info('Loading graphs...')

    if args.timing:
        logging.info('Choosing at most 100 graphs at random for timing')

        random.seed(42)
        args.FILE = random.sample(args.FILE, 100)


    graphs = [
        ig.read(filename, format='picklez') for filename in
        tqdm(args.FILE, desc='File')
    ]

    if args.same_size:
        values, counts = np.unique([len(G.vs) for G in graphs], return_counts=True)
        n_nodes = values[np.argmax(counts)]
        graphs = [G for G in graphs if len(G.vs) == n_nodes]
        print(f"Use only graphs of size {n_nodes}")

    # Sample graphs
    dataset = args.output.split('/')[-1]
    n_graphs = {
        'BZR': 405,
        'MUTAG': 188,
        'PTC_MR': 344,
        'KKI': 83,
        'ENZYMES': 300,
        'PROTEINS': 200,
        'AIDS': 500,
    }
    if n_graphs[dataset] < len(graphs):
        rng = np.random.default_rng(403371)
        graphs = rng.choice(graphs, n_graphs[dataset], replace=False)

    graphs = [
        preprocess(graph) for graph in tqdm(graphs, desc='Preprocessing')
    ]

    y = np.array([g['label'] for g in graphs])

    algorithms = {
        # Histogram kernels (baselines)
        'VH': gk.CalculateVertexHistKernel,
        'EH': gk.CalculateEdgeHistKernel,
        # Other kernels
        'GL': gk.CalculateGraphletKernel,
        'SP': gk.CalculateShortestPathKernel,
        'RW': gk.CalculateExponentialRandomWalkKernel,
        'WL': gk.CalculateWLKernel,
        # 'VEH': gk.CalculateVertexEdgeHistKernel,
        # 'VVEH': gk.CalculateVertexVertexEdgeHistKernel,
    }

    param_grid = {
        'WL': [0, 1, 2, 3, 4, 5, 6, 7],   # $h$ = number of iterations
        'GL': [3, 4],                     # $k$ = size of graphlet
        'VVEH': 10.0 * np.arange(-2, 3),  # $l$ = regularisation term
    }

    # Remove algorithms that have not been specified by the user; this
    # makes it possible to run only a subset of all configurations.
    algorithms = {
        k: v for k, v in algorithms.items() if k in args.algorithm
    }

    os.makedirs(args.output, exist_ok=True)

    for algorithm in sorted(tqdm(algorithms.keys(), desc='Algorithm')):

        start_time = time.process_time()

        # Filename for the current algorithm. We create this beforehand
        # in order to check whether we would overwrite something.
        filename = os.path.join(args.output, f'{algorithm}.npz')

        if os.path.exists(filename):
            if not args.force and not args.timing:
                logging.info('Output path already exists. Skipping.')
                continue

        # Function to apply to the list of graphs in order to obtain
        # a kernel matrix.
        f = algorithms[algorithm]

        if algorithm in param_grid.keys():
            print("still going!")
            try:
                matrices = {
                    str(param): f(graphs, par=param)
                    for param in param_grid[algorithm]
                }
            except NotImplementedError:
                logging.warning(f'''
Caught exception for {algorithm}; continuing with the next algorithm and
its corresponding parameter grid.
                ''')

                traceback.print_exc()
                continue

            # Store the label vector of the graph data set along with
            # the set of matrices.
            matrices['y'] = y

            # We only save matrices if we are not in timing mode. In
            # some sense, the calculations will thus be lost, but we
            # should not account for the save time anyway.
            if not args.timing:
                np.savez(filename, **matrices)

        else:
            K = f(graphs)

            # We only save the matrix if we are not in timing mode; see
            # above for the rationale.
            if not args.timing:
                np.savez(filename, K=K, y=y)
        
        stop_time = time.process_time()

        # We overwrite this *all* the time because the information can
        # always be replaced easily.
        with open(os.path.join(args.output, f'Time_{algorithm}.txt'), 'w') as f:
            print(stop_time - start_time, file=f)
