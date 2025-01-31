#!/usr/bin/env python3

# create_kernel_matrices.py: Given a set of graphs, applies a number of
# grakel graph kernels and stores the resulting kernel matrices. Moreover, the
# script also stored the labels inside each set of matrices (under `y`)
# in order to make the output self-contained.


import argparse
import logging
import os
import random
import sys
import traceback

import grakel
from grakel import GraphletSampling
from grakel.kernels import ShortestPath, WeisfeilerLehman, VertexHistogram
from grakel.kernels import EdgeHistogram, RandomWalkLabeled, GraphHopper
import igraph as ig
import numpy as np

from timeit import time
from tqdm import tqdm

from grakel_util import *


def preprocess(graph):
    ''' Relabels nodes in graph with degree '''

    if 'label' not in graph.vs.attributes():
        graph.vs['label'] = graph.degree()

    return(graph)


def relabel_edges(graph, edge_labels):
    ''' Relabels edges using the concatenated node labels hashed to an
    integer'''
    for e in graph.es:
            
        a = graph.vs['label'][e.source]
        b = graph.vs['label'][e.target] 
        k = str(min(a,b)) + "." + str(max(a,b))
        e['label'] = edge_labels[k]

        # uniform label
        e['label'] = 1

    return(graph)


def gk_function(algorithm, graphs, par):
    """ Function to run the kernel on the param grid. Since different
    kernels have different numbers of parameters, this is necessary. """
    print("parameters", par)
    if algorithm == "SP_gkl":
        gk = ShortestPath(with_labels=True).fit_transform(graphs)
    elif algorithm == "EH_gkl":
        gk = EdgeHistogram().fit_transform(graphs)
    elif algorithm == "WL_gkl":
        gk = WeisfeilerLehman(n_iter=par).fit_transform(graphs)
    elif algorithm == "RW_gkl":
        lam, p = par
        gk = RandomWalkLabeled(lamda=lam, p=p).fit_transform(graphs)
    elif algorithm == "CSM_gkl":
        c, k = par
        # testing lambda function. c should reset for each iteration
        gk = SubgraphMatching(
                k=k, 
                ke=lambda p1, p2: ke_kernel(p1, p2, c), # inline lambda 
                kv=kv_kernel
                ).fit_transform(graphs)
    elif algorithm == "GL_gkl":
        k = par
        gk = GraphletSampling(k=k).fit_transform(graphs)
    elif algorithm == "GH_gkl":
        gk = GraphHopper().fit_transform(graphs)
    return(gk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'FILE', 
            type=str,
            help="Input file(s)"
        )

    parser.add_argument(
            "-a", "--algorithm",
            nargs="+",
            default=[],
            type=str,
            help="Indicates which algorithms to run"
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
    input_files = [f"{args.FILE}/{file}" for file in os.listdir(args.FILE) if file.endswith('.pickle')]

    logging.basicConfig(level=logging.INFO, format=None)

    logging.info('Loading graphs...')

    if args.timing:
        logging.info("Choosing at most 100 graphs at random for timing")

        random.seed(42)
        input_files = random.sample(input_files, 100)

    graph_attributes = {
        "SP_gkl": {"vertex": "label", "edge": []},
        "EH_gkl": {"vertex": [], "edge": "label"},
        "RW_gkl": {"vertex": "label", "edge": []},
        "WL_gkl": {"vertex": "label", "edge": []},
        "CSM_gkl": {"vertex": "both", "edge": "both"},
        "GH_gkl": {"vertex": [], "edge": []},  # TODO fix
        'GL_gkl': {'vertex': [], 'edge': []},
    }

    graphs = [ig.read(filename, format='picklez') for filename in tqdm(input_files, desc='File')]

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
        'FRANKENSTEIN': 500,
        'DHFR': 500,
    }
    if n_graphs[dataset] < len(graphs):
        rng = np.random.default_rng(403371)
        graphs = rng.choice(graphs, n_graphs[dataset], replace=False)

    graphs = [preprocess(graph) for graph in tqdm(graphs, desc="Preprocessing")]
    
    # check if the graph has edge labels, and if not, relabel edges
    if 'label' not in graphs[0].es.attributes():
        edge_labels = set_of_edge_labels(graphs)
        graphs = [relabel_edges(graph, edge_labels) for graph in graphs]
    
    # convert to grakel format
    graphs, y = igraph_to_grakel(graphs, attr=graph_attributes[args.algorithm[0]])

    param_grid = {
        "SP_gkl": [1],
        "GL_gkl": [3, 4, 5],
        "WL_gkl": [1, 2, 3, 4, 5, 6, 7],  # 0 returns an error
        "RW_gkl":  [(l, p) for l in [0.001] for p in [2, 3, 4, 5, 6, 7, None]],
        "CSM_gkl": [(c, k) for c in [0.1, 0.5, 1.0] for k in [3, 4, 5]],
    }

    algorithms = {
        "SP_gkl": "Notused",  # legacy item, I need a value
        "EH_gkl": "Notused",  # legacy item, I need a value
        "WL_gkl": "Notused",  # legacy item, I need a value
        "RW_gkl": "Notused",  # legacy item, I need a value
        "CSM_gkl": "Notused",  # legacy item, I need a value
        "MP_gkl": "Not used",
        "GH_gkl": "Not used",
        "GL_gkl": "Not used",
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
        #f = algorithm[algorithm]

        if algorithm in param_grid.keys():
            try:
                matrices = {
                        str(param): gk_function(
                            algorithm=algorithm, 
                            graphs=graphs, 
                            par=param) 
                        for param in param_grid[algorithm]
                        }
                print([matrices[a].shape for a in matrices])
            except NotImplementedError:
                logging.warning(f'''Caught exception for {algorithm};
                continuing with the next algorithm and its corresponding
                parameter grid.''')

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
            K = gk_function(algorithm=algorithm, graphs=graphs, par=None)
            print(K.shape) 
            # We only save the matrix if we are not in timing mode; see
            # above for the rationale.
            if not args.timing:
                np.savez(filename, K=K, y=y)

        stop_time = time.process_time()

        # We overwrite this *all* the time because the information can
        # always be replaced easily.
        with open(os.path.join(args.output, f'Time_{algorithm}.txt'), 'w') as f:
            print(stop_time - start_time, file=f)
