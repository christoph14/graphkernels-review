#!/usr/bin/env python3
#
# train.py: given a set of kernel matrices, which are assumed to belong
# to the *same* data set, fits and trains a classifier, while reporting
# the best results.

import argparse
import logging
import os

import numpy as np

from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score

from tqdm import tqdm


def normalize(matrix):
    '''
    Normalizes a kernel matrix by dividing through the square root
    product of the corresponding diagonal entries. This is *not* a
    linear operation, so it should be treated as a hyperparameter.

    :param matrix: Matrix to normalize
    :return: Normalized matrix
    '''

    k = 1.0 / np.sqrt(np.diagonal(matrix))
    return np.multiply(matrix, np.outer(k, k))


def grid_search_cv(
    clf,
    train_indices,
    n_folds,
    param_grid,
    kernel_matrices,
):
    '''
    Internal grid search routine for a set of kernel matrices. The
    routine will use a pre-defined set of train indices to use for
    the grid search. Other indices will *not* be considered. Thus,
    information leakage is prevented.


    :param clf: Classifier to fit
    :param train_indices: Indices permitted to be used for cross-validation
    :param n_folds: Number of folds for the cross-validation
    :param param_grid: Parameters for the grid search
    :param kernel_matrices: Kernel matrices to check; each one of them
    is assumed to represent a different choice of parameter. They will
    *all* be checked iteratively by the routine.

    :return: Best classifier, i.e. the classifier with the best
    parameters. Needs to be refit prior to predicting labels on
    the test data set. Moreover, the best-performing matrix, in
    terms of the grid search, is returned. It has to be used in
    all subsequent prediction tasks.
    '''

    y = kernel_matrices['y'][train_indices]

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42  # TODO: make configurable
    )

    best_clf = None
    best_accuracy = 0.0
    best_parameters = {}

    # From this point on, `train_index` and `test_index` are supposed to
    # be understood *relative* to the input training indices.
    for train_index, test_index in cv.split(train_indices, y):
        for K_param, K in kernel_matrices.items():

            # Skip labels; we could also remove them from the set of
            # matrices but this would make the function inconsistent
            # because it should *not* fiddle with the input data set
            # if it can be avoided.
            if K_param == 'y':
                continue

            # This ensures that we *cannot* access the test indices,
            # even if we try :)
            K = K[train_indices][:, train_indices]

            # Starts enumerating all 'inner' parameters, i.e. the ones
            # that pertain to the classifier (and potentially how this
            # matrix should be treated)
            for parameters in list(param_grid):
                if parameters['normalize']:
                    K = normalize(K)

                # Remove the parameter because it does not pertain to
                # the classifier below.
                clf_parameters = {
                    key: value for key, value in parameters.items()
                    if key not in ['normalize']
                }

                accuracy, params = _fit_and_score(
                    clone(clf),
                    K, y,
                    scorer=make_scorer(accuracy_score),
                    train=train_index,
                    test=test_index,
                    verbose=0,
                    parameters=clf_parameters,
                    fit_params=None,  # No additional parameters for `fit()`
                    return_parameters=True,
                )

                # Note that when storing the parameters, we can re-use
                # the original grid because we want to know about this
                # normalization.
                if accuracy > best_accuracy:
                    best_clf = clone(clf).set_params(**params)
                    best_accuracy = accuracy
                    best_parameters = parameters

                    # Update kernel matrix parameter to indicate which
                    # matrix was used to obtain these results. The key
                    # will also be returned later on.
                    best_parameters['K'] = K_param

    return clf, kernel_matrices[best_parameters['K']]


def train_and_test(train_indices, test_indices, matrices):
    '''
    Trains the classifier on a set of kernel matrices (that are all
    assumed to come from the same algorithm). This uses pre-defined
    splits to prevent information leakage.

    :param train_indices: Indices to be used for training
    :param test_indices: Indices to be used for testing
    :param matrices: Kernel matrices belonging to some algorithm

    :return: Dictionary containing information about the training
    process and the trained model.
    '''

    # Parameter grid for the classifier, but also for the 'pre-processing'
    # of a kernel matrix.
    param_grid = {
        'C': 10. ** np.arange(-3, 4),  # 10^{-3}..10^{3}
        'normalize': [False, True]
    }

    clf, K = grid_search_cv(
        SVC(kernel='precomputed'),
        train_indices,
        n_folds=5,
        param_grid=ParameterGrid(param_grid),
        kernel_matrices=matrices
    )

    # Refit the classifier on the test data set; using the kernel matrix
    # that performed best in the hyperparameter search.
    K_train = K[train_indices][:, train_indices]
    clf.fit(K_train, y[train_indices])

    y_test = y[test_indices]
    K_test = K[test_indices][:, train_indices]
    y_pred = clf.predict(K_test)

    accuracy = accuracy_score(y_test, y_pred)

    results = dict()
    results['train_indices'] = train_indices
    results['test_indices'] = test_indices
    results['accuracy'] = accuracy
    results['best_model'] = clf.best_params_
    results['y_test'] = y_test
    results['y_pred'] = y_pred

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'MATRIX', nargs='+',
        type=str,
        help='Input kernel matrix'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=None
    )

    logging.info('Loading input data...')

    # Load *all* matrices; each one of the is assumed to represent
    # a certain input data set.
    matrices = {
        os.path.splitext(os.path.basename(filename))[0]:
            np.load(filename) for filename in tqdm(args.MATRIX, desc='File')
    }

    logging.info('Checking input data and preparing splits...')

    n_graphs = None
    y = None

    # Check input data by looking at the shape of matrices and their
    # corresponding label vectors.
    for name, matrix in tqdm(matrices.items(), 'File'):
        for parameter in matrix:

            M = matrix[parameter]

            if parameter != 'y':
                # A kernel matrix needs to be square
                assert M.shape[0] == M.shape[1]
            else:
                if y is None:
                    y = M
                else:
                    assert y.shape == M.shape

            # Either set the number of graphs, or check that each matrix
            # contains the same number of them.
            if n_graphs is None:
                n_graphs = M.shape[0]
            else:
                assert n_graphs == M.shape[0]

    clf = SVC(kernel='precomputed')

    # Prepare cross-validated indices for the training data set.
    # Ideally, they should be loaded from *outside*.
    all_indices = np.arange(n_graphs)
    n_iterations = 10
    n_folds = 10

    for name, matrix in matrices.items():

        print(f'Kernel name: {name}')

        # Every matrix, i.e. every kernel, gets the *same* folds so that
        # these results do not have to be stored multiple times.
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42  # TODO: make configurable?
        )

        for iteration in range(n_iterations):
            for train_index, test_index in cv.split(all_indices, y):
                train_indices = all_indices[train_index]
                test_indices = all_indices[test_index]

                # Main function for training and testing a certain kernel
                # matrix on the data set.
                train_and_test(
                    train_indices,
                    test_indices,
                    matrix
                )
