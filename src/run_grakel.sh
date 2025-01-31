#!/bin/bash
#
# run_grakel.sh: basic job running script for clusters with LSF support.
# Given a data set name and an algorithm, will create a kernel matrix,
# and do the training of said matrix.
#
# All results will be stored in `results_run`.
#
# Usage:
#   run_grakel.sh DATA ALGORITHM [MAX_ITERATIONS]
#
# Parameters:
#   DATA: Essentially, just a folder name in `data`. All files are taken
#   from there.
#
#   ALGORITHM: An abbreviation of one of the algorithms to run. One of
#   these days, I will document all of them in the repo.
#
#   MAX_ITERATIONS: An optional argument indicating the maximum number
#   of iterations for the SVM.


python3 ../src/grakel_create_kernel_matrices.py -a $2 -o ../matrices/$1 ../data/$1

if [ -n "$3" ]; then
  python3 ../src/train.py ../matrices/$1/$2.npz -n $1 -o ../results/$1_$2.json -I $3
else
  python3 ../src/train.py ../matrices/$1/$2.npz -n $1 -o ../results/$1_$2.json
fi
