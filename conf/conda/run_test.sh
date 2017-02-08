#!/bin/bash

pushd tests

if [ "$(uname)" == "Darwin" ]; then
    py.test -v
fi

if [ "$(uname)" == "Linux" ]; then
    mpirun -np 4 py.test
fi
# if [ "${CONDA_PY:0:1}" == "3" ]; then
#     mpirun -np 4 py.test
# fi
#
# if [ "${CONDA_PY:0:1}" == "2" ]; then
#     mpirun -np 1 py.test
# fi
#
