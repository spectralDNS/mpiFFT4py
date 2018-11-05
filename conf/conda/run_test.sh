#!/bin/bash

pushd tests

export OMPI_MCA_plm=isolated
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=yes

if [ "$(uname)" == "Darwin" ]; then
    mpirun -np 2 py.test -v
fi

if [ "$(uname)" == "Linux" ]; then
    mpirun -np 2 py.test -v
fi
# if [ "${CONDA_PY:0:1}" == "3" ]; then
#     mpirun -np 4 py.test
# fi
#
# if [ "${CONDA_PY:0:1}" == "2" ]; then
#     mpirun -np 1 py.test
# fi
#
