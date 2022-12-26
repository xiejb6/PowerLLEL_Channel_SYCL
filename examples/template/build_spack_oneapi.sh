#!/bin/bash

# 1. Load environments.
# 1) By package management tool.
#   Spack is a handy package management tool designed for large supercomputing centers
#   (https://github.com/spack/spack). It is used here for reference. Another similar
#   tool is Environment Modules (https://modules.readthedocs.io/).
# 2) Manually.
#   Set environment variables ($PATH, $LD_LIBRARY_PATH, $LIBRARY_PATH, $CPATH, $PKG_CONFIG_PATH, etc.)
#   directly using the following commands:
#     export PATH=${PATH_TO_ADD}:$PATH
#     export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_TO_ADD}:$LD_LIBRARY_PATH
#     ...

spack load \
    intel-oneapi-compilers \
    intel-oneapi-mpi%dpcpp \
    intel-oneapi-mkl%dpcpp \
    hdf5%dpcpp \
    fmt%dpcpp \
    rapidjson%dpcpp
# If you plan to use GPTL for profiling, please modify the variable $GPTL_INSTALL_PATH properly.
export PKG_CONFIG_PATH=$HOME/xjb/software/gptl-8.1.1-dpcpp-impi/lib/pkgconfig:$PKG_CONFIG_PATH

# 2. Modify CMake options shown below as needed

rm -rf build
cmake \
    -DCMAKE_INSTALL_PREFIX=build/install \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=dpcpp \
    `# IMPORTANT options` \
    -DUSE_GPTL=ON \
    -DUSE_PDD=ON \
    -B build
cmake --build build -j
cmake --install build
(mkdir -p ./test &&
   cd ./test &&
   cp ../examples/template/param.json param.json &&
   mpirun -n 1 ../build/install/bin/PowerLLEL 2>&1 | tee test.log)