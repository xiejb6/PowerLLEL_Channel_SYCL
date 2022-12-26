# A SYCL port of PowerLLEL_Channel

This is a port of PowerLLEL_Channel from MPI+OpenMP to MPI+SYCL. To do this, we
- rewrite almost all of the original Fortran code in C++ 17;
- replace OpenMP directives with SYCL;
- replace FFTW with oneMKL (DPC++ interface).

## Building

The prerequisites are as follows:
- [CMake](https://cmake.org/) (3.21.0 or newer)
- [Intel oneAPI Base Toolkit + HPC Toolkit](https://www.intel.cn/content/www/cn/zh/developer/tools/oneapi/toolkits.html) (2022.1.0 or newer)
    - Intel oneAPI DPC++/C++ Compiler
    - Intel oneAPI DPC++ Library
    - Intel oneAPI Math Kernel Library
    - Intel MPI Library
- [HDF5](https://www.hdfgroup.org/downloads/hdf5) (1.12.2 or newer)
- [fmt](https://github.com/fmtlib/fmt) (8.1.1 or newer)
- [RapidJSON](http://rapidjson.org) (1.1.0 or newer)
- [GPTL](https://github.com/jmrosinski/GPTL) (optional, 8.1.1 or newer)

Assume that the PowerLLEL_Channel source package is unzipped to the directory `$POWERLLEL_DIR`.

First, copy the template build script `$POWERLLEL_DIR/examples/template/build_spack_oneapi.sh` to `$POWERLLEL_DIR`.

Then, modify the build script according to the prompts in it.

Finally, execute the script.
The script will firstly build the executable `PowerLLEL` in the newly created directory `$POWERLLEL_DIR/build/install/bin`,
and then launch a simple test. Without any errors, the test will end successfully.

## Running

In general, the main `PowerLLEL` executable reads the parameter file `param.json` and outputs results `*.out/*.h5` at working directory `$WORK_DIR`. The best practice is:

- After a successful build, copy the `PowerLLEL` executable to the working directory `$WORK_DIR`.
- Copy the template parameter file `$POWERLLEL_DIR/examples/template/param.json` to `$WORK_DIR`.
- Modify simulation parameters in `param.json` as required.
- Launch a parallel simulation with the following command:
```bash
mpirun -n 1 ./PowerLLEL
```
Note that the total number of MPI processes should be equal to the product of parameters `p_row` and `p_col` in `param.json`.

## Tested configurations

The program was built and tested using the following configurations.

**Intel oneAPI DPC++ 2022.1.0**

- Intel CPU
    - 2 x Xeon Gold 6346
- Intel GPU
    - 2 x Intel Graphics

## License

PowerLLEL_Channel is distributed under the MIT license.