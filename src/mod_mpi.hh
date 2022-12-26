#pragma once

#include <array>
#include <string_view>

#include <mpi.h>

#ifdef SINGLE_PREC
#define MPI_REAL_FP MPI_FLOAT
#else
#define MPI_REAL_FP MPI_DOUBLE
#endif

namespace mod_mpi {
inline int myrank;
inline MPI_Comm comm_cart;
inline MPI_Comm comm_cart_xpen, comm_cart_ypen, comm_cart_zpen;
inline std::array<MPI_Datatype, 6> halotype_vel, halotype_one;
inline std::array<std::array<int, 6>, 3> neighbor_xyz;
inline std::array<int, 6> neighbor;
inline std::array<int, 2> coord_xpen, coord_ypen, coord_zpen;
inline std::array<int, 2> coord_pen;
inline std::array<int, 3> st, en, sz;
inline std::array<int, 3> xst, xen, xsz;
inline std::array<int, 3> yst, yen, ysz;
inline std::array<int, 3> zst, zen, zsz;
void initMPI();
void freeMPI();
} // namespace mod_mpi
