#pragma once

#include "mod_type.hh"

void init_poisson_solver(int nx_global, int ny_global, int nz_global, double dx,
                         double dy, double *dzf_global, char bctype_x[],
                         char bctype_y[], char bctype_z[],
                         std::array<std::array<int, 6>, 3> &neighbor_xyz);
void execute_poisson_solver(Array3DH1& p);
void free_poisson_solver();
