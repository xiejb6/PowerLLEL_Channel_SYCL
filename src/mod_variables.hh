#pragma once

#include "mod_type.hh"

namespace mod_variables {

inline Array3DH3 u, v, w;
inline Array3DH1 p;
inline Array3DH3 u1, v1, w1;

void allocVariables(std::array<int, 3> sz);
void freeVariables();
} // namespace mod_variables
