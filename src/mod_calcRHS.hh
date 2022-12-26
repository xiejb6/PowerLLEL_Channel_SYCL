#pragma once

#include "mod_type.hh"

namespace mod_calcRHS {
void calcRHS(const Array3DH3 &u, const Array3DH3 &v, const Array3DH3 &w,
             Array3DH1 &rhs);
} // namespace mod_calcRHS
