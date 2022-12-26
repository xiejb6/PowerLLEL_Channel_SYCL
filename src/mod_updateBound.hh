#pragma once
#include "mod_type.hh"
#include "mpi.h"

namespace mod_updateBound {
void imposeBCVel(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w, fp u_crf);
void updateBoundVel(fp u_crf, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w);
void updateBoundP(Array3DH1 &p);
} // namespace mod_updateBound
