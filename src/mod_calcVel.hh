#pragma once

#include "mod_type.hh"

namespace mod_calcVel {
void transform2CRF(fp vel_crf, Array3DH3 &vel, fp &vel_force);
void timeIntVelRK1(fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
                   const Array3DH3 &w, Array3DH3 &u1, Array3DH3 &v1,
                   Array3DH3 &w1);
void timeIntVelRK2(fp u_crf, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w,
                   const Array3DH3 &u1, const Array3DH3 &v1,
                   const Array3DH3 &w1);
void correctVel(const Array3DH1 &p, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w);
void forceVel(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w);
} // namespace mod_calcVel
