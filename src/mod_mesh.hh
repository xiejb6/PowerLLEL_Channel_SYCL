#pragma once

#include "mod_type.hh"

namespace mod_mesh {
// 'c' refers to '(mesh cell) center', 'f' refers to '(mesh cell) face'
inline fp dx, dy;
inline fp dx_inv, dy_inv;
inline Array1DH1 xc_global; // redundant, but used in the output process
inline Array1DH1 yc_global; // redundant, but used in the output process
inline Array1DH1 zc_global; // redundant, but used in the output process
inline Array1DH1 dzf_global;
inline Array1DH1 dzc, dzc_inv;
inline Array1DH1 dzf, dzf_inv;
inline Array1DH1 dzflzi;
inline Array1DH1 visc_dzf_inv;

void initMesh();
void inputMesh(int nz_global, Array1DH1 &dzf_global);
void initMeshByFunc(int mesh_type, fp stretch_ratio, int nz_global, fp lz,
                    Array1DH1 &dzf_global);
void outputMesh();
void freeMesh();
} // namespace mod_mesh
