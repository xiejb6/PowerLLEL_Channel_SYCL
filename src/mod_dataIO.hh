#pragma once

#include "mod_parameters.hh"
#include "mod_type.hh"

#include <string_view>

namespace mod_dataIO {
// ret: nt
int inputData(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w);
void outputData(int nt, fp u_crf, Array3DH3 &u, const Array3DH3 &v,
                const Array3DH3 &w, const Array3DH1 &p);
void inputStatData(int nt_in_inst);
void outputStatData(int nt);
} // namespace mod_dataIO
