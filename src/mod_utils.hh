#pragma once

#include "mod_type.hh"
#include <string_view>

namespace mod_utils {

struct value_index_pair_t {
  fp value;
  int rank;
  int ig, jg, kg;
};

fp Mean(int nx_global, int ny_global, const Array1DH1 &dzflzi,
        const Array3DH3 &var);
void initCheckCFLAndDiv(value_index_pair_t &cfl_max,
                        value_index_pair_t &div_max);
void freeCheckCFLAndDiv();
bool checkNaN(const Array3DH3 &var, std::string_view tag);
void calcMaxCFL(fp dt, fp dxi, fp dyi, const Array1DH1 &dzfi,
                const Array3DH3 &u, const Array3DH3 &v, const Array3DH3 &w,
                value_index_pair_t &cfl_max);
bool checkCFL(fp cfl_limit, const value_index_pair_t &cfl_max);
void calcMaxDiv(fp dxi, fp dyi, const Array1DH1 &dzfi, const Array3DH3 &u,
                const Array3DH3 &v, const Array3DH3 &w,
                value_index_pair_t &div_max);
bool checkDiv(fp div_limit, const value_index_pair_t &div_max);
} // namespace mod_utils
