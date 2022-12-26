#pragma once

#include "mod_type.hh"

namespace mod_statistics {
struct StatInfo {
  int nts;
  int nte;
  int nspl;
};

inline StatInfo stat_info;
inline std::array<Array3DH0, 11> stat_vec;

void allocStat(std::array<int, 3> sz);
void freeStat();
void initStat();
void calcStat(int nt, fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
              const Array3DH3 &w, const Array3DH1 &p);
} // namespace mod_statistics
