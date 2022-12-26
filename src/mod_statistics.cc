#include "mod_statistics.hh"
#include "mod_parameters.hh"
#include "mod_type.hh"

namespace mod_statistics {

void allocStat(std::array<int, 3> sz) {
  for (int i = 0; i < 11; ++i) {
    if (stat_which_var[i]) {
      stat_vec[i].allocate(sz);
    }
  }
}

void freeStat() {
  for (int i = 0; i < 11; ++i) {
    if (stat_which_var[i]) {
      stat_vec[i].deallocate();
    }
  }
}

void initStat() {
  stat_info.nts = nt_init_stat + 1;
  stat_info.nte = nt_init_stat;
  stat_info.nspl = 0;
  // automatically setZero when allocated
}

void calcStat(int nt, fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
              const Array3DH3 &w, const Array3DH1 &p) {
  if ((nt - nt_init_stat) % sample_interval == 0) {
    stat_info.nte = stat_info.nte + sample_interval;
    stat_info.nspl = stat_info.nspl + 1;
    auto &sz = u.size();
    std::array<int, 3> temp_sz = {sz[0], sz[1], 1};
    {
      Array3DH0 uc, vc, wc;
      uc.allocate(temp_sz);
      vc.allocate(temp_sz);
      wc.allocate(temp_sz);
      for (int k = 1; k <= sz[2]; ++k) {
        for (int j = 1; j <= sz[1]; ++j) {
          for (int i = 1; i <= sz[0]; ++i) {
            uc(i, j, 1) = (u(i - 1, j, k) + u(i, j, k)) * 0.5_fp + u_crf;
            vc(i, j, 1) = (v(i, j - 1, k) + v(i, j, k)) * 0.5_fp;
            wc(i, j, 1) = (w(i, j, k - 1) + w(i, j, k)) * 0.5_fp;
          }
        }

        auto plus = [k, &sz](Array3DH0 &lhs, const Array3DH0 &rhs) {
          for (int j = 1; j <= sz[1]; ++j) {
            for (int i = 1; i <= sz[0]; ++i) {
              lhs(i, j, k) += rhs(i, j, 1);
            }
          }
        };
        auto plusSquare = [k, &sz](Array3DH0 &lhs, const Array3DH0 &rhs1,
                                   const Array3DH0 &rhs2) {
          for (int j = 1; j <= sz[1]; ++j) {
            for (int i = 1; i <= sz[0]; ++i) {
              lhs(i, j, k) += rhs1(i, j, 1) * rhs2(i, j, 1);
            }
          }
        };
        if (stat_which_var[0])
          plus(stat_vec[0], uc);
        if (stat_which_var[1])
          plus(stat_vec[1], vc);
        if (stat_which_var[2])
          plus(stat_vec[2], wc);
        if (stat_which_var[3])
          plusSquare(stat_vec[3], uc, uc);
        if (stat_which_var[4])
          plusSquare(stat_vec[4], vc, vc);
        if (stat_which_var[5])
          plusSquare(stat_vec[5], wc, wc);
        if (stat_which_var[6])
          plusSquare(stat_vec[6], uc, vc);
        if (stat_which_var[7])
          plusSquare(stat_vec[7], uc, wc);
        if (stat_which_var[8])
          plusSquare(stat_vec[8], vc, wc);
      }
    }

    if (stat_which_var[9]) {
      for (int k = 1; k <= sz[2]; ++k) {
        for (int j = 1; j <= sz[1]; ++j) {
          for (int i = 1; i <= sz[0]; ++i) {
            stat_vec[9](i, j, k) += p(i, j, k);
          }
        }
      }
    }
    if (stat_which_var[10]) {
      for (int k = 1; k <= sz[2]; ++k) {
        for (int j = 1; j <= sz[1]; ++j) {
          for (int i = 1; i <= sz[0]; ++i) {
            stat_vec[10](i, j, k) += p(i, j, k) * p(i, j, k);
          }
        }
      }
    }
  }
}

} // namespace mod_statistics
