#include "mod_utils.hh"
#include "mod_mpi.hh"
#include "mpi.h"
#include <array>
#include <atomic>
#include <cmath>
#include <fmt/core.h>

namespace mod_utils {
namespace {
constexpr int blockcount = 5;
std::array<int, blockcount> blocklengths;
std::array<MPI_Aint, blockcount> displacements;
std::array<MPI_Datatype, blockcount> types;
MPI_Datatype ValueIndexPair_type;
MPI_Op MPI_MAX_INDEX;

void findMPIMaxIndex(void *vinvec, void *vinoutvec, int *len,
                     MPI_Datatype *dtype) {
  auto invec = reinterpret_cast<value_index_pair_t *>(vinvec);
  auto inoutvec = reinterpret_cast<value_index_pair_t *>(vinoutvec);
  for (int i = 0; i < *len; ++i) {
    if (invec[i].value > inoutvec[i].value) {
      inoutvec[i] = invec[i];
    }
  }
}
} // namespace

fp Mean(int nx_global, int ny_global, const Array1DH1 &dzflzi,
        const Array3DH3 &var) {
  using namespace sycl;
  auto &sz = var.size();
  auto r = range(sz[2], sz[1], sz[0]);
  fp *mean = var.get_mean_buf();
  *mean = 0;
  auto ev = mod_oneapi::queue.submit([&](handler &h) {
    auto _var = var.get_view();
    auto _dzflzi = dzflzi.get_view();
    auto sum_reduction = reduction(mean, plus<>());
    h.parallel_for(r, sum_reduction, [=](id<3> idx, auto &sum) {
      int i = idx[2] + 1;
      int j = idx[1] + 1;
      int k = idx[0] + 1;
      sum += _var(i, j, k) * _dzflzi(k);
    });
  });
  ev.wait();
  MPI_Allreduce(MPI_IN_PLACE, mean, 1, MPI_REAL_FP, MPI_SUM, MPI_COMM_WORLD);
  return *mean / nx_global / ny_global;
}

void initCheckCFLAndDiv(value_index_pair_t &cfl_max,
                        value_index_pair_t &div_max) {
  blocklengths = {1, 1, 1, 1, 1};
  types = {MPI_REAL_FP, MPI_INTEGER, MPI_INTEGER, MPI_INTEGER, MPI_INTEGER};
  displacements[0] = 0;
  for (int i = 1; i < blockcount; ++i) {
    MPI_Aint disp, lb;
    MPI_Type_get_extent(types[i - 1], &lb, &disp);
    displacements[i] = displacements[i - 1] + disp;
  }
  MPI_Type_create_struct(blockcount, blocklengths.data(), displacements.data(),
                         types.data(), &ValueIndexPair_type);
  MPI_Type_commit(&ValueIndexPair_type);
  MPI_Op_create(findMPIMaxIndex, 1, &MPI_MAX_INDEX);

  div_max = cfl_max = {
      .value = 0.0_fp, .rank = mod_mpi::myrank, .ig = 1, .jg = 1, .kg = 1};
}

void freeCheckCFLAndDiv() {
  MPI_Op_free(&MPI_MAX_INDEX);
  MPI_Type_free(&ValueIndexPair_type);
}

bool checkNaN(const Array3DH3 &var, std::string_view tag) {
  auto &sz = var.size();
  auto &nhalo = var.nhalo;
  std::atomic<bool> passed(true);
  for (int k = 1 - nhalo[4]; k <= sz[2] + nhalo[5]; ++k) {
    // we can't break the loop in omp for, right?
    if (passed) {
      for (int j = 1 - nhalo[2]; j <= sz[1] + nhalo[3]; ++j) {
        for (int i = 1 - nhalo[0]; i <= sz[0] + nhalo[1]; ++i) {
          if (std::isnan(var(i, j, k))) {
            passed = false;
            break;
          }
        }
        if (!passed)
          break;
      }
    }
  }

  int flag = passed;
  MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  passed = (flag != 0);

  if (!passed && mod_mpi::myrank == 0) {
    fmt::print("PowerLLEL.ERROR.checkNaN: NaN has been detected in <{}>!\n",
               tag);
  }
  return passed;
}

void calcMaxCFL(fp dt, fp dxi, fp dyi, const Array1DH1 &dzfi,
                const Array3DH3 &u, const Array3DH3 &v, const Array3DH3 &w,
                value_index_pair_t &cfl_max) {
  cfl_max = {
      .value = 0.0_fp, .rank = mod_mpi::myrank, .ig = 1, .jg = 1, .kg = 1};
  {
    fp cfl_max_thread = cfl_max.value;
    fp i_tmp = cfl_max.ig;
    fp j_tmp = cfl_max.jg;
    fp k_tmp = cfl_max.kg;
    auto &sz = u.size();
    for (int k = 1; k <= sz[2]; ++k) {
      for (int j = 1; j <= sz[1]; ++j) {
        for (int i = 1; i <= sz[0]; ++i) {
          fp cfl_tmp = fabs((u(i, j, k) + u(i - 1, j, k)) * 0.5_fp) * dxi +
                       fabs((v(i, j, k) + v(i, j - 1, k)) * 0.5_fp) * dyi +
                       fabs((w(i, j, k) + w(i, j, k - 1)) * 0.5_fp) * dzfi(k);
          if (cfl_tmp > cfl_max_thread) {
            cfl_max_thread = cfl_tmp;
            i_tmp = i;
            j_tmp = j;
            k_tmp = k;
          }
        }
      }
    }
    {
      if (cfl_max_thread > cfl_max.value) {
        cfl_max.value = cfl_max_thread;
        cfl_max.ig = i_tmp;
        cfl_max.jg = j_tmp;
        cfl_max.kg = k_tmp;
      }
    }
  }

  cfl_max.value = cfl_max.value * dt;
  cfl_max.ig = cfl_max.ig + mod_mpi::st[0] - 1;
  cfl_max.jg = cfl_max.jg + mod_mpi::st[1] - 1;
  cfl_max.kg = cfl_max.kg + mod_mpi::st[2] - 1;
  MPI_Allreduce(MPI_IN_PLACE, &cfl_max, 1, ValueIndexPair_type, MPI_MAX_INDEX,
                MPI_COMM_WORLD);
}

bool checkCFL(fp cfl_limit, const value_index_pair_t &cfl_max) {
  bool passed = true;
  if (cfl_max.value > cfl_limit || std::isnan(cfl_max.value)) {
    passed = false;
    if (mod_mpi::myrank == 0) {
      fmt::print("PowerLLEL.ERROR.checkCFL: Maximum CFL number exceeds the "
                 "threshold specified by the user!\n");
    }
  }
  return passed;
}

void calcMaxDiv(fp dxi, fp dyi, const Array1DH1 &dzfi, const Array3DH3 &u,
                const Array3DH3 &v, const Array3DH3 &w,
                value_index_pair_t &div_max) {
  div_max = {
      .value = 0.0_fp, .rank = mod_mpi::myrank, .ig = 1, .jg = 1, .kg = 1};
  {
    fp div_max_thread = div_max.value;
    fp i_tmp = div_max.ig;
    fp j_tmp = div_max.jg;
    fp k_tmp = div_max.kg;
    auto &sz = u.size();
    for (int k = 1; k <= sz[2]; ++k) {
      for (int j = 1; j <= sz[1]; ++j) {
        for (int i = 1; i <= sz[0]; ++i) {
          fp div_tmp = fabs((u(i, j, k) - u(i - 1, j, k)) * dxi +
                            (v(i, j, k) - v(i, j - 1, k)) * dyi +
                            (w(i, j, k) - w(i, j, k - 1)) * dzfi(k));
          if (div_tmp > div_max_thread) {
            div_max_thread = div_tmp;
            i_tmp = i;
            j_tmp = j;
            k_tmp = k;
          }
        }
      }
    }
    {
      if (div_max_thread > div_max.value) {
        div_max.value = div_max_thread;
        div_max.ig = i_tmp;
        div_max.jg = j_tmp;
        div_max.kg = k_tmp;
      }
    }
  }

  div_max.ig = div_max.ig + mod_mpi::st[0] - 1;
  div_max.jg = div_max.jg + mod_mpi::st[1] - 1;
  div_max.kg = div_max.kg + mod_mpi::st[2] - 1;
  MPI_Allreduce(MPI_IN_PLACE, &div_max, 1, ValueIndexPair_type, MPI_MAX_INDEX,
                MPI_COMM_WORLD);
}

bool checkDiv(fp div_limit, const value_index_pair_t &div_max) {
  bool passed = true;
  if (div_max.value > div_limit || std::isnan(div_max.value)) {
    passed = false;
    if (mod_mpi::myrank == 0) {
      fmt::print("PowerLLEL.ERROR.checkDiv: Maximum divergence exceeds the "
                 "threshold specified by the user!\n");
    }
  }
  return passed;
}
} // namespace mod_utils