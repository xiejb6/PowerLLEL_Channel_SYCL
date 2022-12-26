#include "mod_initFlow.hh"
#include "mod_mesh.hh"
#include "mod_mpi.hh"
#include "mod_oneapi.hh"
#include "mod_parameters.hh"
#include "mod_type.hh"
#include "mod_utils.hh"

#include <fmt/core.h>
#include <fmt/os.h>
#include <oneapi/dpl/random>
#include <random>

namespace mod_initFlow {
namespace {
void setProfile_Poiseuille(int n, const Array1D<0, 0> &z_norm, fp factor,
                           Array1D<0, 0> &vel) {
  for (int k = 1; k <= n; ++k) {
    vel(k) = 6.0_fp * (1.0_fp - z_norm(k)) * z_norm(k) * factor;
  }
}

void setProfile_Log(int n, const Array1D<0, 0> &z_norm, fp re,
                    Array1D<0, 0> &vel) {
  fp retau = 0.09_fp * pow(re, 0.88_fp);
  for (int k = 1; k <= n; ++k) {
    fp z = fmin(z_norm(k), 1.0_fp - z_norm(k)) * 2.0_fp * retau;
    vel(k) = (z <= 11.6_fp ? z : 2.5_fp * log(z) + 5.5_fp);
  }
}
void addNoise(int iseed, fp norm, Array3DH3 &var) {
  int seed = iseed + mod_mpi::myrank;
  auto &sz = var.size();
  auto _var = var.get_view();
  auto r = sycl::range(sz[2], sz[1], sz[0]);
  auto ev = mod_oneapi::queue.parallel_for(r, [=](sycl::item<3> idx) {
    auto [i, j, k] = mod_oneapi::unpack<1>(idx);
    auto offset = idx.get_linear_id();
    oneapi::dpl::minstd_rand rng(seed, offset);
    oneapi::dpl::uniform_real_distribution<double> dist;
    _var(i, j, k) += 2.0 * (dist(rng) - 0.5) * norm;
  });
  ev.wait();
}
} // namespace

void initFlow(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w) {
  int sz = mod_mpi::sz[2];
  Array1D<0, 0> zclzi(sz);
  Array1D<0, 0> u_prof(sz);
  bool is_mean;
  const auto &field = initial_field;
  if (field == "uni") {
    for (int i = 1; i <= sz; ++i) {
      u_prof(i) = u0;
    }
    is_mean = false;
  } else if (field == "poi") {
    for (int i = 1; i <= sz; ++i) {
      zclzi(i) = mod_mesh::zc_global(mod_mpi::st[2] + i - 1) / lz;
    }
    setProfile_Poiseuille(sz, zclzi, u_ref, u_prof);
    is_mean = true;
  } else if (field == "log") {
    for (int i = 1; i <= sz; ++i) {
      zclzi(i) = mod_mesh::zc_global(mod_mpi::st[2] + i - 1) / lz;
    }
    setProfile_Log(sz, zclzi, re, u_prof);
    is_mean = true;
  } else if (field == "mpi") {
    for (int i = 1; i <= sz; ++i) {
      u_prof(i) = static_cast<fp>(mod_mpi::myrank);
    }
  }

  auto _u = u.get_view();
  auto _u_prof = u_prof.get_view();
  auto r = sycl::range(mod_mpi::sz[2], mod_mpi::sz[1], mod_mpi::sz[0]);
  auto ev = mod_oneapi::queue.parallel_for(r, [=](sycl::item<3> idx) {
    auto [i, j, k] = mod_oneapi::unpack<1>(idx);
    _u(i, j, k) = _u_prof(k);
  });
  ev.wait();

  if (init_with_noise) {
    addNoise(123, noise_intensity, u);
    addNoise(456, noise_intensity, v);
    addNoise(789, noise_intensity, w);
  }

  if (is_mean) {
    fp meanold = mod_utils::Mean(nz, ny, mod_mesh::dzflzi, u);
    if (meanold != 0.0_fp) {
      fp multiplier = 1.0_fp / (meanold * u_ref);
      auto ev = mod_oneapi::queue.parallel_for(r, [=](sycl::item<3> idx) {
        auto [i, j, k] = mod_oneapi::unpack<1>(idx);
        _u(i, j, k) *= multiplier;
      });
      ev.wait();
    }
  }
}
} // namespace mod_initFlow