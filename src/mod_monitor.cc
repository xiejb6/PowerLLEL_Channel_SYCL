#include "mod_monitor.hh"
#include "mod_hdf5.hh"
#include "mod_mesh.hh"
#include "mod_mpi.hh"
#include "mod_parameters.hh"
#include "mod_type.hh"
#include "mod_utils.hh"

#include <algorithm>
#include <array>
#include <fstream>

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace mod_monitor {
namespace {
struct probe_point_t {
  fp u, v, w, p;        // physical quantities at the point
  int i, j, k;          // probe point local index
  int ig, jg, kg;       // probe point global index
  bool have_this_point; // decide whether the local mpi rank has the probe point
};

struct region_t {
  std::array<int, 3> gsize, offset;
  std::array<int, 3> stg, eng;
  std::array<int, 3> st, sz;
  bool have_this_region;
};

probe_point_t probe_point;
std::array<MPI_Comm, 2> comm_skf_z;
std::array<int, 2> myrank_skf_z;

std::ofstream force_file;
std::ofstream probe_file;
std::array<std::ofstream, 2> skf_z_file;

region_t ext_region;
Array3DH0 ext_buffer;

void initCommForSkf(int k_wall, int ks, int ke, MPI_Comm &comm_new,
                    int &myrank_new) {
  int key = 0;
  int color = MPI_UNDEFINED;
  if (k_wall >= ks && k_wall <= ke)
    color = 1;
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm_new);
  if (comm_new == MPI_COMM_NULL) {
    myrank_new = mod_mpi::myrank;
  } else {
    MPI_Comm_rank(comm_new, &myrank_new);
  }
}

void calcAndWriteForcing(int nt, fp u_crf, const Array3DH3 &u,
                         const Array3DH3 &v, const Array3DH3 &w) {
  if (std::any_of(is_forced.cbegin(), is_forced.cend(),
                  [](bool item) { return item; })) {
    std::array<fp, 3> vel_mean;
    vel_mean.fill(0.0_fp);
    if (is_forced[0])
      vel_mean[0] = mod_utils::Mean(nx, ny, mod_mesh::dzflzi, u);
    if (is_forced[1])
      vel_mean[1] = mod_utils::Mean(nx, ny, mod_mesh::dzflzi, v);
    if (is_forced[2])
      vel_mean[2] = mod_utils::Mean(nx, ny, mod_mesh::dzflzi, w);

    if (mod_mpi::myrank == 0) {
      fmt::print(force_file, "{:9}{:15.7E}{:15.7E}{:15.7E}\n", nt,
                 vel_mean[0] + u_crf, vel_mean[1], vel_mean[2]);
    }
  }
}

void extractAndWriteProbePoint(int nt, fp u_crf, const Array3DH3 &u,
                               const Array3DH3 &v, const Array3DH3 &w,
                               const Array3DH1 &p) {
  if (probe_point.have_this_point) {
    int i = probe_point.i;
    int j = probe_point.j;
    int k = probe_point.k;
    probe_point.u = u(i, j, k) + u_crf;
    probe_point.v = v(i, j, k);
    probe_point.w = w(i, j, k);
    probe_point.p = p(i, j, k);
    fmt::print(probe_file, "{:9}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n", nt,
               probe_point.u, probe_point.v, probe_point.w, probe_point.p);
  }
}

void calcAndWriteSkf(std::ofstream &ofile, MPI_Comm comm_skf, int myrank_skf,
                     int nt, int k_wall, fp dz, const Array3DH3 &vel,
                     fp vel_crf) {
  if (comm_skf != MPI_COMM_NULL) {
    fp tau_w;
    fp my_tau_w = 0.0_fp;
    auto sz = vel.size();
    for (int j = 1; j <= sz[1]; ++j) {
      for (int i = 1; i <= sz[0]; ++i) {
        my_tau_w += re_inv * (vel(i, j, k_wall) + vel_crf) / dz;
      }
    }
    MPI_Reduce(&my_tau_w, &tau_w, 1, MPI_REAL_FP, MPI_SUM, 0, comm_skf);
    if (myrank_skf == 0) {
      tau_w = tau_w / nx / ny;
      fp u_tau = std::sqrt(tau_w);
      fp cf = tau_w / (0.5_fp * u_ref * u_ref);
      fmt::print(ofile, "{:9}{:15.7E}{:15.7E}\n", nt, u_tau, cf);
    }
  }
}

template <typename T>
void extractRegion(const T &var, int offset_dir, int reg_st[3], int reg_sz[3],
                   Array3DH0 &buffer) {
  int is = reg_st[0] - 1;
  int js = reg_st[1] - 1;
  int ks = reg_st[2] - 1;

  if (offset_dir == 0) {
    for (int k = 1; k <= reg_sz[2]; ++k) {
      for (int j = 1; j <= reg_sz[1]; ++j) {
        for (int i = 1; i <= reg_sz[0]; ++i) {
          buffer(i, j, k) = var(i + is, j + js, k + ks);
        }
      }
    }
  } else if (offset_dir == 1) {
    for (int k = 1; k <= reg_sz[2]; ++k) {
      for (int j = 1; j <= reg_sz[1]; ++j) {
        for (int i = 1; i <= reg_sz[0]; ++i) {
          buffer(i, j, k) = 0.5_fp * (var(i - 1 + is, j + js, k + ks) +
                                      var(i + is, j + js, k + ks));
        }
      }
    }
  } else if (offset_dir == 2) {
    for (int k = 1; k <= reg_sz[2]; ++k) {
      for (int j = 1; j <= reg_sz[1]; ++j) {
        for (int i = 1; i <= reg_sz[0]; ++i) {
          buffer(i, j, k) = 0.5_fp * (var(i + is, j - 1 + js, k + ks) +
                                      var(i + is, j + js, k + ks));
        }
      }
    }
  } else if (offset_dir == 3) {
    for (int k = 1; k <= reg_sz[2]; ++k) {
      for (int j = 1; j <= reg_sz[1]; ++j) {
        for (int i = 1; i <= reg_sz[0]; ++i) {
          buffer(i, j, k) = 0.5_fp * (var(i + is, j + js, k - 1 + ks) +
                                      var(i + is, j + js, k + ks));
        }
      }
    }
  }
}

void extractAndWriteRegion(int nt, fp u_crf, const Array3DH3 &u,
                           const Array3DH3 &v, const Array3DH3 &w,
                           const Array3DH1 &p) {
  auto string_dump = fmt::format("{}_{:08}.h5", fn_prefix_region, nt);
  auto fh = mod_hdf5::createFile(string_dump);
  mod_hdf5::writeAttribute(fh, "nt", nt);

  std::array<int, 3> offset;
  auto map = [](int val) { return val + 1; };
  std::transform(ext_region.offset.cbegin(), ext_region.offset.cend(),
                 offset.begin(), map);

  if (ext_region.have_this_region) {
    extractRegion(u, 1, ext_region.st.data(), ext_region.sz.data(), ext_buffer);
    auto sz = ext_buffer.size();
    for (int k = 1; k <= sz[2]; ++k) {
      for (int j = 1; j <= sz[1]; ++j) {
        for (int i = 1; i <= sz[0]; ++i) {
          ext_buffer(i, j, k) += u_crf;
        }
      }
    }
  }
  mod_hdf5::write3d(fh, "u", ext_region.gsize, offset, ext_buffer,
                    ext_region.have_this_region);

  if (ext_region.have_this_region) {
    extractRegion(v, 2, ext_region.st.data(), ext_region.sz.data(), ext_buffer);
  }
  mod_hdf5::write3d(fh, "v", ext_region.gsize, offset, ext_buffer,
                    ext_region.have_this_region);

  if (ext_region.have_this_region) {
    extractRegion(w, 3, ext_region.st.data(), ext_region.sz.data(), ext_buffer);
  }
  mod_hdf5::write3d(fh, "w", ext_region.gsize, offset, ext_buffer,
                    ext_region.have_this_region);

  if (ext_region.have_this_region) {
    extractRegion(p, 0, ext_region.st.data(), ext_region.sz.data(), ext_buffer);
  }
  mod_hdf5::write3d(fh, "p", ext_region.gsize, offset, ext_buffer,
                    ext_region.have_this_region);

  mod_hdf5::closeFile(fh);
}
} // namespace

void initMonitor() {
  using mod_mpi::en;
  using mod_mpi::st;
  auto mode = std::ios_base::out;
  if (is_restart)
    mode |= std::ios_base::ate;

  if (out_forcing) {
    force_file.open(fn_forcing, mode);
    if (mod_mpi::myrank == 0) {
      auto str = fmt::format("{:>9}{:>15}{:>15}{:>15}\n", "nt", "u_mean",
                             "v_mean", "w_mean");
      force_file.write(str.c_str(), str.size());
    }
  }

  if (out_probe_point) {
    auto [ig, jg, kg] = probe_ijk;
    if ((ig >= st[0] && ig <= en[0]) && (jg >= st[1] && jg <= en[1]) &&
        (kg >= st[2] && kg <= en[2])) {
      probe_point.have_this_point = true;
      probe_point.ig = ig;
      probe_point.jg = jg;
      probe_point.kg = kg;
      probe_point.i = ig - st[0] + 1;
      probe_point.j = jg - st[1] + 1;
      probe_point.k = kg - st[2] + 1;
    }

    probe_file.open(fn_probe, mode);
    if (probe_point.have_this_point) {
      auto str = fmt::format("Probe point ({:4},{:4},{:4}):\n", probe_point.ig,
                             probe_point.jg, probe_point.kg);
      probe_file.write(str.c_str(), str.size());
      str = fmt::format("{:>9}{:>15}{:>15}{:>15}{:>15}\n", "nt", "u", "v", "w",
                        "p");
      probe_file.write(str.c_str(), str.size());
    }
  }

  // initialize new MPI communicators for the calculation of skin friction
  // coefficients, at the bottom/top wall of the computational domain
  if (out_skf_z[0]) {
    initCommForSkf(1, st[2], en[2], comm_skf_z[0], myrank_skf_z[0]);
    skf_z_file[0].open(fn_skf_z[0], mode);
    if (myrank_skf_z[0] == 0) {
      auto str = fmt::format("{:>9}{:>15}{:>15}\n", "nt", "u_tau", "cf");
      skf_z_file[0].write(str.c_str(), str.size());
    }
  }
  if (out_skf_z[1]) {
    initCommForSkf(nz, st[2], en[2], comm_skf_z[1], myrank_skf_z[1]);
    skf_z_file[1].open(fn_skf_z[1], mode);
    if (myrank_skf_z[1] == 0) {
      auto str = fmt::format("{:>9}{:>15}{:>15}\n", "nt", "u_tau", "cf");
      skf_z_file[1].write(str.c_str(), str.size());
    }
  }

  if (out_region) {
    // exclude the points out of the domain
    region_ijk[0] = std::max(region_ijk[0], 1);
    region_ijk[1] = std::min(region_ijk[1], nx);
    region_ijk[2] = std::max(region_ijk[2], 1);
    region_ijk[3] = std::min(region_ijk[3], ny);
    region_ijk[4] = std::max(region_ijk[4], 1);
    region_ijk[5] = std::min(region_ijk[5], nz);

    ext_region.have_this_region = false;
    ext_region.gsize[0] = region_ijk[1] - region_ijk[0] + 1;
    ext_region.gsize[1] = region_ijk[3] - region_ijk[2] + 1;
    ext_region.gsize[2] = region_ijk[5] - region_ijk[4] + 1;
    ext_region.stg.fill(1);
    ext_region.eng.fill(1);
    ext_region.offset.fill(0);
    ext_region.st.fill(1);
    ext_region.sz.fill(1);

    if ((region_ijk[0] <= en[0] && region_ijk[1] >= st[0]) &&
        (region_ijk[2] <= en[1] && region_ijk[3] >= st[1]) &&
        (region_ijk[4] <= en[2] && region_ijk[5] >= st[2])) {
      ext_region.have_this_region = true;
      ext_region.stg[0] = std::max(region_ijk[0], st[0]);
      ext_region.stg[1] = std::max(region_ijk[2], st[1]);
      ext_region.stg[2] = std::max(region_ijk[4], st[2]);
      ext_region.eng[0] = std::min(region_ijk[1], en[0]);
      ext_region.eng[1] = std::min(region_ijk[3], en[1]);
      ext_region.eng[2] = std::min(region_ijk[5], en[2]);
      ext_region.offset[0] = ext_region.stg[0] - region_ijk[0];
      ext_region.offset[1] = ext_region.stg[1] - region_ijk[2];
      ext_region.offset[2] = ext_region.stg[2] - region_ijk[4];
      ext_region.st[0] = ext_region.stg[0] - st[0] + 1;
      ext_region.st[1] = ext_region.stg[1] - st[1] + 1;
      ext_region.st[2] = ext_region.stg[2] - st[2] + 1;
      ext_region.sz[0] = ext_region.eng[0] - ext_region.stg[0] + 1;
      ext_region.sz[1] = ext_region.eng[1] - ext_region.stg[1] + 1;
      ext_region.sz[2] = ext_region.eng[2] - ext_region.stg[2] + 1;

      ext_buffer.allocate(
          {ext_region.sz[0], ext_region.sz[1], ext_region.sz[2]});
    }
  }
}

void outputMonitor(int nt, fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
                   const Array3DH3 &w, const Array3DH1 &p) {
  if (out_forcing)
    calcAndWriteForcing(nt, u_crf, u, v, w);
  if (out_probe_point)
    extractAndWriteProbePoint(nt, u_crf, u, v, w, p);
  if (out_skf_z[0])
    calcAndWriteSkf(skf_z_file[0], comm_skf_z[0], myrank_skf_z[0], nt, 1,
                    0.5_fp * mod_mesh::dzf(1), u, u_crf);
  if (out_skf_z[1])
    calcAndWriteSkf(skf_z_file[1], comm_skf_z[1], myrank_skf_z[1], nt,
                    u.size()[2], 0.5_fp * mod_mesh::dzf(u.size()[2]), u, u_crf);
  if (out_region && nt % nt_out_region == 0)
    extractAndWriteRegion(nt, u_crf, u, v, w, p);
}

void freeMonitor() {
  if (out_forcing)
    force_file.close();
  if (out_probe_point)
    probe_file.close();
  if (out_skf_z[0])
    skf_z_file[0].close();
  if (out_skf_z[1])
    skf_z_file[1].close();
  if (out_region)
    ext_buffer.deallocate();
}
} // namespace mod_monitor