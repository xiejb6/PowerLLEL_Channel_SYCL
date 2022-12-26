#include "mod_mesh.hh"
#include "mod_hdf5.hh"
#include "mod_mpi.hh"
#include "mod_parameters.hh"

#include <cmath>
#include <fstream>

#include <fmt/core.h>
#include <fmt/os.h>

namespace mod_mesh {
void initMesh() {
  int sz = mod_mpi::sz[2];
  xc_global.allocate(nx);
  yc_global.allocate(ny);
  zc_global.allocate(nz);
  dzf_global.allocate(nz);
  dzc.allocate(sz);
  dzc_inv.allocate(sz);
  dzf.allocate(sz);
  dzf_inv.allocate(sz);
  dzflzi.allocate(sz);
  visc_dzf_inv.allocate(sz);

  // mesh in x & y direction
  dx = lx / nx;
  dy = ly / ny;
  dx_inv = 1.0 / dx;
  dy_inv = 1.0 / dy;
  xc_global(0) = -0.5_fp * dx;
  for (int i = 1; i <= nx + 1; ++i) {
    xc_global(i) = xc_global(0) + i * dx;
  }
  yc_global(0) = -0.5_fp * dy;
  for (int j = 1; j <= ny + 1; ++j) {
    yc_global(j) = yc_global(0) + j * dy;
  }

  // mesh in z direction
  if (read_mesh) {
    // read the mesh spacing <dzf_global>
    inputMesh(nz, dzf_global);
  } else {
    // generate <dzf_global> by mesh functions
    initMeshByFunc(mesh_type, stretch_ratio, nz, lz, dzf_global);
  }

  zc_global(0) = -0.5_fp * dzf_global(0);
  for (int k = 1; k <= nz + 1; ++k) {
    zc_global(k) =
        zc_global(k - 1) + 0.5_fp * (dzf_global(k - 1) + dzf_global(k));
  }
  for (int k = 0; k <= sz + 1; ++k) {
    dzf(k) = dzf_global(mod_mpi::st[2] - 1 + k);
    dzf_inv(k) = 1.0_fp / dzf(k);
    visc_dzf_inv(k) = 1.0_fp / dzf(k);
  }
  for (int k = 0; k <= sz; ++k) {
    dzc(k) = 0.5_fp * (dzf(k) + dzf(k + 1));
    dzc_inv(k) = 1.0_fp / dzc(k);
  }
  for (int k = 0; k <= sz + 1; ++k) {
    dzflzi(k) = dzf(k) / lz;
  }

  if (smooth_wall_visc) {
    if (mod_mpi::neighbor[4] == MPI_PROC_NULL)
      visc_dzf_inv(1) = 0.5_fp * (dzc(0) + dzc(1));
    if (mod_mpi::neighbor[5] == MPI_PROC_NULL) {
      visc_dzf_inv(sz) = 0.5_fp * (dzc(sz - 1) + dzc(sz));
    }
  }
  // output mesh files for post-processing and check
  outputMesh();
}

void inputMesh(int nz_global, Array1DH1 &dzf_global) {
  auto fn_mesh = "mesh.in";
  std::ifstream ifile(fn_mesh);
  if (!ifile) {
    if (mod_mpi::myrank == 0) {
      fmt::print("PowerLLEL.ERROR.inputMesh: File {} doesn't exist!\n",
                 fn_mesh);
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  for (int k = 1; k <= nz_global; ++k) {
    ifile >> dzf_global(k);
  }
  dzf_global(0) = dzf_global(1);
  dzf_global(nz_global + 1) = dzf_global(nz_global);
}

void initMeshByFunc(int mesh_type, fp stretch_ratio, int nz_global, fp lz,
                    Array1DH1 &dzf_global) {
  if (stretch_ratio == 0.0_fp || mesh_type == 0) {
    // uniform
    for (int k = 0; k <= nz_global + 1; ++k) {
      dzf_global(k) = lz / nz_global;
    }
  } else {
    // nonuniform, clustered at both ends
    Array1DH1 zf(nz_global);
    zf(0) = 0.0_fp;
    if (stretch_func == 0) {
      for (int k = 1; k <= nz_global; ++k) {
        fp z0 = k / (1.0_fp * nz_global);
        zf(k) = 0.5_fp * (1.0_fp + tanh((z0 - 0.5_fp) * stretch_ratio) /
                                       tanh(stretch_ratio / 2.0_fp));
      }
    } else {
      for (int k = 1; k <= nz_global; ++k) {
        fp z0 = k / (1.0_fp * nz_global);
        zf(k) = 0.5_fp * (1.0_fp + sin((z0 - 0.5_fp) * stretch_ratio * pi) /
                                       sin(stretch_ratio * pi / 2.0_fp));
      }
    }
    for (int k = 1; k <= nz_global; ++k) {
      dzf_global(k) = lz * (zf(k) - zf(k - 1));
    }
    dzf_global(0) = dzf_global(1);
    dzf_global(nz_global + 1) = dzf_global(nz_global);
  }
}

void outputMesh() {
  auto fn_mesh = fmt::format("mesh_{}-{}-{}.h5", xc_global.size(),
                             yc_global.size(), zc_global.size());
  auto fh = mod_hdf5::createFile(fn_mesh);
  mod_hdf5::write1d(fh, "xc", xc_global);
  mod_hdf5::write1d(fh, "yc", yc_global);
  mod_hdf5::write1d(fh, "zc", zc_global);
  mod_hdf5::closeFile(fh);
  if (mod_mpi::myrank == 0) {
    fmt::print("PowerLLEL.NOTE.outputMesh: Finish writing file {}!\n", fn_mesh);
    auto string_dump = fmt::format("mesh_{:05}.out", nz);
    auto ofile = fmt::output_file(
        string_dump, fmt::file::WRONLY | fmt::file::CREATE | fmt::file::TRUNC);
    ofile.print("{0:4}{1}{0:21}{2}{0:22}{3}\n", "", "k", "dzf", "zc");
    for (int k = 1; k <= nz; ++k) {
      ofile.print("{:05} {:23.15E} {:23.15E}\n", k, dzf_global(k), zc_global(k));
    }
    ofile.close();
    fmt::print("PowerLLEL.NOTE.outputMesh: Finish writing file {}!\n", string_dump);
  }
}

void freeMesh() {
  xc_global.deallocate();
  yc_global.deallocate();
  zc_global.deallocate();
  dzf_global.deallocate();
  dzc.deallocate();
  dzc_inv.deallocate();
  dzf.deallocate();
  dzf_inv.deallocate();
  dzflzi.deallocate();
  visc_dzf_inv.deallocate();
}
} // namespace mod_mesh