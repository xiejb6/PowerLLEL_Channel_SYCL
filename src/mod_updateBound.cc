#include "mod_updateBound.hh"
#include "gptl.hh"
#include "mod_mpi.hh"
#include "mod_type.hh"
#include "mpi.h"
#include <array>
#include <optional>

namespace mod_updateBound {
namespace {
template <typename T>
void updateHalo(const std::array<MPI_Datatype, 6> &halotype, T &var) {
  constexpr auto &nhalo = T::nhalo;
  auto &send_buf = var.get_send_buf();
  auto &recv_buf = var.get_recv_buf();
  // *** south/north ***
  MPI_Sendrecv(send_buf[0], 1, halotype[3], mod_mpi::neighbor[2], 0,
               recv_buf[1], 1, halotype[3], mod_mpi::neighbor[3], 0,
               mod_mpi::comm_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_buf[1], 1, halotype[2], mod_mpi::neighbor[3], 0,
               recv_buf[0], 1, halotype[2], mod_mpi::neighbor[2], 0,
               mod_mpi::comm_cart, MPI_STATUS_IGNORE);
  // *** bottom/top ***
  MPI_Sendrecv(send_buf[2], 1, halotype[5], mod_mpi::neighbor[4], 0,
               recv_buf[3], 1, halotype[5], mod_mpi::neighbor[5], 0,
               mod_mpi::comm_cart, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_buf[3], 1, halotype[4], mod_mpi::neighbor[5], 0,
               recv_buf[2], 1, halotype[4], mod_mpi::neighbor[4], 0,
               mod_mpi::comm_cart, MPI_STATUS_IGNORE);
}

template <typename T> void imposePeriodicBC(int ibound, int idir, T &var) {
  using namespace sycl;
  auto &nhalo = var.nhalo;
  auto &sz = var.size();
  if (idir == 1) {
    if (ibound == 0) {
      mod_oneapi::queue
          .submit([&](handler &h) {
            int zlen = sz[2] + nhalo[4] + nhalo[5];
            int ylen = sz[1] + nhalo[2] + nhalo[3];
            int xlen = nhalo[0];
            int xsz = sz[0];
            auto r = range(zlen, ylen, xlen);
            auto _var = var.get_view();
            h.parallel_for(r, [=](item<3> idx) {
              int i = idx[2] + 1 - nhalo[0];
              int j = idx[1] + 1 - nhalo[2];
              int k = idx[0] + 1 - nhalo[4];
              _var(i, j, k) = _var(xsz + i, j, k);
            });
          })
          .wait();
    } else {
      mod_oneapi::queue
          .submit([&](handler &h) {
            int zlen = sz[2] + nhalo[4] + nhalo[5];
            int ylen = sz[1] + nhalo[2] + nhalo[3];
            int xlen = nhalo[1];
            int xsz = sz[0];
            auto r = range(zlen, ylen, xlen);
            auto _var = var.get_view();
            h.parallel_for(r, [=](item<3> idx) {
              int i = idx[2] + 1;
              int j = idx[1] + 1 - nhalo[2];
              int k = idx[0] + 1 - nhalo[4];
              _var(xsz + i, j, k) = _var(i, j, k);
            });
          })
          .wait();
    }
  }
}

void imposeNoSlipBC(int ibound, bool centered, Array3DH3 &var,
                    const fp *vel_crf = nullptr) {
  using namespace sycl;
  auto &nhalo = var.nhalo;
  auto &sz = var.size();
  fp bcvalue = (vel_crf != nullptr ? 0.0_fp - *vel_crf : 0.0_fp);
  if (ibound == 0) {
    if (centered) {
      mod_oneapi::queue
          .submit([&](handler &h) {
            int ylen = sz[1] + nhalo[2] + nhalo[3];
            int xlen = sz[0] + nhalo[0] + nhalo[1];
            auto r = range(ylen, xlen);
            auto _var = var.get_view();
            h.parallel_for(r, [=](item<2> idx) {
              int i = idx[1] + 1 - nhalo[0];
              int j = idx[0] + 1 - nhalo[2];
              _var(i, j, 0) = 2.0 * bcvalue - _var(i, j, 1);
            });
          })
          .wait();
    } else {
      mod_oneapi::queue
          .submit([&](handler &h) {
            int ylen = sz[1] + nhalo[2] + nhalo[3];
            int xlen = sz[0] + nhalo[0] + nhalo[1];
            auto r = range(ylen, xlen);
            auto _var = var.get_view();
            h.parallel_for(r, [=](item<2> idx) {
              int i = idx[1] + 1 - nhalo[0];
              int j = idx[0] + 1 - nhalo[2];
              _var(i, j, 0) = bcvalue;
            });
          })
          .wait();
    }
  } else {
    int n = sz[2];
    if (centered) {
      mod_oneapi::queue
          .submit([&](handler &h) {
            int ylen = sz[1] + nhalo[2] + nhalo[3];
            int xlen = sz[0] + nhalo[0] + nhalo[1];
            auto r = range(ylen, xlen);
            auto _var = var.get_view();
            h.parallel_for(r, [=](item<2> idx) {
              int i = idx[1] + 1 - nhalo[0];
              int j = idx[0] + 1 - nhalo[2];
              _var(i, j, n + 1) = 2.0 * bcvalue - _var(i, j, n);
            });
          })
          .wait();
    } else {
      mod_oneapi::queue
          .submit([&](handler &h) {
            int ylen = sz[1] + nhalo[2] + nhalo[3];
            int xlen = sz[0] + nhalo[0] + nhalo[1];
            auto r = range(ylen, xlen);
            auto _var = var.get_view();
            h.parallel_for(r, [=](item<2> idx) {
              int i = idx[1] + 1 - nhalo[0];
              int j = idx[0] + 1 - nhalo[2];
              _var(i, j, n) = bcvalue;
              _var(i, j, n + 1) = bcvalue;
            });
          })
          .wait();
    }
  }
}

void imposeZeroGradBC(int ibound, Array3DH1 &var) {
  using namespace sycl;
  auto &nhalo = var.nhalo;
  auto &sz = var.size();
  if (ibound == 0) {
    mod_oneapi::queue
        .submit([&](handler &h) {
          int ylen = sz[1] + nhalo[2] + nhalo[3];
          int xlen = sz[0] + nhalo[0] + nhalo[1];
          auto r = range(ylen, xlen);
          auto _var = var.get_view();
          h.parallel_for(r, [=](item<2> idx) {
            int i = idx[1] + 1 - nhalo[0];
            int j = idx[0] + 1 - nhalo[2];
            _var(i, j, 0) = _var(i, j, 1);
          });
        })
        .wait();
  } else {
    int n = sz[2];
    mod_oneapi::queue
        .submit([&](handler &h) {
          int ylen = sz[1] + nhalo[2] + nhalo[3];
          int xlen = sz[0] + nhalo[0] + nhalo[1];
          auto r = range(ylen, xlen);
          auto _var = var.get_view();
          h.parallel_for(r, [=](item<2> idx) {
            int i = idx[1] + 1 - nhalo[0];
            int j = idx[0] + 1 - nhalo[2];
            _var(i, j, n + 1) = _var(i, j, n);
          });
        })
        .wait();
  }
}
} // namespace

void imposeBCVel(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w, fp u_crf) {
  int ibound, idir;
  // B.C. in x direction
  idir = 1;
  if (mod_mpi::neighbor[0] == MPI_PROC_NULL) {
    ibound = 0;
    imposePeriodicBC(ibound, idir, u);
    imposePeriodicBC(ibound, idir, v);
    imposePeriodicBC(ibound, idir, w);
  }
  if (mod_mpi::neighbor[1] == MPI_PROC_NULL) {
    ibound = 1;
    imposePeriodicBC(ibound, idir, u);
    imposePeriodicBC(ibound, idir, v);
    imposePeriodicBC(ibound, idir, w);
  }
  // B.C. in y direction
  idir = 2;
  if (mod_mpi::neighbor[2] == MPI_PROC_NULL) {
    ibound = 0;
    imposePeriodicBC(ibound, idir, u);
    imposePeriodicBC(ibound, idir, v);
    imposePeriodicBC(ibound, idir, w);
  }
  if (mod_mpi::neighbor[3] == MPI_PROC_NULL) {
    ibound = 1;
    imposePeriodicBC(ibound, idir, u);
    imposePeriodicBC(ibound, idir, v);
    imposePeriodicBC(ibound, idir, w);
  }
  // B.C. in z direction
  idir = 3;
  if (mod_mpi::neighbor[4] == MPI_PROC_NULL) {
    ibound = 0;
    imposeNoSlipBC(ibound, true, u, &u_crf);
    imposeNoSlipBC(ibound, true, v);
    imposeNoSlipBC(ibound, false, w);
  }
  if (mod_mpi::neighbor[5] == MPI_PROC_NULL) {
    ibound = 1;
    imposeNoSlipBC(ibound, true, u, &u_crf);
    imposeNoSlipBC(ibound, true, v);
    imposeNoSlipBC(ibound, false, w);
  }
}

void updateBoundVel(fp u_crf, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w) {
  GPTLstart("--Update halo vel");
  auto ev1 = u.sync_send_buf();
  auto ev2 = v.sync_send_buf();
  auto ev3 = w.sync_send_buf();
  sycl::event::wait(ev1);
  updateHalo(mod_mpi::halotype_vel, u);
  ev1 = u.sync_recv_buf();
  sycl::event::wait(ev2);
  updateHalo(mod_mpi::halotype_vel, v);
  ev2 = v.sync_recv_buf();
  sycl::event::wait(ev3);
  updateHalo(mod_mpi::halotype_vel, w);
  ev3 = w.sync_recv_buf();
  sycl::event::wait(ev1);
  sycl::event::wait(ev2);
  sycl::event::wait(ev3);
  GPTLstop("--Update halo vel");

  GPTLstart("--Impose BC vel");
  imposeBCVel(u, v, w, u_crf);
  GPTLstop("--Impose BC vel");
}

void updateBoundP(Array3DH1 &p) {
  int ibound, idir;

  GPTLstart("--Update halo pres");
  auto ev = p.sync_send_buf();
  sycl::event::wait(ev);
  updateHalo(mod_mpi::halotype_one, p);
  ev = p.sync_recv_buf();
  sycl::event::wait(ev);
  GPTLstop("--Update halo pres");

  GPTLstart("--Impose BC pres");
  // B.C. in x direction
  idir = 1;
  if (mod_mpi::neighbor[0] == MPI_PROC_NULL) {
    ibound = 0;
    imposePeriodicBC(ibound, idir, p);
  }
  if (mod_mpi::neighbor[1] == MPI_PROC_NULL) {
    ibound = 1;
    imposePeriodicBC(ibound, idir, p);
  }
  // B.C. in y direction
  idir = 2;
  if (mod_mpi::neighbor[2] == MPI_PROC_NULL) {
    ibound = 0;
    imposePeriodicBC(ibound, idir, p);
  }
  if (mod_mpi::neighbor[3] == MPI_PROC_NULL) {
    ibound = 1;
    imposePeriodicBC(ibound, idir, p);
  }
  // B.C. in z direction
  idir = 3;
  if (mod_mpi::neighbor[4] == MPI_PROC_NULL) {
    ibound = 0;
    imposeZeroGradBC(ibound, p);
  }
  if (mod_mpi::neighbor[5] == MPI_PROC_NULL) {
    ibound = 1;
    imposeZeroGradBC(ibound, p);
  }
  GPTLstop("--Impose BC pres");
}
} // namespace mod_updateBound