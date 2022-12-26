#include "mod_calcVel.hh"
#include "gptl.hh"
#include "mod_mesh.hh"
#include "mod_oneapi.hh"
#include "mod_parameters.hh"
#include "mod_updateBound.hh"
#include "mod_utils.hh"
#include "mpi.h"

#include <array>

namespace mod_calcVel {
void transform2CRF(fp vel_crf, Array3DH3 &vel, fp &vel_force) {
  if (vel_crf != 0) {
    auto &sz = vel.size();
    auto _vel = vel.get_view();
    auto ev = mod_oneapi::queue.parallel_for(
        sycl::range(sz[2], sz[1], sz[0]), [=](sycl::item<3> idx) {
          auto [i, j, k] = mod_oneapi::unpack<1>(idx);
          _vel(i, j, k) -= vel_crf;
        });
    ev.wait();
    vel_force -= vel_crf;
  }
}

void timeIntVelRK1Kernel(std::array<int, 3> &st, std::array<int, 3> &en,
                         fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
                         const Array3DH3 &w, Array3DH3 &u1, Array3DH3 &v1,
                         Array3DH3 &w1) {
  using namespace mod_mesh;
  using namespace sycl;
  fp _dx_inv = dx_inv;
  fp _dy_inv = dy_inv;
  fp _re_inv = re_inv;
  fp _dt = dt;

  auto &sz = u.size();
  auto r = range(sz[2], sz[1], sz[0]);
  mod_oneapi::queue
      .submit([&](handler &h) {
        auto _u = u.get_view();
        auto _v = v.get_view();
        auto _w = w.get_view();
        auto _u1 = u1.get_view();
        auto _v1 = v1.get_view();
        auto _w1 = w1.get_view();
        auto _dzf = dzf.get_view();
        auto _dzf_inv = dzf_inv.get_view();
        auto _dzc_inv = dzc_inv.get_view();
        h.parallel_for(r, [=](item<3> idx) {
          int i = idx[2] + 1;
          int j = idx[1] + 1;
          int k = idx[0] + 1;
          fp uw, ue, us, un, ub, ut;
          fp ww, we, ws, wn, wb, wt;
          fp vw, ve, vs, vn, vb, vt;
          fp dqw, dqe, dqs, dqn, dqb, dqt;
          fp conv, visc;

          ue  = (_u(i  ,j  ,k  )+_u(i+1,j  ,k  ));
          uw  = (_u(i-1,j  ,k  )+_u(i  ,j  ,k  ));
          un  = (_u(i  ,j  ,k  )+_u(i  ,j+1,k  ));
          us  = (_u(i  ,j-1,k  )+_u(i  ,j  ,k  ));
          ut  = (_u(i  ,j  ,k  )*_dzf(k+1) + _u(i,j,k+1)*_dzf(k))*_dzc_inv(k  );
          ub  = (_u(i  ,j  ,k  )*_dzf(k-1) + _u(i,j,k-1)*_dzf(k))*_dzc_inv(k-1);
          vn  = (_v(i  ,j  ,k  )+_v(i+1,j  ,k  ));
          vs  = (_v(i  ,j-1,k  )+_v(i+1,j-1,k  ));
          wt  = (_w(i  ,j  ,k  )+_w(i+1,j  ,k  ));
          wb  = (_w(i  ,j  ,k-1)+_w(i+1,j  ,k-1));
          dqe = (_u(i+1,j  ,k  )-_u(i  ,j  ,k  ))*_dx_inv;
          dqw = (_u(i  ,j  ,k  )-_u(i-1,j  ,k  ))*_dx_inv;
          dqn = (_u(i  ,j+1,k  )-_u(i  ,j  ,k  ))*_dy_inv;
          dqs = (_u(i  ,j  ,k  )-_u(i  ,j-1,k  ))*_dy_inv;
          dqt = (_u(i  ,j  ,k+1)-_u(i  ,j  ,k  ))*_dzc_inv(k  );
          dqb = (_u(i  ,j  ,k  )-_u(i  ,j  ,k-1))*_dzc_inv(k-1);
          conv = 0.25*( (ue*ue-uw*uw)*_dx_inv + (un*vn-us*vs)*_dy_inv + (ut*wt-ub*wb)*_dzf_inv(k) );
          // add a term induced by the convecting reference frame;
          conv = conv + 0.5*u_crf*(ue-uw)*_dx_inv;
          visc = ((dqe-dqw)*_dx_inv + (dqn-dqs)*_dy_inv + (dqt-dqb)*_dzf_inv(k))*_re_inv;
          _u1(i, j, k) = _u(i, j, k) + _dt * (visc - conv);

          ve  = (_v(i  ,j  ,k  )+_v(i+1,j  ,k  ));
          vw  = (_v(i-1,j  ,k  )+_v(i  ,j  ,k  ));
          vn  = (_v(i  ,j  ,k  )+_v(i  ,j+1,k  ));
          vs  = (_v(i  ,j-1,k  )+_v(i  ,j  ,k  ));
          vt  = (_v(i  ,j  ,k  )*_dzf(k+1) + _v(i,j,k+1)*_dzf(k))*_dzc_inv(k  );
          vb  = (_v(i  ,j  ,k  )*_dzf(k-1) + _v(i,j,k-1)*_dzf(k))*_dzc_inv(k-1);
          ue  = (_u(i  ,j  ,k  )+_u(i  ,j+1,k  ));
          uw  = (_u(i-1,j  ,k  )+_u(i-1,j+1,k  ));
          wt  = (_w(i  ,j  ,k  )+_w(i  ,j+1,k  ));
          wb  = (_w(i  ,j  ,k-1)+_w(i  ,j+1,k-1));
          dqe = (_v(i+1,j  ,k  )-_v(i  ,j  ,k  ))*_dx_inv;
          dqw = (_v(i  ,j  ,k  )-_v(i-1,j  ,k  ))*_dx_inv;
          dqn = (_v(i  ,j+1,k  )-_v(i  ,j  ,k  ))*_dy_inv;
          dqs = (_v(i  ,j  ,k  )-_v(i  ,j-1,k  ))*_dy_inv;
          dqt = (_v(i  ,j  ,k+1)-_v(i  ,j  ,k  ))*_dzc_inv(k  );
          dqb = (_v(i  ,j  ,k  )-_v(i  ,j  ,k-1))*_dzc_inv(k-1);
          conv = 0.25*( (ue*ve-uw*vw)*_dx_inv + (vn*vn-vs*vs)*_dy_inv + (wt*vt-wb*vb)*_dzf_inv(k) );
          // add a term induced by the convecting reference frame;
          conv = conv + 0.5*u_crf*(ve-vw)*_dx_inv;
          visc = ((dqe-dqw)*_dx_inv + (dqn-dqs)*_dy_inv + (dqt-dqb)*_dzf_inv(k))*_re_inv;
          _v1(i, j, k) = _v(i, j, k) + _dt * (visc - conv);

          we  = (_w(i  ,j  ,k  )+_w(i+1,j  ,k  ));
          ww  = (_w(i-1,j  ,k  )+_w(i  ,j  ,k  ));
          wn  = (_w(i  ,j  ,k  )+_w(i  ,j+1,k  ));
          ws  = (_w(i  ,j-1,k  )+_w(i  ,j  ,k  ));
          wt  = (_w(i  ,j  ,k  )+_w(i  ,j  ,k+1));
          wb  = (_w(i  ,j  ,k  )+_w(i  ,j  ,k-1));
          ue  = (_u(i  ,j  ,k  )*_dzf(k+1) + _u(i  ,j  ,k+1)*_dzf(k))*_dzc_inv(k  );
          uw  = (_u(i-1,j  ,k  )*_dzf(k+1) + _u(i-1,j  ,k+1)*_dzf(k))*_dzc_inv(k  );
          vn  = (_v(i  ,j  ,k  )*_dzf(k+1) + _v(i  ,j  ,k+1)*_dzf(k))*_dzc_inv(k  );
          vs  = (_v(i  ,j-1,k  )*_dzf(k+1) + _v(i  ,j-1,k+1)*_dzf(k))*_dzc_inv(k  );
          dqe = (_w(i+1,j  ,k  )-_w(i  ,j  ,k  ))*_dx_inv;
          dqw = (_w(i  ,j  ,k  )-_w(i-1,j  ,k  ))*_dx_inv;
          dqn = (_w(i  ,j+1,k  )-_w(i  ,j  ,k  ))*_dy_inv;
          dqs = (_w(i  ,j  ,k  )-_w(i  ,j-1,k  ))*_dy_inv;
          dqt = (_w(i  ,j  ,k+1)-_w(i  ,j  ,k  ))*_dzf_inv(k+1);
          dqb = (_w(i  ,j  ,k  )-_w(i  ,j  ,k-1))*_dzf_inv(k);
          conv = 0.25*( (ue*we-uw*ww)*_dx_inv + (vn*wn-vs*ws)*_dy_inv + (wt*wt-wb*wb)*_dzc_inv(k) );
          // add a term induced by the convecting reference frame;
          conv = conv + 0.5*u_crf*(we-ww)*_dx_inv;
          visc = ((dqe-dqw)*_dx_inv + (dqn-dqs)*_dy_inv + (dqt-dqb)*_dzc_inv(k))*_re_inv;
          _w1(i, j, k) = _w(i, j, k) + _dt * (visc - conv);
        });
      })
      .wait();
}

void timeIntVelRK1(fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
                   const Array3DH3 &w, Array3DH3 &u1, Array3DH3 &v1,
                   Array3DH3 &w1) {
  std::array<int, 3> st, en;
  st.fill(1);
  en = u.size();
  timeIntVelRK1Kernel(st, en, u_crf, u, v, w, u1, v1, w1);
}

void timeIntVelRK2Kernel(std::array<int, 3> &st, std::array<int, 3> &en,
                         fp u_crf, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w,
                         const Array3DH3 &u1, const Array3DH3 &v1,
                         const Array3DH3 &w1) {
  using namespace mod_mesh;
  using namespace sycl;
  fp _dx_inv = dx_inv;
  fp _dy_inv = dy_inv;
  fp _re_inv = re_inv;
  fp _dt = dt;

  auto &sz = u.size();
  auto r = range(sz[2], sz[1], sz[0]);
  mod_oneapi::queue
      .submit([&](handler &h) {
        auto _u = u.get_view();
        auto _v = v.get_view();
        auto _w = w.get_view();
        auto _u1 = u1.get_view();
        auto _v1 = v1.get_view();
        auto _w1 = w1.get_view();
        auto _dzf = dzf.get_view();
        auto _dzf_inv = dzf_inv.get_view();
        auto _dzc_inv = dzc_inv.get_view();
        h.parallel_for(r, [=](item<3> idx) {
          int i = idx[2] + 1;
          int j = idx[1] + 1;
          int k = idx[0] + 1;
          fp uw, ue, us, un, ub, ut;
          fp ww, we, ws, wn, wb, wt;
          fp vw, ve, vs, vn, vb, vt;
          fp dqw, dqe, dqs, dqn, dqb, dqt;
          fp conv, visc;

          ue  = (_u1(i  ,j  ,k  )+_u1(i+1,j  ,k  ));
          uw  = (_u1(i-1,j  ,k  )+_u1(i  ,j  ,k  ));
          un  = (_u1(i  ,j  ,k  )+_u1(i  ,j+1,k  ));
          us  = (_u1(i  ,j-1,k  )+_u1(i  ,j  ,k  ));
          ut  = (_u1(i  ,j  ,k  )*_dzf(k+1) + _u1(i,j,k+1)*_dzf(k))*_dzc_inv(k  );
          ub  = (_u1(i  ,j  ,k  )*_dzf(k-1) + _u1(i,j,k-1)*_dzf(k))*_dzc_inv(k-1);
          vn  = (_v1(i  ,j  ,k  )+_v1(i+1,j  ,k  ));
          vs  = (_v1(i  ,j-1,k  )+_v1(i+1,j-1,k  ));
          wt  = (_w1(i  ,j  ,k  )+_w1(i+1,j  ,k  ));
          wb  = (_w1(i  ,j  ,k-1)+_w1(i+1,j  ,k-1));
          dqe = (_u1(i+1,j  ,k  )-_u1(i  ,j  ,k  ))*_dx_inv;
          dqw = (_u1(i  ,j  ,k  )-_u1(i-1,j  ,k  ))*_dx_inv;
          dqn = (_u1(i  ,j+1,k  )-_u1(i  ,j  ,k  ))*_dy_inv;
          dqs = (_u1(i  ,j  ,k  )-_u1(i  ,j-1,k  ))*_dy_inv;
          dqt = (_u1(i  ,j  ,k+1)-_u1(i  ,j  ,k  ))*_dzc_inv(k  );
          dqb = (_u1(i  ,j  ,k  )-_u1(i  ,j  ,k-1))*_dzc_inv(k-1);
          conv = 0.25*( (ue*ue-uw*uw)*_dx_inv + (un*vn-us*vs)*_dy_inv + (ut*wt-ub*wb)*_dzf_inv(k) );
          // add a term induced by the convecting reference frame;
          conv = conv + 0.5*u_crf*(ue-uw)*_dx_inv;
          visc = ((dqe-dqw)*_dx_inv + (dqn-dqs)*_dy_inv + (dqt-dqb)*_dzf_inv(k))*_re_inv;
          _u(i, j, k) = (_u1(i, j, k) + _dt * (visc - conv) + _u(i, j, k)) * 0.5;

          ve  = (_v1(i  ,j  ,k  )+_v1(i+1,j  ,k  ));
          vw  = (_v1(i-1,j  ,k  )+_v1(i  ,j  ,k  ));
          vn  = (_v1(i  ,j  ,k  )+_v1(i  ,j+1,k  ));
          vs  = (_v1(i  ,j-1,k  )+_v1(i  ,j  ,k  ));
          vt  = (_v1(i  ,j  ,k  )*_dzf(k+1) + _v1(i,j,k+1)*_dzf(k))*_dzc_inv(k  );
          vb  = (_v1(i  ,j  ,k  )*_dzf(k-1) + _v1(i,j,k-1)*_dzf(k))*_dzc_inv(k-1);
          ue  = (_u1(i  ,j  ,k  )+_u1(i  ,j+1,k  ));
          uw  = (_u1(i-1,j  ,k  )+_u1(i-1,j+1,k  ));
          wt  = (_w1(i  ,j  ,k  )+_w1(i  ,j+1,k  ));
          wb  = (_w1(i  ,j  ,k-1)+_w1(i  ,j+1,k-1));
          dqe = (_v1(i+1,j  ,k  )-_v1(i  ,j  ,k  ))*_dx_inv;
          dqw = (_v1(i  ,j  ,k  )-_v1(i-1,j  ,k  ))*_dx_inv;
          dqn = (_v1(i  ,j+1,k  )-_v1(i  ,j  ,k  ))*_dy_inv;
          dqs = (_v1(i  ,j  ,k  )-_v1(i  ,j-1,k  ))*_dy_inv;
          dqt = (_v1(i  ,j  ,k+1)-_v1(i  ,j  ,k  ))*_dzc_inv(k  );
          dqb = (_v1(i  ,j  ,k  )-_v1(i  ,j  ,k-1))*_dzc_inv(k-1);
          conv = 0.25*( (ue*ve-uw*vw)*_dx_inv + (vn*vn-vs*vs)*_dy_inv + (wt*vt-wb*vb)*_dzf_inv(k) );
          // add a term induced by the convecting reference frame;
          conv = conv + 0.5*u_crf*(ve-vw)*_dx_inv;
          visc = ((dqe-dqw)*_dx_inv + (dqn-dqs)*_dy_inv + (dqt-dqb)*_dzf_inv(k))*_re_inv;
          _v(i, j, k) = (_v1(i, j, k) + _dt * (visc - conv) + _v(i, j, k)) * 0.5;

          we  = (_w1(i  ,j  ,k  )+_w1(i+1,j  ,k  ));
          ww  = (_w1(i-1,j  ,k  )+_w1(i  ,j  ,k  ));
          wn  = (_w1(i  ,j  ,k  )+_w1(i  ,j+1,k  ));
          ws  = (_w1(i  ,j-1,k  )+_w1(i  ,j  ,k  ));
          wt  = (_w1(i  ,j  ,k  )+_w1(i  ,j  ,k+1));
          wb  = (_w1(i  ,j  ,k  )+_w1(i  ,j  ,k-1));
          ue  = (_u1(i  ,j  ,k  )*_dzf(k+1) + _u1(i  ,j  ,k+1)*_dzf(k))*_dzc_inv(k  );
          uw  = (_u1(i-1,j  ,k  )*_dzf(k+1) + _u1(i-1,j  ,k+1)*_dzf(k))*_dzc_inv(k  );
          vn  = (_v1(i  ,j  ,k  )*_dzf(k+1) + _v1(i  ,j  ,k+1)*_dzf(k))*_dzc_inv(k  );
          vs  = (_v1(i  ,j-1,k  )*_dzf(k+1) + _v1(i  ,j-1,k+1)*_dzf(k))*_dzc_inv(k  );
          dqe = (_w1(i+1,j  ,k  )-_w1(i  ,j  ,k  ))*_dx_inv;
          dqw = (_w1(i  ,j  ,k  )-_w1(i-1,j  ,k  ))*_dx_inv;
          dqn = (_w1(i  ,j+1,k  )-_w1(i  ,j  ,k  ))*_dy_inv;
          dqs = (_w1(i  ,j  ,k  )-_w1(i  ,j-1,k  ))*_dy_inv;
          dqt = (_w1(i  ,j  ,k+1)-_w1(i  ,j  ,k  ))*_dzf_inv(k+1);
          dqb = (_w1(i  ,j  ,k  )-_w1(i  ,j  ,k-1))*_dzf_inv(k);
          conv = 0.25*( (ue*we-uw*ww)*_dx_inv + (vn*wn-vs*ws)*_dy_inv + (wt*wt-wb*wb)*_dzc_inv(k) );
          // add a term induced by the convecting reference frame;
          conv = conv + 0.5*u_crf*(we-ww)*_dx_inv;
          visc = ((dqe-dqw)*_dx_inv + (dqn-dqs)*_dy_inv + (dqt-dqb)*_dzc_inv(k))*_re_inv;
          _w(i, j, k) = (_w1(i, j, k) + _dt * (visc - conv) + _w(i, j, k)) * 0.5;
        });
      })
      .wait();
}

void timeIntVelRK2(fp u_crf, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w,
                   const Array3DH3 &u1, const Array3DH3 &v1,
                   const Array3DH3 &w1) {
  std::array<int, 3> st, en;
  st.fill(1);
  en = u.size();
  timeIntVelRK2Kernel(st, en, u_crf, u, v, w, u1, v1, w1);
}

void correctVel(const Array3DH1 &p, Array3DH3 &u, Array3DH3 &v, Array3DH3 &w) {
  using namespace mod_mesh;
  using namespace sycl;
  fp dtdxi = dt * dx_inv;
  fp dtdyi = dt * dy_inv;
  fp _dt = dt;
  auto &sz = u.size();
  auto r = range(sz[2], sz[1], sz[0]);
  mod_oneapi::queue
      .submit([&](handler &h) {
        auto _u = u.get_view();
        auto _v = v.get_view();
        auto _w = w.get_view();
        auto _p = p.get_view();
        auto _dzc_inv = dzc_inv.get_view();
        h.parallel_for(r, [=](item<3> idx) {
          int i = idx[2] + 1;
          int j = idx[1] + 1;
          int k = idx[0] + 1;
          _u(i, j, k) -= (_p(i + 1, j, k) - _p(i, j, k)) * dtdxi;
          _v(i, j, k) -= (_p(i, j + 1, k) - _p(i, j, k)) * dtdyi;
          _w(i, j, k) -= (_p(i, j, k + 1) - _p(i, j, k)) * _dt * _dzc_inv(k);
        });
      })
      .wait();
}

void forceVel(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w) {
  using namespace mod_mesh;
  using namespace sycl;
  fp vel_mean_x, vel_mean_y, vel_mean_z;
  fp force_x, force_y, force_z;
  // TODO: hope we can handle MPI with a more async way
  if (is_forced[0])
    vel_mean_x = mod_utils::Mean(nx, ny, dzflzi, u);
  if (is_forced[1])
    vel_mean_y = mod_utils::Mean(nx, ny, dzflzi, v);
  if (is_forced[2])
    vel_mean_z = mod_utils::Mean(nx, ny, dzflzi, w);

  sycl::event u_ev, v_ev, w_ev;
  auto &sz = u.size();
  auto r = range(sz[2], sz[1], sz[0]);
  if (is_forced[0]) {
    force_x = vel_force[0] - vel_mean_x;
    auto _u = u.get_view();
    u_ev = mod_oneapi::queue.parallel_for(r, [=](item<3> idx) {
      _u(idx[2] + 1, idx[1] + 1, idx[0] + 1) += force_x;
    });
  }
  if (is_forced[1]) {
    force_y = vel_force[1] - vel_mean_y;
    auto _v = v.get_view();
    v_ev = mod_oneapi::queue.parallel_for(r, [=](item<3> idx) {
      _v(idx[2] + 1, idx[1] + 1, idx[0] + 1) += force_y;
    });
  }
  if (is_forced[2]) {
    force_z = vel_force[2] - vel_mean_z;
    auto _w = w.get_view();
    w_ev = mod_oneapi::queue.parallel_for(r, [=](item<3> idx) {
      _w(idx[2] + 1, idx[1] + 1, idx[0] + 1) += force_z;
    });
  }
  u_ev.wait();
  v_ev.wait();
  w_ev.wait();
}
} // namespace mod_calcVel