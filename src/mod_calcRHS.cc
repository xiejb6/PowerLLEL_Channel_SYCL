#include "mod_calcRHS.hh"
#include "mod_mesh.hh"
#include "mod_mpi.hh"
#include "mod_parameters.hh"

namespace mod_calcRHS {
void calcRHS(const Array3DH3 &u, const Array3DH3 &v, const Array3DH3 &w,
             Array3DH1 &rhs) {
  using namespace mod_mesh;
  using namespace sycl;
  fp dti = 1.0_fp / dt;
  fp dtidxi = dti * dx_inv;
  fp dtidyi = dti * dy_inv;
  auto &sz = u.size();
  auto r = range(sz[2], sz[1], sz[0]);
  mod_oneapi::queue
      .submit([&](handler &h) {
        auto _u = u.get_view();
        auto _v = v.get_view();
        auto _w = w.get_view();
        auto _dzf_inv = dzf_inv.get_view();
        auto _rhs = rhs.get_view();
        h.parallel_for(r, [=](item<3> idx) {
          int i = idx[2] + 1;
          int j = idx[1] + 1;
          int k = idx[0] + 1;
          _rhs(i, j, k) = (_u(i, j, k) - _u(i - 1, j, k)) * dtidxi +
                          (_v(i, j, k) - _v(i, j - 1, k)) * dtidyi +
                          (_w(i, j, k) - _w(i, j, k - 1)) * dti * _dzf_inv(k);
        });
      })
      .wait();
}

} // namespace mod_calcRHS