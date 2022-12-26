#include "mod_fft.hh"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mod_oneapi.hh"
#include <oneapi/mkl/dfti.hpp>

using namespace oneapi::mkl::dft;
using onemkl_desc = descriptor<precision::DOUBLE, domain::REAL>;

namespace {
std::unique_ptr<onemkl_desc> x_desc, y_desc;
double *xpen_buffer, *ypen_buffer;
std::array<int, 3> xsz, ysz;
int xstride, ystride;
int x_howmany, y_howmany;
double fft_normfactor;
} // namespace

static bool bctype_valid(const char *bctype) {
  return strcmp(bctype, "PP") == 0;
}

void init_fft(int _xsz[3], int _ysz[3], char bctype_x[], char bctype_y[]) {
  assert(bctype_valid(bctype_x));
  assert(bctype_valid(bctype_y));
  memcpy(xsz.data(), _xsz, sizeof(int) * 3);
  memcpy(ysz.data(), _ysz, sizeof(int) * 3);
  int x_n = xsz[0];
  int y_n = ysz[1];
  x_howmany = xsz[1] * xsz[2];
  y_howmany = ysz[0] * ysz[2];
  xstride = x_n + 2;
  ystride = y_n + 2;
  MKL_LONG strides[2] = {0, 1};
  x_desc = std::make_unique<onemkl_desc>(x_n);
  y_desc = std::make_unique<onemkl_desc>(y_n);
  x_desc->set_value(config_param::NUMBER_OF_TRANSFORMS, x_howmany);
  x_desc->set_value(config_param::INPUT_STRIDES, strides);
  x_desc->set_value(config_param::OUTPUT_STRIDES, strides);
  x_desc->set_value(config_param::FWD_DISTANCE, x_n + 2);
  x_desc->set_value(config_param::BWD_DISTANCE, x_n / 2 + 1);
  x_desc->commit(mod_oneapi::queue);

  y_desc->set_value(config_param::NUMBER_OF_TRANSFORMS, y_howmany);
  y_desc->set_value(config_param::INPUT_STRIDES, strides);
  y_desc->set_value(config_param::OUTPUT_STRIDES, strides);
  y_desc->set_value(config_param::FWD_DISTANCE, y_n + 2);
  y_desc->set_value(config_param::BWD_DISTANCE, y_n / 2 + 1);
  y_desc->commit(mod_oneapi::queue);

  xpen_buffer =
      sycl::malloc_device<double>(xstride * x_howmany, mod_oneapi::queue);
  ypen_buffer =
      sycl::malloc_device<double>(ystride * y_howmany, mod_oneapi::queue);

  fft_normfactor = 1.0 / (xsz[0] * ysz[1]);
}

void execute_fft_fwd_xpen(const Array3DH1 &p, double *var_xpen) {
  // copy inner part of p into xpen_buffer
  auto _p = p.get_view();
  auto _xpen_buffer = xpen_buffer;
  auto _xstride = xstride;
  auto &sz = p.size();
  auto r = sycl::range(sz[2], sz[1], sz[0]);
  auto ev1 = mod_oneapi::queue.parallel_for(r, [=](sycl::item<3> idx) {
    int i = idx[2];
    int j = idx[1];
    int k = idx[0];
    int ysz = idx.get_range()[1];
    _xpen_buffer[(k * ysz + j) * _xstride + i] = _p(i + 1, j + 1, k + 1);
  });
  ev1.wait();
  // do fft
  auto ev2 = oneapi::mkl::dft::compute_forward(*x_desc, _xpen_buffer, {ev1});
  ev2.wait();
  // copy from xpen_buffer to var_xpen and do post-processing
  int num = xsz[0];
  auto ev3 = mod_oneapi::queue.parallel_for(
      sycl::range(x_howmany, num), ev2, [=](sycl::item<2> idx) {
        int i = idx[1];
        int j = idx[0];
        int idx_m = (i == 1 ? num : i);
        var_xpen[idx.get_linear_id()] = _xpen_buffer[j * _xstride + idx_m];
      });
  ev3.wait();
}

void execute_fft_bwd_xpen(const double *var_xpen, Array3DH1 &p) {
  int num = xsz[0];
  auto _xstride = xstride;
  auto _xpen_buffer = xpen_buffer;
  auto ev1 = mod_oneapi::queue.memset(xpen_buffer, 0,
                                      _xstride * x_howmany * sizeof(double));
  ev1.wait();
  auto ev2 = mod_oneapi::queue.parallel_for(
      sycl::range(x_howmany, xsz[0]), ev1, [=](sycl::item<2> idx) {
        int i = idx[1];
        int j = idx[0];
        int idx_m = (i == 1 ? num : i);
        _xpen_buffer[j * _xstride + idx_m] = var_xpen[idx.get_linear_id()];
      });
  ev2.wait();
  auto ev3 = oneapi::mkl::dft::compute_backward(*x_desc, _xpen_buffer, {ev2});
  ev3.wait();
  auto _fft_normfactor = fft_normfactor;
  auto _p = p.get_view();
  auto &sz = p.size();
  auto r = sycl::range(sz[2], sz[1], sz[0]);
  auto ev4 = mod_oneapi::queue.parallel_for(r, ev3, [=](sycl::item<3> idx) {
    int i = idx[2];
    int j = idx[1];
    int k = idx[0];
    int ysz = idx.get_range()[1];
    _p(i + 1, j + 1, k + 1) =
        _fft_normfactor * _xpen_buffer[(k * ysz + j) * _xstride + i];
  });
  ev4.wait();
}

void execute_fft_fwd_ypen(double *var_ypen) {
  // transpose var_ypen into ypen_buffer
  auto _ypen_buffer = ypen_buffer;
  auto ypen_stride1 = ysz[0];
  auto ypen_stride2 = ysz[0] * ysz[1];
  auto buffer_stride1 = ystride;
  auto buffer_stride2 = ystride * ysz[0];
  auto r = sycl::range(ysz[2], ysz[0], ysz[1]);
  auto ev1 = mod_oneapi::queue.parallel_for(r, [=](sycl::item<3> idx) {
    int i = idx[2];
    int j = idx[1];
    int k = idx[0];
    _ypen_buffer[k * buffer_stride2 + j * buffer_stride1 + i] =
        var_ypen[k * ypen_stride2 + j + i * ypen_stride1];
  });
  ev1.wait();
  // do fft
  auto ev2 = oneapi::mkl::dft::compute_forward(*y_desc, _ypen_buffer, {ev1});
  ev2.wait();
  // do post-processing and transpose back to var_ypen
  r = sycl::range(ysz[2], ysz[1], ysz[0]);
  int num = ysz[1];
  auto ev3 = mod_oneapi::queue.parallel_for(r, ev2, [=](sycl::item<3> idx) {
    int i = idx[2];
    int j = idx[1];
    int k = idx[0];
    int idx_m = (j == 1 ? num : j);
    var_ypen[k * ypen_stride2 + j * ypen_stride1 + i] =
        _ypen_buffer[k * buffer_stride2 + idx_m + i * buffer_stride1];
  });
  ev3.wait();
}

void execute_fft_bwd_ypen(double *var_ypen) {
  auto _ypen_buffer = ypen_buffer;
  auto ypen_stride1 = ysz[0];
  auto ypen_stride2 = ysz[0] * ysz[1];
  auto buffer_stride1 = ystride;
  auto buffer_stride2 = ystride * ysz[0];
  auto ev1 = mod_oneapi::queue.memset(ypen_buffer, 0,
                                      ystride * y_howmany * sizeof(double));
  ev1.wait();
  auto r = sycl::range(ysz[2], ysz[0], ysz[1]);
  int num = ysz[1];
  auto ev2 = mod_oneapi::queue.parallel_for(r, ev1, [=](sycl::item<3> idx) {
    int i = idx[2];
    int j = idx[1];
    int k = idx[0];
    int idx_m = (i == 1 ? num : i);
    _ypen_buffer[k * buffer_stride2 + j * buffer_stride1 + idx_m] =
        var_ypen[k * ypen_stride2 + j + i * ypen_stride1];
  });
  ev2.wait();
  // do fft
  auto ev3 = oneapi::mkl::dft::compute_backward(*y_desc, _ypen_buffer, {ev2});
  ev3.wait();
  r = sycl::range(ysz[2], ysz[1], ysz[0]);
  auto ev4 = mod_oneapi::queue.parallel_for(r, ev3, [=](sycl::item<3> idx) {
    int i = idx[2];
    int j = idx[1];
    int k = idx[0];
    var_ypen[k * ypen_stride2 + j * ypen_stride1 + i] =
        _ypen_buffer[k * buffer_stride2 + j + i * buffer_stride1];
  });
  ev4.wait();
}

void free_fft() {
  x_desc.reset();
  y_desc.reset();
  sycl::free(xpen_buffer, mod_oneapi::queue);
  sycl::free(ypen_buffer, mod_oneapi::queue);
}

void get_eigen_values(int ist, int isz, int isz_global, char bctype[],
                      double *lambda) {
  assert(bctype_valid(bctype));
  const double pi = acos(-1.0);
  int ien = ist + isz - 1;

  int n = isz_global;
  double *lambda_glb = (double *)malloc(n * sizeof(double));
  double *lambda_aux = (double *)malloc(n * sizeof(double));

  for (int i = 0; i < n; i++) {
    lambda_aux[i] = 2.0 * (cos(2.0 * pi * (i - 0.0) / n) - 1.0);
  }

  lambda_glb[0] = lambda_aux[0];
  lambda_glb[1] = lambda_aux[n / 2];
  for (int i = 2; i < n; i++) {
    lambda_glb[i] = lambda_aux[i / 2];
  }

  for (int i = ist; i <= ien; i++) {
    lambda[i - ist] = lambda_glb[i - 1];
  }

  free(lambda_glb);
  free(lambda_aux);
}
