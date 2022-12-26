#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <memory>

#include "mod_oneapi.hh"

#ifdef SINGLE_PREC
using fp = float;
#else
using fp = double;
#endif

template <typename T, int lhalo, int rhalo> class _Array1D {
  static_assert(lhalo >= 0);
  static_assert(rhalo >= 0);

public:
  constexpr static std::array<int, 2> nhalo = {lhalo, rhalo};
  // call allocate() next
  ~_Array1D() { deallocate(); }
  _Array1D() : owner(nullptr), ptr(nullptr), sz(0) {}
  _Array1D(int sz) : owner(nullptr), ptr(nullptr), sz(0) { allocate(sz); }
  _Array1D(const _Array1D &rhs) = delete;
  _Array1D(_Array1D &&rhs) = delete;
  _Array1D &operator=(_Array1D &&rhs) = delete;
  void allocate(int sz) {
    assert(sz > 0);
    this->sz = sz;
    owner = sycl::malloc_shared<T>(sz + lhalo + rhalo, mod_oneapi::queue);
    assert(owner);
    memset(owner, 0, sizeof(T) * (sz + lhalo + rhalo));
    ptr = owner + lhalo - 1;
  }
  void deallocate() {
    if (owner) {
      sycl::free(owner, mod_oneapi::queue);
      owner = nullptr;
    }
  }

  // size without halos
  int size() const { return sz; }
  // total buffer with halos
  T *data() { return owner; }
  const T *data() const { return owner; }
  // start with 1
  T &operator()(int x) { return ptr[x]; }
  const T &operator()(int x) const { return ptr[x]; }

  struct View {
    T *ptr;

    T &operator()(int x) { return ptr[x]; }
    T &operator()(int x) const { return ptr[x]; }
  };

  View get_view() { return View{ptr}; }
  View get_view() const { return View{ptr}; }

private:
  T *owner;
  T *ptr;
  int sz;
};

// For host send/recv buffer: [0]north, [1]south, [2]top, [3]bottom
//    [0]north
//    [1]south
//    [2]top
//    [3]bottom
// For device gather/scatter buffer:
//    [0]north send
//    [1]south send
//    [2]north recv
//    [3]south recv
template <typename T> using HaloBuffer = std::array<T *, 4>;

template <typename T, int xlhalo, int xrhalo, int ylhalo, int yrhalo,
          int zlhalo, int zrhalo>
class _Array3D {
  static_assert(xlhalo >= 0);
  static_assert(xrhalo >= 0);
  static_assert(ylhalo >= 0);
  static_assert(yrhalo >= 0);
  static_assert(zlhalo >= 0);
  static_assert(zrhalo >= 0);

public:
  constexpr static std::array<int, 6> nhalo = {xlhalo, xrhalo, ylhalo,
                                               yrhalo, zlhalo, zrhalo};
  // call allocate() next
  ~_Array3D() { deallocate(); }
  _Array3D()
      : owner(nullptr), x_len(0), y_len(0), z_len(0), y_stride(0), z_stride(0) {
  }
  _Array3D(std::array<int, 3> sz)
      : owner(nullptr), x_len(0), y_len(0), z_len(0), y_stride(0), z_stride(0) {
    allocate(sz);
  }
  _Array3D(const _Array3D &rhs) = delete;
  _Array3D &operator=(_Array3D &&rhs) = delete;
  _Array3D(_Array3D &&rhs) = delete;
  _Array3D &operator=(const _Array3D &rhs) = delete;
  void allocate(std::array<int, 3> sz) {
    assert(sz[0] > 0);
    assert(sz[1] > 0);
    assert(sz[2] > 0);
    this->sz = sz;
    x_len = sz[0] + xlhalo + xrhalo;
    y_len = sz[1] + ylhalo + yrhalo;
    z_len = sz[2] + zlhalo + zrhalo;
    y_stride = x_len;
    z_stride = x_len * y_len;
    owner = sycl::malloc_shared<T>(x_len * y_len * z_len, mod_oneapi::queue);
    assert(owner);
    memset(owner, 0, sizeof(T) * x_len * y_len * z_len);

    auto host_alloc = [](int num) {
      auto ptr = sycl::malloc_host<T>(num, mod_oneapi::queue);
      assert(ptr);
      return ptr;
    };
    send_buf[0] = host_alloc(x_len * z_len * nhalo[3]);
    send_buf[1] = host_alloc(x_len * z_len * nhalo[2]);
    send_buf[2] = host_alloc(x_len * y_len * nhalo[5]);
    send_buf[3] = host_alloc(x_len * y_len * nhalo[4]);
    recv_buf[0] = host_alloc(x_len * z_len * nhalo[2]);
    recv_buf[1] = host_alloc(x_len * z_len * nhalo[3]);
    recv_buf[2] = host_alloc(x_len * y_len * nhalo[4]);
    recv_buf[3] = host_alloc(x_len * y_len * nhalo[5]);

    auto device_alloc = [](int num) {
      auto ptr = sycl::malloc_device<T>(num, mod_oneapi::queue);
      assert(ptr);
      return ptr;
    };
    shuffle_buf[0] = device_alloc(x_len * z_len * nhalo[3]);
    shuffle_buf[1] = device_alloc(x_len * z_len * nhalo[2]);
    shuffle_buf[2] = device_alloc(x_len * z_len * nhalo[2]);
    shuffle_buf[3] = device_alloc(x_len * z_len * nhalo[3]);

    mean = sycl::malloc_shared<T>(1, mod_oneapi::queue);
    assert(mean);
  }
  void deallocate() {
    if (owner) {
      sycl::free(owner, mod_oneapi::queue);
      owner = nullptr;
      auto release = [](auto &bufs) {
        for (auto item : bufs) {
          if (item) {
            sycl::free(item, mod_oneapi::queue);
          }
        }
      };
      release(send_buf);
      release(recv_buf);
      release(shuffle_buf);
      sycl::free(mean, mod_oneapi::queue);
    }
  }

  // size without halos
  const std::array<int, 3> &size() const { return sz; }
  // total buffer with halos
  T *data() { return owner; }
  const T *data() const { return owner; }
  // start with 1
  T &operator()(int x, int y, int z) {
    return owner[x - 1 + xlhalo + (y - 1 + ylhalo) * y_stride +
                 (z - 1 + zlhalo) * z_stride];
  }
  const T &operator()(int x, int y, int z) const {
    return owner[x - 1 + xlhalo + (y - 1 + ylhalo) * y_stride +
                 (z - 1 + zlhalo) * z_stride];
  }
  void operator+=(fp val) {
    for (int k = 1; k <= sz[2]; ++k) {
      for (int j = 1; j <= sz[1]; ++j) {
        for (int i = 1; i <= sz[0]; ++i) {
          (*this)(i, j, k) += val;
        }
      }
    }
  }
  void operator-=(fp val) {
    for (int k = 1; k <= sz[2]; ++k) {
      for (int j = 1; j <= sz[1]; ++j) {
        for (int i = 1; i <= sz[0]; ++i) {
          (*this)(i, j, k) -= val;
        }
      }
    }
  }

  struct View {
    T *ptr;
    int y_stride;
    int z_stride;

    T &operator()(int x, int y, int z) {
      return ptr[x - 1 + xlhalo + (y - 1 + ylhalo) * y_stride +
                 (z - 1 + zlhalo) * z_stride];
    }
    T &operator()(int x, int y, int z) const {
      return ptr[x - 1 + xlhalo + (y - 1 + ylhalo) * y_stride +
                 (z - 1 + zlhalo) * z_stride];
    }
  };

  View get_view() { return View{owner, y_stride, z_stride}; }
  View get_view() const { return View{owner, y_stride, z_stride}; }
  T *get_mean_buf() const { return mean; }
  const HaloBuffer<T> &get_send_buf() const { return send_buf; }
  const HaloBuffer<T> &get_recv_buf() const { return recv_buf; }
  std::vector<sycl::event> sync_send_buf() const {
    std::vector<sycl::event> ret(4);
    // top and bottom
    ret[0] = mod_oneapi::queue.memcpy(send_buf[2], &owner[nhalo[4] * z_stride],
                                      x_len * y_len * nhalo[5] * sizeof(T));
    ret[1] = mod_oneapi::queue.memcpy(send_buf[3], &owner[sz[2] * z_stride],
                                      x_len * y_len * nhalo[4] * sizeof(T));
    // gather north and south
    int _y_stride = y_stride;
    int _z_stride = z_stride;
    int ysz = sz[1];
    auto src = owner;
    auto dst1 = shuffle_buf[0];
    auto dst2 = shuffle_buf[1];
    auto ev1 = mod_oneapi::queue.parallel_for(
        sycl::range(z_len, nhalo[3], x_len), [=](sycl::item<3> idx) {
          int i = idx[2];
          int j = idx[1] + nhalo[2];
          int k = idx[0];
          dst1[idx.get_linear_id()] = src[i + j * _y_stride + k * _z_stride];
        });
    auto ev2 = mod_oneapi::queue.parallel_for(
        sycl::range(z_len, nhalo[2], x_len), [=](sycl::item<3> idx) {
          int i = idx[2];
          int j = idx[1] + ysz;
          int k = idx[0];
          dst2[idx.get_linear_id()] = src[i + j * _y_stride + k * _z_stride];
        });
    // transfer north and south
    ret[2] = mod_oneapi::queue.memcpy(
        send_buf[0], dst1, x_len * z_len * nhalo[3] * sizeof(T), ev1);
    ret[3] = mod_oneapi::queue.memcpy(
        send_buf[1], dst2, x_len * z_len * nhalo[2] * sizeof(T), ev2);
    return ret;
  }
  std::vector<sycl::event> sync_recv_buf() const {
    std::vector<sycl::event> ret(4);
    int ysz = sz[1];
    // top and bottom
    ret[0] = mod_oneapi::queue.memcpy(&owner[0], recv_buf[2],
                                      x_len * y_len * nhalo[4] * sizeof(T));
    ret[1] = mod_oneapi::queue.memcpy(&owner[(nhalo[4] + sz[2]) * z_stride],
                                      recv_buf[3],
                                      x_len * y_len * nhalo[5] * sizeof(T));
    // transfer north and south
    auto dst = owner;
    auto src1 = shuffle_buf[2];
    auto src2 = shuffle_buf[3];
    auto ev1 = mod_oneapi::queue.memcpy(src1, recv_buf[0],
                                        x_len * z_len * nhalo[2] * sizeof(T));
    auto ev2 = mod_oneapi::queue.memcpy(src2, recv_buf[1],
                                        x_len * z_len * nhalo[3] * sizeof(T));
    // scatter north and south
    int _y_stride = y_stride;
    int _z_stride = z_stride;
    ret[2] = mod_oneapi::queue.parallel_for(
        sycl::range(z_len, nhalo[2], x_len), ev1, [=](sycl::item<3> idx) {
          int i = idx[2];
          int j = idx[1];
          int k = idx[0];
          dst[i + j * _y_stride + k * _z_stride] = src1[idx.get_linear_id()];
        });
    ret[3] = mod_oneapi::queue.parallel_for(
        sycl::range(z_len, nhalo[3], x_len), ev2, [=](sycl::item<3> idx) {
          int i = idx[2];
          int j = idx[1] + ysz + nhalo[2];
          int k = idx[0];
          dst[i + j * _y_stride + k * _z_stride] = src2[idx.get_linear_id()];
        });
    return ret;
  }

private:
  T *owner;
  // host buffer for MPI operations
  HaloBuffer<T> send_buf;
  HaloBuffer<T> recv_buf;
  // device buffer for north/south gather/scatter
  HaloBuffer<T> shuffle_buf;
  // attach the buffer with Array3D, so only need to malloc/free once
  T *mean;
  std::array<int, 3> sz;
  int x_len;
  int y_len;
  int z_len;
  int y_stride;
  int z_stride;
};

template <int lhalo, int rhalo> using Array1D = _Array1D<fp, lhalo, rhalo>;
template <int xlhalo, int xrhalo, int ylhalo, int yrhalo, int zlhalo,
          int zrhalo>
using Array3D = _Array3D<fp, xlhalo, xrhalo, ylhalo, yrhalo, zlhalo, zrhalo>;

using Array3DH3 = Array3D<1, 1, 1, 1, 1, 1>;
using Array3DH1 = Array3D<1, 1, 1, 1, 1, 1>;
using Array3DH0 = Array3D<0, 0, 0, 0, 0, 0>;
using Array1DH1 = Array1D<1, 1>;

constexpr fp operator"" _fp(long double val) { return static_cast<fp>(val); }
