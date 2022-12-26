#pragma once

#include <CL/sycl.hpp>

namespace mod_oneapi {
inline sycl::queue queue;

template <int B, int N> constexpr auto unpack(const sycl::item<N> &item) {
  if constexpr (N == 1) {
    return std::tuple{item[0] + B};
  } else if constexpr (N == 2) {
    int i = item[1] + B;
    int j = item[0] + B;
    return std::tuple{i, j};
  } else if constexpr (N == 3) {
    int i = item[2] + B;
    int j = item[1] + B;
    int k = item[0] + B;
    return std::tuple{i, j, k};
  } else {
    __builtin_unreachable();
  }
}

} // namespace mod_oneapi
