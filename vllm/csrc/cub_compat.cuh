#pragma once

// CUDA 13 removed the old CUB functors cub::Sum, cub::Max, and cub::Min.
// These aliases keep this tree building against the installed nvcc.
#ifndef USE_ROCM
  #include <cuda/functional>
  #include <cuda/std/functional>

namespace cub {
using Sum = ::cuda::std::plus<>;
using Max = ::cuda::maximum<>;
using Min = ::cuda::minimum<>;
}  // namespace cub
#endif
