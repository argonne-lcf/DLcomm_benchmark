// pci_fixed.cpp -- host-device transfer bandwidth, SYCL.
//
// Derived from the reference pci.cpp with one correctness fix and the MPI
// dependency made optional so it can run on a single rank.
//
// FIX: the reference computes
//     const int   N_byte = N * sizeof(int);          // 2^30
//     const double bw    = (N_byte * world_size) / time;
// `N_byte * world_size` is an int*int product evaluated BEFORE the conversion
// to double, so it overflows for world_size > 1 and is exactly 0 whenever
// world_size is a multiple of 4 -- i.e. 0 GB/s on a 12-rank Aurora node.
// Here every byte count is `unsigned long long` and the division is done in
// double.
//
// BUILD  mpicxx -fsycl -O2 pci_fixed.cpp -o pci_fixed
//        (or: icpx -fsycl -O2 -DNO_MPI pci_fixed.cpp -o pci_fixed)
// RUN    mpiexec -n 12 -ppn 12 ./pci_fixed
//
// Emits one "LAYER=cpp" line per pattern, parseable by dl_comm.transfer.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#ifndef NO_MPI
#include <mpi.h>
#endif

using u64 = unsigned long long;

// Shuffled iota, not a constant: a zero or repeated-value buffer can be
// compressed or served from a zero page, inflating the measured bandwidth.
void fill_randomly(sycl::queue Q, int N, std::vector<int *> ptrs) {
  std::vector<int> v(N);
  std::iota(v.begin(), v.end(), 0);
  std::minstd_rand g;
  for (auto &ptr : ptrs) {
    std::shuffle(v.begin(), v.end(), g);
    Q.memcpy(ptr, v.data(), static_cast<size_t>(N) * sizeof(int)).wait();
  }
}

u64 datatransfer(sycl::queue Q, size_t N_byte,
                 std::vector<std::pair<int *, int *>> ptrs, int iters) {
  u64 min_time = std::numeric_limits<u64>::max();
  for (int r = 0; r < iters; r++) {
#ifndef NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const u64 l_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (auto [dest, src] : ptrs)
      Q.memcpy(dest, src, N_byte);
    Q.wait();
    const u64 l_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    u64 start = l_start, end = l_end;
#ifndef NO_MPI
    MPI_Reduce(&l_start, &start, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&l_end, &end, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
#endif
    min_time = std::min(end - start, min_time);
  }
  return min_time;
}

// All arithmetic in u64/double so the product cannot wrap.
void report(const char *pattern, size_t N_byte, int world_size, u64 time_ns,
            int copies) {
  if (time_ns == 0) {
    std::cout << "LAYER=cpp PATTERN=" << pattern
              << " ERROR=zero_elapsed_time" << std::endl;
    return;
  }
  const u64 total = static_cast<u64>(N_byte) * static_cast<u64>(world_size) *
                    static_cast<u64>(copies);
  const double gbps = static_cast<double>(total) / static_cast<double>(time_ns);
  std::cout << "LAYER=cpp PATTERN=" << pattern << " BYTES=" << total
            << " TIME_NS=" << time_ns << " GBPS=" << gbps << std::endl;
}

int main(int argc, char **argv) {
  int world_size = 1, world_rank = 0;
#ifndef NO_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

  sycl::queue Q;
  const int N = 1 << 28;                                  // 2^28 ints
  const size_t N_byte = static_cast<size_t>(N) * sizeof(int);  // 1 GiB
  const int iters = 10;

  if (world_rank == 0) {
    std::cout << "LAYER=cpp DEVICE="
              << Q.get_device().get_info<sycl::info::device::name>()
              << " RANKS=" << world_size
              << " BUFFER_BYTES=" << N_byte << std::endl;
  }

  int *a_cpu = sycl::malloc_host<int>(N, Q);
  int *b_cpu = sycl::malloc_host<int>(N, Q);
  int *a_gpu = sycl::malloc_device<int>(N, Q);
  int *b_gpu = sycl::malloc_device<int>(N, Q);
  if (!a_cpu || !b_cpu || !a_gpu || !b_gpu) {
    std::cout << "LAYER=cpp ERROR=allocation_failed" << std::endl;
#ifndef NO_MPI
    MPI_Finalize();
#endif
    return 1;
  }
  fill_randomly(Q, N, {a_cpu, b_cpu, a_gpu, b_gpu});

  u64 t_h2d = datatransfer(Q, N_byte, {{a_gpu, a_cpu}}, iters);
  if (world_rank == 0) report("h2d", N_byte, world_size, t_h2d, 1);

  u64 t_d2h = datatransfer(Q, N_byte, {{a_cpu, a_gpu}}, iters);
  if (world_rank == 0) report("d2h", N_byte, world_size, t_d2h, 1);

  u64 t_bi = datatransfer(Q, N_byte, {{a_gpu, a_cpu}, {b_cpu, b_gpu}}, iters);
  if (world_rank == 0) report("bidirectional", N_byte, world_size, t_bi, 2);

  // Device-to-device: the HBM ceiling the PCIe numbers above should be read
  // against. Both endpoints are device allocations, so nothing crosses PCIe.
  u64 t_d2d = datatransfer(Q, N_byte, {{b_gpu, a_gpu}}, iters);
  if (world_rank == 0) report("d2d", N_byte, world_size, t_d2d, 1);

  sycl::free(a_cpu, Q); sycl::free(b_cpu, Q);
  sycl::free(a_gpu, Q); sycl::free(b_gpu, Q);
#ifndef NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
