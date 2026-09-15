// pci_fixed_cuda.cu -- host-device transfer bandwidth, CUDA.
//
// A CUDA translation of dl_comm/transfer/pci_fixed.cpp, which is written in
// SYCL and therefore cannot build on an NVIDIA machine. The measurement is
// kept identical so the two can be compared directly:
//
//   * the same four patterns, in the same order: h2d, d2h, bidirectional, d2d
//   * the same 2^28 int buffer (1 GiB) and the same 10 iterations
//   * the same min-over-iterations timing, reduced across ranks with
//     MPI_MIN on the start and MPI_MAX on the end
//   * the same u64/double arithmetic, so the byte product cannot wrap
//   * the same "LAYER=cpp ..." output contract, so the existing parser in
//     dl_comm.transfer reads this without modification
//
// Mapping of the SYCL constructs:
//   sycl::malloc_host   -> cudaHostAlloc      (pinned host memory)
//   sycl::malloc_device -> cudaMalloc
//   Q.memcpy + Q.wait   -> cudaMemcpyAsync + cudaStreamSynchronize
//   per-tile queue      -> cudaSetDevice(local_rank % ndev)
//
// Aurora exposes one root device per tile under FLAT hierarchy, so the SYCL
// version indexes tiles. Polaris has four discrete GPUs per node, so the same
// index selects a GPU. The intent -- one rank per device, no two ranks
// sharing a PCIe link -- is preserved.
//
// BUILD  nvcc -O2 -ccbin mpicxx pci_fixed_cuda.cu -o pci_fixed_cuda
//        (or: nvcc -O2 -DNO_MPI pci_fixed_cuda.cu -o pci_fixed_cuda)
// RUN    mpiexec -n 4 -ppn 4 ./pci_fixed_cuda

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

#ifndef NO_MPI
#include <mpi.h>
#endif

using u64 = unsigned long long;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::cout << "LAYER=cpp ERROR=cuda_" << cudaGetErrorName(_e)            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// Fill the buffers so the copies move real data rather than zero pages.
static void fill_randomly(int N, int *a_cpu, int *b_cpu) {
  std::mt19937 gen(1234);
  std::uniform_int_distribution<int> dist(0, 1 << 20);
  for (int i = 0; i < N; ++i) {
    a_cpu[i] = dist(gen);
    b_cpu[i] = dist(gen);
  }
}

// Minimum elapsed time over `iters` repetitions of the given copy set.
// Each pair is {destination, source}, matching the SYCL original.
// Returns the cross-rank window in `agg` and this rank's own elapsed time in
// `local`.
//
// The original reduces MIN(start) and MAX(end) across ranks, so the window it
// measures spans from the earliest rank entering the copy to the latest rank
// leaving it. That interval contains rank skew as well as the transfer. For
// the PCIe patterns the copy dominates and the difference is small, but d2d
// runs at HBM speed, so on Polaris the skew is larger than the copy itself:
// the aggregate window reports 17 GB/s for a transfer a single rank completes
// at 687 GB/s. Both numbers are kept -- the aggregate for comparability with
// the SYCL original, the rank-local one because it is what the device did.
static void datatransfer(cudaStream_t stream, size_t N_byte,
                         const std::vector<std::pair<void *, void *>> &ptrs,
                         int iters, u64 *agg, u64 *local) {
  u64 min_agg = std::numeric_limits<u64>::max();
  u64 min_local = std::numeric_limits<u64>::max();
  for (int i = 0; i < iters; ++i) {
    const u64 l_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (auto &pr : ptrs)
      cudaMemcpyAsync(pr.first, pr.second, N_byte, cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    const u64 l_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    min_local = std::min(l_end - l_start, min_local);

    u64 start = l_start, end = l_end;
#ifndef NO_MPI
    MPI_Reduce(&l_start, &start, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&l_end, &end, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
#endif
    min_agg = std::min(end - start, min_agg);
  }
  *agg = min_agg;
  *local = min_local;
}

// All arithmetic in u64/double so the product cannot wrap.
static void report(const char *pattern, size_t N_byte, int world_size,
                   u64 time_ns, int copies, u64 local_ns) {
  if (time_ns == 0) {
    std::cout << "LAYER=cpp PATTERN=" << pattern << " ERROR=zero_elapsed_time"
              << std::endl;
    return;
  }
  const u64 total = static_cast<u64>(N_byte) * static_cast<u64>(world_size) *
                    static_cast<u64>(copies);
  const double gbps = static_cast<double>(total) / static_cast<double>(time_ns);

  // Per-rank view: this rank's own bytes over its own elapsed time, free of
  // cross-rank skew.
  const u64 own = static_cast<u64>(N_byte) * static_cast<u64>(copies);
  const double local_gbps =
      local_ns ? static_cast<double>(own) / static_cast<double>(local_ns) : 0.0;

  std::cout << "LAYER=cpp PATTERN=" << pattern << " BYTES=" << total
            << " TIME_NS=" << time_ns << " GBPS=" << gbps
            << " LOCAL_TIME_NS=" << local_ns
            << " LOCAL_GBPS_PER_RANK=" << local_gbps << std::endl;
}

int main(int argc, char **argv) {
  int world_size = 1, world_rank = 0;
#ifndef NO_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

  // Per-rank device selection, for the same reason as the SYCL version: a
  // default device choice puts every rank on GPU 0, so the measurement
  // becomes contention on one PCIe link rather than the node's capability.
  // PALS exports PALS_LOCAL_RANKID on both Aurora and Polaris; the MPI
  // split is the fallback.
  int local_rank = 0;
  if (const char *p = std::getenv("PALS_LOCAL_RANKID")) {
    local_rank = std::atoi(p);
  } else {
#ifndef NO_MPI
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
#endif
  }

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev == 0) {
    std::cerr << "no GPU devices visible\n";
#ifndef NO_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }
  const int dev_idx = local_rank % ndev;
  CUDA_CHECK(cudaSetDevice(dev_idx));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  {
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    std::cout << "MAP rank=" << world_rank << " local_rank=" << local_rank
              << " host=" << host << " ndev=" << ndev
              << " dev_idx=" << dev_idx << "\n"
              << std::flush;
  }

  const int N = 1 << 28;                                       // 2^28 ints
  const size_t N_byte = static_cast<size_t>(N) * sizeof(int);  // 1 GiB
  const int iters = 10;

  if (world_rank == 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_idx);
    std::cout << "LAYER=cpp DEVICE=" << prop.name << " RANKS=" << world_size
              << " BUFFER_BYTES=" << N_byte << std::endl;
  }

  // Pinned host memory, matching sycl::malloc_host. Pageable memory would
  // measure the driver's staging copy instead of the link.
  int *a_cpu = nullptr, *b_cpu = nullptr, *a_gpu = nullptr, *b_gpu = nullptr;
  if (cudaHostAlloc(&a_cpu, N_byte, cudaHostAllocDefault) != cudaSuccess ||
      cudaHostAlloc(&b_cpu, N_byte, cudaHostAllocDefault) != cudaSuccess ||
      cudaMalloc(&a_gpu, N_byte) != cudaSuccess ||
      cudaMalloc(&b_gpu, N_byte) != cudaSuccess) {
    std::cout << "LAYER=cpp ERROR=allocation_failed" << std::endl;
#ifndef NO_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  fill_randomly(N, a_cpu, b_cpu);
  CUDA_CHECK(cudaMemcpy(a_gpu, a_cpu, N_byte, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_gpu, b_cpu, N_byte, cudaMemcpyHostToDevice));

  u64 t_h2d = 0, l_h2d = 0;
  datatransfer(stream, N_byte, {{a_gpu, a_cpu}}, iters, &t_h2d, &l_h2d);
  if (world_rank == 0) report("h2d", N_byte, world_size, t_h2d, 1, l_h2d);

  u64 t_d2h = 0, l_d2h = 0;
  datatransfer(stream, N_byte, {{a_cpu, a_gpu}}, iters, &t_d2h, &l_d2h);
  if (world_rank == 0) report("d2h", N_byte, world_size, t_d2h, 1, l_d2h);

  u64 t_bi = 0, l_bi = 0;
  datatransfer(stream, N_byte, {{a_gpu, a_cpu}, {b_cpu, b_gpu}}, iters, &t_bi,
               &l_bi);
  if (world_rank == 0)
    report("bidirectional", N_byte, world_size, t_bi, 2, l_bi);

  // Device-to-device: the HBM ceiling the PCIe numbers above should be read
  // against. Both endpoints are device allocations, so nothing crosses PCIe.
  // Read LOCAL_GBPS_PER_RANK here: the aggregate window is dominated by rank
  // skew at HBM speed.
  u64 t_d2d = 0, l_d2d = 0;
  datatransfer(stream, N_byte, {{b_gpu, a_gpu}}, iters, &t_d2d, &l_d2d);
  if (world_rank == 0) report("d2d", N_byte, world_size, t_d2d, 1, l_d2d);

  cudaFreeHost(a_cpu);
  cudaFreeHost(b_cpu);
  cudaFree(a_gpu);
  cudaFree(b_gpu);
  cudaStreamDestroy(stream);
#ifndef NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
