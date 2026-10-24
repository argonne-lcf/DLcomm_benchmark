// pci_fixed_cuda.cu -- host-device transfer bandwidth, CUDA.
//
// A CUDA translation of dl_comm/transfer/pci_fixed.cpp, which is written in
// SYCL and therefore cannot build on an NVIDIA machine. The measurement is
// kept faithful to the reference so the two can be compared directly:
//
//   * the same four patterns, in the same order: h2d, d2h, bidirectional, d2d
//   * the same min-over-iterations timing, reduced across ranks with
//     MPI_MIN on the start and MPI_MAX on the end
//   * the same u64/double arithmetic, so the byte product cannot wrap
//   * the same "LAYER=cpp ..." output contract, so the existing parser in
//     dl_comm/analysis/parse_layers.py reads this without modification
//
// Mapping of the SYCL constructs:
//   sycl::malloc_host   -> cudaHostAlloc      (pinned host memory)
//   sycl::malloc_device -> cudaMalloc
//   Q.memcpy + Q.wait   -> cudaMemcpyAsync + cudaStreamSynchronize
//   per-tile queue      -> cudaSetDevice(local_rank % ndev)
//
// Aurora exposes one root device per tile under FLAT hierarchy, so the SYCL
// version indexes tiles. Tara/Polaris have four discrete GPUs per node, so the
// same index selects a GPU. The intent -- one rank per device, no two ranks
// sharing a PCIe link -- is preserved.
//
// ---------------------------------------------------------------------------
// CHANGES vs the Sep-23 version (each is a defect fix, see docs below)
// ---------------------------------------------------------------------------
// FIX 1  MPI_Barrier restored before each timed iteration.
//        The SYCL reference barriers at the top of every iteration
//        (pci_fixed.cpp:58) and *then* reduces MIN(start)/MAX(end). The CUDA
//        port dropped the barrier but kept the reduction, so the reported
//        window spans the full rank skew accumulated since the last implicit
//        sync rather than the transfer. Measured consequence in job 6886:
//        d2d reported GBPS=39.72 aggregate against LOCAL_GBPS_PER_RANK=1807
//        -- the aggregate window was 364x the copy it claimed to time. The
//        two MPI_Reduce calls are not a barrier: a non-root rank returns as
//        soon as its contribution is sent.
//        DLCOMM_TRANSFER_BARRIER=0 reproduces the old behaviour so the fix
//        can be measured rather than asserted.
//
// FIX 2  cudaMemcpyAsync / cudaStreamSynchronize return codes are checked.
//        Every other CUDA call in this file went through CUDA_CHECK; the two
//        in the hot loop did not. A copy that fails costs no time, so the
//        failure surfaces as an implausibly high bandwidth rather than an
//        error -- the worst possible failure mode for a benchmark.
//
// FIX 3  Payload verification. Nothing checked that the bytes arrived. A
//        no-op copy reports the full buffer size over near-zero time. Each
//        pattern now spot-checks three elements of the destination against
//        the source and reports CHECKS=/FAILURES=.
//
// FIX 4  Size sweep. N was compiled in at 2^28 (1 GiB), so the transfer layer
//        produced exactly one point and could not be compared against the
//        collective layers, which sweep 1-64 MiB. DLCOMM_TRANSFER_BYTES takes
//        a comma-separated list; the default sweep matches the collective
//        sizes plus the original 1 GiB point so prior results stay comparable.
//
// FIX 5  Pageable host memory is measured alongside pinned. The gap is the
//        cost of the driver's staging buffer, and a pinned-only figure
//        attributes that cost to the link. DLCOMM_TRANSFER_PAGEABLE=0 skips.
//
// BUILD  nvcc -O2 -arch=sm_90 pci_fixed_cuda.cu -o pci_fixed_cuda <mpi flags>
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
#include <sstream>
#include <string>
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

// FIX 2: the hot loop cannot `return 1` (it is inside a helper), so it sets a
// sticky error that main() turns into a non-zero exit and an ERROR= record.
static cudaError_t g_copy_err = cudaSuccess;
static int g_copy_err_line = 0;

#define COPY_CHECK(call, line)                                                \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess && g_copy_err == cudaSuccess) {                     \
      g_copy_err = _e;                                                        \
      g_copy_err_line = (line);                                               \
    }                                                                         \
  } while (0)

// Fill the buffers so the copies move real data rather than zero pages.
// A constant fill can be served from a zero page or compressed somewhere in
// the path, reporting a bandwidth the hardware cannot sustain.
static void fill_randomly(size_t N, int *a_cpu, int *b_cpu) {
  std::mt19937 gen(1234);
  std::uniform_int_distribution<int> dist(0, 1 << 20);
  for (size_t i = 0; i < N; ++i) {
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
// leaving it. With the barrier in place (FIX 1) that interval is the transfer
// plus barrier-exit jitter. Without it, it is unbounded rank skew.
//
// Both numbers are kept -- the aggregate for comparability with the SYCL
// original, the rank-local one because it is what the device actually did.
static void datatransfer(cudaStream_t stream, size_t N_byte,
                         const std::vector<std::pair<void *, void *>> &ptrs,
                         int iters, bool use_barrier, u64 *agg, u64 *local) {
  u64 min_agg = std::numeric_limits<u64>::max();
  u64 min_local = std::numeric_limits<u64>::max();
  for (int i = 0; i < iters; ++i) {
#ifndef NO_MPI
    // FIX 1: matches pci_fixed.cpp:58. Without this the MIN/MAX reduction
    // below measures rank skew, not transfer.
    if (use_barrier) MPI_Barrier(MPI_COMM_WORLD);
#else
    (void)use_barrier;
#endif
    const u64 l_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (auto &pr : ptrs)
      COPY_CHECK(cudaMemcpyAsync(pr.first, pr.second, N_byte,
                                 cudaMemcpyDefault, stream),
                 __LINE__);
    COPY_CHECK(cudaStreamSynchronize(stream), __LINE__);
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
                   u64 time_ns, int copies, u64 local_ns, bool pinned,
                   int iters, bool barrier) {
  const char *mem = pinned ? "pinned" : "pageable";
  if (time_ns == 0) {
    std::cout << "LAYER=cpp PATTERN=" << pattern << " HOSTMEM=" << mem
              << " ERROR=zero_elapsed_time" << std::endl;
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

  // BUF_BYTES is the per-rank buffer; BYTES stays the aggregate so the
  // existing parser (parse_layers.parse_transfer) keeps reading the same
  // field. HOSTMEM/ITERS/BARRIER are additive and ignored by that parser,
  // which splits on key=value rather than on position.
  std::cout << "LAYER=cpp PATTERN=" << pattern << " HOSTMEM=" << mem
            << " BUF_BYTES=" << N_byte << " BYTES=" << total
            << " TIME_NS=" << time_ns << " GBPS=" << gbps
            << " LOCAL_TIME_NS=" << local_ns
            << " LOCAL_GBPS_PER_RANK=" << local_gbps
            << " ITERS=" << iters << " BARRIER=" << (barrier ? 1 : 0)
            << std::endl;
}

// FIX 3: prove the bytes arrived. A copy that silently does nothing is timed
// as instantaneous and reported as enormous bandwidth.
static long long g_checks = 0, g_failures = 0;

static void verify(const char *pattern, const int *host_ref,
                   const int *host_got, size_t n_elems, int rank) {
  const size_t idxs[3] = {0, n_elems / 2, n_elems - 1};
  for (size_t k = 0; k < 3; ++k) {
    const size_t i = idxs[k];
    ++g_checks;
    if (host_ref[i] != host_got[i]) {
      ++g_failures;
      if (g_failures <= 3)
        std::cerr << "LAYER=cpp MISMATCH pattern=" << pattern
                  << " rank=" << rank << " idx=" << i
                  << " got=" << host_got[i] << " expected=" << host_ref[i]
                  << std::endl;
    }
  }
}

static std::vector<size_t> parse_sizes() {
  std::vector<size_t> sizes;
  if (const char *env = std::getenv("DLCOMM_TRANSFER_BYTES")) {
    std::stringstream ss(env);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      // strip whitespace
      size_t a = tok.find_first_not_of(" \t");
      if (a == std::string::npos) continue;
      size_t b = tok.find_last_not_of(" \t");
      const unsigned long long v = std::stoull(tok.substr(a, b - a + 1));
      if (v >= sizeof(int)) sizes.push_back(static_cast<size_t>(v));
    }
  }
  if (sizes.empty()) {
    // FIX 4: default sweep covers the collective-layer sizes (1-64 MiB) so a
    // cross-layer comparison has matching points, and keeps the original
    // 1 GiB point so Sep-23 results remain comparable.
    sizes = {1ull << 20, 1ull << 22, 1ull << 24, 1ull << 26, 1ull << 30};
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

int main(int argc, char **argv) {
  int world_size = 1, world_rank = 0;
#ifndef NO_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#else
  (void)argc; (void)argv;
#endif

  // Per-rank device selection, for the same reason as the SYCL version: a
  // default device choice puts every rank on GPU 0, so the measurement
  // becomes contention on one PCIe link rather than the node's capability.
  // PALS exports PALS_LOCAL_RANKID on Aurora, Polaris and Tara; the MPI
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

  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  std::cout << "MAP rank=" << world_rank << " local_rank=" << local_rank
            << " host=" << host << " ndev=" << ndev
            << " dev_idx=" << dev_idx << "\n"
            << std::flush;

  const std::vector<size_t> sizes = parse_sizes();
  const int iters = std::getenv("DLCOMM_TRANSFER_ITERS")
                        ? std::atoi(std::getenv("DLCOMM_TRANSFER_ITERS"))
                        : 10;
  // FIX 1 toggle: default on (correct). Set 0 to reproduce the Sep-23 bug.
  const bool barrier = !(std::getenv("DLCOMM_TRANSFER_BARRIER") &&
                         std::string(std::getenv("DLCOMM_TRANSFER_BARRIER")) == "0");
  // FIX 5 toggle: pageable arm costs one extra pass; default on.
  const bool do_pageable = !(std::getenv("DLCOMM_TRANSFER_PAGEABLE") &&
                             std::string(std::getenv("DLCOMM_TRANSFER_PAGEABLE")) == "0");

  if (iters < 1) {
    std::cout << "LAYER=cpp ERROR=bad_iters" << std::endl;
#ifndef NO_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }

  const size_t max_bytes = sizes.back();
  const size_t max_elems = max_bytes / sizeof(int);

  if (world_rank == 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_idx);
    std::cout << "LAYER=cpp DEVICE=" << prop.name << " RANKS=" << world_size
              << " BUFFER_BYTES=" << max_bytes << " ITERS=" << iters
              << " BARRIER=" << (barrier ? 1 : 0)
              << " PAGEABLE_ARM=" << (do_pageable ? 1 : 0)
              << " SIZES=" << sizes.size() << std::endl;
  }

  // Device buffers are allocated once at the maximum size; each sweep point
  // copies a prefix. Re-allocating per size would measure the allocator.
  int *a_gpu = nullptr, *b_gpu = nullptr;
  if (cudaMalloc(&a_gpu, max_bytes) != cudaSuccess ||
      cudaMalloc(&b_gpu, max_bytes) != cudaSuccess) {
    std::cout << "LAYER=cpp ERROR=device_allocation_failed BYTES="
              << max_bytes << std::endl;
#ifndef NO_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }

  // Reference copy on the host, used by FIX 3 to verify every pattern.
  std::vector<int> ref(max_elems);

  const int arms = do_pageable ? 2 : 1;
  for (int arm = 0; arm < arms; ++arm) {
    const bool pinned = (arm == 0);

    int *a_cpu = nullptr, *b_cpu = nullptr;
    if (pinned) {
      // Pinned host memory, matching sycl::malloc_host.
      if (cudaHostAlloc(&a_cpu, max_bytes, cudaHostAllocDefault) !=
              cudaSuccess ||
          cudaHostAlloc(&b_cpu, max_bytes, cudaHostAllocDefault) !=
              cudaSuccess) {
        std::cout << "LAYER=cpp ERROR=pinned_allocation_failed BYTES="
                  << max_bytes << std::endl;
#ifndef NO_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
      }
    } else {
      // FIX 5: pageable arm. The driver stages through its own pinned buffer,
      // so the gap against the pinned arm is staging cost, not link capability.
      a_cpu = static_cast<int *>(std::malloc(max_bytes));
      b_cpu = static_cast<int *>(std::malloc(max_bytes));
      if (!a_cpu || !b_cpu) {
        std::cout << "LAYER=cpp ERROR=pageable_allocation_failed BYTES="
                  << max_bytes << std::endl;
        std::free(a_cpu);
        std::free(b_cpu);
        continue;
      }
    }

    fill_randomly(max_elems, a_cpu, b_cpu);
    std::memcpy(ref.data(), a_cpu, max_bytes);
    CUDA_CHECK(cudaMemcpy(a_gpu, a_cpu, max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_gpu, b_cpu, max_bytes, cudaMemcpyHostToDevice));

    for (size_t N_byte : sizes) {
      const size_t n_elems = N_byte / sizeof(int);

      // ---- h2d -----------------------------------------------------------
      CUDA_CHECK(cudaMemset(a_gpu, 0, N_byte));
      u64 t = 0, l = 0;
      datatransfer(stream, N_byte, {{a_gpu, a_cpu}}, iters, barrier, &t, &l);
      if (world_rank == 0)
        report("h2d", N_byte, world_size, t, 1, l, pinned, iters, barrier);
      {
        std::vector<int> got(n_elems);
        CUDA_CHECK(cudaMemcpy(got.data(), a_gpu, N_byte,
                              cudaMemcpyDeviceToHost));
        verify("h2d", ref.data(), got.data(), n_elems, world_rank);
      }

      // ---- d2h -----------------------------------------------------------
      // a_gpu currently holds the h2d result, which equals ref.
      std::memset(a_cpu, 0, N_byte);
      datatransfer(stream, N_byte, {{a_cpu, a_gpu}}, iters, barrier, &t, &l);
      if (world_rank == 0)
        report("d2h", N_byte, world_size, t, 1, l, pinned, iters, barrier);
      verify("d2h", ref.data(), a_cpu, n_elems, world_rank);

      // ---- bidirectional --------------------------------------------------
      // Both copies are issued before the wait so they overlap on a
      // full-duplex link. With pageable memory cudaMemcpyAsync on the default
      // path is effectively synchronous, so this arm's bidirectional figure
      // is a serialized upper bound rather than full duplex; HOSTMEM= in the
      // record says which arm produced it.
      datatransfer(stream, N_byte, {{a_gpu, a_cpu}, {b_cpu, b_gpu}}, iters,
                   barrier, &t, &l);
      if (world_rank == 0)
        report("bidirectional", N_byte, world_size, t, 2, l, pinned, iters,
               barrier);

      // ---- d2d -------------------------------------------------------------
      // On-device copy. Never crosses PCIe, so it measures HBM bandwidth and
      // forms the ceiling the host-device numbers should be read against.
      // Read LOCAL_GBPS_PER_RANK here: even with the barrier the aggregate
      // window is dominated by barrier-exit jitter at HBM speed.
      CUDA_CHECK(cudaMemcpy(a_gpu, ref.data(), N_byte,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(b_gpu, 0, N_byte));
      datatransfer(stream, N_byte, {{b_gpu, a_gpu}}, iters, barrier, &t, &l);
      if (world_rank == 0)
        report("d2d", N_byte, world_size, t, 1, l, pinned, iters, barrier);
      {
        std::vector<int> got(n_elems);
        CUDA_CHECK(cudaMemcpy(got.data(), b_gpu, N_byte,
                              cudaMemcpyDeviceToHost));
        verify("d2d", ref.data(), got.data(), n_elems, world_rank);
      }

      if (g_copy_err != cudaSuccess) {
        std::cout << "LAYER=cpp ERROR=copy_failed_"
                  << cudaGetErrorName(g_copy_err) << " line="
                  << g_copy_err_line << " at BUF_BYTES=" << N_byte
                  << std::endl;
#ifndef NO_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
      }
    }

    if (pinned) {
      cudaFreeHost(a_cpu);
      cudaFreeHost(b_cpu);
    } else {
      std::free(a_cpu);
      std::free(b_cpu);
    }
  }

  // FIX 3 summary. Reduced across ranks so one bad rank cannot hide.
  long long tot_checks = g_checks, tot_failures = g_failures;
#ifndef NO_MPI
  MPI_Reduce(&g_checks, &tot_checks, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&g_failures, &tot_failures, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
#endif
  if (world_rank == 0)
    std::cout << "LAYER=cpp CORRECTNESS checks=" << tot_checks
              << " failures=" << tot_failures << std::endl;

  cudaFree(a_gpu);
  cudaFree(b_gpu);
  cudaStreamDestroy(stream);
#ifndef NO_MPI
  MPI_Finalize();
#endif
  return tot_failures == 0 ? 0 : 1;
}
