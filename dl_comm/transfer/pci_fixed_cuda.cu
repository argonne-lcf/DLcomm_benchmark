// pci_fixed_cuda.cu -- host-device transfer bandwidth, CUDA.
//
// A CUDA translation of dl_comm/transfer/pci_fixed.cpp, which is written in
// SYCL and therefore cannot build on an NVIDIA machine. The measurement is
// kept faithful to the reference so the two can be compared directly:
//
//   * the same four patterns, in the same order: h2d, d2h, bidirectional, d2d
//   * the same min-over-iterations timing, reduced across ranks with
//     MPI_MIN on the start and MPI_MAX on the end
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
// ---------------------------------------------------------------------------
// REVIEW FIXES (independent review 20261024T071020Z-dlcomm3c)
// ---------------------------------------------------------------------------
// RF1    Bidirectional used ONE stream, so the two copies serialized and the
//        "full-duplex overlap" claim was false. Each concurrent copy now gets
//        its own stream; timing brackets the whole set (all streams started,
//        then all synchronized), which is what a duplex figure must measure.
//
// RF2    Warmup. The header claimed warmup was added; it was not. Every copy
//        was timed from the first touch, so first-use page faults, lazy
//        context init and the driver's pageable staging setup landed in the
//        sample. An untimed warmup pass now precedes every timed set.
//        DLCOMM_TRANSFER_WARMUP=N sets the count (default 3, 0 disables).
//
// RF3    Byte sizes not divisible by sizeof(int) overflowed the host buffers:
//        max_elems rounded *down*, so DLCOMM_TRANSFER_BYTES=5 allocated 4
//        bytes and copied 5 (confirmed under AddressSanitizer). Sizes must now
//        be a positive multiple of sizeof(int); anything else is rejected with
//        a diagnostic. std::stoull exceptions are caught rather than escaping.
//
// RF4    Bidirectional was never verified. h2d/d2h/d2d checked their payload
//        but the two bidirectional destinations did not, so a dropped or
//        mis-sized direction still reported failures=0. Both are now checked.
//
// RF5    A failed copy was reported too late: the sticky error was examined
//        only after report() had already printed a bandwidth line for the
//        failed transfer. The error is now checked immediately after each
//        timed set and before any report(), so no number is ever printed for
//        a transfer that did not happen.
//
// RF6    Error paths stranded peers and leaked resources. CUDA_CHECK did a
//        bare `return 1` after MPI_Init, leaving other ranks blocked in the
//        next barrier/reduction; partial allocation failures leaked whatever
//        had already succeeded. All failure paths now go through
//        fail_fast(), which MPI_Aborts, and allocations are released by a
//        single cleanup path.
//
// RF7    Timing used high_resolution_clock::now().time_since_epoch().count(),
//        whose period is implementation-defined, while the output field is
//        named TIME_NS. Replaced with an explicit duration_cast<nanoseconds>
//        against steady_clock, which is also monotonic (high_resolution_clock
//        may alias system_clock and jump).
//
// RF8    The aggregate byte product N_byte * world_size * copies was asserted
//        to be unwrappable. It is checked instead.
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
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

#ifndef NO_MPI
#include <mpi.h>
#endif

using u64 = unsigned long long;

// RF6: a rank-local failure must take the whole job down. Returning 1 from
// one rank leaves the others blocked in the next MPI_Barrier/MPI_Reduce until
// the scheduler's walltime kills them, which looks like a hang rather than an
// error.
[[noreturn]] static void fail_fast(const std::string &what) {
  std::cout << "LAYER=cpp ERROR=" << what << std::endl;
  std::cout.flush();
#ifndef NO_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(1);
}

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fail_fast(std::string("cuda_") + cudaGetErrorName(_e) + " at " +        \
                __FILE__ + ":" + std::to_string(__LINE__));                   \
    }                                                                         \
  } while (0)

// FIX 2: the hot loop cannot fail_fast directly (it must let every rank reach
// the same decision point), so it sets a sticky error that the caller checks
// immediately after the timed set -- see RF5.
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

// RF7: monotonic, and explicitly converted to nanoseconds rather than trusting
// the clock's native period to be nano.
static inline u64 now_ns() {
  return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

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
// RF1: `streams` must hold at least ptrs.size() entries. Copy i is issued on
// stream i, so a multi-copy set runs concurrently instead of serializing
// behind one stream. The timed region covers "all copies issued" to "all
// copies complete", which is the only meaningful bracket for a duplex figure.
//
// RF2: `warmup` untimed repetitions run first. They execute the identical copy
// set, so first-touch faults, context init and pageable staging allocation are
// paid before the clock starts.
//
// The original reduces MIN(start) and MAX(end) across ranks, so the window it
// measures spans from the earliest rank entering the copy to the latest rank
// leaving it. With the barrier in place (FIX 1) that interval is the transfer
// plus barrier-exit jitter. Without it, it is unbounded rank skew.
static void datatransfer(const std::vector<cudaStream_t> &streams,
                         size_t N_byte,
                         const std::vector<std::pair<void *, void *>> &ptrs,
                         int iters, bool use_barrier, int warmup, u64 *agg,
                         u64 *local) {
  // RF2: untimed warmup.
  for (int w = 0; w < warmup; ++w) {
    for (size_t k = 0; k < ptrs.size(); ++k)
      COPY_CHECK(cudaMemcpyAsync(ptrs[k].first, ptrs[k].second, N_byte,
                                 cudaMemcpyDefault, streams[k]),
                 __LINE__);
    for (size_t k = 0; k < ptrs.size(); ++k)
      COPY_CHECK(cudaStreamSynchronize(streams[k]), __LINE__);
  }

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
    const u64 l_start = now_ns();
    // RF1: issue every copy before waiting on any of them, each on its own
    // stream, so independent copies actually overlap.
    for (size_t k = 0; k < ptrs.size(); ++k)
      COPY_CHECK(cudaMemcpyAsync(ptrs[k].first, ptrs[k].second, N_byte,
                                 cudaMemcpyDefault, streams[k]),
                 __LINE__);
    for (size_t k = 0; k < ptrs.size(); ++k)
      COPY_CHECK(cudaStreamSynchronize(streams[k]), __LINE__);
    const u64 l_end = now_ns();

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

// FIX 3: prove the bytes arrived. A copy that silently does nothing is timed
// as instantaneous and reported as enormous bandwidth.
static long long g_checks = 0, g_failures = 0;

// R2-1: verification must gate the report, not follow it.
//
// Round-2 review: every report() ran before the verify() for the same
// transfer, so a copy whose API calls all succeeded while the data arrived
// corrupt still printed a plausible GBPS line. The non-zero exit code at the
// end does not retract a line a log scraper has already ingested.
//
// verify_now() returns the failure count for this transfer; report() takes it
// and emits either a bandwidth line (clean) or an explicit ERROR line
// (corrupt), so a corrupted transfer can never produce a number that looks
// usable.
static long long verify_now(const char *pattern, const int *host_ref,
                            const int *host_got, size_t n_elems, int rank) {
  if (n_elems == 0) return 0;
  long long local_failures = 0;
  const size_t idxs[3] = {0, n_elems / 2, n_elems - 1};
  for (size_t k = 0; k < 3; ++k) {
    const size_t i = idxs[k];
    ++g_checks;
    if (host_ref[i] != host_got[i]) {
      ++g_failures;
      ++local_failures;
      if (g_failures <= 3)
        std::cerr << "LAYER=cpp MISMATCH pattern=" << pattern
                  << " rank=" << rank << " idx=" << i
                  << " got=" << host_got[i] << " expected=" << host_ref[i]
                  << std::endl;
    }
  }
  return local_failures;
}

// RF5: call immediately after each timed set, before report(). A transfer that
// errored must never produce a bandwidth line.
static void abort_if_copy_failed(size_t N_byte, const char *pattern) {
  if (g_copy_err == cudaSuccess) return;
  std::ostringstream os;
  os << "copy_failed_" << cudaGetErrorName(g_copy_err) << " pattern="
     << pattern << " line=" << g_copy_err_line << " BUF_BYTES=" << N_byte;
  fail_fast(os.str());
}

// R2-2: the pageable arm cannot measure duplex overlap.
//
// cudaMemcpyAsync from unpinned memory is synchronous with respect to the
// host, so the two "concurrent" copies of the bidirectional pattern run back
// to back no matter how many streams they are issued on. Round-1 documented
// that in a comment; round 2 correctly rejected that as insufficient, because
// the emitted record still said PATTERN=bidirectional with two copies and no
// downstream consumer reads source comments.
//
// The pattern name itself now carries the distinction, so an analysis script
// cannot mistake one for the other:
//   pinned   -> PATTERN=bidirectional            (true overlap)
//   pageable -> PATTERN=bidirectional_serialized (upper bound, no overlap)
// OVERLAP=1|0 is emitted on every record as a machine-readable flag.
static const char *bidir_pattern(bool pinned) {
  return pinned ? "bidirectional" : "bidirectional_serialized";
}

static void report(const char *pattern, size_t N_byte, int world_size,
                   u64 time_ns, int copies, u64 local_ns, bool pinned,
                   int iters, bool barrier, long long failures) {
  const char *mem = pinned ? "pinned" : "pageable";
  // R2-1: a corrupted transfer emits an error, never a bandwidth number.
  if (failures != 0) {
    std::cout << "LAYER=cpp PATTERN=" << pattern << " HOSTMEM=" << mem
              << " BUF_BYTES=" << N_byte
              << " ERROR=payload_verification_failed"
              << " FAILURES=" << failures << std::endl;
    return;
  }
  if (time_ns == 0) {
    std::cout << "LAYER=cpp PATTERN=" << pattern << " HOSTMEM=" << mem
              << " ERROR=zero_elapsed_time" << std::endl;
    return;
  }
  // RF8: check the product instead of asserting it cannot wrap.
  const u64 umax = std::numeric_limits<u64>::max();
  const u64 nb = static_cast<u64>(N_byte);
  const u64 ws = static_cast<u64>(world_size);
  const u64 cp = static_cast<u64>(copies);
  if (ws != 0 && nb > umax / ws) fail_fast("byte_product_overflow");
  u64 total = nb * ws;
  if (cp != 0 && total > umax / cp) fail_fast("byte_product_overflow");
  total *= cp;

  const double gbps = static_cast<double>(total) / static_cast<double>(time_ns);

  // Per-rank view: this rank's own bytes over its own elapsed time, free of
  // cross-rank skew.
  const u64 own = nb * cp;
  const double local_gbps =
      local_ns ? static_cast<double>(own) / static_cast<double>(local_ns) : 0.0;

  // BUF_BYTES is the per-rank buffer; BYTES stays the aggregate so the
  // existing parser (parse_layers.parse_transfer) keeps reading the same
  // field. HOSTMEM/ITERS/BARRIER/OVERLAP are additive and ignored by that
  // parser, which splits on key=value rather than on position.
  //
  // R2-2: OVERLAP says whether concurrent copies could actually overlap.
  // Only a multi-copy pattern from pinned memory can; everything else is a
  // serialized measurement.
  const int overlap = (copies > 1 && pinned) ? 1 : 0;
  std::cout << "LAYER=cpp PATTERN=" << pattern << " HOSTMEM=" << mem
            << " BUF_BYTES=" << N_byte << " BYTES=" << total
            << " TIME_NS=" << time_ns << " GBPS=" << gbps
            << " LOCAL_TIME_NS=" << local_ns
            << " LOCAL_GBPS_PER_RANK=" << local_gbps
            << " OVERLAP=" << overlap
            << " ITERS=" << iters << " BARRIER=" << (barrier ? 1 : 0)
            << std::endl;
}

// RF3: every size must be a positive multiple of sizeof(int). The element
// counts downstream are byte/sizeof(int), which rounds down, so a size that is
// not a multiple makes the host buffers smaller than the copies that write
// into them. Rejecting is correct: silently rounding would report a bandwidth
// for a size the user did not ask for.
static std::vector<size_t> parse_sizes() {
  std::vector<size_t> sizes;
  if (const char *env = std::getenv("DLCOMM_TRANSFER_BYTES")) {
    std::stringstream ss(env);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      size_t a = tok.find_first_not_of(" \t");
      if (a == std::string::npos) continue;
      size_t b = tok.find_last_not_of(" \t");
      const std::string t = tok.substr(a, b - a + 1);

      unsigned long long v = 0;
      try {
        size_t consumed = 0;
        v = std::stoull(t, &consumed);
        if (consumed != t.size())
          fail_fast("bad_transfer_bytes_trailing_chars value=" + t);
      } catch (const std::exception &) {
        fail_fast("bad_transfer_bytes_not_a_number value=" + t);
      }

      if (v == 0) fail_fast("bad_transfer_bytes_zero value=" + t);
      if (v % sizeof(int) != 0)
        fail_fast("bad_transfer_bytes_not_multiple_of_" +
                  std::to_string(sizeof(int)) + " value=" + t);
      if (v > static_cast<unsigned long long>(
                  std::numeric_limits<size_t>::max()))
        fail_fast("bad_transfer_bytes_too_large value=" + t);
      sizes.push_back(static_cast<size_t>(v));
    }
    if (sizes.empty()) fail_fast("bad_transfer_bytes_empty");
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
  if (ndev == 0) fail_fast("no_gpu_devices_visible");
  const int dev_idx = local_rank % ndev;
  CUDA_CHECK(cudaSetDevice(dev_idx));

  // RF1: two streams, because the widest copy set (bidirectional) has two
  // concurrent copies.
  const size_t kMaxConcurrent = 2;
  std::vector<cudaStream_t> streams(kMaxConcurrent);
  for (size_t i = 0; i < kMaxConcurrent; ++i)
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

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
  // RF2 toggle: warmup repetitions, untimed. 0 reproduces the old behaviour.
  const int warmup = std::getenv("DLCOMM_TRANSFER_WARMUP")
                         ? std::atoi(std::getenv("DLCOMM_TRANSFER_WARMUP"))
                         : 3;

  if (iters < 1) fail_fast("bad_iters");
  if (warmup < 0) fail_fast("bad_warmup");

  const size_t max_bytes = sizes.back();
  const size_t max_elems = max_bytes / sizeof(int);  // exact: RF3 guarantees it

  if (world_rank == 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_idx));
    std::cout << "LAYER=cpp DEVICE=" << prop.name << " RANKS=" << world_size
              << " BUFFER_BYTES=" << max_bytes << " ITERS=" << iters
              << " WARMUP=" << warmup
              << " BARRIER=" << (barrier ? 1 : 0)
              << " PAGEABLE_ARM=" << (do_pageable ? 1 : 0)
              << " SIZES=" << sizes.size() << std::endl;
  }

  // Device buffers are allocated once at the maximum size; each sweep point
  // copies a prefix. Re-allocating per size would measure the allocator.
  // RF6: free whatever succeeded before aborting on a later failure.
  int *a_gpu = nullptr, *b_gpu = nullptr;
  if (cudaMalloc(&a_gpu, max_bytes) != cudaSuccess) {
    fail_fast("device_allocation_failed BYTES=" + std::to_string(max_bytes));
  }
  if (cudaMalloc(&b_gpu, max_bytes) != cudaSuccess) {
    cudaFree(a_gpu);
    fail_fast("device_allocation_failed BYTES=" + std::to_string(max_bytes));
  }

  // Reference copy on the host, used by FIX 3 to verify every pattern.
  std::vector<int> ref(max_elems);
  // RF4: bidirectional writes b_gpu's content back to b_cpu, so its expected
  // payload is b_cpu's original content, captured here.
  std::vector<int> ref_b(max_elems);

  const int arms = do_pageable ? 2 : 1;
  for (int arm = 0; arm < arms; ++arm) {
    const bool pinned = (arm == 0);

    int *a_cpu = nullptr, *b_cpu = nullptr;
    if (pinned) {
      // Pinned host memory, matching sycl::malloc_host.
      if (cudaHostAlloc(&a_cpu, max_bytes, cudaHostAllocDefault) !=
          cudaSuccess) {
        cudaFree(a_gpu); cudaFree(b_gpu);
        fail_fast("pinned_allocation_failed BYTES=" +
                  std::to_string(max_bytes));
      }
      if (cudaHostAlloc(&b_cpu, max_bytes, cudaHostAllocDefault) !=
          cudaSuccess) {
        cudaFreeHost(a_cpu); cudaFree(a_gpu); cudaFree(b_gpu);
        fail_fast("pinned_allocation_failed BYTES=" +
                  std::to_string(max_bytes));
      }
    } else {
      // FIX 5: pageable arm. The driver stages through its own pinned buffer,
      // so the gap against the pinned arm is staging cost, not link capability.
      a_cpu = static_cast<int *>(std::malloc(max_bytes));
      b_cpu = static_cast<int *>(std::malloc(max_bytes));
      if (!a_cpu || !b_cpu) {
        std::free(a_cpu);
        std::free(b_cpu);
        cudaFree(a_gpu); cudaFree(b_gpu);
        fail_fast("pageable_allocation_failed BYTES=" +
                  std::to_string(max_bytes));
      }
    }

    fill_randomly(max_elems, a_cpu, b_cpu);
    std::memcpy(ref.data(), a_cpu, max_bytes);
    std::memcpy(ref_b.data(), b_cpu, max_bytes);
    CUDA_CHECK(cudaMemcpy(a_gpu, a_cpu, max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_gpu, b_cpu, max_bytes, cudaMemcpyHostToDevice));

    for (size_t N_byte : sizes) {
      const size_t n_elems = N_byte / sizeof(int);  // exact: RF3
      u64 t = 0, l = 0;
      long long f = 0;

      // R2-1: for every pattern the order is now
      //   copy -> check CUDA error -> verify payload -> report
      // so a transfer that moved corrupt bytes emits an ERROR record instead
      // of a bandwidth number. Previously report() ran before verify() and a
      // silently corrupt copy produced a plausible GBPS line.

      // ---- h2d -----------------------------------------------------------
      CUDA_CHECK(cudaMemset(a_gpu, 0, N_byte));
      datatransfer(streams, N_byte, {{a_gpu, a_cpu}}, iters, barrier, warmup,
                   &t, &l);
      abort_if_copy_failed(N_byte, "h2d");          // RF5: before report
      {
        std::vector<int> got(n_elems);
        CUDA_CHECK(cudaMemcpy(got.data(), a_gpu, N_byte,
                              cudaMemcpyDeviceToHost));
        f = verify_now("h2d", ref.data(), got.data(), n_elems, world_rank);
      }
      if (world_rank == 0)
        report("h2d", N_byte, world_size, t, 1, l, pinned, iters, barrier, f);

      // ---- d2h -----------------------------------------------------------
      // a_gpu currently holds the h2d result, which equals ref.
      std::memset(a_cpu, 0, N_byte);
      datatransfer(streams, N_byte, {{a_cpu, a_gpu}}, iters, barrier, warmup,
                   &t, &l);
      abort_if_copy_failed(N_byte, "d2h");
      f = verify_now("d2h", ref.data(), a_cpu, n_elems, world_rank);
      if (world_rank == 0)
        report("d2h", N_byte, world_size, t, 1, l, pinned, iters, barrier, f);

      // ---- bidirectional --------------------------------------------------
      // RF1: the two copies run on separate streams, so a full-duplex link can
      // actually overlap them.
      //
      // R2-2: the pageable arm cannot overlap -- cudaMemcpyAsync from unpinned
      // memory is synchronous with respect to the host -- so it is reported
      // under a DIFFERENT pattern name (bidirectional_serialized) with
      // OVERLAP=0, rather than as a duplex figure with a caveat in a comment
      // no downstream parser reads.
      //
      // d2h into a_cpu and h2d out of b_cpu, so the two directions touch
      // different buffers and can be verified independently (RF4).
      std::memset(a_cpu, 0, N_byte);
      CUDA_CHECK(cudaMemset(b_gpu, 0, N_byte));
      datatransfer(streams, N_byte, {{a_cpu, a_gpu}, {b_gpu, b_cpu}}, iters,
                   barrier, warmup, &t, &l);
      abort_if_copy_failed(N_byte, bidir_pattern(pinned));
      // RF4: check BOTH destinations. Previously neither was checked, so a
      // dropped direction still reported failures=0.
      f = verify_now("bidirectional_d2h", ref.data(), a_cpu, n_elems,
                     world_rank);
      {
        std::vector<int> got(n_elems);
        CUDA_CHECK(cudaMemcpy(got.data(), b_gpu, N_byte,
                              cudaMemcpyDeviceToHost));
        f += verify_now("bidirectional_h2d", ref_b.data(), got.data(), n_elems,
                        world_rank);
      }
      if (world_rank == 0)
        report(bidir_pattern(pinned), N_byte, world_size, t, 2, l, pinned,
               iters, barrier, f);

      // ---- d2d -------------------------------------------------------------
      // On-device copy. Never crosses PCIe, so it measures HBM bandwidth and
      // forms the ceiling the host-device numbers should be read against.
      // Read LOCAL_GBPS_PER_RANK here: even with the barrier the aggregate
      // window is dominated by barrier-exit jitter at HBM speed.
      CUDA_CHECK(cudaMemcpy(a_gpu, ref.data(), N_byte,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(b_gpu, 0, N_byte));
      datatransfer(streams, N_byte, {{b_gpu, a_gpu}}, iters, barrier, warmup,
                   &t, &l);
      abort_if_copy_failed(N_byte, "d2d");
      {
        std::vector<int> got(n_elems);
        CUDA_CHECK(cudaMemcpy(got.data(), b_gpu, N_byte,
                              cudaMemcpyDeviceToHost));
        f = verify_now("d2d", ref.data(), got.data(), n_elems, world_rank);
      }
      if (world_rank == 0)
        report("d2d", N_byte, world_size, t, 1, l, pinned, iters, barrier, f);

      // a_gpu must hold ref again for the next sweep point's d2h.
      CUDA_CHECK(cudaMemcpy(a_gpu, ref.data(), N_byte,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(b_gpu, ref_b.data(), N_byte,
                            cudaMemcpyHostToDevice));
    }

    if (pinned) {
      CUDA_CHECK(cudaFreeHost(a_cpu));
      CUDA_CHECK(cudaFreeHost(b_cpu));
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

  // RF6: checked teardown. A failure here is a real error, not noise.
  CUDA_CHECK(cudaFree(a_gpu));
  CUDA_CHECK(cudaFree(b_gpu));
  for (size_t i = 0; i < streams.size(); ++i)
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
#ifndef NO_MPI
  MPI_Finalize();
#endif
  return tot_failures == 0 ? 0 : 1;
}
