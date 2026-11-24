// ccl_bench.cpp -- C++-level CCL benchmark for DLcomm.
//
// Measures collectives and point-to-point directly against the vendor CCL
// library, with no Python and no torch in the path. This is the layer between
// OSU (MPI) and torch.distributed: it shows how much of the gap between MPI
// and PyTorch is the CCL itself versus the framework above it.
//
// BACKENDS -- selected at compile time, exactly one:
//   -DDLCOMM_XCCL   oneCCL  (Intel, Aurora)        default
//   -DDLCOMM_NCCL   NCCL    (NVIDIA)
//   -DDLCOMM_RCCL   RCCL    (AMD)
//
// NCCL and RCCL are behind compile guards because Aurora has no NVIDIA or AMD
// device: the code is kept compilable and reviewable, but only the XCCL path
// can be executed here. Building the others requires a machine with the
// corresponding runtime.
//
// BUILD (Aurora)
//   mpicxx -fsycl -O2 -DDLCOMM_XCCL ccl_bench.cpp -o ccl_bench \
//       -I$CCL_ROOT/include -L$CCL_ROOT/lib -lccl
//
// OUTPUT  one "LAYER=cpp_ccl" line per (collective, size), parseable by
//         dl_comm.osu.compare.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <set>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#include <mpi.h>

#if defined(DLCOMM_NCCL) || defined(DLCOMM_RCCL)
#include "ccl_correctness.hpp"
#if defined(DLCOMM_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#define GPU_MALLOC(p, n) cudaMalloc((p), (n))
#define GPU_FREE(p) cudaFree(p)
#define GPU_SYNC() cudaDeviceSynchronize()
#define GPU_SET_DEVICE(d) cudaSetDevice(d)
#define GPU_GET_DEVICE_COUNT(p) cudaGetDeviceCount(p)
#define GPU_COPY_H2D(d, s, n) cudaMemcpy((d), (s), (n), cudaMemcpyHostToDevice)
#define GPU_COPY_D2H(d, s, n) cudaMemcpy((d), (s), (n), cudaMemcpyDeviceToHost)
#define GPU_MEMSET(p, v, n) cudaMemset((p), (v), (n))
#define GPU_ERROR_STRING(e) cudaGetErrorString(e)
#define BACKEND_NAME "nccl"
#else
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#define GPU_MALLOC(p, n) hipMalloc((p), (n))
#define GPU_FREE(p) hipFree(p)
#define GPU_SYNC() hipDeviceSynchronize()
#define GPU_SET_DEVICE(d) hipSetDevice(d)
#define GPU_GET_DEVICE_COUNT(p) hipGetDeviceCount(p)
#define GPU_COPY_H2D(d, s, n) hipMemcpy((d), (s), (n), hipMemcpyHostToDevice)
#define GPU_COPY_D2H(d, s, n) hipMemcpy((d), (s), (n), hipMemcpyDeviceToHost)
#define GPU_MEMSET(p, v, n) hipMemset((p), (v), (n))
#define GPU_ERROR_STRING(e) hipGetErrorString(e)
#define BACKEND_NAME "rccl"
#endif
#else
#define DLCOMM_XCCL 1
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#define BACKEND_NAME "xccl"
#endif

using clk = std::chrono::high_resolution_clock;

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// busbw factor per collective, matching dl_comm/analysis/bandwidth.py so the
// C++ numbers are directly comparable with the Python layer's.
static double busbw_factor(const std::string &op, int n) {
  if (op == "allreduce") return 2.0 * (n - 1) / n;
  if (op == "allgather" || op == "alltoall" || op == "alltoallv" ||
      op == "reduce_scatter")
    return double(n - 1) / n;
  return 1.0;  // broadcast, reduce, barrier, sendrecv
}

// Bytes that belong in the algbw numerator for a given collective.
//
// `nbytes` is the buffer size the caller asked for. Whether that is the
// per-rank slice or the total volume depends on how each call below is
// parameterised, so this must be read against the call sites:
//
//   allgather:      passes `count` (per-rank input) and produces count*world
//                   of output -> total volume is nbytes*world.
//   reduce_scatter: passes `count/world` (per-rank OUTPUT), so the operation
//                   consumes `count` == nbytes of input -> nbytes is ALREADY
//                   the total volume. Multiplying again over-reported it by
//                   world (242 GB/s at 12 ranks, physically impossible).
//   alltoall:       passes `count/world` per peer, total moved is nbytes.
//
// Flagged by A-Bot-CELS review point 10 (verify the busbw numerator uses the
// intended per-rank vs aggregate byte definition).
static size_t traffic_bytes(const std::string &op, size_t nbytes, int n) {
  if (op == "allgather") return nbytes * static_cast<size_t>(n);
  return nbytes;
}

struct Row {
  std::string op;
  size_t bytes;
  double t_med;
};

static void emit(const Row &r, int world, int rank) {
  if (rank != 0) return;
  size_t moved = traffic_bytes(r.op, r.bytes, world);
  double algbw = r.t_med > 0 ? double(moved) / r.t_med : 0.0;
  double busbw = algbw * busbw_factor(r.op, world);
  // BYTES stays the per-rank buffer size (what the caller asked for);
  // MOVED_BYTES is the volume the algbw numerator actually used.
  std::cout << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME << " OP=" << r.op
            << " BYTES=" << r.bytes << " MOVED_BYTES=" << moved
            << " RANKS=" << world
            << " T_MED=" << r.t_med << " ALGBW=" << algbw
            << " BUSBW=" << busbw << std::endl;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
  const int warmup = 5;

  // Sizes and ops are overridable so a sweep can locate a threshold without
  // recompiling. DLCOMM_SIZES is a comma-separated byte list; DLCOMM_OPS is a
  // comma-separated op list. Unset means the default three-point sweep and
  // all ops.
  std::vector<size_t> sizes;
  if (const char* env_sizes = std::getenv("DLCOMM_SIZES")) {
    std::stringstream ss(env_sizes);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        sizes.push_back(static_cast<size_t>(std::stoull(tok)));
      }
    }
  }
  if (sizes.empty()) {
    sizes = {1 << 20, 1 << 21, 1 << 22};
  }

  std::set<std::string> only_ops;
  if (const char* env_ops = std::getenv("DLCOMM_OPS")) {
    std::stringstream ss(env_ops);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        only_ops.insert(tok);
      }
    }
  }
  // Empty set means "run everything"; a non-empty set restricts the run.
  auto want = [&only_ops](const std::string& op) {
    return only_ops.empty() || only_ops.count(op) > 0;
  };

  if (rank == 0) {
    std::cout << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME
              << " RANKS=" << world << " ITERS=" << iters << std::endl;
  }

  // Device selection must be per-rank, and the node-local rank is how every
  // backend here decides which device it owns. The computation uses only PALS
  // and MPI, so it is shared rather than duplicated per backend.
  //
  // The local rank comes from PALS (PALS_LOCAL_RANKID), with an MPI
  // shared-memory split as the fallback so the binary is not tied to one
  // launcher. Pattern follows argonne-lcf/HPC-Patterns (concurency/bench_ccl.cpp,
  // p2p/tile_mapping.sh).
  int local_rank = -1;
  if (const char *p = std::getenv("PALS_LOCAL_RANKID")) {
    local_rank = std::atoi(p);
  } else {
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
  }

#ifdef DLCOMM_XCCL
  ccl::init();
  // `sycl::queue{sycl::gpu_selector_v}` returns the SAME device on every rank,
  // so all 12 ranks on a node drove tile 0 while the other 11 tiles idled:
  // every "collective" was really a self-copy contending on one tile's
  // memory. Aurora runs with ZE_FLAT_DEVICE_HIERARCHY=FLAT, so get_devices()
  // lists all 12 tiles as separate root devices and indexing by node-local
  // rank is correct.

  const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (gpus.empty()) {
    if (rank == 0) std::cerr << "no GPU devices visible\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  sycl::device sel_dev = gpus.at(local_rank % gpus.size());
  sycl::queue q{sel_dev};

  // Report the mapping so a silent collapse onto one tile cannot recur.
  {
    std::string nm = sel_dev.get_info<sycl::info::device::name>();
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    std::string line = "MAP rank=" + std::to_string(rank) +
                       " local_rank=" + std::to_string(local_rank) +
                       " host=" + std::string(host) +
                       " ndev=" + std::to_string(gpus.size()) +
                       " dev_idx=" + std::to_string(local_rank % gpus.size()) +
                       " dev=" + nm + "\n";
    std::cout << line << std::flush;
  }

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type addr;
  if (rank == 0) {
    // Only rank 0 creates the main KVS; creating it on every rank would
    // produce a different address than the one broadcast below.
    kvs = ccl::create_main_kvs();
    addr = kvs->get_address();
    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    kvs = ccl::create_kvs(addr);
  }
  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(world, rank, dev, ctx, kvs);
  auto stream = ccl::create_stream(q);

  for (size_t nbytes : sizes) {
    size_t count = nbytes / sizeof(float);
    float *sbuf = sycl::malloc_device<float>(count, q);
    float *rbuf = sycl::malloc_device<float>(count * world, q);
    q.memset(sbuf, 1, nbytes).wait();
    std::vector<double> ts;

    // ---- allreduce ----
    if (want("allreduce")) {
      // (timing vector hoisted below the buffer allocation)
      for (int i = 0; i < warmup + iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        ccl::allreduce(sbuf, rbuf, count, ccl::reduction::sum, comm, stream)
            .wait();
        auto t1 = clk::now();
        if (i >= warmup)
          ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      emit({"allreduce", nbytes, median(ts)}, world, rank);
    }

    // ---- allgather ----
    if (want("allgather")) {
      ts.clear();
      for (int i = 0; i < warmup + iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        ccl::allgather(sbuf, rbuf, count, comm, stream).wait();
        auto t1 = clk::now();
        if (i >= warmup)
          ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      emit({"allgather", nbytes, median(ts)}, world, rank);
    }

    // ---- alltoall ----
    if (want("alltoall")) {
      ts.clear();
      size_t per = count / world;
      if (per > 0) {
        for (int i = 0; i < warmup + iters; i++) {
          MPI_Barrier(MPI_COMM_WORLD);
          auto t0 = clk::now();
          ccl::alltoall(sbuf, rbuf, per, comm, stream).wait();
          auto t1 = clk::now();
          if (i >= warmup)
            ts.push_back(std::chrono::duration<double>(t1 - t0).count());
        }
        emit({"alltoall", nbytes, median(ts)}, world, rank);
    }
    }

    // ---- broadcast ----
    if (want("broadcast")) {
      ts.clear();
      for (int i = 0; i < warmup + iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        ccl::broadcast(sbuf, count, 0, comm, stream).wait();
        auto t1 = clk::now();
        if (i >= warmup)
          ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      emit({"broadcast", nbytes, median(ts)}, world, rank);
    }

    // ---- reduce ----
    if (want("reduce")) {
      ts.clear();
      for (int i = 0; i < warmup + iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        ccl::reduce(sbuf, rbuf, count, ccl::reduction::sum, 0, comm, stream)
            .wait();
        auto t1 = clk::now();
        if (i >= warmup)
          ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      emit({"reduce", nbytes, median(ts)}, world, rank);
    }

    // ---- reduce_scatter ----
    if (want("reduce_scatter")) {
      ts.clear();
      // Per-rank output count. Declared here because the want() guard scopes
      // this block; it was previously shared with the loop above.
      const size_t per = count / static_cast<size_t>(world);
      if (per > 0) {
        for (int i = 0; i < warmup + iters; i++) {
          MPI_Barrier(MPI_COMM_WORLD);
          auto t0 = clk::now();
          ccl::reduce_scatter(sbuf, rbuf, per, ccl::reduction::sum, comm, stream)
              .wait();
          auto t1 = clk::now();
          if (i >= warmup)
            ts.push_back(std::chrono::duration<double>(t1 - t0).count());
        }
        emit({"reduce_scatter", nbytes, median(ts)}, world, rank);
    }
    }

    // ---- point-to-point: rank 0 <-> rank 1 ----
    if (world >= 2 && want("sendrecv")) {
      ts.clear();
      for (int i = 0; i < warmup + iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        if (rank == 0) {
          ccl::send(sbuf, count, 1, comm, stream).wait();
          ccl::recv(rbuf, count, 1, comm, stream).wait();
        } else if (rank == 1) {
          ccl::recv(rbuf, count, 0, comm, stream).wait();
          ccl::send(sbuf, count, 0, comm, stream).wait();
        }
        auto t1 = clk::now();
        if (i >= warmup)
          ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      emit({"sendrecv", nbytes, median(ts)}, world, rank);
    }

    // ---- barrier (no payload) ----
    ts.clear();
    for (int i = 0; i < warmup + iters; i++) {
      auto t0 = clk::now();
      ccl::barrier(comm, stream);
      auto t1 = clk::now();
      if (i >= warmup)
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    if (rank == 0) {
      std::cout << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME
                << " OP=barrier BYTES=0 RANKS=" << world
                << " T_MED=" << median(ts) << " ALGBW=0 BUSBW=0" << std::endl;
    }

    sycl::free(sbuf, q);
    sycl::free(rbuf, q);
  }
#else
  // NCCL / RCCL path. No GPU validation is claimed without executing here.
  auto fatal = [&](const char* api, const char* detail) {
    std::cerr << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME
              << " ERROR=api_or_config rank=" << rank << " API=" << api
              << " CHECKS=0 FAILURES=1 detail=" << detail << std::endl;
    // Do not attempt a collective after an API failure: peers may already
    // be blocked in NCCL. MPI_Abort terminates the entire MPI job nonzero.
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
  };
#define GPU_CHECK(call) do { auto status = (call); if (status != 0) \
    fatal(#call, GPU_ERROR_STRING(status)); } while (0)
#define NCCL_CHECK(call) do { ncclResult_t status = (call); if (status != ncclSuccess) \
    fatal(#call, ncclGetErrorString(status)); } while (0)
#define MPI_CHECK(call) do { if ((call) != MPI_SUCCESS) \
    fatal(#call, "MPI failure"); } while (0)

  if (iters <= 0 || iters > std::numeric_limits<int>::max() - warmup ||
      world > dlcomm_correctness::max_exact_world || local_rank < 0)
    fatal("configuration", "invalid iteration/rank/exact-reduction bound");
  const std::set<std::string> supported = {"allreduce", "allgather", "broadcast",
      "reduce", "reduce_scatter", "sendrecv", "alltoall"};
  for (const auto& op : only_ops)
    if (!supported.count(op)) fatal("DLCOMM_OPS", "unsupported operation");
  for (size_t nbytes : sizes) {
    if (nbytes < sizeof(float) || nbytes % sizeof(float) != 0 ||
        nbytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(world))
      fatal("DLCOMM_SIZES", "sizes must be nonzero float multiples without overflow");
  }
  // DLCOMM_INJECT_CORRUPTION=1 corrupts exactly one device output tail after
  // the first measured iteration. Pick the last participating rank to test
  // non-root failure aggregation (reduce only defines root's output).
  bool inject = false, injected = false;
  if (const char* option = std::getenv("DLCOMM_INJECT_CORRUPTION")) {
    if (std::string(option) != "0" && std::string(option) != "1")
      fatal("DLCOMM_INJECT_CORRUPTION", "expected 0 or 1");
    inject = std::string(option) == "1";
  }

  ncclComm_t comm;
  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  int dev_count = 0;
  GPU_CHECK(GPU_GET_DEVICE_COUNT(&dev_count));
  if (dev_count <= 0) fatal("GPU_GET_DEVICE_COUNT", "no visible GPUs");
  const int dev_idx = local_rank % dev_count;
  GPU_CHECK(GPU_SET_DEVICE(dev_idx));
  std::cout << "LAYER=cpp_ccl MAP rank=" << rank
            << " local_rank=" << local_rank << " dev_idx=" << dev_idx
            << " dev_count=" << dev_count << std::endl;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  unsigned long long tot_checks = 0, tot_failures = 0, tot_injected = 0;
  for (size_t nbytes : sizes) {
    const size_t count = nbytes / sizeof(float);
    void *sbuf = nullptr, *rbuf = nullptr;
    GPU_CHECK(GPU_MALLOC(&sbuf, nbytes));
    GPU_CHECK(GPU_MALLOC(&rbuf, nbytes * world));
    std::vector<float> host_in(count);
    for (const std::string op : {"allreduce", "allgather", "broadcast", "reduce",
                                 "reduce_scatter", "sendrecv", "alltoall"}) {
      if (!want(op)) continue;
      if ((op == "reduce_scatter" || op == "alltoall") && count % world != 0) {
        if (rank == 0)
          std::cout << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME << " OP=" << op
                    << " BYTES=" << nbytes << " STATUS=SKIP REASON=nondivisible_count"
                    << " CHECKS=0 FAILURES=0" << std::endl;
        continue;  // Never silently truncate the requested buffer.
      }
      if (op == "sendrecv" && world < 2) {
        if (rank == 0)
          std::cout << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME << " OP=" << op
                    << " BYTES=" << nbytes << " STATUS=SKIP REASON=requires_two_ranks"
                    << " CHECKS=0 FAILURES=0" << std::endl;
        continue;
      }
      unsigned long long local[3] = {0, 0, 0}, global[3] = {0, 0, 0};
      std::vector<double> ts;
      for (int i = 0; i < warmup + iters; ++i) {
        const dlcomm_correctness::Expected expected(op, count, world, rank, i);
        const size_t output_count = expected.output_count();
        std::vector<float> host_out(output_count);
        for (size_t j = 0; j < count; ++j)
          host_in[j] = dlcomm_correctness::input_value(rank, j, i);
        GPU_CHECK(GPU_COPY_H2D(sbuf, host_in.data(), nbytes));
        // Poison all defined output elements to catch missing writes, even
        // when the expected pattern repeats after its 1021-element period.
        if (output_count) GPU_CHECK(GPU_MEMSET(rbuf, 0xff, output_count * sizeof(float)));
        GPU_CHECK(GPU_SYNC());
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        auto t0 = clk::now();
        if (op == "allreduce") {
          NCCL_CHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, comm, 0));
        } else if (op == "allgather") {
          NCCL_CHECK(ncclAllGather(sbuf, rbuf, count, ncclFloat, comm, 0));
        } else if (op == "broadcast") {
          NCCL_CHECK(ncclBroadcast(sbuf, rbuf, count, ncclFloat, 0, comm, 0));
        } else if (op == "reduce") {
          NCCL_CHECK(ncclReduce(sbuf, rbuf, count, ncclFloat, ncclSum, 0, comm, 0));
        } else if (op == "reduce_scatter") {
          NCCL_CHECK(ncclReduceScatter(sbuf, rbuf, count / world, ncclFloat, ncclSum, comm, 0));
        } else if (op == "sendrecv") {
          NCCL_CHECK(ncclGroupStart());
          if (rank == 0 || rank == 1) {
            const int peer = 1 - rank;
            NCCL_CHECK(ncclSend(sbuf, count, ncclFloat, peer, comm, 0));
            NCCL_CHECK(ncclRecv(rbuf, count, ncclFloat, peer, comm, 0));
          }
          NCCL_CHECK(ncclGroupEnd());
        } else if (op == "alltoall") {
          const size_t per = count / world;
#ifdef DLCOMM_NCCL_HAS_ALLTOALL
          // Opt in only when the installed NCCL/RCCL declares ncclAllToAll.
          NCCL_CHECK(ncclAllToAll(sbuf, rbuf, per, ncclFloat, comm, 0));
#else
          // Portable NCCL/RCCL implementation using the same peer layout.
          NCCL_CHECK(ncclGroupStart());
          for (int peer = 0; peer < world; ++peer) {
            NCCL_CHECK(ncclSend(static_cast<float*>(sbuf) + peer * per,
                                per, ncclFloat, peer, comm, 0));
            NCCL_CHECK(ncclRecv(static_cast<float*>(rbuf) + peer * per,
                                per, ncclFloat, peer, comm, 0));
          }
          NCCL_CHECK(ncclGroupEnd());
#endif
        }
        GPU_CHECK(GPU_SYNC());
        auto t1 = clk::now();
        // Error inspection, injection, copy and verification are NOT timed.
        ncclResult_t async_status = ncclSuccess;
        NCCL_CHECK(ncclCommGetAsyncError(comm, &async_status));
        if (async_status != ncclSuccess) fatal("nccl_async", ncclGetErrorString(async_status));
        const int injection_rank = op == "reduce" ? 0 : (op == "sendrecv" ? 1 : world - 1);
        if (inject && !injected && i == warmup && rank == injection_rank && output_count) {
          const float corrupt = expected.value(output_count - 1) + 1.0f;
          GPU_CHECK(GPU_COPY_H2D(static_cast<float*>(rbuf) + output_count - 1,
                                 &corrupt, sizeof(float)));
          injected = true;
          ++local[2];
        }
        if (output_count)
          GPU_CHECK(GPU_COPY_D2H(host_out.data(), rbuf, output_count * sizeof(float)));
        const auto checked = dlcomm_correctness::verify(host_out.data(), output_count, expected);
        if (checked.failures && local[1] == 0)
          std::cerr << "LAYER=cpp_ccl MISMATCH rank=" << rank << " OP=" << op
                    << " BYTES=" << nbytes << " ITER=" << i
                    << " IDX=" << checked.first_index << " GOT=" << checked.first_got
                    << " EXPECTED=" << checked.first_expected << std::endl;
        local[0] += checked.checks;
        local[1] += checked.failures;
        if (i >= warmup) ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      MPI_CHECK(MPI_Allreduce(local, global, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
      // All ranks remember injection completion; inject once per whole run.
      if (global[2]) injected = true;
      tot_checks += global[0];
      tot_failures += global[1];
      tot_injected += global[2];
      if (rank == 0) {
        const double t_med = median(ts);
        const size_t moved = traffic_bytes(op, nbytes, world);
        const double algbw = t_med > 0 ? double(moved) / t_med : 0;
        std::cout << "LAYER=cpp_ccl BACKEND=" << BACKEND_NAME << " OP=" << op
                  << " BYTES=" << nbytes << " MOVED_BYTES=" << moved << " RANKS=" << world
                  << " T_MED=" << t_med << " ALGBW=" << algbw
                  << " BUSBW=" << algbw * busbw_factor(op, world)
                  << " CHECKS=" << global[0] << " FAILURES=" << global[1]
                  << " INJECTED=" << global[2]
                  << " STATUS=" << (global[1] ? "FAIL" : "PASS")
                  << " VERIFIED_ITERS=" << warmup + iters << std::endl;
      }
    }
    GPU_CHECK(GPU_FREE(sbuf));
    GPU_CHECK(GPU_FREE(rbuf));
  }
  if (inject && tot_injected == 0) fatal("injection", "no eligible output was exercised");
  NCCL_CHECK(ncclCommDestroy(comm));
  if (rank == 0)
    std::cout << "LAYER=cpp_ccl CORRECTNESS checks=" << tot_checks
              << " failures=" << tot_failures << " INJECTED=" << tot_injected << std::endl;
  MPI_CHECK(MPI_Finalize());
#undef GPU_CHECK
#undef NCCL_CHECK
#undef MPI_CHECK
  return tot_failures > 0 ? 1 : 0;
#endif

  MPI_Finalize();
  return 0;
}
