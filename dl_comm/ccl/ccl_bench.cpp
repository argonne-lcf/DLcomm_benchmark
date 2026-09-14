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
#if defined(DLCOMM_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#define GPU_MALLOC(p, n) cudaMalloc((p), (n))
#define GPU_FREE(p) cudaFree(p)
#define GPU_SYNC() cudaDeviceSynchronize()
#define GPU_SET_DEVICE(d) cudaSetDevice(d)
#define BACKEND_NAME "nccl"
#else
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#define GPU_MALLOC(p, n) hipMalloc((p), (n))
#define GPU_FREE(p) hipFree(p)
#define GPU_SYNC() hipDeviceSynchronize()
#define GPU_SET_DEVICE(d) hipSetDevice(d)
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
// `nbytes` is the PER-RANK buffer size. For allgather the operation produces
// nbytes*world of output, and the standard (NCCL/OSU) convention divides the
// total moved volume by time -- not the per-rank slice. Reporting the per-rank
// size understated allgather by exactly world_size (12x at 12 ranks) and made
// it look like the slowest collective on the machine by two orders of
// magnitude. reduce_scatter is the mirror case: nbytes*world of input is
// consumed to produce nbytes per rank.
//
// Flagged by A-Bot-CELS review point 10 (verify the busbw numerator uses the
// intended per-rank vs aggregate byte definition).
static size_t traffic_bytes(const std::string &op, size_t nbytes, int n) {
  if (op == "allgather" || op == "reduce_scatter")
    return nbytes * static_cast<size_t>(n);
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

#ifdef DLCOMM_XCCL
  ccl::init();
  // Device selection must be per-rank. `sycl::queue{sycl::gpu_selector_v}`
  // returns the SAME device on every rank, so all 12 ranks on a node drove
  // tile 0 while the other 11 tiles idled: every "collective" was really a
  // self-copy contending on one tile's memory. Aurora runs with
  // ZE_FLAT_DEVICE_HIERARCHY=FLAT, so get_devices() lists all 12 tiles as
  // separate root devices and indexing by node-local rank is correct.
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
  // NCCL / RCCL path. Compiled only where the runtime exists.
  ncclComm_t comm;
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  GPU_SET_DEVICE(rank % 8);
  ncclCommInitRank(&comm, world, id, rank);

  for (size_t nbytes : sizes) {
    size_t count = nbytes / sizeof(float);
    void *sbuf = nullptr, *rbuf = nullptr;
    GPU_MALLOC(&sbuf, nbytes);
    GPU_MALLOC(&rbuf, nbytes * world);

    std::vector<double> ts;
    for (int i = 0; i < warmup + iters; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      auto t0 = clk::now();
      ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, comm, 0);
      GPU_SYNC();
      auto t1 = clk::now();
      if (i >= warmup)
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    emit({"allreduce", nbytes, median(ts)}, world, rank);

    ts.clear();
    for (int i = 0; i < warmup + iters; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      auto t0 = clk::now();
      ncclAllGather(sbuf, rbuf, count, ncclFloat, comm, 0);
      GPU_SYNC();
      auto t1 = clk::now();
      if (i >= warmup)
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    emit({"allgather", nbytes, median(ts)}, world, rank);

    ts.clear();
    for (int i = 0; i < warmup + iters; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      auto t0 = clk::now();
      ncclBroadcast(sbuf, rbuf, count, ncclFloat, 0, comm, 0);
      GPU_SYNC();
      auto t1 = clk::now();
      if (i >= warmup)
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    emit({"broadcast", nbytes, median(ts)}, world, rank);

    if (world >= 2 && want("sendrecv")) {
      ts.clear();
      for (int i = 0; i < warmup + iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        ncclGroupStart();
        if (rank == 0) {
          ncclSend(sbuf, count, ncclFloat, 1, comm, 0);
          ncclRecv(rbuf, count, ncclFloat, 1, comm, 0);
        } else if (rank == 1) {
          ncclRecv(rbuf, count, ncclFloat, 0, comm, 0);
          ncclSend(sbuf, count, ncclFloat, 0, comm, 0);
        }
        ncclGroupEnd();
        GPU_SYNC();
        auto t1 = clk::now();
        if (i >= warmup)
          ts.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      emit({"sendrecv", nbytes, median(ts)}, world, rank);
    }

    GPU_FREE(sbuf);
    GPU_FREE(rbuf);
  }
  ncclCommDestroy(comm);
#endif

  MPI_Finalize();
  return 0;
}
