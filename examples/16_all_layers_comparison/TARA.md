# Two-node Tara validation profile

The Aurora example's Intel oneCCL/SYCL layers are not applicable on Tara.
Use the CUDA counterparts with an eight-rank, two-node allocation (four
GH200 GPUs per node). Do not report unsupported layers as passed.

## Explicit NCCL protocol restriction

```bash
source examples/16_all_layers_comparison/tara_nccl_env.sh
```

This profile selects `NCCL_PROTO=Simple`, visibly excludes LL128, and refuses
unsafe/unknown protocol overrides. Explicit `LL` is accepted. In two-node
Tara validation, forced LL128 produced actual 16-float partial-sum corruption
in both PyTorch/NCCL and torchcomms/NCCL; forced LL and Simple passed the
same checks. This is a deployment restriction, **not a vendor root-cause
fix**. It is not silently applied on other platforms. Preserve the profile
message and effective environment with benchmark results.

## Native torchcomms gather

The adapter now executes CUDA blocking gather through native asynchronous
gather followed by an explicit work wait and device synchronization. Native
blocking gather previously stranded a peer after an allreduce/broadcast
sequence. This is **not** an all-gather fallback; traffic remains gather.
The synchronization can affect timing and must remain included in reported
blocking-call cost. Asynchronous callers still receive a work handle without
an eager wait. CPU and XPU blocking behavior remains unchanged.

## NIXL control rendezvous

With exactly two MPI ranks and a mpi4py build compatible with the launcher:

```bash
mpiexec -n 2 -ppn 1 python \
  examples/21_cuda_layer_sweeps/nixl_putget_bench.py \
  --rendezvous mpi --backend LIBFABRIC --op READ --mem VRAM \
  --sizes 1048576,2097152,4194304 --warmup 5 --iters 20 \
  --expect cross-node --json-out read-vram.json
```

Repeat with `--mem DRAM`. MPI moves control metadata only; NIXL still moves
payloads. This avoids the original NFS file rendezvous's long visibility
waits. File rendezvous remains the default and requires a fresh shared
`--sync-dir`. Supply a launcher timeout/job walltime for either mode.
Verification covers the whole buffer and corruption exits nonzero. The
constant pattern does not detect every possible permutation.

READ works with Tara's current LIBFABRIC/CXI NIXL build. **WRITE is still
unsupported by that build**; requests remain errors, not synthetic passes or
silent READ substitutions. Small-payload performance remains limited and is
not a performance-parity claim.

## Validation evidence

Two GH200 nodes, eight GPU ranks, float32, 1/2/4 MiB, five warmup and twenty
timed iterations. Native gather plus the explicit Simple profile:

- PyTorch/NCCL: 39/39 sequential tasks, 6,240 global checks, zero failures.
- torchcomms/NCCL: 39/39 sequential tasks, 6,240 global checks, zero failures.
- AWS Libfabric/CXI verified in logs, not Socket fallback.
- OSU host/CUDA: 20 cases passed; raw libfabric host pingpong: three sizes passed.
- CUDA transfers: five patterns passed.
- C++ NCCL ran with limited sampled preflight checks, not full certification.
- NIXL READ VRAM/DRAM: three sizes passed whole-buffer checks.
- Separate NIXL 1 GiB READ control passed at about 13 GB/s.

PBS jobs 11736 (native regression and protocol discriminator) and 11738
(all-layer regression) retained configurations, stdout and transport logs
under `/home/kaushikvelusamy/dlcomm-tara-fix-runs/results/` on Tara.
The all-layer harness's old wheel import probe rejects a source checkout;
that is a probe-scope mismatch, not a benchmark correctness failure. Source
hash verification is required when testing this branch instead of PyPI.

These are scoped tested fixes/profile restrictions, not a claim that every
backend operation or every payload size is certified. PyPI 0.4.1 does not
contain these branch changes.
