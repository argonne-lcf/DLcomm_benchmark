# Tara deep validation follow-up

## Small-message NIXL READ

Select `--rendezvous mpi --progress-mode manual` explicitly for the tested
LIBFABRIC/CXI setup. This disables the background progress thread and drives
completion through the benchmark's bounded polling. Default `thread` remains
unchanged for compatibility. Effective mode is recorded in logs/JSON.

Two-node job 11744: 4 KiB/64 KiB/1 MiB READ took approximately 35–80 us in
manual mode versus 28–32 ms threaded, with whole-buffer verification passing
for DRAM and VRAM. This is a validated configuration improvement, not a
universal NIXL backend repair. READ/WRITE and backend support still differ;
WRITE remains unsupported in this installation.

## C++ correctness

The NCCL/RCCL path now checks entire outputs after every warmup and measured
iteration, outside timing. Exact rank/index/iteration-dependent inputs cover
allreduce, allgather, broadcast, reduce, reduce_scatter, sendrecv, and alltoall.
GPU/NCCL calls are checked; failures are aggregated across MPI ranks and cause
nonzero exit. Rows report CHECKS/FAILURES/VERIFIED_ITERS/STATUS.
`DLCOMM_INJECT_CORRUPTION=1` deliberately changes one participating output tail
and is for negative validation only; never use it in performance runs.

Job 11748 (two nodes/eight GPUs, five sizes, 25 verified iterations):
35 operation/size cases, 4,260,556,800 element checks, zero failures with Simple.
Injection produced exactly one failure and exit 1. Forced LL128 Ring produced
352 bad elements and exit 1, proving the issue without torch/Python.
RCCL and XCCL are not hardware-certified by these Tara runs; XCCL unchanged.

## LL128 remains unresolved

Installed stack: NCCL 2.29.7; aws-ofi-nccl v1.9.2-aws; CXI.
Ring/LL128 corruption reproduced in native C++. Socket, disabling GPU P2P,
host-staged CXI, and allreduce-only Tree passed diagnostic runs; none is a
vendor repair. Changing P2P read selection, disabling SHM, CUDA flush, and
SENDRECV did not resolve corruption. A passing default run was intermittent,
not evidence of repair. Global Tree also rejects unsupported operations;
do not recommend it as an all-layer setting. Keep the explicit Simple profile.

## Final all-layer regression

Job 11749: two GH200 nodes/eight GPUs, 6m56s; all supported cases passed.
Both Python collective backends: 39/39 tasks, 6,240 global checks, zero failures.
C++: 4,220,518,400 whole-output element checks, zero failures at 1/2/4 MiB.
OSU 20/20; CUDA five patterns; three host libfabric sizes.
NIXL manual READ whole-buffer checks passed all sizes:
VRAM 1/2/4 MiB: 54.96/96.03/182.64 us;
DRAM 1/2/4 MiB: 21.69/32.43/53.66 us.
The separate 1 GiB READ control measured 13.86 GB/s, not small-message parity.
Raw evidence: `/home/kaushikvelusamy/dlcomm-tara-fix-runs/results/` on Tara.
119 CSV rows include summaries: 114 PASS, three platform NA, two unsupported
WRITE rows. They are not 119 independent tests.
