# DLcomm hardening — status

Branch `hardening`. This document records what was implemented, what was
fixed, what is verified on hardware, and what remains open.

Evidence standard used throughout: a claim counts as verified only when it was
produced by a completed run on Aurora XPUs. Results from the CPU/gloo test
suite are development signal, not evidence — section 6 gives a concrete case
where gloo passed code that deadlocks on XCCL.

## 1. Features implemented

### 1.1 New collectives

The registry grew from 10 dispatchable collectives to 13.

| Collective | Kind | busbw factor | Notes |
|---|---|---|---|
| `alltoallv` | vector all-to-all | `(n-1)/n` | uneven splits via `all_to_all_single` |
| `sendrecv` | point-to-point | `1.0` | blocking, parity-ordered pairs |
| `sendrecv_async` | point-to-point | `2.0` | non-blocking `isend`/`irecv` |

Pairing for both point-to-point operations is `(0,1), (2,3), …`. On an
odd-sized group the final rank has no partner and records a **skip**, not a
pass.

Split sizes for `alltoallv` are deliberately skewed rather than equal, so a
implementation that ignores the split vector cannot pass. At `ws=4` the splits
are `[102, 204, 307, 411]`.

Each new collective has a real correctness checker that compares against a
computed expected value. None returns a bare "ran without raising".

### 1.2 Verification infrastructure

- Payload generation and failure recording split into `dl_comm/verify/`.
- Correctness handlers rewritten in `dl_comm/analysis/correctness.py`.
- A sabotage hook (`DLCOMM_SABOTAGE=1`) replaces a collective with a no-op so
  the detection path itself can be tested.
- `tests/test_correctness_detects_breakage.py` — 48 tests that break each
  collective deliberately and assert the checker reports a failure.

### 1.3 Diagnostics

- **Verdict barrier timeout** (`DLCOMM_VERDICT_TIMEOUT`, default 120 s): the
  final cross-rank reduction is guarded by a bounded non-blocking `Ibarrier`
  instead of blocking forever.
- **Global watchdog** (`DLCOMM_WATCHDOG`): a wall-clock budget that dumps every
  rank's Python stack via `faulthandler` and exits non-zero. This is what
  located the `sendrecv_async` deadlock to an exact line.

### 1.4 Test suite

149 tests locally. Files: `test_correctness_detects_breakage.py`,
`test_bandwidth.py`, `test_payload.py`, `test_timer_sync.py`,
`test_topology_validation.py`, `test_failures_and_results.py`,
`test_config_spec_matches_registry.py`.

## 2. Defects fixed

Each has a document under `docs/fixes/`.

| # | Defect | Doc |
|---|---|---|
| 1 | `torch.ones()` payloads made verification vacuous — a rank-independent payload cannot detect a wrong reduction | `01` |
| 2 | Bandwidth off by 256× — byte count used element count | `02` |
| 3 | `-ppn 12` on a 2×4 topology silently orphaned ranks `[4..11, 16..23]` and still exited 0 | `03` |
| 4 | Failures logged but not propagated to exit status | `04` |
| 5 | busbw used the wrong group size and a fixed factor | `05` |
| 6 | Packaging metadata, Apache-2.0 | `06` |
| 7 | Log directory creation failed silently | `07` |
| 8 | Unconditional `oneccl_bindings_for_pytorch` import | `08` |
| 9 | Teardown segfault destroyed the exit status | `09` |
| 10 | p2p and vector collectives absent | `10` |
| 11 | Verdict reduction could deadlock with no diagnostic | `11` |
| 12 | A hang produced no evidence at all | `12` |
| 13 | Pickle-based `MPI.allreduce` deadlocked after XCCL init | `13` |
| 14 | `results.json` reported `"enabled": false` on verified runs; the "requested but zero checks ran" guard could never fire | `14` |
| 15 | `sendrecv_async` deadlock on XCCL | `15` |

Two of these were pre-existing bugs found while adding features rather than
introduced by this work: `reducescatter` was missing from `config_spec.json`
despite being implemented and verified, and the collective lookup raised a bare
`KeyError` that named neither the offending config field nor the valid choices.

## 3. Verified on hardware

Job `8824490`, 2 nodes, 24 ranks, `frameworks/2025.3.1`, torch 2.10, XCCL.

```
A pytest      exit=0        140 passed, 2 skipped
B healthy     exit=0        [CORRECTNESS] checks=480 failures=0
C sabotaged   exit=1        [CORRECTNESS] checks=480 failures=480
VALIDATION: PASS
```

Run C is the load-bearing result: with `allreduce` replaced by a no-op, every
one of the 480 checks that passes in run B fails. Before this work the same
sabotage passed silently.

Bandwidth accounting was confirmed separately on job `8824234`:
`n=12 bytes=4194304 t_med=0.000419 algbw_med=1.000e+10 busbw_med=1.834e+10`,
matching the expected `2(n-1)/n = 1.8333` for allreduce.

## 4. Collective status

Verified means a completed hardware run reported correctness for it.

| Collective | Implemented | Checker | CPU/gloo | XPU/XCCL |
|---|---|---|---|---|
| `allreduce` | yes | yes | pass | **verified** (480 checks, job 8824490) |
| `allgather` | yes | yes | pass | not exercised |
| `alltoall` | yes | yes | pass | not exercised |
| `alltoallsingle` | yes | yes | pass | not exercised |
| `broadcast` | yes | yes | pass | not exercised |
| `gather` | yes | yes | pass | not exercised |
| `scatter` | yes | yes | pass | not exercised |
| `reduce` | yes | yes | pass | not exercised |
| `reducescatter` | yes | yes | pass | not exercised |
| `barrier` | yes | shape-only | pass | not exercised |
| `alltoallv` | yes | yes | pass | **ran, correctness unknown** |
| `sendrecv` | yes | yes | pass | **ran, correctness unknown** |
| `sendrecv_async` | yes | yes | pass | **deadlocks** |

### 4.1 Point-to-point detail

`sendrecv` completed on 24 ranks and produced consistent numbers across the two
node groups (job `8824561`):

```
group              n     bytes    t_med      algbw_med   busbw_med
(Within-Group-0)  12   4194304   0.000583    7.196e+09   7.196e+09
```

busbw factor `1.000000`, correct for a pairwise exchange.

`alltoallv` likewise:

```
(Within-Group-0)  12   4194288   0.001328    3.159e+09   2.896e+09
```

busbw factor `0.916667`, correct for `(n-1)/n` at `n=12`.

**"Correctness unknown" is deliberate wording.** Passing checks are printed only
in the end-of-run summary. Both jobs died in the third task before reaching it,
so no correctness verdict exists for these two collectives yet. They are not
known to be wrong; they are unproven.

### 4.2 `sendrecv_async` — open defect

Deadlocks on XCCL. The watchdog localised it exactly:

```
Timeout (0:05:00)!
  File ".../torch/distributed/distributed_c10d.py", line 2536 in irecv
  File ".../dl_comm/comm/collectives.py", line 416 in _send_recv_async
  File ".../dl_comm/dl_comm_main.py", line 618 in main
```

Line 618 is the warmup loop, so it hangs on the first call, before any timed
iteration.

Two attempts, both wrong:

| Job | Ordering | Result |
|---|---|---|
| `8824532` | both peers `isend` first | 22 ranks hang in `isend` |
| `8824561` | both peers `irecv` first | 22 ranks hang in `irecv` |

The second job disproved the first diagnosis. The hang simply moved to whichever
call came first, which rules out "the receive must be posted first" and points
instead at a symmetry problem: both peers issue the *same* call first, so
nothing can match. XCCL's `isend`/`irecv` evidently do not return before the
peer posts the matching operation.

Current patch — untested on hardware — orders by rank parity, as the working
blocking implementation does:

```python
if group_rank % 2 == 0:
    reqs = [dist.isend(...), dist.irecv(...)]
else:
    reqs = [dist.irecv(...), dist.isend(...)]
```

22 of 24 ranks hanging is expected, not a separate anomaly: the odd rank in each
odd-sized group returns early without entering the exchange.

## 5. Pending issues

1. **`sendrecv_async` deadlock.** Parity-ordering patch written, not yet run on
   hardware. Until a job completes, the fix is unproven.
2. **Correctness verdict for `alltoallv` and `sendrecv`.** Requires a run that
   survives all three tasks.
3. **Fix 14 never executed.** The `results.json` `"enabled"` correction is
   deployed on Aurora but no job has run the full verdict path since it landed.
4. **Defect 9 unproven.** The XCCL `destroy_process_group()` segfault
   (`rank 5 died from signal 11`) has not recurred since the teardown change,
   but absence of recurrence is not proof.
5. **`config_spec.json` is not enforced.** The validator reads `framework` and
   `backend` only; `spec["collective"]`, `spec["op"]` and `spec["dtype"]` are
   never consulted. The file reads like enforcement but is documentation.
   `tests/test_config_spec_matches_registry.py` pins it to the registry so the
   two cannot drift, but wiring it into real validation is a separate change.
6. **Nine collectives never exercised on XPUs.** Only `allreduce` appears in the
   hardware validation config. The others are CPU-verified only.
7. **Missing API coverage.** No `monitored_barrier`, no object-based variants,
   no fused `all_gather_into_tensor` / `reduce_scatter_tensor`, no
   `async_op=True` path.
8. **`barrier` has a shape-only checker.** It cannot fail, which by the standard
   used here means it is untested rather than passing.
9. **LICENSE copyright holder unset.** Apache-2.0 metadata is in place; the
   holder line needs a decision.
10. **Nothing pushed.** Four commits sit on local branch `hardening`; no PR.

## 6. Why the CPU suite is not sufficient

The `sendrecv_async` deadlock is the clearest case. The 149-test gloo suite
passes on all three orderings — the original, the first wrong fix, and the
current patch — because gloo buffers a send rather than requiring a matched
receive at enqueue time. No CPU-side test distinguishes working code from code
that hangs every rank on XCCL.

The same applies to fix 13: a pickle-based `MPI.COMM_WORLD.allreduce` works
locally and deadlocks after XCCL initialisation on Aurora.

Both defects were found only by running on hardware, and one of them was found
only because the watchdog dumped a stack.

## 7. Reproducing

```
# full three-phase validation
qsub validate_on_aurora.sh

# the three newer collectives on XPUs
qsub run_newcolls.sh
```

Artefacts are written to
`/lus/flare/projects/datascience/kaushik/DLcomm/validation/<timestamp>/`.
