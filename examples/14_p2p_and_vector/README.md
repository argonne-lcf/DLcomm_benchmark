# Example 14 — point-to-point and vector collectives

Exercises the three operations added in fix 10: `sendrecv`, `sendrecv_async`
and `alltoallv`. One Aurora node, 12 XPU ranks.

## Contents

| File | Purpose |
|---|---|
| `14_p2p_and_vector_xccl.yaml` | configuration for the three operations |
| `jobscript_p2p_and_vector_xccl.sh` | PBS submission script |

## Running

```
qsub jobscript_p2p_and_vector_xccl.sh
```

Results are written to `logs/run_<timestamp>/`.

## Why these three are separated from the collective examples

Each has a property the collectives do not.

`sendrecv` pairs ranks by parity: `(0,1)`, `(2,3)`, and so on. An odd-sized
group leaves the final rank without a partner, and that rank records a skip
rather than a pass. A run reporting fewer checks than ranks is expected here.

`sendrecv_async` uses non-blocking `isend`/`irecv`. It deadlocks on the
torchcomms 0.1.0 stack that ships with `frameworks/2025.3.1`, and passes on
0.3.0. It is ordered last in `order_of_run` so that a hang does not discard the
other two results.

`alltoallv` uses deliberately skewed split sizes rather than equal ones. An
implementation that ignores the split vector produces the right total byte
count and the wrong distribution, which an even split would not detect.

## Watchdog

The jobscript sets `DLCOMM_WATCHDOG=600`. On expiry every rank's Python stack
is dumped through `faulthandler` and the job exits non-zero. Without it, a
`sendrecv_async` deadlock consumes the full walltime and produces no diagnostic
output. This is the mechanism that located the deadlock to an exact line; see
`docs/fixes/12`.
