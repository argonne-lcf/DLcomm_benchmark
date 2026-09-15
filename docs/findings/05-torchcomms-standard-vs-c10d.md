# torchcomms standard API vs c10d: reductions agree

**Status:** resolved; both paths measured correct on hardware (job 8826293)

## Question

DLcomm drives torchcomms through its native `TorchComm` API, not through
`torch.distributed`. Job 8826159 had measured a reduction returning the wrong
value, which raised the question of whether the standard path was at fault and
whether the c10d path would behave differently.

## Which API DLcomm uses

`TorchCommsDist` calls `torchcomms.new_comm(backend, device, name)` and issues
collectives directly on the returned `TorchComm`
(`dl_comm/comm/torchcomms_backend.py`). No `ProcessGroup` is involved:

- `init_process_group` on the facade is a no-op, present only so the rest of
  DLcomm has one call path.
- `TorchCommsGroup` is a DLcomm wrapper holding a communicator and a rank list.
  It is not a `ProcessGroup`, which is why `dist.get_world_size(group)` raises
  on it and why `_group_info` reads `group.ranks` directly.
- Subcommunicators use `new_comm` over a `PrefixStore`, because `split` hangs
  on this build.

The `c10d` in the upstream directory name `pytorch_c10d_torchcomms` refers to
the torch build the library was compiled against, not to the path DLcomm takes.

## Measurement

Job 8826293, one node, 12 ranks, one XPU tile per rank. Every rank contributes
`1.0` except rank 3, which contributes `0.0`, so each reduction has a distinct
expected value and a wrong operator cannot coincidentally produce a right
answer. Each op is run twice on identical input: once on the native
`TorchComm`, once on a c10d `ProcessGroup` with the XCCL backend.

| op | expected | standard `TorchComm` | c10d / XCCL |
|---|---|---|---|
| SUM | 11.00 | 11.00 | 11.00 |
| MIN | 0.00 | 0.00 | 0.00 |
| MAX | 1.00 | 1.00 | 1.00 |
| PRODUCT | 0.00 | 0.00 | 0.00 |

8 of 8 correct. The two APIs agree with each other and with the expected
values.

The same job also held all four `ReduceOp` members live simultaneously and
compared them: `SUM`, `MIN`, `MAX` and `PRODUCT` are mutually distinct.

## Conclusion

The standard torchcomms path is not defective, and it does not differ from
c10d for reductions. Both are usable; DLcomm's choice of the native API is not
a correctness risk.

## What the earlier wrong reading was

Two false conclusions were drawn before this measurement and are recorded here
so they are not repeated:

1. *"`SUM`, `MIN`, `MAX` and `AVG` are the same object."* This came from
   printing `id()`-style addresses of values fetched by four sequential
   `getattr` calls. Each temporary was freed before the next was created, so
   pybind11 reused the address. Identical addresses across sequential
   temporaries are not evidence of identical objects.
2. *"Every torchcomms reduction number is suspect."* Not supported. The
   incorrect reduction in job 8826159 was caused by DLcomm's own adapter: `_op`
   ended in `getattr(tc.ReduceOp, text, tc.ReduceOp.SUM)`, so any op name that
   failed to match was silently executed as a sum. That default has been
   removed and the op mapping is covered by
   `tests/test_torchcomms_reduce_op_mapping.py`.

## Reproducing

`tools/probe_standard_vs_c10d.py`, submitted with
`tools/probe_standard_vs_c10d.sh`. Note that the job may exit non-zero from the
teardown race documented in `docs/torchcomms-stack.md`; read the printed table,
not the exit status.
